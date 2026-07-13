"""
Production capture loop -- one duty-cycle wake window, per
docs/data-pipeline.md's "Duty-cycle sampling and near-real-time on-device
flow" (steps 1-8), run against a HardwareBundle (edge/hal/factory.py)
instead of synthetic generators.

This intentionally reuses simulation/pipeline/{signal_conditioning,
feature_extraction, anomaly_detection, storage}.py unchanged -- that code
was never simulation-specific, only simulation/data_generator/ stands in for
real hardware (see edge/__init__.py). CaptureLoop is the orchestration glue
between the HAL (edge/hal/) and that already-built, already-tested pipeline
code.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from scipy.io import wavfile

from edge.config import EdgeConfig
from edge.hal.factory import HardwareBundle
from edge.telemetry import build_payload
from simulation.pipeline.anomaly_detection import BaselineAnomalyDetector
from simulation.pipeline.feature_extraction import (
    build_joint_feature_vector,
    extract_acoustic_features,
    extract_environmental_features,
)
from simulation.pipeline.signal_conditioning import bandpass_filter, spectral_denoise
from simulation.pipeline.storage import insert_system_health_log, insert_window_record

logger = logging.getLogger(__name__)

_ENV_PARAMS = ("temperature_c", "ph", "turbidity_ntu", "salinity_psu")
_ROC_COLUMN_MAP = {
    "temperature_c": "temp_roc",
    "ph": "ph_roc",
    "turbidity_ntu": "turbidity_roc",
    "salinity_psu": "salinity_roc",
}

# Fallback readings used when a non-critical sensor (env/IMU/power) fails --
# keep the window completing (audio capture + storage still happen) rather
# than losing the whole window over e.g. one I2C hiccup. Environmental
# fallback prefers the previous window's raw reading (see run_one_window);
# these are the last resort when there's no previous reading yet (first
# window of a run).
_DEFAULT_ENV_READING = {"temperature_c": 0.0, "ph": 0.0, "turbidity_ntu": 0.0, "salinity_psu": 0.0}
_DEFAULT_IMU_READING = {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "accel_magnitude_g": 1.0}
_DEFAULT_POWER_READING = {"battery_voltage": 0.0, "solar_charge_w": 0.0, "enclosure_temp_c": 0.0, "uptime_sec": 0}


class WindowCaptureError(Exception):
    """
    Raised when run_one_window() cannot complete a duty-cycle window.

    Wraps whatever underlying exception (HAL driver, disk I/O, storage)
    caused the failure, so callers (edge/scheduler.py) can catch one type
    instead of every possible HAL/library exception. Per DECISIONS.md's
    duty-cycle model, a failed window is logged and skipped -- not retried
    mid-cycle -- so there's no sub-type here: every caller reacts the same
    way, by moving on to the next scheduled wake window.
    """


class CaptureLoop:
    """
    Runs one duty-cycle window per call to run_one_window(). edge/scheduler.py
    calls this repeatedly on the M-minute duty cycle; edge/main.py's --once
    mode calls it exactly once for a dry run / smoke test.

    Rate-of-change (environmental_readings' *_roc columns) is computed
    against the *previous call's* raw reading, held in memory
    (`self._prev_env_reading`). On construction this is seeded from the
    most recently stored environmental_readings row (see
    _load_last_env_reading()), so a process restart resumes roc continuity
    from the last real reading instead of resetting to 0 for one window --
    None only for a genuinely fresh database with no prior rows.
    """

    def __init__(
        self,
        config: EdgeConfig,
        hardware: HardwareBundle,
        db_conn,
        detector: Optional[BaselineAnomalyDetector] = None,
    ):
        self._config = config
        self._hw = hardware
        self._db = db_conn
        self._detector = detector
        self._prev_env_reading: Optional[Dict[str, float]] = self._load_last_env_reading()

        os.makedirs(self._config.storage.audio_dir, exist_ok=True)

    def _load_last_env_reading(self) -> Optional[Dict[str, float]]:
        """
        Seed rate-of-change tracking from the most recently stored
        environmental_readings row, joined through captures for correct
        timestamp ordering (environmental_readings itself has no timestamp
        column -- see docs/data-pipeline.md's schema). Returns None if the
        table is empty (first-ever run against a fresh database).
        """
        row = self._db.execute(
            """
            SELECT e.temperature_c, e.ph, e.turbidity_ntu, e.salinity_psu
            FROM environmental_readings e
            JOIN captures c ON c.capture_id = e.capture_id
            ORDER BY c.timestamp_utc DESC, c.capture_id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return dict(zip(_ENV_PARAMS, row))

    def set_detector(self, detector: BaselineAnomalyDetector) -> None:
        """Install a fitted calibration baseline (edge/calibration.py's output)."""
        self._detector = detector

    def _safe_read(self, name: str, read_fn, fallback: Dict[str, float]) -> Dict[str, float]:
        """
        Call one non-critical HAL read (env/IMU/power), logging a warning
        and returning `fallback` instead of raising if it fails -- so a
        single failed sensor doesn't stop audio capture and storage from
        still happening this window.
        """
        try:
            return read_fn()
        except Exception as exc:
            logger.warning("%s read failed, using fallback reading: %s", name, exc)
            return dict(fallback)

    def _with_rate_of_change(self, reading: Dict[str, float]) -> Dict[str, float]:
        row = dict(reading)
        for param in _ENV_PARAMS:
            roc_col = _ROC_COLUMN_MAP[param]
            if self._prev_env_reading is None:
                row[roc_col] = 0.0
            else:
                row[roc_col] = reading[param] - self._prev_env_reading[param]
        return row

    def _condition(self, audio, sample_rate: int):
        # Mirrors simulation/pipeline/signal_conditioning.py's
        # condition_signal() two-step sequence (bandpass then spectral
        # denoise), but parameterized by edge/config.yaml's
        # signal_conditioning block instead of condition_signal()'s
        # hardcoded high_hz=12000 default -- lets a deployment site retune
        # the passband without touching pipeline code.
        cfg = self._config.signal_conditioning
        filtered = bandpass_filter(audio, sample_rate, low_hz=cfg.bandpass_low_hz, high_hz=cfg.bandpass_high_hz)
        return spectral_denoise(filtered, sample_rate)

    def run_one_window(self) -> Dict:
        """
        Execute one full wake window: capture -> condition -> extract ->
        detect -> store -> telemeter -> (implicit sleep, handled by the
        caller -- edge/scheduler.py or --once mode).

        Returns:
            dict summary: capture_id, timestamp_utc, audio_filename,
            anomaly_score, is_anomaly, telemetry_sent (bool),
            calibrated (bool -- False if no detector installed yet, in
            which case anomaly_score/is_anomaly are placeholders, matching
            simulation/scripts/run_simulation.py's placeholder-row
            approach for the same not-yet-scored case).

        Raises:
            WindowCaptureError: the window could not be completed (audio
            capture or storage failed). Non-critical sensor reads
            (env/IMU/power) and telemetry send never raise -- they degrade
            to a fallback reading / a failed-send result instead, per
            docs/data-pipeline.md's duty-cycle model (a failed window is
            logged and skipped, not retried mid-cycle).
        """
        timestamp = datetime.now(timezone.utc)
        timestamp_utc = timestamp.isoformat()
        duration_s = self._config.duty_cycle.window_duration_s
        sample_rate = self._config.audio.sample_rate_hz

        # --- Step 1: wake, capture acoustic + environmental + IMU/health readings ---
        # Audio capture has no fallback: without it there's nothing to
        # condition, score, or store, so its failure is unrecoverable.
        try:
            audio = self._hw.hydrophone.capture(duration_s, sample_rate)
        except Exception as exc:
            raise WindowCaptureError(f"hydrophone capture failed: {exc}") from exc

        raw_env_reading = self._safe_read(
            "env_sensors", self._hw.env_sensors.read, self._prev_env_reading or _DEFAULT_ENV_READING
        )
        imu_reading = self._safe_read("imu", self._hw.imu.read, _DEFAULT_IMU_READING)
        power_reading = self._safe_read("power", self._hw.power.read, _DEFAULT_POWER_READING)
        env_row = self._with_rate_of_change(raw_env_reading)

        try:
            # --- Step 2: Tier 1 -- raw audio to a flat file, named by capture timestamp ---
            audio_filename = f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.wav"
            audio_path = os.path.join(self._config.storage.audio_dir, audio_filename)
            wavfile.write(audio_path, sample_rate, audio)

            # --- Step 3: Stage 0 signal conditioning ---
            conditioned_audio = self._condition(audio, sample_rate)

            # --- Step 4: Stage 1 feature extraction + joint vector ---
            acoustic_features = extract_acoustic_features(conditioned_audio, sample_rate)
            environmental_features = extract_environmental_features(env_row)
            joint_vector = build_joint_feature_vector(acoustic_features, environmental_features)

            # --- Step 5: Stage 2 unsupervised anomaly detection ---
            calibrated = self._detector is not None
            if calibrated:
                anomaly_result = self._detector.score(joint_vector)
            else:
                # No calibration baseline installed yet (edge/calibration.py not
                # run) -- write an unscored placeholder rather than block
                # capture/storage on calibration being complete, matching
                # simulation/scripts/run_simulation.py's same placeholder
                # pattern for its (deliberately deferred) detection step.
                anomaly_result = {"anomaly_score": 0.0, "is_anomaly": False}

            # --- Step 6: Tier 2 -- structured data to SQLite ---
            capture_id = insert_window_record(
                self._db,
                timestamp_utc=timestamp_utc,
                audio_filename=audio_filename,
                duration_sec=duration_s,
                sample_rate_hz=sample_rate,
                acoustic_features=acoustic_features,
                environmental_row=env_row,
                anomaly_result=anomaly_result,
                feature_vector_version=self._config.calibration.feature_vector_version,
                baseline_version=self._config.calibration.baseline_version,
            )
            insert_system_health_log(
                self._db,
                timestamp_utc=timestamp_utc,
                battery_voltage=power_reading["battery_voltage"],
                solar_charge_w=power_reading["solar_charge_w"],
                enclosure_temp_c=power_reading["enclosure_temp_c"],
                imu_orientation=imu_reading,
                uptime_sec=power_reading["uptime_sec"],
            )
        except Exception as exc:
            raise WindowCaptureError(f"window processing/storage failed: {exc}") from exc

        # --- Step 7: assemble + transmit compact telemetry payload ---
        payload = build_payload(
            timestamp_utc=timestamp_utc,
            acoustic_features=acoustic_features,
            environmental_row=env_row,
            anomaly_result=anomaly_result,
        )
        try:
            telemetry_sent = self._hw.telemetry.send(payload)
        except Exception as exc:
            # TelemetryLink.send()'s contract is "return False, never raise"
            # (edge/hal/interfaces.py) -- an unexpected raise is treated the
            # same as a returned False rather than failing a window whose
            # audio/features/anomaly score are already safely stored.
            logger.warning("telemetry send failed, treating as failed send this cycle: %s", exc)
            telemetry_sent = False

        self._prev_env_reading = raw_env_reading

        # --- Step 8: sleep until next wake window -- caller's responsibility ---
        return {
            "capture_id": capture_id,
            "timestamp_utc": timestamp_utc,
            "audio_filename": audio_filename,
            "anomaly_score": anomaly_result["anomaly_score"],
            "is_anomaly": anomaly_result["is_anomaly"],
            "telemetry_sent": telemetry_sent,
            "calibrated": calibrated,
            # Not part of the on-device schema/telemetry -- exposed so
            # edge/calibration.py can collect this window's joint vector
            # for baseline fitting without re-running Stages 0/1.
            "feature_vector": joint_vector,
        }
