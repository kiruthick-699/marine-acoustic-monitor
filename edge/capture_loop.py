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

_ENV_PARAMS = ("temperature_c", "ph", "turbidity_ntu", "salinity_psu")
_ROC_COLUMN_MAP = {
    "temperature_c": "temp_roc",
    "ph": "ph_roc",
    "turbidity_ntu": "turbidity_roc",
    "salinity_psu": "salinity_roc",
}


class CaptureLoop:
    """
    Runs one duty-cycle window per call to run_one_window(). edge/scheduler.py
    calls this repeatedly on the M-minute duty cycle; edge/main.py's --once
    mode calls it exactly once for a dry run / smoke test.

    Rate-of-change (environmental_readings' *_roc columns) is computed
    against the *previous call's* raw reading, held in memory
    (`self._prev_env_reading`). This means roc resets to 0 on process
    restart rather than resuming from the last DB row -- acceptable for a
    first implementation (a restart is already a break in the duty-cycle
    continuity the roc signal is meant to capture) but noted here as a
    known simplification; a future improvement would seed
    `_prev_env_reading` from the most recent environmental_readings row on
    startup.
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
        self._prev_env_reading: Optional[Dict[str, float]] = None

        os.makedirs(self._config.storage.audio_dir, exist_ok=True)

    def set_detector(self, detector: BaselineAnomalyDetector) -> None:
        """Install a fitted calibration baseline (edge/calibration.py's output)."""
        self._detector = detector

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
        """
        timestamp = datetime.now(timezone.utc)
        timestamp_utc = timestamp.isoformat()
        duration_s = self._config.duty_cycle.window_duration_s
        sample_rate = self._config.audio.sample_rate_hz

        # --- Step 1: wake, capture acoustic + environmental + IMU/health readings ---
        audio = self._hw.hydrophone.capture(duration_s, sample_rate)
        raw_env_reading = self._hw.env_sensors.read()
        imu_reading = self._hw.imu.read()
        power_reading = self._hw.power.read()
        env_row = self._with_rate_of_change(raw_env_reading)

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

        # --- Step 7: assemble + transmit compact telemetry payload ---
        payload = build_payload(
            timestamp_utc=timestamp_utc,
            acoustic_features=acoustic_features,
            environmental_row=env_row,
            anomaly_result=anomaly_result,
        )
        telemetry_sent = self._hw.telemetry.send(payload)

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
