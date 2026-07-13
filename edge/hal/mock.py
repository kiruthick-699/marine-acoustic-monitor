"""
Mock HAL backend -- every interface in edge/hal/interfaces.py implemented
with realistic stand-in data, zero real hardware required.

This is what lets the *entire* production orchestration (edge/capture_loop.py,
edge/telemetry.py, edge/calibration.py, edge/scheduler.py) be built, run, and
unit-tested today, before hardware-spec.md's BOM is priced and bought
(DECISIONS.md: no hardware purchased yet). Swapping edge/config.yaml's
`hardware.mode` from "mock" to "real" is the only change needed once real
edge/hal/real/*.py drivers exist -- nothing above the HAL layer changes.

MockHydrophone reuses simulation/data_generator/synthetic_audio.py directly.
MockEnvironmentalSensors reproduces the same diel-cycle math as
simulation/data_generator/synthetic_environmental.py, but as a single
stateful per-call reading (that generator produces a whole DataFrame series
at once; a real sensor bus is instead polled once per wake window). Both are
deliberately kept close to those synthetic generators so evaluate.py's
already-validated methodology carries over to end-to-end edge/ testing.
"""

import time
from typing import Dict, Optional

import numpy as np

from edge.hal.interfaces import (
    EnvironmentalSensors,
    HydrophoneSource,
    IMUSensor,
    PowerMonitor,
    TelemetryLink,
)
from simulation.data_generator.synthetic_audio import generate_duty_cycle_sample

MINUTES_PER_DAY = 24 * 60


class MockHydrophone(HydrophoneSource):
    """
    Generates synthetic hydrophone audio per capture(), with a configurable
    chance of injecting a biological call or vessel event -- same event
    types and probabilities philosophy as
    simulation/scripts/run_simulation.py's BIOLOGICAL_CALL_PROBABILITY /
    VESSEL_EVENT_PROBABILITY, so a mock-mode run exercises the same anomaly
    mix a real deployment's calibration/evaluation would.
    """

    def __init__(
        self,
        biological_call_probability: float = 0.12,
        vessel_event_probability: float = 0.08,
        failure_rate: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self._bio_p = biological_call_probability
        self._vessel_p = vessel_event_probability
        self._failure_rate = failure_rate
        self._rng = rng or np.random.default_rng()
        self.last_metadata: Optional[dict] = None

    def capture(self, duration_s: float, sample_rate: int) -> np.ndarray:
        # HydrophoneSource has no "failed" return value (unlike
        # TelemetryLink.send()'s bool), so a simulated failure raises --
        # edge/capture_loop.py treats this as unrecoverable for the window.
        if self._rng.random() < self._failure_rate:
            raise RuntimeError("MockHydrophone: simulated capture failure")

        roll = self._rng.random()
        if roll < self._bio_p:
            anomaly = "biological"
        elif roll < self._bio_p + self._vessel_p:
            anomaly = "vessel"
        else:
            anomaly = None

        audio, metadata = generate_duty_cycle_sample(
            duration_s=duration_s, sample_rate=sample_rate, inject_anomaly=anomaly
        )
        self.last_metadata = metadata
        return audio


class MockEnvironmentalSensors(EnvironmentalSensors):
    """
    Stateful mock: each read() advances an internal window counter by
    `window_interval_minutes` and returns a single diel-cycled reading,
    optionally with a storm/runoff event superimposed once
    `trigger_storm()` has been called -- mirroring
    synthetic_environmental.py's baseline + inject_storm_runoff_event()
    shapes, but evaluated pointwise per call instead of as a batch
    DataFrame (a real sensor bus has no "whole series" to generate ahead of
    time -- it's polled once per wake window).
    """

    def __init__(
        self,
        window_interval_minutes: float = 10.0,
        failure_rate: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self._window_interval_minutes = window_interval_minutes
        self._failure_rate = failure_rate
        self._elapsed_minutes = 0.0
        self._rng = rng or np.random.default_rng()
        self._storm_onset_minutes: Optional[float] = None

    def trigger_storm(self) -> None:
        """Arm a storm/runoff event starting at the next read()."""
        self._storm_onset_minutes = self._elapsed_minutes

    def read(self) -> Dict[str, float]:
        # Time still advances on a failed read (a missed sample, not a
        # rewind) so the diel cycle and storm ramp stay correct once reads
        # resume; see edge/capture_loop.py's fallback-reading handling.
        if self._rng.random() < self._failure_rate:
            self._elapsed_minutes += self._window_interval_minutes
            raise RuntimeError("MockEnvironmentalSensors: simulated read failure")

        time_of_day_frac = (self._elapsed_minutes % MINUTES_PER_DAY) / MINUTES_PER_DAY
        peak_frac = 14.0 / 24.0
        diel_phase = 2 * np.pi * (time_of_day_frac - 0.25 + peak_frac)

        temperature_c = 18.0 + 2.5 * np.sin(diel_phase) + self._rng.normal(0, 0.15)
        ph = 8.05 + 0.08 * np.sin(diel_phase) + self._rng.normal(0, 0.01)
        turbidity_ntu = max(0.0, 3.0 + self._rng.normal(0, 0.3))
        salinity_psu = 35.0 + self._rng.normal(0, 0.05)

        if self._storm_onset_minutes is not None:
            elapsed_since_onset = self._elapsed_minutes - self._storm_onset_minutes
            # ~3 day recovery, same real-world timescale as
            # synthetic_environmental.py's inject_storm_runoff_event()
            recovery_minutes = 3 * MINUTES_PER_DAY
            if 0 <= elapsed_since_onset <= recovery_minutes:
                progress = elapsed_since_onset / recovery_minutes
                envelope = min(1.0, progress / 0.1) if progress < 0.1 else np.exp(-((progress - 0.1) / 0.3))
                turbidity_ntu = max(0.0, turbidity_ntu + 27.0 * envelope)
                salinity_psu -= 4.0 * envelope
                ph -= 0.2 * envelope
            else:
                self._storm_onset_minutes = None  # event finished, clear it

        self._elapsed_minutes += self._window_interval_minutes

        return {
            "temperature_c": float(temperature_c),
            "ph": float(ph),
            "turbidity_ntu": float(turbidity_ntu),
            "salinity_psu": float(salinity_psu),
        }


class MockIMU(IMUSensor):
    """
    Gentle simulated buoy motion -- small roll/pitch oscillation, near-zero
    yaw drift, accel magnitude close to 1g -- enough to exercise
    system_health_log's imu_orientation storage without claiming to model
    real wave dynamics (hardware-spec.md scopes the IMU as system-health/
    context, not physically-accurate motion simulation).
    """

    def __init__(self, failure_rate: float = 0.0, rng: Optional[np.random.Generator] = None):
        self._failure_rate = failure_rate
        self._rng = rng or np.random.default_rng()
        self._t = 0.0

    def read(self) -> Dict[str, float]:
        self._t += 1.0
        if self._rng.random() < self._failure_rate:
            raise RuntimeError("MockIMU: simulated read failure")

        roll_deg = 3.0 * np.sin(self._t * 0.3) + self._rng.normal(0, 0.5)
        pitch_deg = 2.0 * np.sin(self._t * 0.21) + self._rng.normal(0, 0.5)
        yaw_deg = (self._rng.normal(0, 0.2) + self._t * 0.01) % 360
        accel_magnitude_g = 1.0 + self._rng.normal(0, 0.02)

        return {
            "roll_deg": float(roll_deg),
            "pitch_deg": float(pitch_deg),
            "yaw_deg": float(yaw_deg),
            "accel_magnitude_g": float(accel_magnitude_g),
        }


class MockTelemetryLink(TelemetryLink):
    """
    Records every payload passed to send() in `.sent_payloads` (for test
    assertions) instead of actually transmitting. `failure_rate` lets tests
    exercise the "failed send this cycle" path capture_loop.py must handle
    without raising (see TelemetryLink.send()'s contract).
    """

    def __init__(self, failure_rate: float = 0.0, rng: Optional[np.random.Generator] = None):
        self._failure_rate = failure_rate
        self._rng = rng or np.random.default_rng()
        self.sent_payloads = []

    def send(self, payload: Dict) -> bool:
        if self._rng.random() < self._failure_rate:
            return False
        self.sent_payloads.append(payload)
        return True


class MockPowerMonitor(PowerMonitor):
    """
    Plausible battery/solar/enclosure readings: battery voltage slowly
    drains and recharges on a rough diel solar cycle, enclosure temp tracks
    ambient with the Pi's own heat added, uptime increments in wall-clock
    seconds from construction.
    """

    def __init__(self, failure_rate: float = 0.0, rng: Optional[np.random.Generator] = None):
        self._failure_rate = failure_rate
        self._rng = rng or np.random.default_rng()
        self._start_time = time.monotonic()
        self._elapsed_minutes = 0.0

    def read(self) -> Dict[str, float]:
        self._elapsed_minutes += 1.0
        if self._rng.random() < self._failure_rate:
            raise RuntimeError("MockPowerMonitor: simulated read failure")

        time_of_day_frac = (self._elapsed_minutes % MINUTES_PER_DAY) / MINUTES_PER_DAY
        solar_charge_w = max(0.0, 8.0 * np.sin(2 * np.pi * (time_of_day_frac - 0.25)))
        battery_voltage = 12.6 + 0.4 * np.sin(2 * np.pi * time_of_day_frac) + self._rng.normal(0, 0.02)
        enclosure_temp_c = 22.0 + 3.0 * np.sin(2 * np.pi * (time_of_day_frac - 0.25)) + self._rng.normal(0, 0.3)

        return {
            "battery_voltage": float(battery_voltage),
            "solar_charge_w": float(solar_charge_w),
            "enclosure_temp_c": float(enclosure_temp_c),
            "uptime_sec": int(time.monotonic() - self._start_time),
        }
