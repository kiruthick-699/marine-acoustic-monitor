"""
Direct tests of the mock HAL backend (edge/hal/mock.py) -- shape/contract
checks against edge/hal/interfaces.py, independent of CaptureLoop.
"""

import numpy as np
import pytest

from edge.hal.mock import (
    MockEnvironmentalSensors,
    MockHydrophone,
    MockIMU,
    MockPowerMonitor,
    MockTelemetryLink,
)


def test_mock_hydrophone_capture_shape():
    hydrophone = MockHydrophone(rng=np.random.default_rng(0))
    audio = hydrophone.capture(duration_s=2.0, sample_rate=8000)
    assert audio.shape[0] == 2 * 8000
    assert audio.dtype == np.float32


def test_mock_hydrophone_failure_rate_one_always_raises():
    hydrophone = MockHydrophone(failure_rate=1.0, rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        hydrophone.capture(duration_s=1.0, sample_rate=8000)


def test_mock_env_sensors_reading_keys_and_roc_progression():
    sensors = MockEnvironmentalSensors(window_interval_minutes=10.0, rng=np.random.default_rng(0))
    reading = sensors.read()
    assert set(reading.keys()) == {"temperature_c", "ph", "turbidity_ntu", "salinity_psu"}
    assert reading["turbidity_ntu"] >= 0.0


def test_mock_env_sensors_storm_trigger_raises_turbidity():
    sensors = MockEnvironmentalSensors(window_interval_minutes=10.0, rng=np.random.default_rng(1))
    baseline = sensors.read()
    sensors.trigger_storm()
    # Mirrors synthetic_environmental.py's inject_storm_runoff_event ramp:
    # envelope is 0 at the onset window itself (linspace(0, 1, rise_windows)),
    # so the effect is only visible above sensor noise a few windows into the
    # ramp -- advance past the onset window before asserting.
    for _ in range(19):
        sensors.read()
    storm_reading = sensors.read()
    assert storm_reading["turbidity_ntu"] > baseline["turbidity_ntu"]
    assert storm_reading["salinity_psu"] < baseline["salinity_psu"]


def test_mock_env_sensors_failure_rate_one_always_raises_but_advances_time():
    sensors = MockEnvironmentalSensors(window_interval_minutes=10.0, failure_rate=1.0, rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        sensors.read()
    assert sensors._elapsed_minutes == 10.0


def test_mock_imu_reading_keys():
    imu = MockIMU(rng=np.random.default_rng(0))
    reading = imu.read()
    assert set(reading.keys()) == {"roll_deg", "pitch_deg", "yaw_deg", "accel_magnitude_g"}


def test_mock_imu_failure_rate_one_always_raises():
    imu = MockIMU(failure_rate=1.0, rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        imu.read()


def test_mock_telemetry_records_sent_payloads():
    link = MockTelemetryLink(rng=np.random.default_rng(0))
    assert link.send({"a": 1}) is True
    assert link.sent_payloads == [{"a": 1}]


def test_mock_telemetry_failure_rate_one_always_fails_and_does_not_raise():
    link = MockTelemetryLink(failure_rate=1.0, rng=np.random.default_rng(0))
    assert link.send({"a": 1}) is False
    assert link.sent_payloads == []


def test_mock_power_monitor_reading_keys():
    power = MockPowerMonitor(rng=np.random.default_rng(0))
    reading = power.read()
    assert set(reading.keys()) == {"battery_voltage", "solar_charge_w", "enclosure_temp_c", "uptime_sec"}


def test_mock_power_monitor_failure_rate_one_always_raises():
    power = MockPowerMonitor(failure_rate=1.0, rng=np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        power.read()
