"""
Hardware Abstraction Layer interfaces.

Five abstract interfaces, one per component role in docs/hardware-spec.md's
component list. edge/capture_loop.py, edge/telemetry.py, and edge/scheduler.py
are written entirely against these interfaces -- never against a concrete
mock or real implementation directly -- so the same orchestration code runs
unchanged whether edge/hal/mock.py or a future edge/hal/real/*.py backend is
selected (edge/config.yaml: hardware.mode).

Each method's docstring states the exact shape callers can rely on, since
that contract is what keeps mock and real implementations interchangeable.
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class HydrophoneSource(ABC):
    """Acoustic sensor (docs/hardware-spec.md: Hydrophone / ADC)."""

    @abstractmethod
    def capture(self, duration_s: float, sample_rate: int) -> np.ndarray:
        """
        Capture `duration_s` seconds of hydrophone audio at `sample_rate`.

        Returns:
            1D float32 array of length round(duration_s * sample_rate),
            amplitude scaled the same way simulation/data_generator/
            synthetic_audio.py's output is (roughly unit-scale float, not
            raw ADC codes) -- signal_conditioning.py and feature_extraction.py
            both assume this scale.
        """


class EnvironmentalSensors(ABC):
    """Temperature/pH/turbidity/salinity sensors (docs/hardware-spec.md)."""

    @abstractmethod
    def read(self) -> Dict[str, float]:
        """
        Take one instantaneous reading from all four environmental sensors.

        Returns:
            dict with exactly the keys temperature_c, ph, turbidity_ntu,
            salinity_psu (raw absolute values -- NOT normalized, NOT
            including rate-of-change; edge/capture_loop.py computes roc
            against the previous window's stored reading and
            feature_extraction.extract_environmental_features() does the
            normalization). Matches docs/data-pipeline.md's
            environmental_readings absolute-value columns.
        """


class IMUSensor(ABC):
    """IMU (docs/hardware-spec.md: orientation/motion, system-health context)."""

    @abstractmethod
    def read(self) -> Dict[str, float]:
        """
        Take one instantaneous orientation/motion reading.

        Returns:
            dict with roll_deg, pitch_deg, yaw_deg, accel_magnitude_g --
            serialized as JSON into system_health_log.imu_orientation
            (docs/data-pipeline.md) by edge/capture_loop.py. Not part of the
            core acoustic anomaly-detection feature vector (DECISIONS.md /
            hardware-spec.md: IMU is system-health/context only, unless
            later shown useful).
        """


class TelemetryLink(ABC):
    """Low-bandwidth link: LoRa or low-bandwidth cellular (docs/hardware-spec.md)."""

    @abstractmethod
    def send(self, payload: Dict) -> bool:
        """
        Transmit one compact telemetry payload (edge/telemetry.py's
        build_payload() output). Never called with raw audio -- DECISIONS.md
        locks "raw audio never transmitted" at the architecture level, so
        capture_loop.py never constructs a payload containing it.

        Returns:
            True if the transmission succeeded, False otherwise (e.g. link
            unavailable, timeout). A False return must not raise -- callers
            treat a failed send as "retry next duty cycle", per the
            duty-cycle model in DECISIONS.md, not a fatal error mid-loop.
        """


class PowerMonitor(ABC):
    """Power/system-health sensing (docs/hardware-spec.md: power draw / enclosure)."""

    @abstractmethod
    def read(self) -> Dict[str, float]:
        """
        Take one instantaneous system-health reading.

        Returns:
            dict with battery_voltage, solar_charge_w, enclosure_temp_c,
            uptime_sec -- matching system_health_log's columns in
            docs/data-pipeline.md exactly (imu_orientation is added
            separately by capture_loop.py from IMUSensor.read()).
        """
