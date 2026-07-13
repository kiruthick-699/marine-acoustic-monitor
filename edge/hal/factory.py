"""
HAL factory: builds a HardwareBundle from an EdgeConfig.

The single place that knows how to translate `hardware.mode` ("mock" |
"real") into concrete HAL implementations. Everything downstream
(edge/capture_loop.py, edge/scheduler.py, edge/main.py) depends only on
HardwareBundle's interface-typed fields, never on edge.hal.mock or
edge.hal.real directly -- so adding a third mode later (e.g. a
"replay-from-recorded-file" backend for regression testing against a real
past deployment) only touches this one function.
"""

import dataclasses

from edge.config import EdgeConfig
from edge.hal.interfaces import (
    EnvironmentalSensors,
    HydrophoneSource,
    IMUSensor,
    PowerMonitor,
    TelemetryLink,
)


@dataclasses.dataclass
class HardwareBundle:
    hydrophone: HydrophoneSource
    env_sensors: EnvironmentalSensors
    imu: IMUSensor
    telemetry: TelemetryLink
    power: PowerMonitor


def build_hardware(config: EdgeConfig) -> HardwareBundle:
    """
    Construct a HardwareBundle matching `config.hardware.mode`.

    Args:
        config: loaded EdgeConfig (edge/config.py).

    Returns:
        HardwareBundle wired to either edge/hal/mock.py (mode="mock", no
        hardware required) or edge/hal/real.py (mode="real", raises
        NotImplementedError on first use until real drivers are written).

    Raises:
        ValueError: unknown hardware.mode.
    """
    if config.hardware.mode == "mock":
        from edge.hal.mock import (
            MockEnvironmentalSensors,
            MockHydrophone,
            MockIMU,
            MockPowerMonitor,
            MockTelemetryLink,
        )

        return HardwareBundle(
            hydrophone=MockHydrophone(),
            env_sensors=MockEnvironmentalSensors(
                window_interval_minutes=config.duty_cycle.window_interval_minutes
            ),
            imu=MockIMU(),
            telemetry=MockTelemetryLink(),
            power=MockPowerMonitor(),
        )

    if config.hardware.mode == "real":
        from edge.hal.real import (
            RealEnvironmentalSensors,
            RealHydrophone,
            RealIMU,
            RealPowerMonitor,
            RealTelemetryLink,
        )

        return HardwareBundle(
            hydrophone=RealHydrophone(config),
            env_sensors=RealEnvironmentalSensors(config),
            imu=RealIMU(config),
            telemetry=RealTelemetryLink(config),
            power=RealPowerMonitor(config),
        )

    raise ValueError(
        f"Unknown hardware.mode {config.hardware.mode!r} -- expected 'mock' or 'real' (edge/config.yaml)"
    )
