"""
Real hardware HAL backend -- stub drivers implementing edge/hal/interfaces.py
against actual GPIO/I2C/SPI hardware.

Status: NOT IMPLEMENTED. This module exists so the wiring is already in
place -- correct class names, correct interface contracts, correct config
plumbing (I2C addresses, bus numbers) -- for the day hardware-spec.md's BOM
is priced, bought, and wired per docs/pi-implementation.md's address plan.
Every method below raises NotImplementedError with a comment on exactly what
driver work replaces it; nothing here is a guess at final part numbers,
since hardware-spec.md deliberately left ADC/bus choice as "deferred to
implementation".

Hardware libraries (smbus2, spidev, RPi.GPIO, an ADC vendor SDK, a LoRa
serial/SPI driver) are imported lazily inside each class's __init__, not at
module level -- so `import edge.hal.real` never fails on a dev machine
without those packages installed, and edge/main.py can still list this
module's classes for --mode=real wiring even before hardware exists.

Selection of which backend edge/main.py instantiates is entirely driven by
edge/config.yaml's `hardware.mode` ("mock" | "real") -- see edge/config.py.
"""

from typing import Dict

import numpy as np

from edge.config import EdgeConfig
from edge.hal.interfaces import (
    EnvironmentalSensors,
    HydrophoneSource,
    IMUSensor,
    PowerMonitor,
    TelemetryLink,
)


class RealHydrophone(HydrophoneSource):
    """
    TODO once hardware exists: hydrophone -> preamp -> ADC -> Pi.

    hardware-spec.md: "Interface to the Pi is most likely I2S or SPI, not
    USB" -- bus choice deferred to implementation. Concretely this needs:
      1. Select ADC part (e.g. an I2S MEMS-style ADC breakout, or an SPI ADC
         like an ADS1256/similar high-rate 16-24 bit part) once BOM pricing
         (docs/hardware-spec.md -> priced BOM) is done.
      2. Configure the corresponding kernel driver (I2S: dtoverlay in
         /boot/config.txt + ALSA capture via e.g. `sounddevice`/`pyaudio`;
         SPI: `spidev` + manual sample clocking) at `sample_rate_hz` from
         edge/config.yaml.
      3. Implement capture() to block for `duration_s`, return a 1D float32
         array normalized to the same rough amplitude scale
         signal_conditioning.py / feature_extraction.py assume (see
         edge/hal/interfaces.py's docstring).
    """

    def __init__(self, config: EdgeConfig):
        self._config = config
        # e.g.: import sounddevice  (I2S/ALSA path)  -- or --  import spidev  (SPI ADC path)

    def capture(self, duration_s: float, sample_rate: int) -> np.ndarray:
        raise NotImplementedError(
            "RealHydrophone.capture() -- implement once the hydrophone/ADC is wired; "
            "see docs/pi-implementation.md's wiring plan and hardware-spec.md's "
            "'Hydrophone / ADC' interface notes."
        )


class RealEnvironmentalSensors(EnvironmentalSensors):
    """
    TODO once hardware exists: temperature/pH/turbidity/salinity over I2C.

    hardware-spec.md: "I2C is preferred where available for multi-sensor
    sharing over one bus... with attention to address collisions if
    multiple sensors share default addresses." Concretely:
      1. Wire each sensor per the address plan in docs/pi-implementation.md
         (config.yaml: hardware.i2c.env_sensor_addresses).
      2. Use smbus2 (or the sensor vendor's Python SDK, e.g. a DFRobot
         Gravity library) to read each sensor's register(s) and convert raw
         ADC counts to physical units per that sensor's datasheet.
      3. Sensors without native I2C output route through the auxiliary ADC
         per hardware-spec.md -- read via the same or a secondary ADC
         channel instead of a dedicated I2C register in that case.
    """

    def __init__(self, config: EdgeConfig):
        self._config = config
        # e.g.: import smbus2; self._bus = smbus2.SMBus(config.hardware.i2c_bus_number)

    def read(self) -> Dict[str, float]:
        raise NotImplementedError(
            "RealEnvironmentalSensors.read() -- implement once env sensors are wired; "
            "see docs/pi-implementation.md's I2C address plan and hardware-spec.md's "
            "'Environmental sensor bus' interface notes."
        )


class RealIMU(IMUSensor):
    """
    TODO once hardware exists: IMU over I2C (or SPI) -- hardware-spec.md:
    "standard I2C or SPI devices; low pin/power overhead, straightforward to
    share the same I2C bus as environmental sensors if address space
    allows." Concretely, read raw accel/gyro (and mag, if the chosen IMU
    has one) registers and convert to roll/pitch/yaw + accel magnitude per
    the IMU vendor's fusion algorithm or a simple complementary filter.
    """

    def __init__(self, config: EdgeConfig):
        self._config = config
        # e.g.: import smbus2; self._bus = smbus2.SMBus(config.hardware.i2c_bus_number)

    def read(self) -> Dict[str, float]:
        raise NotImplementedError(
            "RealIMU.read() -- implement once the IMU is wired; see "
            "docs/pi-implementation.md's I2C address plan."
        )


class RealTelemetryLink(TelemetryLink):
    """
    TODO once hardware exists: LoRa or low-bandwidth cellular module.
    DECISIONS.md: "Choice between LoRa and low-bandwidth cellular is a
    deployment-specific decision... not fixed at the architecture level" --
    edge/config.yaml's hardware.telemetry.type selects which concrete driver
    this wraps at construction time. Concretely:
      - LoRa: typically a serial (UART) AT-command module or an SPI
        transceiver (e.g. SX127x family) driven directly.
      - Cellular: typically a serial (UART) AT-command modem, similar
        integration shape to LoRa but different AT command set/APN setup.
    send() must serialize `payload` (already a small dict -- see
    edge/telemetry.py's build_payload()) to whatever wire format the link
    uses (e.g. compact JSON, or a packed binary struct if airtime/power
    budget requires it) and return False (not raise) on any failure, per
    the interface contract.
    """

    def __init__(self, config: EdgeConfig):
        self._config = config
        # e.g. (LoRa, UART AT-command): import serial; self._port = serial.Serial(...)
        # e.g. (LoRa, SPI transceiver): import spidev; self._spi = spidev.SpiDev(); ...

    def send(self, payload: Dict) -> bool:
        raise NotImplementedError(
            "RealTelemetryLink.send() -- implement once the telemetry module is wired "
            "and the LoRa-vs-cellular deployment decision (DECISIONS.md) is made for "
            "this specific site."
        )


class RealPowerMonitor(PowerMonitor):
    """
    TODO once hardware exists: battery voltage / solar charge / enclosure
    temp. Concretely, typically a fuel-gauge IC (e.g. I2C) for battery
    voltage/current, a charge controller's I2C/analog telemetry output for
    solar_charge_w, and a simple I2C or 1-Wire temperature sensor inside the
    enclosure for enclosure_temp_c. uptime_sec can be computed in software
    (process start time) without dedicated hardware.
    """

    def __init__(self, config: EdgeConfig):
        self._config = config
        # e.g.: import smbus2; self._bus = smbus2.SMBus(config.hardware.i2c_bus_number)

    def read(self) -> Dict[str, float]:
        raise NotImplementedError(
            "RealPowerMonitor.read() -- implement once power-system telemetry hardware "
            "is wired; see docs/hardware-spec.md's 'Power draw implications' section."
        )
