"""
Power budget estimator -- a software-only planning tool, no hardware
required, for the item docs/hardware-spec.md and DECISIONS.md both flag as
not-yet-done: "Power budget calculation (duty cycle vs solar input vs
battery capacity)."

This does NOT replace a real power budget once parts are chosen -- every
per-component draw figure below is a placeholder sourced from typical
published specs for the *class* of part named (a Raspberry Pi 4, a
SX127x-class LoRa module, etc.), not a datasheet number for a specific SKU
on a priced BOM (which doesn't exist yet -- see docs/hardware-spec.md).
Re-run this with real datasheet numbers (--pi-active-w, etc.) once
components are actually selected; the model/formulas stay valid, only the
inputs change.

Model (deliberately simple, first-pass sizing, not a full simulation):
  - Two power states per docs/hardware-spec.md's "Power draw implications of
    duty-cycle sampling": idle (sensors/compute powered down between wake
    windows) and active (capture + FFT/feature extraction + anomaly
    detection + telemetry transmission, all folded into one active-window
    average for simplicity).
  - avg_power_w = time-weighted average of idle and active power over one
    full duty cycle (window_duration_s active, the rest of
    window_interval_minutes idle).
  - daily_energy_wh = avg_power_w * 24h.
  - solar sizing: daily_energy_wh / peak_sun_hours / system_efficiency.
  - battery sizing: daily_energy_wh * autonomy_days / usable_depth_of_discharge,
    converted to Ah at the given battery bus voltage.
"""

import argparse
import dataclasses
from typing import Optional

# ---- Placeholder per-component draw figures (Watts) ----
# Sourced from typical published specs for the part *class*, not a
# specific chosen SKU -- see module docstring. Override via CLI flags once
# real parts/datasheets exist.
DEFAULT_PI_IDLE_W = 0.6  # Raspberry Pi 4, lightly loaded / near-idle
DEFAULT_PI_ACTIVE_W = 3.4  # Raspberry Pi 4, CPU busy (FFT/feature extraction/anomaly detection burst)
DEFAULT_HYDROPHONE_ADC_W = 0.05  # hydrophone preamp + ADC, active
DEFAULT_ENV_SENSORS_W = 0.15  # temp + pH + turbidity + salinity combined, active
DEFAULT_IMU_W = 0.01  # IMU, active
DEFAULT_LORA_IDLE_W = 0.005  # LoRa module, sleep/idle between transmissions
DEFAULT_LORA_TX_W = 0.4  # LoRa module, transmitting (SX127x-class, ~100mA @ 3.3V-ish)
DEFAULT_CELLULAR_TX_W = 2.0  # low-bandwidth cellular module, transmitting -- much higher than LoRa

DEFAULT_PEAK_SUN_HOURS = 4.0  # conservative coastal-site default; site-specific in reality
DEFAULT_SOLAR_SYSTEM_EFFICIENCY = 0.75  # charge controller + wiring + non-ideal panel angle losses
DEFAULT_BATTERY_DOD = 0.8  # usable depth of discharge (LiFePO4-class; lower for lead-acid, ~0.5)
DEFAULT_BATTERY_VOLTAGE = 12.0
DEFAULT_AUTONOMY_DAYS = 3.0  # days of zero solar input the battery alone must cover


@dataclasses.dataclass
class PowerBudgetInputs:
    window_duration_s: float = 5.0
    window_interval_minutes: float = 10.0
    telemetry_type: str = "lora"  # "lora" | "cellular"

    pi_idle_w: float = DEFAULT_PI_IDLE_W
    pi_active_w: float = DEFAULT_PI_ACTIVE_W
    hydrophone_adc_w: float = DEFAULT_HYDROPHONE_ADC_W
    env_sensors_w: float = DEFAULT_ENV_SENSORS_W
    imu_w: float = DEFAULT_IMU_W
    telemetry_idle_w: float = DEFAULT_LORA_IDLE_W
    telemetry_tx_w: Optional[float] = None  # resolved from telemetry_type if None

    peak_sun_hours: float = DEFAULT_PEAK_SUN_HOURS
    solar_system_efficiency: float = DEFAULT_SOLAR_SYSTEM_EFFICIENCY
    battery_dod: float = DEFAULT_BATTERY_DOD
    battery_voltage: float = DEFAULT_BATTERY_VOLTAGE
    autonomy_days: float = DEFAULT_AUTONOMY_DAYS

    def __post_init__(self):
        if self.telemetry_tx_w is None:
            self.telemetry_tx_w = (
                DEFAULT_CELLULAR_TX_W if self.telemetry_type == "cellular" else DEFAULT_LORA_TX_W
            )
        if self.telemetry_type == "cellular":
            self.telemetry_idle_w = max(self.telemetry_idle_w, 0.02)  # cellular modems idle higher than LoRa


@dataclasses.dataclass
class PowerBudgetResult:
    idle_power_w: float
    active_power_w: float
    avg_power_w: float
    daily_energy_wh: float
    recommended_solar_w: float
    recommended_battery_wh: float
    recommended_battery_ah: float
    duty_cycle_fraction: float


def compute_power_budget(inputs: PowerBudgetInputs) -> PowerBudgetResult:
    interval_s = inputs.window_interval_minutes * 60
    if inputs.window_duration_s > interval_s:
        raise ValueError("window_duration_s cannot exceed window_interval_minutes * 60")

    idle_power_w = inputs.pi_idle_w + inputs.telemetry_idle_w
    active_power_w = (
        inputs.pi_active_w
        + inputs.hydrophone_adc_w
        + inputs.env_sensors_w
        + inputs.imu_w
        + inputs.telemetry_tx_w
    )

    duty_cycle_fraction = inputs.window_duration_s / interval_s
    avg_power_w = active_power_w * duty_cycle_fraction + idle_power_w * (1 - duty_cycle_fraction)
    daily_energy_wh = avg_power_w * 24

    recommended_solar_w = daily_energy_wh / inputs.peak_sun_hours / inputs.solar_system_efficiency
    recommended_battery_wh = daily_energy_wh * inputs.autonomy_days / inputs.battery_dod
    recommended_battery_ah = recommended_battery_wh / inputs.battery_voltage

    return PowerBudgetResult(
        idle_power_w=idle_power_w,
        active_power_w=active_power_w,
        avg_power_w=avg_power_w,
        daily_energy_wh=daily_energy_wh,
        recommended_solar_w=recommended_solar_w,
        recommended_battery_wh=recommended_battery_wh,
        recommended_battery_ah=recommended_battery_ah,
        duty_cycle_fraction=duty_cycle_fraction,
    )


def format_report(inputs: PowerBudgetInputs, result: PowerBudgetResult) -> str:
    lines = [
        "Power budget estimate (PLACEHOLDER component figures -- see module docstring)",
        "=" * 78,
        f"Duty cycle: {inputs.window_duration_s:.0f}s active every {inputs.window_interval_minutes:.0f} min "
        f"({result.duty_cycle_fraction:.4%} duty cycle)",
        f"Telemetry: {inputs.telemetry_type}",
        "",
        f"Idle power draw:      {result.idle_power_w:.3f} W",
        f"Active power draw:    {result.active_power_w:.3f} W",
        f"Time-weighted average: {result.avg_power_w:.3f} W",
        f"Daily energy:          {result.daily_energy_wh:.2f} Wh/day",
        "",
        f"Recommended solar panel:  >= {result.recommended_solar_w:.1f} W "
        f"(at {inputs.peak_sun_hours:.1f} peak sun hours/day, "
        f"{inputs.solar_system_efficiency:.0%} system efficiency)",
        f"Recommended battery:      >= {result.recommended_battery_wh:.1f} Wh "
        f"(~{result.recommended_battery_ah:.1f} Ah @ {inputs.battery_voltage:.0f}V) "
        f"for {inputs.autonomy_days:.0f} days autonomy at {inputs.battery_dod:.0%} usable depth of discharge",
        "",
        "Caveat: every per-component draw figure is a placeholder for the part *class* "
        "(see this file's module docstring), not a priced BOM part's datasheet number. "
        "Re-run with --pi-active-w etc. overridden once real parts are selected "
        "(docs/hardware-spec.md's BOM is not priced yet).",
    ]
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-duration-s", type=float, default=PowerBudgetInputs.window_duration_s)
    parser.add_argument("--window-interval-minutes", type=float, default=PowerBudgetInputs.window_interval_minutes)
    parser.add_argument("--telemetry-type", choices=["lora", "cellular"], default=PowerBudgetInputs.telemetry_type)
    parser.add_argument("--pi-idle-w", type=float, default=DEFAULT_PI_IDLE_W)
    parser.add_argument("--pi-active-w", type=float, default=DEFAULT_PI_ACTIVE_W)
    parser.add_argument("--hydrophone-adc-w", type=float, default=DEFAULT_HYDROPHONE_ADC_W)
    parser.add_argument("--env-sensors-w", type=float, default=DEFAULT_ENV_SENSORS_W)
    parser.add_argument("--imu-w", type=float, default=DEFAULT_IMU_W)
    parser.add_argument("--peak-sun-hours", type=float, default=DEFAULT_PEAK_SUN_HOURS)
    parser.add_argument("--solar-system-efficiency", type=float, default=DEFAULT_SOLAR_SYSTEM_EFFICIENCY)
    parser.add_argument("--battery-dod", type=float, default=DEFAULT_BATTERY_DOD)
    parser.add_argument("--battery-voltage", type=float, default=DEFAULT_BATTERY_VOLTAGE)
    parser.add_argument("--autonomy-days", type=float, default=DEFAULT_AUTONOMY_DAYS)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    inputs = PowerBudgetInputs(
        window_duration_s=args.window_duration_s,
        window_interval_minutes=args.window_interval_minutes,
        telemetry_type=args.telemetry_type,
        pi_idle_w=args.pi_idle_w,
        pi_active_w=args.pi_active_w,
        hydrophone_adc_w=args.hydrophone_adc_w,
        env_sensors_w=args.env_sensors_w,
        imu_w=args.imu_w,
        peak_sun_hours=args.peak_sun_hours,
        solar_system_efficiency=args.solar_system_efficiency,
        battery_dod=args.battery_dod,
        battery_voltage=args.battery_voltage,
        autonomy_days=args.autonomy_days,
    )
    result = compute_power_budget(inputs)
    print(format_report(inputs, result))


if __name__ == "__main__":
    main()
