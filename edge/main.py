"""
Edge entrypoint -- what actually runs on the Pi (or, today, on a dev machine
in mock mode). Wires together config, HAL, storage, calibration, and the
scheduler; this is the module docs/pi-implementation.md's systemd service
unit invokes.

Usage:
    python -m edge.main                    # run forever, mock hardware, duty cycle per config
    python -m edge.main --once             # run exactly one window and print the summary, then exit
    python -m edge.main --mode real        # override hardware.mode (fails loudly until edge/hal/real.py is implemented)
    python -m edge.main --calibrate        # force a fresh calibration period even if a baseline already exists
    python -m edge.main --max-iterations 5 # bounded run (demo/testing) instead of forever
"""

import argparse
import json
from pathlib import Path

from edge.calibration import load_baseline, run_calibration
from edge.capture_loop import CaptureLoop
from edge.config import EdgeConfig, load_config
from edge.hal.factory import build_hardware
from edge.scheduler import DutyCycleScheduler
from simulation.pipeline.storage import init_db


def _print_window_summary(summary: dict) -> None:
    flag = "ANOMALY" if summary["is_anomaly"] else "normal"
    calib = "" if summary["calibrated"] else " [uncalibrated placeholder score]"
    print(
        f"[{summary['timestamp_utc']}] capture_id={summary['capture_id']} "
        f"score={summary['anomaly_score']:.4f} ({flag}){calib} "
        f"telemetry_sent={summary['telemetry_sent']}"
    )


def build_app(config: EdgeConfig, force_calibrate: bool = False) -> CaptureLoop:
    """
    Construct a fully-wired, ready-to-run CaptureLoop: DB initialized,
    hardware built per config.hardware.mode, and a calibration baseline
    either loaded from disk or freshly run.
    """
    db_conn = init_db(config.storage.db_path)
    hardware = build_hardware(config)
    loop = CaptureLoop(config, hardware, db_conn)

    detector = None if force_calibrate else load_baseline(config.storage.baseline_model_path)
    if detector is not None:
        print(f"Loaded existing calibration baseline from {config.storage.baseline_model_path}")
        loop.set_detector(detector)
    else:
        run_calibration(loop, config)

    return loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Marine monitoring edge process.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: edge/config.yaml).")
    parser.add_argument("--mode", choices=["mock", "real"], default=None, help="Override hardware.mode.")
    parser.add_argument("--once", action="store_true", help="Run exactly one duty-cycle window and exit.")
    parser.add_argument(
        "--calibrate", action="store_true", help="Force a fresh calibration period even if a baseline exists."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None, help="Stop after N windows instead of running forever."
    )
    args = parser.parse_args()

    config = load_config(Path(args.config)) if args.config else load_config()
    if args.mode:
        config.hardware.mode = args.mode

    loop = build_app(config, force_calibrate=args.calibrate)

    if args.once:
        summary = loop.run_one_window()
        _print_window_summary(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "feature_vector"}, indent=2))
        return

    scheduler = DutyCycleScheduler(loop, config, on_window=_print_window_summary)
    print(
        f"Starting duty cycle: {config.duty_cycle.window_duration_s}s every "
        f"{config.duty_cycle.window_interval_minutes} min, hardware.mode={config.hardware.mode}"
    )
    scheduler.run_forever(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
