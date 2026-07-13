"""
Duty-cycle scheduler -- wakes CaptureLoop every M minutes, forever, per
DECISIONS.md's locked sampling model ("record N seconds every M minutes").

Kept deliberately simple (a sleep loop, not a cron/systemd-timer-driven
process) so the same process can also run a long-lived duty cycle
standalone under a systemd service (docs/pi-implementation.md) without extra
OS-level scheduling infrastructure. Sleep duration accounts for how long the
window itself took to process, so drift doesn't accumulate window over
window.
"""

import time
from typing import Callable, Optional

from edge.capture_loop import CaptureLoop
from edge.config import EdgeConfig


class DutyCycleScheduler:
    """
    Args:
        loop: a CaptureLoop, already calibrated (detector installed) or not
            -- an uncalibrated loop still runs and stores windows, just with
            placeholder anomaly results (see CaptureLoop.run_one_window()).
        config: EdgeConfig; config.duty_cycle.window_interval_minutes sets
            the wake interval.
        sleep_fn: injected sleep function, defaulting to time.sleep.
            Tests pass a no-op (or a recording stub) so scheduler behavior
            can be verified without real wall-clock waits.
        on_window: optional callback invoked with each run_one_window()
            summary dict, e.g. for logging or test assertions.
    """

    def __init__(
        self,
        loop: CaptureLoop,
        config: EdgeConfig,
        sleep_fn: Callable[[float], None] = time.sleep,
        on_window: Optional[Callable[[dict], None]] = None,
    ):
        self._loop = loop
        self._config = config
        self._sleep_fn = sleep_fn
        self._on_window = on_window

    def run_forever(self, max_iterations: Optional[int] = None) -> None:
        """
        Run the duty cycle indefinitely (or `max_iterations` times, for
        bounded test/demo runs).

        Each iteration: run one window, then sleep for
        (window_interval_minutes * 60 - elapsed_processing_time), clamped
        to >= 0 -- so on a Pi where capture+processing takes a non-trivial
        fraction of the interval, wake windows still land close to the
        configured M-minute cadence rather than drifting later each cycle.
        """
        interval_s = self._config.duty_cycle.window_interval_minutes * 60
        iterations = 0

        while max_iterations is None or iterations < max_iterations:
            start = time.monotonic()

            summary = self._loop.run_one_window()
            if self._on_window is not None:
                self._on_window(summary)

            elapsed = time.monotonic() - start
            sleep_for = max(0.0, interval_s - elapsed)
            self._sleep_fn(sleep_for)

            iterations += 1
