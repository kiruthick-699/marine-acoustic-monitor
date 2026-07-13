"""
Scheduler test: bounded run (max_iterations) with an injected no-op sleep_fn
so the test doesn't actually wait on wall-clock duty-cycle intervals.
"""

from edge.capture_loop import CaptureLoop, WindowCaptureError
from edge.config import EdgeConfig
from edge.hal.factory import build_hardware
from edge.scheduler import DutyCycleScheduler
from simulation.pipeline.storage import init_db


def _make_config(tmp_path) -> EdgeConfig:
    config = EdgeConfig()
    config.duty_cycle.window_duration_s = 1.0
    config.duty_cycle.window_interval_minutes = 10.0
    config.audio.sample_rate_hz = 8000
    config.storage.audio_dir = str(tmp_path / "audio")
    config.storage.db_path = str(tmp_path / "db.sqlite")
    config.storage.baseline_model_path = str(tmp_path / "baseline_model.joblib")
    config.hardware.mode = "mock"
    return config


def test_run_forever_bounded_calls_on_window_and_sleep_each_iteration(tmp_path):
    config = _make_config(tmp_path)
    db_conn = init_db(config.storage.db_path)
    hardware = build_hardware(config)
    loop = CaptureLoop(config, hardware, db_conn)

    sleep_calls = []
    window_summaries = []

    scheduler = DutyCycleScheduler(
        loop,
        config,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
        on_window=window_summaries.append,
    )
    scheduler.run_forever(max_iterations=3)

    assert len(sleep_calls) == 3
    assert len(window_summaries) == 3
    # sleep is clamped to the configured interval, never negative
    assert all(0 <= s <= config.duty_cycle.window_interval_minutes * 60 for s in sleep_calls)

    captures_count = db_conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    assert captures_count == 3


def test_run_forever_skips_failed_window_and_continues_to_next(tmp_path):
    config = _make_config(tmp_path)
    db_conn = init_db(config.storage.db_path)
    hardware = build_hardware(config)
    loop = CaptureLoop(config, hardware, db_conn)

    # Fail only the 2nd of 3 windows -- proves a single bad window doesn't
    # stop the scheduler from reaching the 3rd.
    real_run_one_window = loop.run_one_window
    calls = {"n": 0}

    def flaky_run_one_window():
        calls["n"] += 1
        if calls["n"] == 2:
            raise WindowCaptureError("simulated unrecoverable window failure")
        return real_run_one_window()

    loop.run_one_window = flaky_run_one_window

    sleep_calls = []
    window_summaries = []

    scheduler = DutyCycleScheduler(
        loop,
        config,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
        on_window=window_summaries.append,
    )
    scheduler.run_forever(max_iterations=3)

    # scheduler ran (and slept) all 3 iterations despite the failure...
    assert calls["n"] == 3
    assert len(sleep_calls) == 3
    # ...but only the 2 successful windows produced a summary/capture row.
    assert len(window_summaries) == 2
    captures_count = db_conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    assert captures_count == 2
