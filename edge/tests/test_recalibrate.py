"""
Tests for edge/recalibrate.py -- offline recalibration from an existing
edge/output/db.sqlite, as opposed to edge/calibration.py's live-collection
run_calibration().
"""

import os

import pytest

from edge.capture_loop import CaptureLoop
from edge.config import EdgeConfig
from edge.hal.factory import build_hardware
from edge.recalibrate import next_baseline_version, recalibrate
from simulation.pipeline.storage import init_db


def _make_config(tmp_path, **overrides) -> EdgeConfig:
    config = EdgeConfig()
    config.duty_cycle.window_duration_s = 1.0
    config.audio.sample_rate_hz = 8000
    config.storage.audio_dir = str(tmp_path / "audio")
    config.storage.db_path = str(tmp_path / "db.sqlite")
    config.storage.baseline_model_path = str(tmp_path / "baseline_model.joblib")
    config.hardware.mode = "mock"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _run_windows(config, n: int) -> None:
    db_conn = init_db(config.storage.db_path)
    hardware = build_hardware(config)
    loop = CaptureLoop(config, hardware, db_conn)
    for _ in range(n):
        loop.run_one_window()
    db_conn.close()


def test_next_baseline_version_increments():
    assert next_baseline_version("v1") == "v2"
    assert next_baseline_version("v9") == "v10"


def test_next_baseline_version_rejects_unexpected_format():
    with pytest.raises(ValueError):
        next_baseline_version("version-1")


def test_recalibrate_fits_and_saves_new_versioned_baseline(tmp_path):
    config = _make_config(tmp_path)
    _run_windows(config, n=8)

    result = recalibrate(config.storage.db_path, config, last_n_windows=8)

    assert result.n_windows == 8
    assert result.baseline_version == "v2"  # default config.calibration.baseline_version is "v1"
    assert os.path.exists(result.baseline_model_path)
    # a new file, distinct from (and not overwriting) the original path
    assert result.baseline_model_path != config.storage.baseline_model_path
    assert not os.path.exists(config.storage.baseline_model_path)

    # the fitted detector actually works: scoring a reconstructed vector
    # from one of the same stored windows doesn't raise.
    import sqlite3

    from edge.recalibrate import _fetch_rows, _reconstruct_joint_feature_vector

    conn = sqlite3.connect(config.storage.db_path)
    row = _fetch_rows(conn, last_n_windows=1)[0]
    conn.close()
    joint_vector = _reconstruct_joint_feature_vector(row)
    score = result.detector.score(joint_vector)
    assert isinstance(score["anomaly_score"], float)


def test_recalibrate_last_n_windows_uses_only_the_most_recent(tmp_path):
    config = _make_config(tmp_path)
    _run_windows(config, n=10)

    result_all = recalibrate(config.storage.db_path, config, last_n_windows=10)
    result_recent = recalibrate(config.storage.db_path, config, last_n_windows=3)

    assert result_all.n_windows == 10
    assert result_recent.n_windows == 3


def test_recalibrate_start_end_range_filters_by_timestamp(tmp_path):
    config = _make_config(tmp_path)
    db_conn = init_db(config.storage.db_path)
    hardware = build_hardware(config)
    loop = CaptureLoop(config, hardware, db_conn)
    loop.run_one_window()

    # the sole stored window's timestamp is strictly after this cutoff, so a
    # start bound set to "now" excludes it -- proving the filter is applied,
    # not ignored.
    from datetime import datetime, timezone

    future_start = datetime.now(timezone.utc).isoformat()
    db_conn.close()

    with pytest.raises(ValueError):
        recalibrate(config.storage.db_path, config, start=future_start)


def test_recalibrate_raises_on_empty_selection(tmp_path):
    config = _make_config(tmp_path)
    init_db(config.storage.db_path).close()

    with pytest.raises(ValueError):
        recalibrate(config.storage.db_path, config, last_n_windows=5)


def test_recalibrate_does_not_overwrite_existing_baseline_on_repeated_runs(tmp_path):
    config = _make_config(tmp_path)
    _run_windows(config, n=5)

    first = recalibrate(config.storage.db_path, config, last_n_windows=5)
    assert first.baseline_version == "v2"

    # simulate the operator bumping config.calibration.baseline_version to
    # v2 (as instructed) before recalibrating again
    config.calibration.baseline_version = "v2"
    second = recalibrate(config.storage.db_path, config, last_n_windows=5)

    assert second.baseline_version == "v3"
    assert second.baseline_model_path != first.baseline_model_path
    assert os.path.exists(first.baseline_model_path)
    assert os.path.exists(second.baseline_model_path)
