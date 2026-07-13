"""
Offline recalibration -- re-fit a BaselineAnomalyDetector from data already
stored in an existing edge/output/db.sqlite, instead of edge/calibration.py's
run_calibration() (which fits from a *live* duty-cycle collection period at
first deployment). Matches docs/ml-pipeline.md: "Recalibration itself is an
offline/batch operation performed during bulk data retrieval, not on-device
in real time."

Usage:
    python -m edge.recalibrate --last-n-windows 500
    python -m edge.recalibrate --start 2026-01-01T00:00:00+00:00 --end 2026-02-01T00:00:00+00:00
    python -m edge.recalibrate --db /path/to/retrieved/db.sqlite --last-n-windows 1000

Reconstructing feature vectors from storage (not from a full live re-run):
`feature_vectors` only persists one REAL column per non-MFCC acoustic
feature (docs/data-pipeline.md's schema keeps spectral_centroid/
zero_crossing_rate/rms_energy/spectral_flatness as a single mean value each,
not the mean+std pair simulation/pipeline/feature_extraction.py computes
live) -- so a reconstructed vector is missing those 4 *_std columns compared
to what CaptureLoop.run_one_window() builds at capture time. This is an
intentional Tier-2 storage tradeoff (compact schema, matches
docs/data-pipeline.md exactly), not a bug here, and it's harmless: a newly
fitted BaselineAnomalyDetector.score() reindexes future joint vectors
against whatever feature_names it was fit on, silently ignoring extra
columns it never saw during fit(). MFCC coefficients (mean and std both) and
all four environmental parameters + rate-of-change are stored in full and
reconstruct exactly.
"""

import argparse
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import List, NamedTuple, Optional

import pandas as pd

from edge.calibration import save_baseline
from edge.config import EdgeConfig, load_config
from edge.logging_setup import configure_logging
from simulation.pipeline.anomaly_detection import BaselineAnomalyDetector
from simulation.pipeline.feature_extraction import build_joint_feature_vector, extract_environmental_features

# Not logging.getLogger(__name__): `python -m edge.recalibrate` sets this
# module's __name__ to "__main__" rather than "edge.recalibrate" -- using
# __name__ would silently detach this logger from the "edge" tree
# edge/logging_setup.py configures (see edge/main.py's logger for the same
# fix and the regression test that caught it).
logger = logging.getLogger("edge.recalibrate")

_VERSION_RE = re.compile(r"^v(\d+)$")

_ROW_COLUMNS = (
    "capture_id",
    "timestamp_utc",
    "mfcc",
    "spectral_centroid",
    "zero_crossing_rate",
    "rms_energy",
    "spectral_flatness",
    "temperature_c",
    "ph",
    "turbidity_ntu",
    "salinity_psu",
    "temp_roc",
    "ph_roc",
    "turbidity_roc",
    "salinity_roc",
)


class RecalibrationResult(NamedTuple):
    detector: BaselineAnomalyDetector
    baseline_version: str
    baseline_model_path: str
    n_windows: int


def next_baseline_version(current: str) -> str:
    """
    Increment a "vN" baseline_version string (edge/config.py's
    CalibrationConfig.baseline_version convention) to "v(N+1)".
    """
    match = _VERSION_RE.match(current)
    if not match:
        raise ValueError(f"baseline_version {current!r} doesn't match the expected 'vN' convention")
    return f"v{int(match.group(1)) + 1}"


def _fetch_rows(
    conn: sqlite3.Connection,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    last_n_windows: Optional[int] = None,
) -> List[tuple]:
    """
    Fetch (capture_id, timestamp_utc, ...) rows joined across captures,
    feature_vectors, and environmental_readings, in chronological order.

    Args:
        start: ISO 8601 timestamp, inclusive lower bound on captures.timestamp_utc.
        end: ISO 8601 timestamp, inclusive upper bound on captures.timestamp_utc.
        last_n_windows: if given, ignore start/end and return only the most
            recent N windows (still chronologically ordered in the result).
    """
    query = """
        SELECT c.capture_id, c.timestamp_utc,
               f.mfcc, f.spectral_centroid, f.zero_crossing_rate, f.rms_energy, f.spectral_flatness,
               e.temperature_c, e.ph, e.turbidity_ntu, e.salinity_psu,
               e.temp_roc, e.ph_roc, e.turbidity_roc, e.salinity_roc
        FROM captures c
        JOIN feature_vectors f ON f.capture_id = c.capture_id
        JOIN environmental_readings e ON e.capture_id = c.capture_id
    """
    params: List = []

    if last_n_windows is not None:
        query += " ORDER BY c.timestamp_utc DESC, c.capture_id DESC LIMIT ?"
        params.append(last_n_windows)
        rows = conn.execute(query, params).fetchall()
        return list(reversed(rows))

    clauses = []
    if start is not None:
        clauses.append("c.timestamp_utc >= ?")
        params.append(start)
    if end is not None:
        clauses.append("c.timestamp_utc <= ?")
        params.append(end)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY c.timestamp_utc ASC, c.capture_id ASC"

    return conn.execute(query, params).fetchall()


def _reconstruct_joint_feature_vector(row: tuple) -> pd.Series:
    """
    Rebuild one window's joint feature vector from a _fetch_rows() row --
    see this module's docstring for the acoustic *_std columns this can't
    recover (not persisted by storage.py's schema).
    """
    values = dict(zip(_ROW_COLUMNS, row))

    acoustic_features = json.loads(bytes(values["mfcc"]).decode("utf-8"))
    acoustic_features.update(
        {
            "spectral_centroid_mean": values["spectral_centroid"],
            "zero_crossing_rate_mean": values["zero_crossing_rate"],
            "rms_energy_mean": values["rms_energy"],
            "spectral_flatness_mean": values["spectral_flatness"],
        }
    )

    env_row = pd.Series(
        {
            "temperature_c": values["temperature_c"],
            "ph": values["ph"],
            "turbidity_ntu": values["turbidity_ntu"],
            "salinity_psu": values["salinity_psu"],
            "temp_roc": values["temp_roc"],
            "ph_roc": values["ph_roc"],
            "turbidity_roc": values["turbidity_roc"],
            "salinity_roc": values["salinity_roc"],
        }
    )
    environmental_features = extract_environmental_features(env_row)

    return build_joint_feature_vector(acoustic_features, environmental_features)


def recalibrate(
    db_path: str,
    config: EdgeConfig,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    last_n_windows: Optional[int] = None,
) -> RecalibrationResult:
    """
    Re-fit a BaselineAnomalyDetector from stored capture windows in `db_path`.

    Args:
        db_path: path to an existing db.sqlite (docs/data-pipeline.md schema)
            with at least one capture window in the selected range.
        config: EdgeConfig -- config.calibration.threshold_sigma controls the
            new detector's fit, config.calibration.baseline_version is
            incremented (see next_baseline_version()) for the new model,
            config.storage.baseline_model_path names the new model file.
        start, end: ISO 8601 timestamp bounds (inclusive), mutually exclusive
            with last_n_windows.
        last_n_windows: use only the most recent N stored windows instead of
            a timestamp range.

    Returns:
        RecalibrationResult(detector, baseline_version, baseline_model_path,
        n_windows) -- the new baseline_version and model path are *not*
        written back to config.yaml; the caller (or this module's CLI) is
        responsible for telling the operator to update it deliberately, so
        a recalibration run never silently changes what a running process
        loads next.

    Raises:
        ValueError: no capture windows found in the selected range.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = _fetch_rows(conn, start=start, end=end, last_n_windows=last_n_windows)
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"no capture windows found in {db_path} for the given start/end/last_n_windows selection")

    feature_vectors = [_reconstruct_joint_feature_vector(row) for row in rows]

    detector = BaselineAnomalyDetector(threshold_sigma=config.calibration.threshold_sigma)
    detector.fit(feature_vectors)

    new_version = next_baseline_version(config.calibration.baseline_version)
    old_path = Path(config.storage.baseline_model_path)
    new_path = old_path.with_name(f"{old_path.stem}_{new_version}{old_path.suffix}")
    save_baseline(detector, str(new_path))

    logger.info(
        "Recalibrated on %d stored windows from %s. New baseline (%s) saved to %s -- "
        "not yet active: update edge/config.yaml's calibration.baseline_version to %r "
        "and storage.baseline_model_path to %r to make this the model a running process loads.",
        len(rows),
        db_path,
        new_version,
        new_path,
        new_version,
        str(new_path),
    )

    return RecalibrationResult(
        detector=detector,
        baseline_version=new_version,
        baseline_model_path=str(new_path),
        n_windows=len(rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline recalibration from a stored edge/ database.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: edge/config.yaml).")
    parser.add_argument("--db", type=str, default=None, help="Path to db.sqlite (default: config.storage.db_path).")
    parser.add_argument(
        "--last-n-windows", type=int, default=None, help="Use only the N most recent stored windows."
    )
    parser.add_argument("--start", type=str, default=None, help="ISO 8601 start timestamp, inclusive.")
    parser.add_argument("--end", type=str, default=None, help="ISO 8601 end timestamp, inclusive.")
    args = parser.parse_args()

    if args.last_n_windows is not None and (args.start is not None or args.end is not None):
        parser.error("--last-n-windows cannot be combined with --start/--end")

    config = load_config(Path(args.config)) if args.config else load_config()
    configure_logging(config)
    db_path = args.db or config.storage.db_path

    result = recalibrate(
        db_path, config, start=args.start, end=args.end, last_n_windows=args.last_n_windows
    )

    print(f"Recalibrated on {result.n_windows} windows.")
    print(f"New baseline saved to: {result.baseline_model_path}")
    print("Update edge/config.yaml to make this the active baseline:")
    print(f"  calibration.baseline_version: {result.baseline_version}")
    print(f"  storage.baseline_model_path: {result.baseline_model_path}")


if __name__ == "__main__":
    main()
