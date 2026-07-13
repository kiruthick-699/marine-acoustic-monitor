"""
SQLite storage layer.

Implements Tier 2 of docs/data-pipeline.md's storage tiers: structured data
(feature vectors, environmental readings, anomaly flags, system health log)
in SQLite, WAL mode, with rows referencing a Tier 1 audio filename rather
than embedding audio. Schema matches the sketch in docs/data-pipeline.md
exactly (same table/column names) so this code and that doc stay in sync.
"""

import json
import os
import sqlite3
from typing import Dict, Union

import pandas as pd

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS captures (
    capture_id     INTEGER PRIMARY KEY,
    timestamp_utc  TEXT NOT NULL,
    audio_filename TEXT NOT NULL,
    duration_sec   REAL NOT NULL,
    sample_rate_hz INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_vectors (
    capture_id             INTEGER PRIMARY KEY REFERENCES captures(capture_id),
    mfcc                   BLOB,
    spectral_centroid      REAL,
    zero_crossing_rate     REAL,
    rms_energy             REAL,
    spectral_flatness      REAL,
    feature_vector_version TEXT
);

CREATE TABLE IF NOT EXISTS environmental_readings (
    capture_id    INTEGER PRIMARY KEY REFERENCES captures(capture_id),
    temperature_c REAL,
    ph            REAL,
    turbidity_ntu REAL,
    salinity_psu  REAL,
    temp_roc      REAL,
    ph_roc        REAL,
    turbidity_roc REAL,
    salinity_roc  REAL
);

CREATE TABLE IF NOT EXISTS anomaly_flags (
    capture_id       INTEGER PRIMARY KEY REFERENCES captures(capture_id),
    anomaly_score    REAL NOT NULL,
    is_anomaly       INTEGER NOT NULL,
    baseline_version TEXT
);

CREATE TABLE IF NOT EXISTS system_health_log (
    log_id           INTEGER PRIMARY KEY,
    timestamp_utc    TEXT NOT NULL,
    battery_voltage  REAL,
    solar_charge_w   REAL,
    enclosure_temp_c REAL,
    imu_orientation  BLOB,
    uptime_sec       INTEGER
);
"""


def init_db(path: str) -> sqlite3.Connection:
    """
    Create (or open) the on-device SQLite database and apply the schema.

    WAL (write-ahead logging) mode is enabled per docs/data-pipeline.md, so
    the capture/processing loop's writes don't block concurrent read access
    (e.g. a maintenance-visit export reading the file while the loop is
    still running). Table creation uses `IF NOT EXISTS` so calling this
    against an already-initialized database file is safe and a no-op on the
    schema.

    Args:
        path: filesystem path to the SQLite database file (created if it
            doesn't exist).

    Returns:
        An open sqlite3.Connection with the schema applied, ready for
        insert_window_record() calls.
    """
    # sqlite3.connect() does not create missing parent directories itself --
    # simulation/scripts/run_simulation.py happens to always makedirs() the
    # output dir before calling this, but edge/main.py's first-ever run has
    # no such guarantee (config.storage.db_path's parent may not exist yet
    # on a fresh checkout), so it's handled here once, for every caller.
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def insert_window_record(
    conn: sqlite3.Connection,
    *,
    timestamp_utc: str,
    audio_filename: str,
    duration_sec: float,
    sample_rate_hz: int,
    acoustic_features: Dict[str, float],
    environmental_row: Union[pd.Series, Dict[str, float]],
    anomaly_result: Dict[str, float],
    feature_vector_version: str = "v1",
    baseline_version: str = "v1",
) -> int:
    """
    Write one duty-cycle window's full structured record in one transaction.

    Inserts across `captures`, `feature_vectors`, `environmental_readings`,
    and `anomaly_flags` as a single transaction (one commit at the end) so a
    window's record is always fully written or not written at all -- no
    other table ever ends up with a row for a `capture_id` that doesn't
    exist in the others. `audio_filename` is stored by reference only (Tier
    1 flat file); no audio data is written to SQLite.

    `feature_vectors.mfcc` stores the full set of per-coefficient MFCC
    mean/std statistics (from
    simulation/pipeline/feature_extraction.extract_acoustic_features(),
    keys prefixed `mfcc_`) as a JSON blob, since the schema sketch in
    docs/data-pipeline.md represents the many MFCC coefficients as a single
    serialized column rather than one column per coefficient. The other
    acoustic feature columns (`spectral_centroid`, `zero_crossing_rate`,
    `rms_energy`, `spectral_flatness`) each take that feature's mean-across-
    frames value, matching the schema's one-value-per-feature shape.

    Args:
        conn: open connection from init_db().
        timestamp_utc: ISO 8601 capture timestamp.
        audio_filename: Tier 1 flat audio file this record's features and
            readings were derived from.
        duration_sec: capture window duration, seconds.
        sample_rate_hz: audio sample rate, Hz.
        acoustic_features: output of
            feature_extraction.extract_acoustic_features().
        environmental_row: one window's environmental reading (with rate-
            of-change columns already computed), as produced by
            synthetic_environmental.compute_rate_of_change() or real sensor
            equivalent -- a mapping with temperature_c, ph, turbidity_ntu,
            salinity_psu, temp_roc, ph_roc, turbidity_roc, salinity_roc.
        anomaly_result: output of
            anomaly_detection.BaselineAnomalyDetector.score() -- a mapping
            with anomaly_score and is_anomaly.
        feature_vector_version: tracks which feature-extraction code
            version produced `acoustic_features`, per the schema's comment
            on this column.
        baseline_version: tracks which calibration baseline was active when
            `anomaly_result` was computed, per the schema's comment on this
            column.

    Returns:
        The new row's capture_id (int), the join key across the other three
        tables for this window.
    """
    mfcc_fields = {k: v for k, v in acoustic_features.items() if k.startswith("mfcc_")}
    mfcc_blob = json.dumps(mfcc_fields).encode("utf-8")

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO captures (timestamp_utc, audio_filename, duration_sec, sample_rate_hz)
            VALUES (?, ?, ?, ?)
            """,
            (timestamp_utc, audio_filename, duration_sec, sample_rate_hz),
        )
        capture_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO feature_vectors
                (capture_id, mfcc, spectral_centroid, zero_crossing_rate,
                 rms_energy, spectral_flatness, feature_vector_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                capture_id,
                mfcc_blob,
                acoustic_features["spectral_centroid_mean"],
                acoustic_features["zero_crossing_rate_mean"],
                acoustic_features["rms_energy_mean"],
                acoustic_features["spectral_flatness_mean"],
                feature_vector_version,
            ),
        )

        cursor.execute(
            """
            INSERT INTO environmental_readings
                (capture_id, temperature_c, ph, turbidity_ntu, salinity_psu,
                 temp_roc, ph_roc, turbidity_roc, salinity_roc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                capture_id,
                environmental_row["temperature_c"],
                environmental_row["ph"],
                environmental_row["turbidity_ntu"],
                environmental_row["salinity_psu"],
                environmental_row["temp_roc"],
                environmental_row["ph_roc"],
                environmental_row["turbidity_roc"],
                environmental_row["salinity_roc"],
            ),
        )

        cursor.execute(
            """
            INSERT INTO anomaly_flags (capture_id, anomaly_score, is_anomaly, baseline_version)
            VALUES (?, ?, ?, ?)
            """,
            (
                capture_id,
                anomaly_result["anomaly_score"],
                int(bool(anomaly_result["is_anomaly"])),
                baseline_version,
            ),
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return capture_id


def insert_system_health_log(
    conn: sqlite3.Connection,
    *,
    timestamp_utc: str,
    battery_voltage: float,
    solar_charge_w: float,
    enclosure_temp_c: float,
    imu_orientation: Dict[str, float],
    uptime_sec: int,
) -> int:
    """
    Write one system_health_log row.

    Decoupled from captures/feature_vectors/environmental_readings/
    anomaly_flags by design (docs/data-pipeline.md: "system_health_log is
    intentionally decoupled from captures since health telemetry may be
    logged on a different cadence than acoustic sampling") -- callers may
    invoke this once per capture window (edge/capture_loop.py's current
    cadence) or on an independent schedule without needing a capture_id.

    Args:
        conn: open connection from init_db().
        timestamp_utc: ISO 8601 reading timestamp.
        battery_voltage: volts.
        solar_charge_w: watts.
        enclosure_temp_c: degrees Celsius.
        imu_orientation: mapping (e.g. roll_deg/pitch_deg/yaw_deg/
            accel_magnitude_g, per edge/hal/interfaces.py's IMUSensor)
            serialized to JSON and stored as the imu_orientation BLOB.
        uptime_sec: process/device uptime in seconds.

    Returns:
        The new row's log_id.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO system_health_log
            (timestamp_utc, battery_voltage, solar_charge_w, enclosure_temp_c, imu_orientation, uptime_sec)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp_utc,
            battery_voltage,
            solar_charge_w,
            enclosure_temp_c,
            json.dumps(imu_orientation).encode("utf-8"),
            uptime_sec,
        ),
    )
    conn.commit()
    return cursor.lastrowid
