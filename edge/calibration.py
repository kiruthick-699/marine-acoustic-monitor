"""
Calibration -- establish Stage 2's baseline (docs/ml-pipeline.md), for real
deployment rather than the offline evaluate.py-style batch fit
simulation/scripts/evaluate.py does.

Runs `config.calibration.calibration_windows` duty-cycle windows through an
already-constructed CaptureLoop (uncalibrated -- each window is written to
storage with a placeholder anomaly result, same as
simulation/scripts/run_simulation.py's approach), collects their joint
feature vectors, fits a BaselineAnomalyDetector, installs it on the loop
(so every subsequent run_one_window() call scores for real), and persists
the fitted model to disk so a process restart doesn't require recalibrating
from scratch.

docs/ml-pipeline.md: "Recalibration itself is an offline/batch operation
performed during bulk data retrieval, not on-device in real time" -- this
module's run_calibration() is for the *initial* calibration period at first
deployment; a full recalibration workflow (re-fitting from retrieved data
with a new baseline_version) is future work, noted in
docs/pi-implementation.md.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import joblib

from edge.capture_loop import CaptureLoop
from edge.config import EdgeConfig
from simulation.pipeline.anomaly_detection import BaselineAnomalyDetector

logger = logging.getLogger(__name__)


def run_calibration(loop: CaptureLoop, config: EdgeConfig) -> BaselineAnomalyDetector:
    """
    Run the initial calibration period and install the resulting baseline
    on `loop`.

    Args:
        loop: a CaptureLoop constructed with detector=None (or any
            detector -- it's overwritten once fitting completes).
        config: EdgeConfig; config.calibration controls window count and
            threshold_sigma (see edge/config.py's CalibrationConfig).

    Returns:
        The fitted BaselineAnomalyDetector (also installed on `loop` via
        loop.set_detector() and persisted to
        config.storage.baseline_model_path).
    """
    n_windows = config.calibration.calibration_windows
    feature_vectors = []

    logger.info("Running calibration: %d windows, assumed normal site conditions...", n_windows)
    for i in range(n_windows):
        summary = loop.run_one_window()
        feature_vectors.append(summary["feature_vector"])
        if (i + 1) % max(n_windows // 10, 1) == 0 or i == n_windows - 1:
            logger.info("  calibration window %d/%d", i + 1, n_windows)

    detector = BaselineAnomalyDetector(threshold_sigma=config.calibration.threshold_sigma)
    detector.fit(feature_vectors)
    loop.set_detector(detector)
    save_baseline(detector, config.storage.baseline_model_path)
    logger.info("Calibration complete. Baseline saved to %s", config.storage.baseline_model_path)

    return detector


def save_baseline(detector: BaselineAnomalyDetector, path: str) -> None:
    """Persist a fitted BaselineAnomalyDetector to disk (joblib)."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    joblib.dump(detector, path)


def load_baseline(path: str) -> Optional[BaselineAnomalyDetector]:
    """
    Load a previously-persisted baseline, or None if no file exists yet
    (first run, calibration not yet performed).
    """
    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)
