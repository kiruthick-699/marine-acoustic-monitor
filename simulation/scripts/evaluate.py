"""
Anomaly detector evaluation.

Loads the ground-truth anomaly metadata written by run_simulation.py, fits
simulation/pipeline/anomaly_detection.BaselineAnomalyDetector on the
designated calibration-period windows (assumed normal, matching Stage 2's
calibration period in docs/ml-pipeline.md), scores every remaining window,
and prints precision/recall/F1 against the known injected anomalies
(biological calls, vessel events, storm runoff) recorded in the ground
truth.
"""

import argparse
import json
import os
import sys

import pandas as pd

# allow running as `python simulation/scripts/evaluate.py` (repo root isn't
# on sys.path in that form, only via `python -m simulation.scripts...`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.pipeline.anomaly_detection import BaselineAnomalyDetector

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "ground_truth.json")


def evaluate(ground_truth_path: str = GROUND_TRUTH_PATH) -> None:
    """
    Fit the detector on the calibration period and score the rest.

    Args:
        ground_truth_path: path to the ground_truth.json written by
            run_simulation.py.
    """
    with open(ground_truth_path) as f:
        run = json.load(f)

    windows = run["windows"]
    calibration_windows = run["calibration_windows"]

    calibration_set = windows[:calibration_windows]
    evaluation_set = windows[calibration_windows:]

    if not evaluation_set:
        print("No windows after the calibration period to evaluate against -- run with more --n-windows.")
        return

    calibration_df = pd.DataFrame([w["feature_vector"] for w in calibration_set])
    detector = BaselineAnomalyDetector(random_state=0).fit(calibration_df)

    true_positives = false_positives = false_negatives = true_negatives = 0

    for window in evaluation_set:
        feature_vector = pd.Series(window["feature_vector"])
        result = detector.score(feature_vector)
        predicted = result["is_anomaly"]
        actual = window["true_anomaly"]

        if predicted and actual:
            true_positives += 1
        elif predicted and not actual:
            false_positives += 1
        elif not predicted and actual:
            false_negatives += 1
        else:
            true_negatives += 1

    # Precision: of windows the detector flagged, how many were real
    # anomalies. Recall: of real anomalies, how many the detector caught.
    # F1: harmonic mean of the two, a single number balancing both.
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    n_eval = len(evaluation_set)
    n_true_anomalies = sum(1 for w in evaluation_set if w["true_anomaly"])

    print(f"Calibration period: {calibration_windows} windows (assumed normal, used to fit the baseline)")
    print(f"Evaluation period:  {n_eval} windows ({n_true_anomalies} with a true injected anomaly)")
    print()
    print(f"True positives:  {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"True negatives:  {true_negatives}")
    print()
    print(f"Precision: {precision:.3f}  (of windows flagged anomalous, fraction that really were)")
    print(f"Recall:    {recall:.3f}  (of real injected anomalies, fraction the detector caught)")
    print(f"F1:        {f1:.3f}  (harmonic mean of precision and recall)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the anomaly detector against simulation ground truth."
    )
    parser.add_argument(
        "--ground-truth",
        default=GROUND_TRUTH_PATH,
        help="Path to ground_truth.json written by run_simulation.py",
    )
    args = parser.parse_args()
    evaluate(args.ground_truth)
