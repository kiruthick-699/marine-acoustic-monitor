"""
Feature extraction pipeline.

Implements Stage 1 of the ML pipeline described in docs/ml-pipeline.md: for
each duty-cycle capture window, extract acoustic features from the audio and
normalized environmental features (including rate-of-change), then
concatenate both into a single joint feature vector. This joint vector is
the "correlate, don't isolate" design from DECISIONS.md and docs/related-
work.md -- acoustic and environmental signal are combined on-device into one
feature space before anomaly detection sees either, rather than analyzing
the two streams separately.

This module operates on the synthetic data generators in
simulation/data_generator/ so the extraction and (later) anomaly-detection
stages can be built and tested before any real hardware or field recordings
exist -- see DECISIONS.md, project status is planning/no hardware yet.
"""

from typing import Dict

import librosa
import numpy as np
import pandas as pd

# Reference baseline stats for normalizing environmental readings, matching
# the generation parameters in simulation/data_generator/synthetic_environmental.py
# (baseline mean and diel-swing amplitude / typical spread per parameter).
# In a real deployment these would instead be fit from the initial
# calibration period, per docs/ml-pipeline.md's Stage 2 calibration baseline
# -- fixed reference constants are a placeholder appropriate only for this
# synthetic-data pipeline.
ENV_REFERENCE_STATS = {
    "temperature_c": {"mean": 18.0, "scale": 2.5},
    "ph": {"mean": 8.05, "scale": 0.08},
    "turbidity_ntu": {"mean": 3.0, "scale": 1.0},
    "salinity_psu": {"mean": 35.0, "scale": 0.5},
}

# Maps each absolute-value column to its rate-of-change column, matching the
# naming used by compute_rate_of_change() in synthetic_environmental.py and
# the environmental_readings schema in docs/data-pipeline.md.
_ROC_COLUMN_MAP = {
    "temperature_c": "temp_roc",
    "ph": "ph_roc",
    "turbidity_ntu": "turbidity_roc",
    "salinity_psu": "salinity_roc",
}

N_MFCC = 13


def extract_acoustic_features(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """
    Extract the acoustic feature set from docs/ml-pipeline.md Stage 1.

    Each feature is computed per short analysis frame across the capture
    window (librosa's default frame/hop sizing), then reduced to a mean and
    standard deviation across frames -- the mean captures the feature's
    typical level over the window, the std captures how much it varies
    within the window (e.g. a brief loud event raises RMS std even if mean
    RMS stays close to the ambient floor).

    Feature-by-feature reasoning, in terms of what it captures and why it
    helps separate ambient / biological / vessel sound (see
    synthetic_audio.py for how each of those is synthesized):

    - MFCCs (Mel-frequency cepstral coefficients): a compact representation
      of the spectral envelope ("timbre") of the sound, standard in audio
      and bioacoustic classification. A tonal vessel signature (60/90/120 Hz
      harmonics), a biological FM whistle sweeping 4-12 kHz, and pink-noise
      ambient background each have distinctly different spectral envelopes,
      so they land in different regions of MFCC space even when other
      single-number features (like RMS) are similar.
    - Spectral centroid: the "brightness" / center of mass of the spectrum.
      The synthetic biological whistle (4-12 kHz) sits far higher in
      frequency than the synthetic vessel tonal event (60-120 Hz) or pink
      ambient noise (energy concentrated at low frequencies), so centroid
      alone strongly separates a biological event from the other two.
    - Zero-crossing rate (ZCR): how often the waveform crosses zero,
      correlating with noisiness/percussiveness. Broadband, impulsive
      content (the biological click train, vessel cavitation noise) raises
      ZCR relative to smoother tonal or colored-noise content, helping
      distinguish impulsive events from steady tonal or ambient signal.
    - RMS energy: overall loudness of the window. Both synthetic anomaly
      types (biological call, vessel passage) are deliberately louder than
      the ambient noise floor (see synthetic_audio.py's amplitude choices),
      so RMS is a first-pass signal for "something happened in this window"
      even before considering which kind of event it was.
    - Spectral flatness: how tonal (near 0) vs. noise-like (near 1) the
      spectrum is. The vessel event's narrowband 60/90/120 Hz tones pull
      flatness down; broadband cavitation noise, the biological click
      train, and pink ambient noise all pull it up. Flatness is what lets
      the feature set separate "there's a tonal component present" (vessel)
      from "this is broadband, tonal or not" (ambient/clicks), which
      centroid and RMS alone don't distinguish.

    Args:
        audio: 1D audio array for one capture window (as produced by
            simulation/data_generator/synthetic_audio.py).
        sample_rate: samples per second (Hz) of `audio`.

    Returns:
        Flat dict of named features: mfcc_1_mean..mfcc_13_mean,
        mfcc_1_std..mfcc_13_std, spectral_centroid_mean/std,
        zero_crossing_rate_mean/std, rms_energy_mean/std,
        spectral_flatness_mean/std.
    """
    audio = np.asarray(audio, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    rms = librosa.feature.rms(y=audio)[0]
    flatness = librosa.feature.spectral_flatness(y=audio)[0]

    features: Dict[str, float] = {}

    for i in range(mfcc.shape[0]):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    features["zero_crossing_rate_std"] = float(np.std(zcr))
    features["rms_energy_mean"] = float(np.mean(rms))
    features["rms_energy_std"] = float(np.std(rms))
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))

    return features


def extract_environmental_features(env_row_with_rate_of_change: pd.Series) -> Dict[str, float]:
    """
    Normalize one window's environmental readings (and their pre-computed
    rate-of-change) into the environmental half of the joint feature vector.

    Normalization is a z-score against reference baseline stats
    (ENV_REFERENCE_STATS): (value - mean) / scale. This puts temperature,
    pH, turbidity, and salinity -- which live on completely different
    numeric scales and units -- onto comparable magnitudes, which matters
    once they're concatenated with acoustic features and rate-of-change
    values into one joint vector for Stage 2's anomaly detector: without
    normalization, a parameter with naturally large raw numbers (e.g.
    salinity ~35) could dominate distance/reconstruction-based anomaly
    scoring over a parameter with naturally small numbers (e.g. pH ~8),
    regardless of which one is actually behaving anomalously.

    Rate-of-change values are divided by the same parameter's `scale`
    (not re-centered, since a rate-of-change is already roughly zero-
    centered at baseline) so a given absolute swing and the equivalent
    per-window swing are expressed in the same normalized units.

    Args:
        env_row_with_rate_of_change: one row (pandas Series) from the
            DataFrame produced by
            simulation/data_generator/synthetic_environmental.py's
            compute_rate_of_change(), containing temperature_c, ph,
            turbidity_ntu, salinity_psu, and their *_roc columns.

    Returns:
        Flat dict of named, normalized features: temperature_c_norm,
        ph_norm, turbidity_ntu_norm, salinity_psu_norm, and
        temp_roc_norm, ph_roc_norm, turbidity_roc_norm, salinity_roc_norm.
    """
    row = env_row_with_rate_of_change
    features: Dict[str, float] = {}

    for param, stats in ENV_REFERENCE_STATS.items():
        value = float(row[param])
        features[f"{param}_norm"] = (value - stats["mean"]) / stats["scale"]

        roc_col = _ROC_COLUMN_MAP[param]
        roc_value = float(row[roc_col])
        features[f"{roc_col}_norm"] = roc_value / stats["scale"]

    return features


def build_joint_feature_vector(
    acoustic_features: Dict[str, float], environmental_features: Dict[str, float]
) -> pd.Series:
    """
    Concatenate acoustic and environmental features into one joint vector.

    This is the "correlate, don't isolate" step from DECISIONS.md and
    docs/related-work.md: acoustic and environmental features are combined
    into a single feature space here, before Stage 2's anomaly detector
    sees either, so the detector can learn correlated structure across both
    (e.g. a turbidity spike co-occurring with unusual acoustic content)
    rather than only ever seeing them as two independently-scored streams.

    Acoustic and environmental feature names are disjoint by construction
    (acoustic names come from MFCC/spectral/ZCR/RMS naming, environmental
    names all end in `_norm`), so no prefixing is needed to avoid
    collisions when merging.

    Args:
        acoustic_features: output of extract_acoustic_features().
        environmental_features: output of extract_environmental_features().

    Returns:
        pandas Series indexed by feature name, holding both feature groups
        concatenated into one vector -- this is what would be persisted as
        a `feature_vectors` row (docs/data-pipeline.md) and fed to Stage 2.
    """
    return pd.Series({**acoustic_features, **environmental_features})


if __name__ == "__main__":
    from simulation.data_generator.synthetic_audio import generate_duty_cycle_sample
    from simulation.data_generator.synthetic_environmental import generate_environmental_series

    # One duty-cycle window: synthetic audio with a vessel passage event,
    # paired with one synthetic environmental reading, demonstrating the
    # full Stage 1 flow end-to-end on synthetic data.
    audio, audio_meta = generate_duty_cycle_sample(
        duration_s=30, sample_rate=22050, inject_anomaly="vessel"
    )
    env_readings, env_meta = generate_environmental_series(n_windows=1, inject_anomaly_at=None)
    env_row = env_readings.iloc[0]

    acoustic_features = extract_acoustic_features(audio, sample_rate=22050)
    environmental_features = extract_environmental_features(env_row)
    joint_vector = build_joint_feature_vector(acoustic_features, environmental_features)

    print("Audio ground truth:", audio_meta)
    print("Environmental ground truth:", env_meta)
    print(f"\nJoint feature vector ({len(joint_vector)} features):")
    print(joint_vector)
