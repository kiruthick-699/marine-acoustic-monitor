from edge.telemetry import build_payload, payload_within_budget, serialize_payload


def _sample_inputs():
    acoustic_features = {
        "rms_energy_mean": 0.05,
        "spectral_centroid_mean": 1200.0,
        "spectral_flatness_mean": 0.3,
        "zero_crossing_rate_mean": 0.12,
        # extra full-vector fields that must NOT leak into the payload
        "mfcc_1_mean": -120.0,
        "mfcc_1_std": 8.0,
    }
    environmental_row = {
        "temperature_c": 18.2,
        "ph": 8.05,
        "turbidity_ntu": 3.1,
        "salinity_psu": 35.0,
        "temp_roc": 0.1,
        "ph_roc": 0.0,
        "turbidity_roc": 0.05,
        "salinity_roc": -0.02,
    }
    anomaly_result = {"anomaly_score": 1.23, "is_anomaly": True}
    return acoustic_features, environmental_row, anomaly_result


def test_build_payload_excludes_full_feature_vector_and_raw_audio():
    acoustic_features, environmental_row, anomaly_result = _sample_inputs()
    payload = build_payload(
        timestamp_utc="2026-07-12T00:00:00+00:00",
        acoustic_features=acoustic_features,
        environmental_row=environmental_row,
        anomaly_result=anomaly_result,
    )

    assert set(payload.keys()) == {"t", "acoustic", "env", "anomaly"}
    assert "mfcc_1_mean" not in payload["acoustic"]
    assert set(payload["acoustic"].keys()) == {"rms", "centroid", "flatness", "zcr"}
    assert set(payload["env"].keys()) == {"temp_c", "ph", "turbidity_ntu", "salinity_psu"}
    assert payload["anomaly"] == {"score": 1.23, "flag": True}


def test_payload_within_budget_for_default_max_bytes():
    acoustic_features, environmental_row, anomaly_result = _sample_inputs()
    payload = build_payload(
        timestamp_utc="2026-07-12T00:00:00+00:00",
        acoustic_features=acoustic_features,
        environmental_row=environmental_row,
        anomaly_result=anomaly_result,
    )
    assert payload_within_budget(payload)


def test_payload_within_budget_flags_oversized_payload():
    acoustic_features, environmental_row, anomaly_result = _sample_inputs()
    payload = build_payload(
        timestamp_utc="2026-07-12T00:00:00+00:00",
        acoustic_features=acoustic_features,
        environmental_row=environmental_row,
        anomaly_result=anomaly_result,
    )
    assert not payload_within_budget(payload, max_bytes=10)


def test_serialize_payload_round_trips_as_json():
    import json

    acoustic_features, environmental_row, anomaly_result = _sample_inputs()
    payload = build_payload(
        timestamp_utc="2026-07-12T00:00:00+00:00",
        acoustic_features=acoustic_features,
        environmental_row=environmental_row,
        anomaly_result=anomaly_result,
    )
    raw = serialize_payload(payload)
    assert json.loads(raw) == payload
