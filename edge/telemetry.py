"""
Telemetry payload assembly.

DECISIONS.md / docs/data-pipeline.md step 7: "Assemble a compact telemetry
payload (feature summary + environmental readings + anomaly alert if
flagged) and transmit over the low-bandwidth link." Two hard constraints
this module exists to enforce:

1. Compact only -- never the full feature vector. The joint feature vector
   (simulation/pipeline/feature_extraction.py) includes 13 MFCC
   coefficients x mean/std (26 floats) plus 4 more acoustic stats x
   mean/std -- reasonable for local SQLite storage, too heavy for a
   LoRa-class link's per-message budget. build_payload() picks a small
   fixed summary (4 acoustic scalars) rather than forwarding the full
   vector.
2. Raw audio never appears here, structurally -- build_payload()'s
   signature has no audio parameter at all, so it's not possible to
   accidentally wire a raw audio array into a payload this module produces.
"""

import json
from datetime import datetime
from typing import Dict

# A JSON dict payload with 4 field groups plus a timestamp comfortably fits
# within a single LoRa message at low-to-mid spreading factors (SF7-SF9 give
# roughly 220+ bytes of usable payload in most regional plans; only the
# slowest SF10-12 settings get tighter than this). 220, not the frequently-
# quoted 222/242-byte SF7 ceiling, leaves a little headroom for whatever
# framing a real link driver (edge/hal/real.py's RealTelemetryLink) adds on
# top of this payload's raw bytes.
DEFAULT_MAX_PAYLOAD_BYTES = 220


def build_payload(
    *,
    timestamp_utc: str,
    acoustic_features: Dict[str, float],
    environmental_row: Dict[str, float],
    anomaly_result: Dict[str, float],
) -> Dict:
    """
    Build one duty-cycle window's compact telemetry payload.

    Args:
        timestamp_utc: ISO 8601 capture timestamp.
        acoustic_features: output of
            simulation.pipeline.feature_extraction.extract_acoustic_features()
            -- only a small summary subset is forwarded (see module
            docstring), not the full dict.
        environmental_row: this window's environmental reading including
            rate-of-change (edge/capture_loop.py's env_row).
        anomaly_result: output of
            simulation.pipeline.anomaly_detection.BaselineAnomalyDetector.score()
            (or capture_loop's uncalibrated placeholder).

    Returns:
        Small JSON-serializable dict:
            {
                "t": int (Unix epoch seconds, not the full ISO 8601 string
                    -- shaves ~15-19 bytes versus a timestamp string, which
                    matters against a LoRa-class per-message budget; Tier 2
                    SQLite keeps the full-precision ISO timestamp),
                "acoustic": {"rms", "centroid", "flatness", "zcr"},
                "env": {"temp_c", "ph", "turbidity_ntu", "salinity_psu"},
                "anomaly": {"score", "flag"},
            }
    """
    # Rounded to a few significant digits -- telemetry is airtime/power
    # constrained (docs/hardware-spec.md: transmission is typically the
    # highest instantaneous draw event in the cycle), and full float
    # precision buys nothing here: SQLite (Tier 2, local) already holds the
    # unrounded values for every window this payload summarizes.
    epoch_seconds = int(datetime.fromisoformat(timestamp_utc).timestamp())
    return {
        "t": epoch_seconds,
        "acoustic": {
            "rms": round(acoustic_features["rms_energy_mean"], 4),
            "centroid": round(acoustic_features["spectral_centroid_mean"], 1),
            "flatness": round(acoustic_features["spectral_flatness_mean"], 3),
            "zcr": round(acoustic_features["zero_crossing_rate_mean"], 4),
        },
        "env": {
            "temp_c": round(environmental_row["temperature_c"], 2),
            "ph": round(environmental_row["ph"], 3),
            "turbidity_ntu": round(environmental_row["turbidity_ntu"], 2),
            "salinity_psu": round(environmental_row["salinity_psu"], 2),
        },
        "anomaly": {
            "score": round(anomaly_result["anomaly_score"], 3),
            "flag": bool(anomaly_result["is_anomaly"]),
        },
    }


def serialize_payload(payload: Dict) -> bytes:
    """
    Serialize a payload to compact JSON bytes -- the wire format a
    TelemetryLink implementation (edge/hal/mock.py or a future
    edge/hal/real.py driver) would actually transmit.

    A real LoRa/cellular driver may prefer a packed binary struct over JSON
    if airtime/power budget is tight (docs/hardware-spec.md: telemetry
    transmission is typically the highest instantaneous draw event in the
    cycle) -- this function is the seam where that swap would happen,
    without touching build_payload() or its callers.
    """
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def payload_within_budget(payload: Dict, max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES) -> bool:
    """
    Check a payload's serialized size against `max_bytes`
    (edge/config.yaml: hardware.telemetry.max_payload_bytes).

    Returns:
        True if serialize_payload(payload) fits within max_bytes.
    """
    return len(serialize_payload(payload)) <= max_bytes
