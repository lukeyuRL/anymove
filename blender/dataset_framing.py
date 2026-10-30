"""Per-creature Dataset framing policies and reproducibility guards.

Pure Python: importable in GitHub CI without Blender, numpy or CUDA.
"""
import hashlib
import json
import math


def validate_yaw_limit(yaw_max_deg):
    if yaw_max_deg is None:
        return None
    if (isinstance(yaw_max_deg, bool)
            or not isinstance(yaw_max_deg, (float, int))
            or not math.isfinite(yaw_max_deg)
            or not 0 <= yaw_max_deg <= 70):
        raise ValueError("dataset.yaw_max_deg must be finite and in [0,70]")
    return float(yaw_max_deg)


def choose_yaw_degrees(rng, yaw_max_deg=None):
    """Preserve original dragon/wolf yaw policy when no override is set."""
    cap = validate_yaw_limit(yaw_max_deg)
    roll = rng.random()
    if cap is None:
        return rng.uniform(-30, 30) if roll < .75 else (
            rng.uniform(-70, 70) if roll < .85 else 0.0)
    return rng.uniform(-cap, cap) if roll < .85 else 0.0


def framing_signature(rig_info, layout, yaw_max_deg):
    """Changing geometry, labels, camera, or yaw cannot silently reuse images."""
    limits = validate_yaw_limit(yaw_max_deg)
    payload = {
        "version": "anymove_dataset_framing_v1",
        "bones": rig_info["bones"],
        "keypoints": rig_info["keypoints"],
        "layout": {
            key: layout[key] for key in
            ("H", "W_FRONT", "W_SIDE", "M_PER_PX", "GROUND_V", "Z_CENTER")
        },
        "yaw_max_deg": limits,
    }
    encoded = json.dumps(payload, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def same_framing(meta, expected_signature):
    return isinstance(meta, dict) and meta.get("framing_signature") == expected_signature
