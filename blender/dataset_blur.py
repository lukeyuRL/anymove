"""Synthetic pose blur policy, independent of Blender and NumPy.

The profile ID is part of every sample's metadata. Changing the policy causes
old samples to be re-rendered instead of silently reusing old PNGs.
"""
import math

BLUR_PROFILE = "soft_blur_v1_20pct_shutter_0.2_to_0.4"
BLUR_PROBABILITY = 0.20
SHUTTER_MIN = 0.20
SHUTTER_MAX = 0.40


def choose_blur(rng):
    """Return (enabled, shutter); rng supports random() and uniform()."""
    enabled = bool(rng.random() < BLUR_PROBABILITY)
    shutter = float(rng.uniform(SHUTTER_MIN, SHUTTER_MAX)) if enabled else None
    return enabled, shutter


def profile_matches(meta):
    return isinstance(meta, dict) and meta.get("blur_profile") == BLUR_PROFILE


def scale_matches(meta, scale):
    try:
        actual = float(meta["scale"])
    except (TypeError, KeyError, ValueError, OverflowError):
        return False
    return math.isfinite(actual) and math.isclose(actual, scale, rel_tol=0, abs_tol=1e-9)
