"""Calibrated two-view orthographic geometry for arbitrary *azimuth* angles.

Convention (Blender world, Z up): 0 deg is the original Front camera at
(0, -distance, z_center), 90 deg the original Side camera at (-distance, 0,
z_center). Camera image-right is (cos(azimuth), -sin(azimuth), 0).
Both cameras remain level, orthographic, same scale, and share a vertical axis.

"front" / "side" are *slot names* for compatibility with existing two-image
Readers and det.npz; they do not imply a physical 0/90 degree camera.
Perspective or camera elevation requires a different calibration contract.
"""
import math

import numpy as np


DEFAULT_ANGLES_DEG = (0.0, 90.0)
MAX_PAIR_CONDITION = 10.0


def _angle(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric azimuth in degrees")
    try:
        angle = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric azimuth in degrees") from exc
    if not math.isfinite(angle):
        raise ValueError(f"{name} must be a finite numeric azimuth in degrees")
    return angle % 360.0


def horizontal_matrix(front_angle_deg, side_angle_deg):
    """Rows project (world X,Y) onto each view's image-right axis."""
    f, s = math.radians(_angle(front_angle_deg, "front")), math.radians(_angle(side_angle_deg, "side"))
    return np.array(((math.cos(f), -math.sin(f)), (math.cos(s), -math.sin(s))), dtype=np.float64)


def pair_condition(front_angle_deg, side_angle_deg):
    """Condition of horizontal triangulation; 1 is optimal, infinity singular."""
    return float(np.linalg.cond(horizontal_matrix(front_angle_deg, side_angle_deg)))


def validate_pair(front_angle_deg, side_angle_deg, max_condition=MAX_PAIR_CONDITION):
    """Return normalized angles or fail closed on ill-conditioned camera pairs."""
    front = _angle(front_angle_deg, "front")
    side = _angle(side_angle_deg, "side")
    condition = pair_condition(front, side)
    if not math.isfinite(condition) or condition > max_condition:
        raise ValueError(
            f"DEGENERATE_CAMERA_PAIR: azimuths {front:g}/{side:g} deg have "
            f"horizontal triangulation condition {condition:.3g} "
            f"(maximum {max_condition:g}). Choose two non-opposite, "
            "non-parallel camera axes, ideally separated by about 60-120 deg."
        )
    return front, side


def resolve_angles(front_angle_deg, side_angle_deg, override=None):
    """Resolve creature.json angles with an optional explicit experiment override.

    MOCAP_CAMERA_AZIMUTHS='315,45' overrides *only* camera azimuths, never
    framing/scale. The override must be used consistently by Blender, Reader
    training/detection and Solver to avoid silently wrong motion.json files.
    """
    if override is not None and str(override).strip():
        values = str(override).split(",")
        if len(values) != 2:
            raise ValueError("MOCAP_CAMERA_AZIMUTHS must be two degrees, e.g. '315,45'")
        return validate_pair(values[0].strip(), values[1].strip())
    return validate_pair(front_angle_deg, side_angle_deg)


def layout_angles(layout):
    """Read configured slot angles, defaulting to the historical 0/90 rig."""
    return validate_pair(getattr(layout, "FRONT_AZIMUTH_DEG", 0.0),
                         getattr(layout, "SIDE_AZIMUTH_DEG", 90.0))


def project_orthographic(points, angle_deg, width, height, metres_per_pixel, z_center):
    """Project world points (...,3) to (u,v) under the level ORTHO contract."""
    points = np.asarray(points, dtype=np.float64)
    a = math.radians(_angle(angle_deg, "camera"))
    u = width / 2 + (math.cos(a) * points[..., 0] - math.sin(a) * points[..., 1]) / metres_per_pixel
    v = height / 2 - (points[..., 2] - z_center) / metres_per_pixel
    return np.stack((u, v), axis=-1)


def triangulate_orthographic(front_uv, side_uv, front_conf, side_conf, *,
                             front_width, side_width, height, metres_per_pixel,
                             z_center, front_angle_deg=0, side_angle_deg=90,
                             front_height_weight=0.25):
    """Reconstruct world XYZ using both horizontal projections and weighted Z.

    This inverts the calibrated 2x2 horizontal matrix; it does not assume that
    image u means world X or -Y. Confidence changes only the Z fusion here.
    """
    front, side = validate_pair(front_angle_deg, side_angle_deg)
    f = np.asarray(front_uv, dtype=np.float64)
    s = np.asarray(side_uv, dtype=np.float64)
    fc = np.asarray(front_conf, dtype=np.float64)
    sc = np.asarray(side_conf, dtype=np.float64)
    if f.shape != s.shape or f.ndim < 2 or f.shape[-1] != 2 or fc.shape != f.shape[:-1] or sc.shape != f.shape[:-1]:
        raise ValueError("Two-view UV/confidence shapes are incompatible")
    if not all(np.isfinite(v).all() for v in (f, s, fc, sc)):
        raise ValueError("Non-finite UV or confidence in two-view detections")
    if not math.isfinite(front_height_weight) or front_height_weight < 0:
        raise ValueError("front_height_weight must be finite and nonnegative")
    q = np.stack(((f[..., 0] - front_width / 2) * metres_per_pixel,
                  (s[..., 0] - side_width / 2) * metres_per_pixel), axis=-1)
    xy = q @ np.linalg.inv(horizontal_matrix(front, side)).T
    fz = z_center + (height / 2 - f[..., 1]) * metres_per_pixel
    sz = z_center + (height / 2 - s[..., 1]) * metres_per_pixel
    wf = front_height_weight * np.clip(fc, 1e-3, 1)
    ws = np.clip(sc, 1e-3, 1)
    z = (fz * wf + sz * ws) / np.maximum(wf + ws, 1e-9)
    return np.concatenate((xy, z[..., None]), axis=-1)


def verify_record_angles(record_angles, expected_angles, source):
    """Reject stale Reader/detection/dataset metadata under custom camera views.

    Historical records without angles are supported only for original 0/90.
    Compare modulo 360, so -45 and 315 describe the same camera.
    """
    exp = validate_pair(*expected_angles)
    if record_angles is None:
        if np.allclose(exp, DEFAULT_ANGLES_DEG, atol=1e-6, rtol=0):
            return "legacy_default_0_90"
        raise ValueError(f"CAMERA_ANGLE_METADATA_MISSING: {source} has no camera angles; "
                         f"the active layout is {exp}. Regenerate the {source} with matching cameras.")
    obs = np.asarray(record_angles, dtype=np.float64)
    if obs.shape != (2,) or not np.isfinite(obs).all():
        raise ValueError(f"CAMERA_ANGLE_METADATA_INVALID: {source} requires exactly two finite angles")
    obs = np.mod(obs, 360.0)
    delta = np.mod(obs - np.asarray(exp) + 180.0, 360.0) - 180.0
    if not np.all(np.abs(delta) < 1e-5):
        raise ValueError(f"CAMERA_ANGLE_MISMATCH: {source} contains {obs.tolist()} "
                         f"but active layout requires {list(exp)}")
    return "camera_angles_verified"

def verify_reference_file(reference_path, expected_angles):
    """Check that the actual reference image is tied to the configured cameras."""
    import hashlib
    import json
    import pathlib

    path = pathlib.Path(reference_path)
    manifest = path.with_suffix(".camera.json")
    if not manifest.is_file():
        return verify_record_angles(None, expected_angles, f"reference {path.name}")
    with manifest.open(encoding="utf-8") as fh:
        record = json.load(fh)
    result = verify_record_angles(record.get("camera_azimuth_deg"), expected_angles,
                                  f"reference {manifest.name}")
    expected_sha = record.get("reference_sha256")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if not isinstance(expected_sha, str) or expected_sha != actual_sha:
        raise ValueError(f"CAMERA_REFERENCE_SHA_MISMATCH: reference image {path.name} "
                         "was modified or its camera manifest is stale.")
    return result
