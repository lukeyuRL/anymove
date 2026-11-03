"""Two-camera orthographic frame, composition and calibrated pixel/world mapping.

Default camera slots reproduce the original front 0deg (u=+X) and side 90deg
(u=-Y). With explicit camera angles the slots retain their legacy names but
represent arbitrary horizontal azimuths. Both views share world Z, level tilt,
scale and ground row; mirrors blender/common.py.
"""
import json, os, pathlib

import numpy as np

from mocap.camera_geometry import project_orthographic, resolve_angles

H, W_FRONT, W_SIDE = 720, 420, 860
M_PER_PX = 1.9 / H
GROUND_V = 630
FRONT_AZIMUTH_DEG, SIDE_AZIMUTH_DEG = 0.0, 90.0
# MOCAP_CREATURE=<assets dir> with a creature.json "layout" overrides the wolf's framing (see blender/common.py)
_cfg = pathlib.Path(os.environ.get("MOCAP_CREATURE", "")) / "creature.json"
if os.environ.get("MOCAP_CREATURE") and _cfg.exists():
    _l = json.load(open(_cfg)).get("layout", {})
    W_FRONT, W_SIDE = _l.get("W_FRONT", W_FRONT), _l.get("W_SIDE", W_SIDE)
    M_PER_PX, GROUND_V = _l.get("M_PER_PX", M_PER_PX), _l.get("GROUND_V", GROUND_V)
    FRONT_AZIMUTH_DEG = _l.get("FRONT_AZIMUTH_DEG", FRONT_AZIMUTH_DEG)
    SIDE_AZIMUTH_DEG = _l.get("SIDE_AZIMUTH_DEG", SIDE_AZIMUTH_DEG)
FRONT_AZIMUTH_DEG, SIDE_AZIMUTH_DEG = resolve_angles(
    FRONT_AZIMUTH_DEG, SIDE_AZIMUTH_DEG, os.environ.get("MOCAP_CAMERA_AZIMUTHS"))
W = W_FRONT + W_SIDE
Z_CENTER = (GROUND_V - H / 2) * M_PER_PX
BG, FLOOR, DIVIDER = (128, 130, 134), (152, 154, 158), (34, 34, 38)


def compose(front_rgba, side_rgba, bg=BG, floor=FLOOR, divider=DIVIDER):
    """Alpha-composite the two RGBA renders over the studio backdrop (grey wall, lighter floor band, divider)."""
    out = np.zeros((H, W, 3), np.float32)
    out[:] = bg
    out[GROUND_V:] = floor
    for img, x0 in ((front_rgba, 0), (side_rgba, W_FRONT)):
        img = img.astype(np.float32)
        a = img[..., 3:4] / 255.0
        h, w = img.shape[:2]
        out[:h, x0:x0 + w] = img[..., :3] * a + out[:h, x0:x0 + w] * (1 - a)
    out[:, W_FRONT - 2:W_FRONT + 2] = divider
    return out.clip(0, 255).astype(np.uint8)


def split(frame):
    """A 1280 x 720 frame (resize first if needed) into its front and side views."""
    return frame[:, :W_FRONT], frame[:, W_FRONT:]


def project(p):
    """World points (..., 3) to pixel coordinates in each view: front (u, v) and side (u, v)."""
    p = np.asarray(p, np.float64)
    front = project_orthographic(p, FRONT_AZIMUTH_DEG, W_FRONT, H, M_PER_PX, Z_CENTER)
    side = project_orthographic(p, SIDE_AZIMUTH_DEG, W_SIDE, H, M_PER_PX, Z_CENTER)
    return front, side


def unproject_front(uv):
    """Front-slot pixels to (local image-right coordinate, z), not world X for arbitrary angles."""
    uv = np.asarray(uv, np.float64)
    return np.stack([(uv[..., 0] - W_FRONT / 2) * M_PER_PX, Z_CENTER + (H / 2 - uv[..., 1]) * M_PER_PX], -1)


def unproject_side(uv):
    """Legacy (-image-right, z) coordinates; this is world Y only at 90deg."""
    uv = np.asarray(uv, np.float64)
    return np.stack([-(uv[..., 0] - W_SIDE / 2) * M_PER_PX, Z_CENTER + (H / 2 - uv[..., 1]) * M_PER_PX], -1)
