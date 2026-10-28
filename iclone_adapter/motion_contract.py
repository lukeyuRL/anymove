"""Strict P1 FK channel contract: NOT Blender Euler, retargeting or quaternions."""
import math
from .schema import ContractError, skeleton_hash, validate_skeleton

MOTION_VERSION = "anymove.iclone.fk_channels/1"


def validate_motion(motion, skeleton):
    validate_skeleton(skeleton)
    if motion.get("schema_version") != MOTION_VERSION:
        raise ContractError("expected explicit iClone FK channels; do not use Blender motion.json")
    if motion.get("skeleton_hash") != skeleton_hash(skeleton):
        raise ContractError("skeleton identity mismatch")
    fps, count = motion.get("fps"), motion.get("frame_count")
    if type(fps) is not int or not 1 <= fps <= 240:
        raise ContractError("integer fps 1..240 required")
    if type(count) is not int or not 1 <= count <= 50000:
        raise ContractError("frame_count out of range")
    tracks = motion.get("tracks")
    if not isinstance(tracks, dict) or not tracks:
        raise ContractError("at least one bone track required")
    known = {b["id"] for b in skeleton["bones"]}
    for bone_id, values in tracks.items():
        if bone_id not in known:
            raise ContractError("unknown bone ID: " + bone_id)
        if not isinstance(values, list) or len(values) != count:
            raise ContractError("inconsistent track frame_count")
        for frame in values:
            if not isinstance(frame, list) or len(frame) != 3:
                raise ContractError("rotation must be XYZ radians")
            if not all(type(v) in (int, float) and math.isfinite(v) and
                       abs(v) <= 32 * math.pi for v in frame):
                raise ContractError("invalid rotation channel")
    if motion.get("space") != "iclone_fk_layer_xyz_radians":
        raise ContractError("requires explicit iClone FK layer channel space")
    return True


def make_sine_test(skeleton, bone_id, fps=24, frames=48, amplitude_deg=10.0):
    """Single-axis synthetic motion to verify host channel semantics."""
    validate_skeleton(skeleton)
    if bone_id not in {b["id"] for b in skeleton["bones"]}:
        raise ContractError("unknown bone ID")
    if frames < 2:
        raise ContractError("requires at least two frames")
    samples = [[math.radians(amplitude_deg) *
                math.sin(2.0 * math.pi * i / (frames - 1)), 0., 0.]
               for i in range(frames)]
    motion = {
        "schema_version": MOTION_VERSION,
        "space": "iclone_fk_layer_xyz_radians",
        "skeleton_hash": skeleton_hash(skeleton),
        "fps": fps, "frame_count": frames,
        "tracks": {bone_id: samples},
        "notes": "synthetic FK X-channel diagnostic, not AnyMove mocap",
    }
    validate_motion(motion, skeleton)
    return motion
