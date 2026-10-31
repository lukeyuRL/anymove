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

    # Experimental single root-layer position/rotation, with explicit
    # source identity and XYZ channel semantics. The original pelvis_layer
    # remains supported for the already validated Dragon profile.
    positions=motion.get("position_tracks")
    root=motion.get("root_motion")
    if (positions is not None or root is not None):
        if not isinstance(root,dict) or root.get("mode") not in ("pelvis_layer","root_layer"):
            raise ContractError("position/root track requires explicit root-layer metadata")
        rid=root.get("target_bone_id")
        source_root=root.get("source_root")
        if rid not in tracks or rid not in known:
            raise ContractError("root rotation track/identity missing")
        if root["mode"]=="pelvis_layer":
            if source_root!="pelvis":
                raise ContractError("root Pelvis rotation track/identity missing")
        elif (not isinstance(source_root,str) or not source_root or
              not any(b.get("id")==rid and b.get("name")==source_root
                      for b in skeleton["bones"])):
            raise ContractError("generic root target Bone name/identity mismatch")
        if root.get("calibrated") is not False:
            raise ContractError("experimental root must remain calibrated=false")
        components=root.get("components")
        if components not in ("rotation_only","rotation_and_translation"):
            raise ContractError("invalid experimental root components")
        if components=="rotation_and_translation":
            scale=root.get("root_position_scale")
            if (type(scale) not in (int,float) or not math.isfinite(scale) or
                    scale<=0 or scale>10000):
                raise ContractError("root Position scale must be explicit and positive")
            if not isinstance(positions,dict) or set(positions)!={rid}:
                raise ContractError("root Position track must target only pelvis")
            values=positions[rid]
            if not isinstance(values,list) or len(values)!=count:
                raise ContractError("root Position frame count mismatch")
            for row in values:
                if (not isinstance(row,list) or len(row)!=3 or
                    not all(type(x) in (int,float) and math.isfinite(x)
                            and abs(x)<1000000 for x in row)):
                    raise ContractError("invalid root Position sample")
        elif positions is not None:
            raise ContractError("rotation-only root cannot include Position tracks")
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
