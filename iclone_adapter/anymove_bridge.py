"""Explicitly calibrated AnyMove/Blender Euler -> iClone FK channel bridge.

This does NOT perform automatic retargeting. Every mapping and axis transform
must be explicitly approved. Root motion is deliberately excluded in this P1
slice and reported as such.
"""
import argparse
import json
import math

from .motion_contract import MOTION_VERSION, validate_motion
from .schema import ContractError, skeleton_hash, validate_skeleton

MAP_VERSION = "anymove.iclone.bone_map/1"


def qnormalize(q):
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        raise ContractError("quaternion needs w,x,y,z")
    a = [float(x) for x in q]
    if not all(math.isfinite(x) for x in a):
        raise ContractError("nonfinite quaternion")
    length = math.sqrt(sum(x*x for x in a))
    if length < 1e-10:
        raise ContractError("zero-length quaternion")
    return [x / length for x in a]


def qmul(a, b):
    w, x, y, z = a
    p, q, r, s = b
    return [w*p-x*q-y*r-z*s, w*q+x*p+y*s-z*r,
            w*r-x*s+y*p+z*q, w*s+x*r-y*q+z*p]


def qinverse_unit(q):
    w, x, y, z = qnormalize(q)
    return [w, -x, -y, -z]


def from_euler_xyz(a):
    """Blender XYZ: R = Rz @ Ry @ Rx, angles in radians."""
    if len(a) != 3 or not all(math.isfinite(float(x)) for x in a):
        raise ContractError("invalid Blender Euler XYZ")
    x, y, z = [float(v)*.5 for v in a]
    qx = [math.cos(x), math.sin(x), 0., 0.]
    qy = [math.cos(y), 0., math.sin(y), 0.]
    qz = [math.cos(z), 0., 0., math.sin(z)]
    return qnormalize(qmul(qmul(qz, qy), qx))


def to_euler_xyz(q):
    w, x, y, z = qnormalize(q)
    r20 = 2*(x*z - w*y)
    pitch = math.asin(max(-1., min(1., -r20)))
    if abs(abs(r20) - 1.) < 1e-7:
        # Gimbal singularity: choose X=0 and solve Z.
        yaw = math.atan2(-2*(x*y - w*z), 1 - 2*(x*x+y*y))
        return [0., pitch, yaw]
    roll = math.atan2(2*(y*z + w*x), 1 - 2*(x*x+y*y))
    yaw = math.atan2(2*(x*y + w*z), 1 - 2*(y*y+z*z))
    return [roll, pitch, yaw]


def make_mapping_template(motion, target):
    validate_skeleton(target)
    source_bones = motion.get("bones")
    if not isinstance(source_bones, list) or not source_bones:
        raise ContractError("AnyMove motion missing ordered bones")
    return {
        "schema_version": MAP_VERSION,
        "target_skeleton_hash": skeleton_hash(target),
        "root_policy": "skip",
        "bone_map": {
            name: {"target_id": None, "axis_quaternion_wxyz": [1., 0., 0., 0.],
                   "calibrated": False}
            for name in source_bones[1:]
        },
        "note": "Set target_id and calibrated=true ONLY after checking Bone axis/rest alignment.",
    }


def convert(motion, target, mapping):
    """Convert mapped local Euler XYZ to target Layer channel XYZ.

    axis_quaternion_wxyz represents an externally calibrated change of basis
    from each Blender source bone local axes into iClone FK channel axes.
    No bone length/IK/Root retargeting is claimed.
    """
    validate_skeleton(target)
    if mapping.get("schema_version") != MAP_VERSION:
        raise ContractError("unsupported mapping schema")
    if mapping.get("target_skeleton_hash") != skeleton_hash(target):
        raise ContractError("mapping target skeleton mismatch")
    if mapping.get("root_policy") != "skip":
        raise ContractError("P1 supports ONLY explicit root_policy=skip")
    names = motion.get("bones")
    frames = motion.get("euler")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ContractError("AnyMove bones invalid or duplicate")
    if not isinstance(frames, list) or not frames:
        raise ContractError("AnyMove euler frames missing")
    if any(not isinstance(row, list) or len(row) != len(names) for row in frames):
        raise ContractError("AnyMove frame/bone count inconsistent")
    spec = mapping.get("bone_map")
    if not isinstance(spec, dict) or not spec:
        raise ContractError("missing bone mapping")
    target_ids = {b["id"] for b in target["bones"]}
    used, tracks = set(), {}
    for source, entry in spec.items():
        if source not in names[1:]:
            raise ContractError("unknown or root source bone: " + source)
        if not isinstance(entry, dict) or not entry.get("calibrated"):
            raise ContractError("bone not axis-calibrated: " + source)
        dest = entry.get("target_id")
        if dest not in target_ids or dest in used:
            raise ContractError("target Bone missing or reused: " + str(dest))
        used.add(dest)
        basis = qnormalize(entry.get("axis_quaternion_wxyz"))
        basis_inv = qinverse_unit(basis)
        source_index = names.index(source)
        out = []
        for row in frames:
            q = from_euler_xyz(row[source_index])
            target_q = qmul(qmul(basis, q), basis_inv)
            out.append(to_euler_xyz(target_q))
        tracks[dest] = out
    fps = motion.get("fps", 24)
    if type(fps) is float and fps.is_integer():
        fps = int(fps)
    out = {
        "schema_version": MOTION_VERSION,
        "space": "iclone_fk_layer_xyz_radians",
        "skeleton_hash": skeleton_hash(target),
        "fps": fps,
        "frame_count": len(frames),
        "tracks": tracks,
        "limitations": ["root_translation_skipped", "root_rotation_skipped",
                        "no_length_retargeting", "no_fk_ik_reconciliation"],
        "notes": "requires runtime verification of per-bone iClone FK axis calibration",
    }
    validate_motion(out, target)
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert AnyMove Euler motion with explicit per-bone calibration")
    parser.add_argument("mode", choices=("map", "convert"))
    parser.add_argument("motion_json")
    parser.add_argument("skeleton_json")
    parser.add_argument("output_json")
    parser.add_argument("--map-json", default=None)
    args = parser.parse_args()
    def read(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    source, skeleton = read(args.motion_json), read(args.skeleton_json)
    if args.mode == "map":
        result = make_mapping_template(source, skeleton)
    else:
        if not args.map_json:
            parser.error("convert requires --map-json FILE")
        result = convert(source, skeleton, read(args.map_json))
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
