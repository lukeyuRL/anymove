"""Rig-agnostic, same-rig AnyMove -> iClone FK preview package.

Any source creature is accepted if its motion/rig_info/snapshot agree. This
does NOT retarget between different rigs, infer iClone bind matrices, or
certify root Position or FK axes. It produces a guarded, hashed preview
that the existing iClone "Write FK motion" panel can load.
"""
import argparse
import hashlib
import json
import math
import pathlib

from .anymove_bridge import from_euler_xyz, qmul, qinverse_unit, to_euler_xyz
from .motion_contract import MOTION_VERSION, validate_motion
from .rest_basis import relative_rest_basis
from .root_motion import convert_pelvis_root
from .schema import ContractError, skeleton_hash, validate_skeleton
from .verify_import import audit_rig

PROFILE_VERSION = "anymove.iclone.generic_same_rig/1"
FK_FILENAME = "creature_fk.json"
MANIFEST_FILENAME = "creature_import_manifest.json"
REPORT_FILENAME = "creature_import_report.json"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _write(path, data):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _select_bones(requested, allowed):
    if requested == "all":
        chosen = list(allowed)
    elif isinstance(requested, str):
        chosen = [s.strip() for s in requested.split(",") if s.strip()]
    else:
        chosen = list(requested)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ContractError("select unique, nonempty source Bone names")
    unknown = set(chosen) - set(allowed)
    if unknown:
        raise ContractError("unknown or source root Bone(s): " + ", ".join(sorted(unknown)))
    return chosen


def _validated_rig(source_info, target):
    validate_skeleton(target)
    bones = source_info.get("bones")
    if not isinstance(bones, list) or len(bones) < 2:
        raise ContractError("same-rig preview requires at least root + one child Bone")
    names = [b.get("name") for b in bones]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ContractError("source rig Bone names must be nonempty and unique")
    for i, bone in enumerate(bones):
        parent = bone.get("parent")
        if type(parent) is not int or (parent != -1 if i == 0 else not (0 <= parent < i)):
            raise ContractError("source skeleton requires one first root and parent-before-child order")
        rows = bone.get("matrix_local")
        if (not isinstance(rows, list) or len(rows) != 4 or
                any(not isinstance(row, list) or len(row) != 4 for row in rows)):
            raise ContractError("source Bone rest matrix must be 4x4")
    report = audit_rig(source_info, target)
    if report["result"] != "compatible_static_rest_snapshot":
        raise ContractError(
            "Same-rig rest preflight FAILED; missing %s, ambiguous %s, parents %s, "
            "max rotation component %s, max position error %s. "
            "Use a newly imported unposed source rig, not an arbitrary retarget."
            % (report["missing_names"], report["ambiguous_names"],
               report["mismatched_parents"],
               report["max_rotation_matrix_component_error"],
               report["max_position_error_source_units"]))
    if not math.isfinite(float(report["source_to_target_uniform_scale"])):
        raise ContractError("nonfinite source-to-target import scale")
    return names, report


def _source_motion(motion, names):
    if motion.get("bones") != names:
        raise ContractError("source motion Bone names/order differ from source rig_info.json")
    frames = motion.get("euler")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ContractError("source motion needs at least two Euler frames")
    if any(not isinstance(frame, list) or len(frame) != len(names) for frame in frames):
        raise ContractError("source motion frames have incorrect Bone counts")
    for frame in frames:
        for row in frame:
            if (not isinstance(row, list) or len(row) != 3 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in row)):
                raise ContractError("source motion requires finite XYZ Euler radians")
    fps = motion.get("fps")
    if (type(fps) not in (float, int) or not math.isfinite(fps) or
            not 1 <= fps <= 240 or int(fps) != fps):
        raise ContractError("iClone FK writer requires integer FPS 1..240")
    return frames, int(fps)


def _motion_variation(values):
    return any(max(row[k] for row in values) - min(row[k] for row in values) > 1e-6
               for k in range(3))


def make_creature_preview(motion, source_info, target, bones="all",
                          root_mode="none", root_position_scale=None,
                          root_position_space="armature"):
    """Generate explicit FK Layer XYZ tracks for any *same* source/target rig."""
    names, report = _validated_rig(source_info, target)
    samples, fps = _source_motion(motion, names)
    if root_mode not in ("none", "rotation", "full"):
        raise ContractError("root_mode must be none/rotation/full")
    if root_mode != "full" and root_position_scale is not None:
        raise ContractError("root_position_scale is only valid with full root mode")
    if root_mode != "full" and root_position_space != "armature":
        raise ContractError("root_position_space is only valid with full root mode")
    if root_position_space not in ("armature", "root_local"):
        raise ContractError("generic root space must be armature or root_local")
    selected = _select_bones(bones, names[1:])
    by_name = {}
    for entry in target["bones"]:
        by_name.setdefault(entry.get("name"), []).append(entry)
    if any(len(by_name.get(name, [])) != 1 for name in names):
        raise ContractError("source Bone target identity is ambiguous or missing")
    tracks = {}
    basis_info = {}
    varying = []
    index = {name: i for i, name in enumerate(names)}
    for name in selected:
        rest = relative_rest_basis(source_info, name)
        rest_inv = qinverse_unit(rest)
        rows = []
        for frame in samples:
            xyz = frame[index[name]]
            if max(abs(v) for v in xyz) < 1e-10:
                rows.append([0., 0., 0.])
            else:
                q = from_euler_xyz(xyz)
                rows.append(to_euler_xyz(qmul(qmul(rest, q), rest_inv)))
        if _motion_variation(rows):
            varying.append(name)
        bid = by_name[name][0]["id"]
        tracks[bid] = rows
        basis_info[name] = rest

    root_name = names[0]
    root_meta, root_positions = None, None
    if root_mode != "none":
        root_id = by_name[root_name][0]["id"]
        angles, root_positions, root_meta = convert_pelvis_root(
            motion, source_info, root_id, mode=root_mode,
            position_scale=root_position_scale,
            position_space=root_position_space,
            source_root=root_name,
        )
        tracks[root_id] = angles
        if _motion_variation(angles):
            varying.append(root_name)
        if root_positions is not None and _motion_variation(root_positions):
            varying.append(root_name + ":position")

    if not varying:
        raise ContractError("all selected FK and root Position tracks are static")
    result = {
        "schema_version": MOTION_VERSION,
        "space": "iclone_fk_layer_xyz_radians",
        "skeleton_hash": skeleton_hash(target),
        "fps": fps,
        "frame_count": len(samples),
        "tracks": tracks,
        "calibration_status": "unverified_generic_same_rig_rest_basis",
        "rotation_basis_mode": "parent_relative_rest_quaternion_conjugation",
        "expected_unposed_world_matrices": {
            b["id"]: list(b["snapshot_world_matrix"]) for b in target["bones"]
        },
        "generic_source_root_name": root_name,
        "generic_source_bone_count": len(names),
        "generic_source_joint_count": len(source_info.get("keypoints") or []),
        "mapped_source_bones": selected,
        "preview_bones_with_rotation_change": varying,
        "unverified_rest_basis_quaternion_wxyz": basis_info,
        "import_validation": {
            "result": report["result"],
            "matched_bones": report["matched_unique_names"],
            "target_nodes": report["target_nodes"],
            "source_to_target_uniform_rest_scale": report["source_to_target_uniform_scale"],
        },
        "limitations": [
            "same_rig_only_no_arbitrary_retargeting",
            "uncalibrated_iclone_fk_axes",
            "uncalibrated_iclone_root_position",
            "fresh_unposed_scene_required",
            "ik_and_skin_runtime_not_verified",
        ],
        "warning": (
            "EXPERIMENTAL same-rig preview only. Target rest alignment proves "
            "neither iClone FK axis calibration nor evaluated skin/Root position. "
            "Use a fresh scene and compare the entire clip against Blender."
        ),
    }
    if root_meta is not None:
        result["root_motion"] = root_meta
        if root_positions is not None:
            result["position_tracks"] = {root_id: root_positions}
    result["pipeline"] = {
        "schema_version": PROFILE_VERSION,
        "source_motion_sha256": _digest(motion),
        "source_rig_sha256": _digest(source_info),
        "target_skeleton_hash": skeleton_hash(target),
        "target_unposed_world_sha256": _digest(result["expected_unposed_world_matrices"]),
    }
    validate_motion(result, target)
    return result, report


def prepare_package(folder, motion, source_info, target, **kwargs):
    """Create read-only signed-by-hash package, refusing to overwrite files."""
    folder = pathlib.Path(folder)
    if folder.exists():
        raise ContractError("output package directory already exists: " + str(folder))
    fk, report = make_creature_preview(motion, source_info, target, **kwargs)
    manifest = {
        "schema_version": PROFILE_VERSION,
        "fk_filename": FK_FILENAME,
        "fk_sha256": _digest(fk),
        "source": fk["pipeline"],
    }
    folder.mkdir(parents=True)
    _write(folder / FK_FILENAME, fk)
    _write(folder / MANIFEST_FILENAME, manifest)
    _write(folder / REPORT_FILENAME, report)
    return folder / FK_FILENAME


def verify_generic_fk(path, data=None):
    """Verify offline FK package integrity before iClone writes any keys."""
    path = pathlib.Path(path)
    payload = _read(path) if data is None else data
    meta = payload.get("pipeline")
    if not isinstance(meta, dict) or meta.get("schema_version") != PROFILE_VERSION:
        raise ContractError("not a generic same-rig FK package")
    if path.name != FK_FILENAME:
        raise ContractError("generic package FK filename changed")
    manifest_path = path.parent / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ContractError("generic package manifest missing")
    manifest = _read(manifest_path)
    if manifest.get("schema_version") != PROFILE_VERSION or manifest.get("fk_filename") != FK_FILENAME:
        raise ContractError("generic package manifest profile mismatch")
    if manifest.get("source") != meta or manifest.get("fk_sha256") != _digest(payload):
        raise ContractError("generic FK differs from the verified manifest")
    if meta.get("target_skeleton_hash") != payload.get("skeleton_hash"):
        raise ContractError("generic FK target skeleton hash mismatch")
    if meta.get("target_unposed_world_sha256") != _digest(
            payload.get("expected_unposed_world_matrices")):
        raise ContractError("generic FK original unposed pose hash mismatch")
    if payload.get("calibration_status") != "unverified_generic_same_rig_rest_basis":
        raise ContractError("generic FK must stay explicitly uncalibrated")
    return {"status": "generic_fk_payload_matches_manifest",
            "profile": PROFILE_VERSION, "motion_sha256": _digest(payload)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--rig-info", required=True)
    parser.add_argument("--skeleton", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bones", default="all", help="all or comma-separated Bone names")
    parser.add_argument("--root-mode", choices=("none", "rotation", "full"), default="none")
    parser.add_argument("--root-position-scale", type=float, default=None)
    parser.add_argument("--root-position-space", choices=("armature", "root_local"),
                        default="armature")
    args = parser.parse_args(argv)
    output = prepare_package(
        args.output_dir, _read(args.motion), _read(args.rig_info), _read(args.skeleton),
        bones=args.bones, root_mode=args.root_mode,
        root_position_scale=args.root_position_scale,
        root_position_space=args.root_position_space)
    print("Generic creature FK preview ready (UNCALIBRATED):", output)


if __name__ == "__main__":
    main()
