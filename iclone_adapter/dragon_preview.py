"""Convert the SAME AnyMove Dragon rig to an uncalibrated iClone FK preview.

This is an explicit EXPLORATORY test, not certified Euler retargeting.
Before making a preview we verify all 42 source Bone rest-space world matrices
and parent paths against the selected iClone snapshot (under one global scale).
No PyTorch or Blender required for this step.

Example:
  python -m iclone_adapter.dragon_preview \
    runs/dragon/clips/roar/motion.json F:/anymove_test/skeleton.json \
    F:/anymove_test/dragon_roar_fk_preview.json --bones neck1,head,tail1,tail2

To attempt all 41 non-root bones AFTER visually verifying the preview,
repeat with --bones all. This does not copy root translation or rotation.
"""
import argparse
import json
import math

from .anymove_bridge import (from_euler_xyz, qmul, qinverse_unit,
                             to_euler_xyz)
from .rest_basis import relative_rest_basis
from .motion_contract import MOTION_VERSION, validate_motion
from .schema import ContractError, skeleton_hash, validate_skeleton
from .verify_import import DEFAULT_SOURCE, audit_rig, create_mapping

DEFAULT_BONES = ("neck1", "head", "tail1", "tail2")


def _bone_selection(bones, allowed):
    if bones == "all":
        selected = list(allowed)
    elif isinstance(bones, str):
        selected = [s.strip() for s in bones.split(",") if s.strip()]
    else:
        selected = list(bones)
    if not selected:
        raise ContractError("choose at least one named non-root Bone")
    if len(set(selected)) != len(selected):
        raise ContractError("duplicate source Bone in preview selection")
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ContractError("unknown or root source Bone(s): " + ", ".join(invalid))
    return selected


def make_preview(motion, target, source_info, bones=DEFAULT_BONES,
                 basis="raw"):
    """Return a test-only FK JSON, optionally trying relative-rest conjugation.

    basis="raw" is the historical direct-Euler experiment;
    basis="rest" applies O_bone @ R_blender @ inverse(O_bone) to every selected
    non-root Bone, matching the earlier controlled test B. This does NOT
    automatically calibrate FK Layer axes or enable production conversion.
    """
    if basis not in ("raw", "rest"):
        raise ContractError("basis must be raw or rest")
    validate_skeleton(target)
    report = audit_rig(source_info, target)
    if report["result"] != "compatible_static_rest_snapshot":
        raise ContractError(
            "Dragon import differs from original rest rig; "
            "check dragon_import_report.json or re-export an unposed snapshot: " +
            str(report["missing_names"] or report["ambiguous_names"] or
                report["mismatched_parents"] or
                report["max_position_error_source_units"]))
    src_names = [b["name"] for b in source_info["bones"]]
    if motion.get("bones") != src_names:
        raise ContractError("Dragon motion.json Bone names/order differ from rig_info.json")
    samples = motion.get("euler")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ContractError("motion.json euler must contain at least two frames")
    if any(not isinstance(frame, list) or len(frame) != len(src_names)
           for frame in samples):
        raise ContractError("motion.json euler has inconsistent Bone array size")
    fps = motion.get("fps")
    if (type(fps) not in (int, float) or not math.isfinite(fps) or
            not (1 <= fps <= 240) or int(fps) != fps):
        raise ContractError("iClone P1 requires an integer project FPS 1..240, motion has %r" % fps)
    fps = int(fps)

    selected = _bone_selection(bones, src_names[1:])
    # This mapping is deliberately still CALIBRATED=FALSE. We only borrow the
    # mechanically verified hierarchy target IDs, never certify FK axes.
    draft = create_mapping(source_info, target, report)
    for entry in draft["bone_map"].values():
        if entry["calibrated"]:
            raise ContractError("unexpected calibrated flag in static-only mapping")
    indexes = {name: src_names.index(name) for name in selected}
    tracks = {}
    rest_quaternions = {}
    movement = []
    for name in selected:
        col = indexes[name]
        values = [frame[col] for frame in samples]
        if any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in values):
            raise ContractError("invalid XYZ samples on source Bone " + name)
        rows = [[float(v) for v in row] for row in values]
        if any(not math.isfinite(v) for row in rows for v in row):
            raise ContractError("non-finite source rotation for " + name)
        delta = max(max(row[axis] for row in rows) -
                    min(row[axis] for row in rows) for axis in range(3))
        if delta > 1e-6:
            movement.append(name)
        if basis == "rest":
            # Per-bone relative-to-parent rest orientation; NOT armature world.
            # This is precisely the same formula as probe_fk_rest_conjugate.
            rest_q = relative_rest_basis(source_info, name)
            rest_q_inv = qinverse_unit(rest_q)
            rest_quaternions[name] = rest_q
            converted = []
            for xyz in rows:
                if max(abs(x) for x in xyz) < 1e-10:
                    converted.append([0., 0., 0.])
                else:
                    q = from_euler_xyz(xyz)
                    transformed = qmul(qmul(rest_q, q), rest_q_inv)
                    converted.append(to_euler_xyz(transformed))
            rows = converted
        tracks[draft["bone_map"][name]["target_id"]] = rows
    if not movement:
        raise ContractError(
            "selected Bone tracks are static across all frames; choose bones with visible motion")
    result = {
        "schema_version": MOTION_VERSION,
        "space": "iclone_fk_layer_xyz_radians",
        "skeleton_hash": skeleton_hash(target),
        "fps": fps,
        "frame_count": len(samples),
        "tracks": tracks,
        "calibration_status": (
            "unverified_identity_same_rig_preview" if basis == "raw" else
            "unverified_rest_basis_conjugation_preview"
        ),
        "rotation_basis_mode": basis,
        "experimental_rest_quaternion_wxyz": rest_quaternions,
        "preview_source_bones": selected,
        "expected_unposed_world_matrices": {
            b["id"]: list(b["snapshot_world_matrix"]) for b in target["bones"]
        },
        "preview_bones_with_rotation_change": movement,
        "import_validation": {
            "result": report["result"],
            "matched_bones": report["matched_unique_names"],
            "scale": report["source_to_target_uniform_scale"],
        },
        "limitations": [
            "experimental_unverified_fk_channel_basis",
            "root_translation_skipped",
            "root_rotation_skipped",
            "no_fk_ik_reconciliation",
        ],
        "warning": (
            "TEST PROJECT ONLY. Blender XYZ Euler -> iClone FK Layer uses " +
            ("RAW direct Euler; NOT CALIBRATED. " if basis == "raw" else
             "PARENT-RELATIVE REST ROTATION CONJUGATION; NOT CALIBRATED. ") +
            "A successful two-bone wing probe does NOT certify all 41 bones. "
            "Compare neck, tail, legs, and wings with Blender same-frame poses; "
            "root rotation/translation are omitted."
        ),
    }
    validate_motion(result, target)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build an experimental same-rig Dragon FK preview (not a calibrated retarget)")
    parser.add_argument("motion_json", help="AnyMove solver output with euler and bones")
    parser.add_argument("skeleton_json", help="iClone Dragon current rest snapshot")
    parser.add_argument("output_json", help="iClone FK preview file for P1 Motion Writer")
    parser.add_argument("--source-rig", default=str(DEFAULT_SOURCE))
    parser.add_argument("--bones", default=",".join(DEFAULT_BONES),
                        help="comma-separated source Bone names; 'all' includes 41 non-root Bones")
    parser.add_argument("--basis", choices=("raw","rest"), default="raw",
                        help="raw: original Euler; rest: experimentally conjugate each "
                             "Bone by its parent-relative rest quaternion (NOT calibrated)")
    args = parser.parse_args()
    def read(path):
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    out = make_preview(read(args.motion_json), read(args.skeleton_json),
                       read(args.source_rig), bones=args.bones,
                       basis=args.basis)
    with open(args.output_json,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    print("EXPERIMENTAL / FK AXES UNVERIFIED; root motion excluded")
    print("Basis conversion:", out["rotation_basis_mode"])
    print("Wrote:",args.output_json)
    print("FPS:",out["fps"],"Frames:",out["frame_count"],
          "Mapped Bones:",len(out["tracks"]),
          "Moving Bones:",", ".join(out["preview_bones_with_rotation_change"]))


if __name__ == "__main__":
    main()
