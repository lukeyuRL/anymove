"""Audit an iClone import against the original AnyMove Rig and prepare a mapping.

This does NOT assert bind-pose or FK animation-channel equivalence. In the
Dragon FBX tested by the project, a 100x iClone scene-unit transform above
the 42 animation bones makes Blender rest matrices match after scaling.

Usage:
    python -m iclone_adapter.verify_import skeleton.json \
      --source assets/dragon/rigged/rig_info.json \
      --report-out dragon_import_report.json \
      --map-out dragon_mapping.json

The 41 mapped non-root bones remain calibrated=false. Verify FK Layer axes
and root motion before enabling conversion for a motion with real poses.
"""
import argparse
import json
import math
import pathlib

from .schema import ContractError, validate_skeleton, skeleton_hash
from .anymove_bridge import MAP_VERSION

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "assets" / "dragon" / "rigged" / "rig_info.json"


def audit_rig(source_info, target, max_component_error=0.002,
              max_position_error=0.01):
    """Measure static bone/world alignment under one uniform positive scale.

    Blender matrix_local in rig_info.json is bone rest matrix in the armature
    space (NOT local-to-parent). Target snapshot_world_matrix is iClone scene
    space. This comparison only handles axis-aligned globally scaled imports.
    """
    validate_skeleton(target)
    src = source_info.get("bones")
    if not isinstance(src, list) or not src:
        raise ContractError("source rig_info requires nonempty bones")
    source_names = [b["name"] for b in src]
    if len(set(source_names)) != len(src):
        raise ContractError("source rig contains duplicate bone names")
    by_name = {}
    for b in target["bones"]:
        by_name.setdefault(b["name"], []).append(b)
    missing = [name for name in source_names if name not in by_name]
    ambiguous = [name for name in source_names if len(by_name.get(name, [])) > 1]
    parents_wrong = []
    for i, bone in enumerate(src):
        matches = by_name.get(bone["name"], [])
        if len(matches) != 1:
            continue
        expected = src[bone["parent"]]["name"] if bone["parent"] >= 0 else "rig"
        p = matches[0].get("parent_id")
        actual = p.rsplit("/", 1)[-1] if p else None
        if actual != expected:
            parents_wrong.append({"bone": bone["name"],
                                  "expected_parent": expected,
                                  "actual_parent": actual})
    unique_matches = [(s, by_name[s["name"]][0]) for s in src
                      if len(by_name.get(s["name"], [])) == 1]
    scale_num, scale_den = 0.0, 0.0
    for s, t in unique_matches:
        a = s["matrix_local"]
        b = t["snapshot_world_matrix"]
        for i in range(3):
            for j in range(3):
                x, y = float(a[i][j]), float(b[i*4+j])
                scale_num += x*y
                scale_den += x*x
    scale = scale_num/scale_den if scale_den else None
    per_bone = []
    for s, t in unique_matches:
        rows = s["matrix_local"]
        vals = t["snapshot_world_matrix"]
        if scale is None or scale < 1e-9:
            continue
        components = [abs(float(vals[i*4+j])/scale - float(rows[i][j]))
                      for i in range(3) for j in range(3)]
        p_target = [float(vals[i*4+3])/scale for i in range(3)]
        p_source = [float(rows[i][3]) for i in range(3)]
        translation_error = math.sqrt(sum((a-b)**2 for a,b in zip(p_target,p_source)))
        per_bone.append({
            "name": s["name"],
            "rotation_component_max_error": max(components),
            "position_error_source_units": translation_error,
        })
    max_mat_error = max((x["rotation_component_max_error"] for x in per_bone),default=None)
    max_pos_error = max((x["position_error_source_units"] for x in per_bone),default=None)
    geometry_ok = (not missing and not ambiguous and not parents_wrong and
                   scale is not None and scale > 0 and
                   max_mat_error is not None and max_mat_error <= max_component_error and
                   max_pos_error is not None and max_pos_error <= max_position_error)
    extras = [b["name"] for b in target["bones"] if b["name"] not in source_names]
    report = {
        "result": "compatible_static_rest_snapshot" if geometry_ok else "needs_mapping_review",
        "source_bones": len(src),
        "target_nodes": len(target["bones"]),
        "matched_unique_names": len(unique_matches),
        "matched_parent_relations": len(src)-len(missing)-len(ambiguous)-len(parents_wrong),
        "extra_target_nodes": extras,
        "missing_names": missing, "ambiguous_names": ambiguous,
        "mismatched_parents": parents_wrong,
        "source_to_target_uniform_scale": scale,
        "max_rotation_matrix_component_error": max_mat_error,
        "max_position_error_source_units": max_pos_error,
        "component_tolerance": max_component_error,
        "position_tolerance": max_position_error,
        "bone_errors": per_bone,
        "target_reference_kind": target.get("reference", {}).get("kind"),
        "fk_channel_axes_calibrated": False,
        "root_motion_supported": False,
        "note": "Static rest-geometry correspondence does not validate iClone FK "
                "Layer Euler channels, true Skin Bind Pose, root motion or IK."
    }
    return report


def create_mapping(source_info, target, report):
    """Prepare identity-basis guesses; do NOT authorize them as calibrated."""
    if report.get("result") != "compatible_static_rest_snapshot":
        raise ContractError("cannot map until static bones, hierarchy and pose match")
    source = source_info["bones"]
    by_name = {b["name"]:b for b in target["bones"]}
    return {
        "schema_version": MAP_VERSION,
        "target_skeleton_hash": skeleton_hash(target),
        "root_policy": "skip",
        "bone_map": {
            s["name"]: {
                "target_id": by_name[s["name"]]["id"],
                "axis_quaternion_wxyz": [1.,0.,0.,0.],
                "calibrated": False,
            } for s in source if s["parent"] >= 0
        },
        "note": "All source names, hierarchy and rest-world transforms align; "
                "identity axis quaternions are hypotheses. Set calibrated=true "
                "only after runtime testing FK channel orientation. "
                "The source root is omitted (root motion not yet supported)."
    }


def main():
    p = argparse.ArgumentParser(description="Verify Dragon iClone Skeleton and prepare uncalibrated bone mapping")
    p.add_argument("target", help="Exported iClone skeleton.json")
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--map-out", default="dragon_mapping.json")
    p.add_argument("--report-out", default="dragon_import_report.json")
    args = p.parse_args()
    with open(args.source,encoding="utf-8") as f:
        source = json.load(f)
    with open(args.target,encoding="utf-8") as f:
        target = json.load(f)
    report = audit_rig(source,target)
    with open(args.report_out,"w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print("Import check:",report["result"],
          "matched:",report["matched_unique_names"],"/",report["source_bones"],
          "scale:",report["source_to_target_uniform_scale"])
    if report["result"] != "compatible_static_rest_snapshot":
        raise SystemExit("Report written, but mapping blocked by incompatible static pose")
    mapping = create_mapping(source,target,report)
    with open(args.map_out,"w",encoding="utf-8") as f:
        json.dump(mapping,f,ensure_ascii=False,indent=2)
    print("Mapping template written (UNVERIFIED FK axes):",args.map_out)


if __name__ == "__main__":
    main()
