"""Export AnyMove rig_info.json and creature.json from an EXISTING skinned .blend.

Blender command (from repo root):
  blender -b -P tools/import_existing_rig.py -- assets/dinosaur/rigged/rigged.blend

Only metadata JSON is written: this script never saves or re-rigs the .blend.
bpy is imported only at runtime, so the data contracts can be tested with CPython.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import tempfile

DEFAULT_LIMITS = [[-20.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]]
FRONT, SIDE, HEIGHT, GROUND = 640, 640, 720, 630


def xyz(value):
    if len(value) != 3:
        raise ValueError("joint must have 3 coordinates")
    p = [float(v) for v in value]
    if any(not math.isfinite(v) for v in p):
        raise ValueError("non-finite joint coordinates")
    return p


def parent_first(records):
    """Parent-before-child; never rely on Blender collection iteration order."""
    by_name = {b["name"]: b for b in records}
    if len(by_name) != len(records):
        raise ValueError("duplicate Bone names")
    roots = [b["name"] for b in records if b["parent"] is None]
    if len(roots) != 1:
        raise ValueError("expected one Skeleton root, got %d" % len(roots))
    children = {name: [] for name in by_name}
    for b in records:
        parent = b["parent"]
        if parent is not None:
            if parent not in by_name:
                raise ValueError("missing Bone parent: " + str(parent))
            children[parent].append(b["name"])
    ordered, visited = [], set()
    def visit(name):
        if name in visited:
            raise ValueError("cycle in Bone hierarchy")
        visited.add(name)
        ordered.append(by_name[name])
        for child in sorted(children[name]):
            visit(child)
    visit(roots[0])
    if len(ordered) != len(records):
        raise ValueError("disconnected/cyclic Bone hierarchy")
    return ordered


def build_rig_info(records, layout):
    """Match existing solver's bones/keypoints/layout schema without changing a Rig."""
    ordered = parent_first(records)
    indices = {b["name"]: i for i, b in enumerate(ordered)}
    bones, keypoints, shared = [], [], {}
    def keypoint(bone, index, end):
        pos = xyz(bone[end])
        quantized = tuple(round(v, 6) for v in pos)
        if quantized in shared:
            name, original = shared[quantized]
            if max(abs(x - y) for x, y in zip(original, pos)) > 1e-6:
                raise ValueError("keypoint precision collision")
            return name
        name = "joint:" + bone["name"] + ":" + end
        shared[quantized] = (name, pos)
        keypoints.append({"name": name, "bone": index, "end": end, "rest": pos})
        return name

    for i, b in enumerate(ordered):
        matrix = b["matrix_local"]
        if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
            raise ValueError("matrix_local must be 4x4: " + b["name"])
        if any(not math.isfinite(float(v)) for row in matrix for v in row):
            raise ValueError("invalid matrix_local: " + b["name"])
        length = float(b["length"])
        if not math.isfinite(length) or length <= 1e-8:
            raise ValueError("zero-length Bone: " + b["name"])
        limits = [[float(v) for v in pair] for pair in b["limits_deg"]]
        if (len(limits) != 3 or any(len(pair) != 2 for pair in limits)
                or any(not all(math.isfinite(v) for v in pair) or pair[0] >= pair[1]
                       for pair in limits)):
            raise ValueError("finite positive rotation limit range required: " + b["name"])
        head = keypoint(b, i, "head")
        tail = keypoint(b, i, "tail")
        parent = b["parent"]
        bones.append({
            "name": b["name"],
            "parent": indices[parent] if parent is not None else -1,
            "matrix_local": [[float(v) for v in row] for row in matrix],
            "length": length,
            "head_joint": head,
            "tail_joint": tail,
            "limits_deg": limits,
        })
    if len(keypoints) < 3:
        raise ValueError("fewer than three unique joints; cannot train reader")
    return {"bones": bones, "keypoints": keypoints, "layout": layout}


def camera_layout(bounds):
    """Calculate metres/pixel for high-res FRONT/SIDE from actual world bounds."""
    pts = [xyz(v) for v in bounds]
    if not pts:
        raise ValueError("missing skinned Mesh bounds")
    x, y, z = zip(*pts)
    if max(z) - min(z) < 1e-5:
        raise ValueError("zero-height Mesh bounds")
    # Cameras in this branch are fixed at X=Y=0, floor at Z=0.
    mpp = max(
        2 * max(map(abs, x)) / (FRONT - 96),
        2 * max(map(abs, y)) / (SIDE - 96),
        max(z) / (GROUND - 48),
        1e-5,
    ) * 1.15
    return {
        "H": HEIGHT, "W_FRONT": FRONT, "W_SIDE": SIDE,
        "M_PER_PX": mpp, "GROUND_V": GROUND,
        "Z_CENTER": (GROUND - HEIGHT / 2) * mpp,
        "Y_CENTER": 0.0, "X_CENTER": 0.0,
    }


def creature_draft(name, bounds, layout):
    x, y, z = zip(*[xyz(v) for v in bounds])
    size = max(max(x)-min(x), max(y)-min(y), max(z)-min(z), 1.0)
    centre_z = (min(z) + max(z)) / 2.0
    return {
        "name": name, "noun": "dinosaur",
        "prep": {"length_m": max(y)-min(y), "rotz_deg": 0, "axis": "y"},
        "layout": {k: layout[k] for k in
                   ("W_FRONT", "W_SIDE", "M_PER_PX", "GROUND_V")},
        "paint": {}, "feet": [],
        "describe": "a rigged dinosaur (REVIEW appearance before video prompts)",
        "colour_code": "REVIEW: limb colour mapping is not configured",
        "count": "REVIEW: number of legs, arms, and tails",
        "flank": "side", "limbs": "limbs",
        "camera": {
            "cam": "%.4f,%.4f,%.4f" % (-1.8*size, -1.6*size, centre_z+.5*size),
            "target": "0,0,%.4f" % centre_z,
            "lens": 40,
        },
    }


def rotation_limits(pose_bone):
    """Read active LOCAL limit-rotation constraints; do not invent anatomy."""
    limits = [row[:] for row in DEFAULT_LIMITS]
    sources = ["draft_default"] * 3
    for c in pose_bone.constraints:
        if (c.type != "LIMIT_ROTATION" or c.owner_space != "LOCAL"
                or getattr(c, "mute", False) or c.influence < .999):
            continue
        for i, axis in enumerate("xyz"):
            if getattr(c, "use_limit_" + axis, False):
                low = math.degrees(getattr(c, "min_" + axis))
                high = math.degrees(getattr(c, "max_" + axis))
                if math.isfinite(low) and math.isfinite(high) and low < high:
                    limits[i], sources[i] = [low, high], "blender_local_constraint"
                # A locked axis is left provisional: current Solver cannot
                # parameterize a zero-width sigmoid rotation interval.
    return limits, sources


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
            json.dump(obj, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Existing Blender Rig to AnyMove JSON drafts")
    parser.add_argument("blend", help="Existing rigged .blend (never modified)")
    parser.add_argument("--asset-dir", help="Output assets/<creature> path")
    parser.add_argument("--name", help="Override creature name")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated JSON only")
    args = parser.parse_args(argv)
    source = Path(args.blend).expanduser().resolve()
    if source.suffix.lower() != ".blend" or not source.is_file():
        raise ValueError("Not an existing Blender file: " + str(source))
    if args.asset_dir:
        asset_dir = Path(args.asset_dir).expanduser().resolve()
    elif source.parent.name == "rigged":
        asset_dir = source.parent.parent
    else:
        raise ValueError("Provide --asset-dir for a .blend outside a rigged/ directory")
    name = args.name or asset_dir.name
    if not name or "/" in name or "\\" in name:
        raise ValueError("invalid creature name")
    targets = (asset_dir / "rigged" / "rig_info.json",
               asset_dir / "creature.json",
               asset_dir / "rigged" / "rig_import_report.json")
    if not args.overwrite and any(p.exists() for p in targets):
        raise FileExistsError("Existing output metadata; back up then use --overwrite: " +
                              ", ".join(str(p) for p in targets if p.exists()))

    import bpy
    from mathutils import Matrix, Vector
    bpy.ops.wm.open_mainfile(filepath=str(source))
    armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError("Expected exactly one Armature; found %d" % len(armatures))
    arm = armatures[0]
    meshes = [o for o in bpy.data.objects if o.type == "MESH"
              and any(m.type == "ARMATURE" and m.object == arm for m in o.modifiers)]
    if not meshes:
        raise ValueError("No Mesh with an Armature modifier bound to selected rig")
    identity = Matrix.Identity(4)
    if max(abs(arm.matrix_world[i][j] - identity[i][j])
           for i in range(4) for j in range(4)) > 1e-4:
        raise ValueError("Armature Object transform is not identity; normalize a COPY")
    if abs(bpy.context.scene.unit_settings.scale_length - 1.0) > 1e-5:
        raise ValueError("Scene unit is not 1 metre/unit; normalize a COPY")

    bounds = [[float(v) for v in (o.matrix_world @ Vector(corner))]
              for o in meshes for corner in o.bound_box]
    layout = camera_layout(bounds)
    records, limits_sources = [], {}
    for b in arm.data.bones:
        pb = arm.pose.bones[b.name]
        limits, sources = rotation_limits(pb)
        limits_sources[b.name] = sources
        records.append({
            "name": b.name, "parent": b.parent.name if b.parent else None,
            "matrix_local": [[float(v) for v in row] for row in b.matrix_local],
            "length": float(b.length),
            "head": [float(v) for v in b.head_local],
            "tail": [float(v) for v in b.tail_local],
            "limits_deg": limits,
        })
    info = build_rig_info(records, layout)
    info["import_provenance"] = {
        "source_blend": str(source),
        "source_armature": arm.name,
        "rotation_limit_sources": limits_sources,
        "status": "draft_needs_review",
    }
    creature = creature_draft(name, bounds, layout)
    bx, by, bz = zip(*bounds)
    warnings = []
    if min(bz) < -0.02 or min(bz) > 0.05:
        warnings.append("Floor is not at Z=0; fix rest placement on a COPY")
    if arm.name != "rig" or len(meshes) != 1 or meshes[0].name != "body":
        warnings.append("Existing render_dataset.py needs ONE Mesh 'body' and Armature 'rig'; "
                        "prepare a COPY of the .blend, or adapt renderer for multiple meshes")
    if any("draft_default" in v for v in limits_sources.values()):
        warnings.append("Rotation limits include temporary +/-20 degree defaults: REVIEW first")
    if any(pb.constraints for pb in arm.pose.bones):
        warnings.append("Existing pose constraints may override Blender FK sampling")
    warnings += [
        "Check actual -Y forward / +Z up / metric units and head/tail labels",
        "Configure creature.json feet/paint/colour_code/appearance before AI video",
        "Generic root and non-paired joint names need motion Solver validation",
    ]
    report = {
        "status": "draft_needs_review",
        "source_blend": str(source),
        "creature": name,
        "armature": arm.name,
        "mesh_names": [m.name for m in meshes],
        "bone_count": len(info["bones"]),
        "joint_count": len(info["keypoints"]),
        "bounds_min": [min(bx), min(by), min(bz)],
        "bounds_max": [max(bx), max(by), max(bz)],
        "current_dataset_loader_compatible": (
            arm.name == "rig" and len(meshes) == 1 and
            meshes[0].name == "body" and -.02 <= min(bz) <= .05
        ),
        "warnings": warnings,
    }
    # Only JSON files are written; original .blend remains unchanged.
    for path, payload in zip(targets, (info, creature, report)):
        atomic_json(path, payload)
        print("created:", path)
    print("existing-rig import: %d bones, %d joints; draft requires review" %
          (len(info["bones"]), len(info["keypoints"])))
    print("dataset loader compatible:", report["current_dataset_loader_compatible"])
    for warning in warnings:
        print("REVIEW:", warning)


if __name__ == "__main__":
    import sys
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(arguments)
