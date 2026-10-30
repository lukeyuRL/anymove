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
    bones, keypoints = [], []
    def keypoint(bone, index, end):
        pos = xyz(bone[end])
        # FBX/Blender float32 Rest endpoints may differ by only a few
        # micrometres. The old 6-decimal rounding assigned different
        # heatmap targets to physically identical joints.
        # Euclidean threshold is tiny relative to a visible pixel.
        for existing in keypoints:
            if math.dist(pos, existing["rest"]) <= 1e-5:
                return existing["name"]
        name = "joint:" + bone["name"] + ":" + end
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


def ensure_similarity(matrix, label):
    """Allow translation/rotation/uniform positive scale; refuse shear or mirroring."""
    a = [[float(matrix[i][j]) for j in range(4)] for i in range(4)]
    if any(not math.isfinite(v) for row in a for v in row):
        raise ValueError(label + " has non-finite transform values")
    if any(abs(a[3][i] - (1.0 if i == 3 else 0.0)) > 1e-6 for i in range(4)):
        raise ValueError(label + " needs an affine object transform")
    cols = [[a[row][col] for row in range(3)] for col in range(3)]
    lengths = [math.sqrt(sum(v*v for v in col)) for col in cols]
    scale = sum(lengths) / 3.0
    if scale < 1e-8 or any(abs(v-scale) > 1e-4*scale for v in lengths):
        raise ValueError(label + " has nonuniform/zero scale; prepare manually on a COPY")
    if any(abs(sum(cols[i][k]*cols[j][k] for k in range(3))) > 1e-4*scale*scale
           for i in range(3) for j in range(i+1, 3)):
        raise ValueError(label + " has shear; prepare manually on a COPY")
    cross = [cols[0][1]*cols[1][2]-cols[0][2]*cols[1][1],
             cols[0][2]*cols[1][0]-cols[0][0]*cols[1][2],
             cols[0][0]*cols[1][1]-cols[0][1]*cols[1][0]]
    det = sum(cross[k]*cols[2][k] for k in range(3))
    if det <= 0:
        raise ValueError(label + " is mirrored (negative determinant); prepare manually on a COPY")
    return scale


def prepare_copy_in_memory(bpy, arm, meshes):
    """Normalize Armature and skin Mesh REST data in a new unsaved in-memory copy.

    Only similarity transforms are supported on the Armature. The caller must
    explicitly save to a new .blend path, never back to the original.
    """
    from mathutils import Matrix
    metric = float(bpy.context.scene.unit_settings.scale_length)
    if not math.isfinite(metric) or metric <= 0:
        raise ValueError("invalid Blender scene unit scale")
    world = arm.matrix_world.copy()
    uniform_scale = ensure_similarity(world, "Armature Object")
    for pb in arm.pose.bones:
        basis = pb.matrix_basis
        if max(abs(basis[i][j] - (1.0 if i == j else 0.0))
               for i in range(4) for j in range(4)) > 1e-4:
            raise ValueError("Existing Armature is not in Rest Pose: " + pb.name)
    original_bones = {
        bone.name: (metric*(world @ bone.head_local),
                    metric*(world @ bone.tail_local))
        for bone in arm.data.bones
    }
    old_meshes = []
    for obj in meshes:
        w = obj.matrix_world.copy()
        if not math.isfinite(w.determinant()) or w.determinant() <= 1e-10:
            raise ValueError("Negative/degenerate Mesh transform: " + obj.name)
        vertices = [metric*(w @ v.co) for v in list(obj.data.vertices)[:12]]
        old_meshes.append((obj, w, vertices))
    original_name = arm.name
    original_mesh_names = [m.name for m in meshes]
    arm.data = arm.data.copy()
    arm.data.transform(Matrix.Scale(metric, 4) @ world)
    arm.parent = None
    arm.matrix_world = Matrix.Identity(4)
    for obj, old_world, before in old_meshes:
        obj.data = obj.data.copy()
        obj.data.transform(Matrix.Scale(metric, 4) @ old_world, shape_keys=True)
        obj.parent = None
        obj.matrix_world = Matrix.Identity(4)
        for index, expected in enumerate(before):
            if (obj.data.vertices[index].co - expected).length > 2e-4*max(1.0, expected.length):
                raise RuntimeError("Normalized Mesh vertex mismatch: " + obj.name)
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.context.view_layer.update()
    for bone in arm.data.bones:
        old_head, old_tail = original_bones[bone.name]
        if ((bone.head_local-old_head).length > 2e-4*max(1.0, old_head.length)
                or (bone.tail_local-old_tail).length > 2e-4*max(1.0, old_tail.length)):
            raise RuntimeError("Normalized Bone Rest Position mismatch: " + bone.name)
    # Only one skinned mesh can be consumed by the current dataset renderer.
    if len(meshes) == 1:
        if bpy.data.objects.get("rig") not in (None, arm):
            raise ValueError("Object name 'rig' already belongs to another object")
        if bpy.data.objects.get("body") not in (None, meshes[0]):
            raise ValueError("Object name 'body' already belongs to another object")
        arm.name = "rig"
        meshes[0].name = "body"
    return {
        "original_armature_name": original_name,
        "original_mesh_names": original_mesh_names,
        "original_armature_world_matrix": [[float(x) for x in row] for row in world],
        "original_scene_unit_scale": metric,
        "original_uniform_armature_scale": uniform_scale,
        "rest_bone_vertex_coordinate_audit": "passed",
        "skin_deformation_runtime_audit": "not_yet_verified",
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
    parser.add_argument("--overwrite", action="store_true", help="Replace generated outputs after backup")
    parser.add_argument("--prepare-copy", action="store_true",
                        help="Save a new normalized .blend and metadata, never the source")
    args = parser.parse_args(argv)
    source = Path(args.blend).expanduser().resolve()
    if source.suffix.lower() != ".blend" or not source.is_file():
        raise ValueError("Not an existing Blender file: " + str(source))
    if args.asset_dir:
        asset_dir = Path(args.asset_dir).expanduser().resolve()
    elif source.parent.name == "rigged":
        asset_dir = source.parent.parent
        if args.prepare_copy:
            asset_dir = asset_dir.with_name(asset_dir.name + "_prepared")
    else:
        raise ValueError("Provide --asset-dir for a .blend outside a rigged/ directory")
    name = args.name or asset_dir.name
    if not name or "/" in name or "\\" in name:
        raise ValueError("invalid creature name")
    prepared_blend = asset_dir / "rigged" / "rigged.blend" if args.prepare_copy else None
    if prepared_blend is not None and prepared_blend.resolve() == source:
        raise ValueError("Prepared copy must not overwrite original .blend")
    targets = (asset_dir / "rigged" / "rig_info.json",
               asset_dir / "creature.json",
               asset_dir / "rigged" / "rig_import_report.json")
    all_outputs = targets + ((prepared_blend,) if prepared_blend else ())
    if not args.overwrite and any(p.exists() for p in all_outputs):
        raise FileExistsError("Existing output metadata; back up then use --overwrite: " +
                              ", ".join(str(p) for p in all_outputs if p.exists()))

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
    normalization_audit = None
    if args.prepare_copy:
        normalization_audit = prepare_copy_in_memory(bpy, arm, meshes)
        prepared_blend.parent.mkdir(parents=True, exist_ok=True)
    else:
        if max(abs(arm.matrix_world[i][j] - identity[i][j])
               for i in range(4) for j in range(4)) > 1e-4:
            matrix = [[round(float(x), 7) for x in row] for row in arm.matrix_world]
            raise ValueError(
                "Armature Object transform is not identity: %s. "
                "This is common for FBX and does not mean Skin Weights are bad. "
                "Do NOT Apply Transform to the original asset. "
                "Rerun with --prepare-copy for a separate normalized blend." % matrix
            )
        if abs(bpy.context.scene.unit_settings.scale_length - 1.0) > 1e-5:
            raise ValueError("Scene units not metres; rerun with --prepare-copy")

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
    if len(meshes) != 1:
        warnings.append("Current dataset renderer requires ONE skinned Mesh; "
                        "arbitrary Blender Object names are now supported")
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
        "prepared_blend": str(prepared_blend) if prepared_blend else None,
        "normalization_audit": normalization_audit,
        "creature": name,
        "armature": arm.name,
        "mesh_names": [m.name for m in meshes],
        "bone_count": len(info["bones"]),
        "joint_count": len(info["keypoints"]),
        "bounds_min": [min(bx), min(by), min(bz)],
        "bounds_max": [max(bx), max(by), max(bz)],
        "current_dataset_loader_compatible": len(meshes) == 1,
        "warnings": warnings,
    }
    # Saving is opt-in and ALWAYS to a different path.
    if prepared_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=str(prepared_blend))
        if not prepared_blend.is_file():
            raise RuntimeError("Blender did not create prepared copy: " + str(prepared_blend))
        print("created prepared .blend:", prepared_blend)
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
