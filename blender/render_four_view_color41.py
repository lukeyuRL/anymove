"""Render four views of the PRE-PAINTED Dino SkinMax41 .blend using Blender Eevee.

This script is separate from the unpushed local P4 visibility renderer.
Replays existing 2,200 source pose metadata; it never samples new poses.

Each pose: 4 RGBA PNGs, 4 labels NPZ (part_id, depth_m, joints_uv),
and one visibility_meta.json. GT is derived from the evaluated skinned mesh,
not RGB, so changing the render engine does not change its geometry.
Joint visibility probabilities are NOT supplied: internal joints are not
equivalent to visible surface pixels.

Use --indices 1,2,3,25,50 for a smoke test, then --indices all --resume
in the SAME output folder. No existing Cycles dataset is overwritten.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import bpy
import numpy as np
from mathutils import Euler, Vector

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "blender"))
import common
from fourview_render_contract import (
    VIEWS, contract_sha256, engine_identifier, png_dimensions, pose_indices,
    sample_is_complete, sha256_file, validate_pose,
)

PROFILE = "anymove.dino_skinmax41_fourview_eevee/1"


def get_args():
    if "--" not in sys.argv:
        raise SystemExit("Pass Blender arguments after --")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("blend", "rig", "source-dataset", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--render-engine", choices=("eevee", "cycles"), default="eevee")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if not 1 <= args.samples <= 4096:
        parser.error("samples must be 1..4096")
    if not math.isfinite(args.scale) or not 0 < args.scale <= 4:
        parser.error("scale must be finite and in (0,4]")
    for field in ("blend", "rig", "source_dataset", "out"):
        setattr(args, field, pathlib.Path(getattr(args, field)).resolve())
    if not args.blend.is_file() or not args.rig.is_file():
        parser.error("Blend or rig_info.json input file is missing")
    if "skinmax" not in args.blend.name.lower():
        parser.error("Input must be the pre-painted SkinMax41 blend, not rigged.blend")
    return args


def configure_scene(scene, engine, samples):
    engine_id = engine_identifier(engine, bpy.app.version)
    if engine == "cycles":
        common.studio(scene, samples=samples)
    else:
        scene.render.engine = engine_id
        if not hasattr(scene.eevee, "taa_render_samples"):
            raise RuntimeError("Eevee render sample setting unavailable")
        scene.eevee.taa_render_samples = samples
        scene.view_settings.view_transform = "Standard"
        world = scene.world or bpy.data.worlds.new("world")
        scene.world = world
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg is None:
            raise ValueError("Expected world Background shader node")
        bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
        bg.inputs["Strength"].default_value = 0.55
        for name, rot, energy in (
            ("key", (math.radians(50), 0, math.radians(-35)), 3.2),
            ("fill", (math.radians(60), 0, math.radians(120)), 1.0),
            ("rim", (math.radians(35), 0, math.radians(200)), 1.6),
        ):
            ob = bpy.data.objects.get(name)
            if ob is None:
                ob = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
                scene.collection.objects.link(ob)
            ob.rotation_euler = rot
            ob.data.energy = energy
            ob.data.angle = math.radians(8)
    scene.render.film_transparent = True
    if hasattr(scene.render.image_settings, "media_type"):
        scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False
    return engine_id


def create_views(scene):
    src = common.cameras(scene)
    cams = {"front": src["cam_front"], "side": src["cam_side"]}
    for name, xyz, width in (
        ("back", (0, 20, common.Z_CENTER), common.W_FRONT),
        ("opposite_side", (20, 0, common.Z_CENTER), common.W_SIDE),
    ):
        ob = bpy.data.objects.get("cam_" + name)
        if ob is None:
            ob = bpy.data.objects.new("cam_" + name, bpy.data.cameras.new("cam_" + name))
            scene.collection.objects.link(ob)
        ob.location = xyz
        ob.rotation_euler = (
            Vector((0, 0, common.Z_CENTER)) - ob.location
        ).to_track_quat("-Z", "Y").to_euler()
        ob.data.type = "ORTHO"
        ob.data.ortho_scale = common.M_PER_PX * max(width, common.H)
        ob.data.clip_end = 100
        cams[name] = (ob, width)
    bpy.context.view_layer.update()
    return cams


def replay_pose(rig, metadata, info, names):
    bpy.context.scene.frame_set(1)
    rig.animation_data_clear()
    for pb in rig.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)
        pb.rotation_euler = Euler(metadata["pose"][pb.name], "XYZ")
    rig.rotation_mode = "XYZ"
    rig.scale = (1, 1, 1)
    rig.rotation_euler = (0, 0, metadata["yaw"])
    rig.location = Vector(metadata["root"])
    bpy.context.view_layer.update()
    mw = rig.matrix_world
    xyz = np.asarray([
        list(mw @ (
            rig.pose.bones[names[kp["bone"]]].head
            if kp["end"] == "head"
            else rig.pose.bones[names[kp["bone"]]].tail
        )) for kp in info["keypoints"]
    ], dtype=np.float64)
    gt = np.asarray(metadata["joints_world"], dtype=np.float64)
    if xyz.shape != gt.shape:
        raise ValueError("Replay / source joint shape mismatch")
    error = float(np.linalg.norm(xyz - gt, axis=-1).max())
    if error > 0.001:
        raise ValueError(f"Source pose FK cannot be reproduced: {error:.6f} m")
    return xyz, error


def mesh_surface(body, names):
    obj = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        xyz = np.empty(len(mesh.vertices) * 3, np.float64)
        mesh.vertices.foreach_get("co", xyz)
        mw = np.asarray(body.matrix_world, dtype=np.float64)
        xyz = xyz.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        triangles = np.asarray([list(t.vertices) for t in mesh.loop_triangles], dtype=np.int32)
        if len(triangles) == 0:
            raise ValueError("No surface triangles")
        index = {name: i for i, name in enumerate(names)}
        group_map = {g.index: index[g.name] for g in body.vertex_groups if g.name in index}
        weights = np.zeros((len(mesh.vertices), len(names)), np.float32)
        for vert in mesh.vertices:
            for group in vert.groups:
                dest = group_map.get(group.group)
                if dest is not None:
                    weights[vert.index, dest] += group.weight
        total = weights[triangles].sum(axis=1)
        part = np.argmax(total, axis=1).astype(np.int16)
        part[total.max(axis=1) < 1e-6] = -1
        return xyz, triangles, part
    finally:
        obj.to_mesh_clear()


def project(xyz, cam, width, height, scale):
    inv = np.asarray(cam.matrix_world.inverted(), dtype=np.float64)
    local = xyz @ inv[:3, :3].T + inv[:3, 3]
    uv = np.stack([
        width / 2 + local[:, 0] / common.M_PER_PX * scale,
        height / 2 - local[:, 1] / common.M_PER_PX * scale,
    ], axis=-1)
    return uv, -local[:, 2]


def z_raster(vertices, triangles, parts, camera, width, height, scale):
    uv, distances = project(vertices, camera, width, height, scale)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    part_id = np.full((height, width), -1, dtype=np.int16)
    for i, triangle in enumerate(triangles):
        p, d = uv[triangle], distances[triangle]
        if d.max() <= 0:
            continue
        x0 = max(0, int(math.floor(float(p[:, 0].min()))))
        x1 = min(width - 1, int(math.ceil(float(p[:, 0].max()))))
        y0 = max(0, int(math.floor(float(p[:, 1].min()))))
        y1 = min(height - 1, int(math.ceil(float(p[:, 1].max()))))
        if x0 > x1 or y0 > y1:
            continue
        ax, ay = p[0]
        bx, by = p[1]
        cx, cy = p[2]
        det = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(float(det)) < 1e-8:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        xx, yy = xs + 0.5 - ax, ys + 0.5 - ay
        b = (xx * (cy - ay) - yy * (cx - ax)) / det
        c = ((bx - ax) * yy - (by - ay) * xx) / det
        a = 1 - b - c
        valid = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        if not valid.any():
            continue
        z = a * d[0] + b * d[1] + c * d[2]
        buf = depth[y0:y1 + 1, x0:x1 + 1]
        update = valid & (z > 0) & (z < buf)
        if update.any():
            buf[update] = z[update].astype(np.float32)
            part_id[y0:y1 + 1, x0:x1 + 1][update] = parts[i]
    return part_id, depth


def atomic_json(path, data):
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_text(json.dumps(data, indent=2), encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as dest:
            np.savez_compressed(dest, **arrays)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main():
    a = get_args()
    to_do = pose_indices(a.source_dataset, a.indices)
    info = json.loads(a.rig.read_text(encoding="utf-8"))
    names = [b["name"] for b in info["bones"]]
    if len(names) != len(set(names)):
        raise ValueError("Rig has duplicate bones")
    bpy.ops.wm.open_mainfile(filepath=str(a.blend))
    scene = bpy.context.scene
    rig, body = common.skinned_character_objects()
    if set(names) != {pb.name for pb in rig.pose.bones}:
        raise ValueError("SkinMax .blend bone list != rig_info.json")
    color_attrs = sorted(attr.name for attr in body.data.color_attributes)
    if not color_attrs:
        raise ValueError("Input must have painted vertex colors")
    engine = configure_scene(scene, a.render_engine, a.samples)
    cameras = create_views(scene)
    dims = {
        view: [round((common.W_FRONT if view in ("front", "back") else common.W_SIDE) * a.scale),
               round(common.H * a.scale)]
        for view in VIEWS
    }
    if any(min(size) <= 0 for size in dims.values()):
        raise ValueError("Invalid image dimensions")
    contract = {
        "schema": PROFILE, "engine": engine, "render_engine": a.render_engine,
        "samples": a.samples, "scale": a.scale, "blender_version": list(bpy.app.version),
        "skinmax_blend_sha256": sha256_file(a.blend), "rig_sha256": sha256_file(a.rig),
        "source_dataset": str(a.source_dataset), "color_attributes": color_attrs,
        "layout": {"w_front": common.W_FRONT, "w_side": common.W_SIDE,
                   "h": common.H, "m_per_px": common.M_PER_PX,
                   "z_center": common.Z_CENTER},
        "views": list(VIEWS), "dimensions": dims,
        "source_motion_blur_reproduced": False,
        "part_id_policy": "max_summed_skin_weight_per_triangle",
    }
    sign = contract_sha256(contract)
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = a.out / "render_contract.json"
    if cfg.is_file():
        if not a.resume or json.loads(cfg.read_text(encoding="utf-8")) != contract:
            raise ValueError("Dataset already exists with different config; use NEW directory or --resume")
    else:
        if any(a.out.iterdir()):
            raise ValueError("Nonempty directory without matching dataset render contract")
        atomic_json(cfg, contract)
    done = skipped = 0
    started = time.monotonic()
    print("COLOR41_FOURVIEW_BEGIN", engine, "poses", len(to_do), "contract", sign, flush=True)
    for pose_id, meta_path in to_do:
        meta_sha = sha256_file(meta_path)
        if a.resume and sample_is_complete(a.out, pose_id, meta_sha, sign, dims):
            skipped += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        validate_pose(meta, names, len(info["keypoints"]))
        jw, fk_error = replay_pose(rig, meta, info, names)
        xyz, tris, part = mesh_surface(body, names)
        view_info, artifact_hashes = {}, {}
        for view in VIEWS:
            cam, _ = cameras[view]
            width, height = dims[view]
            uv, joint_depth = project(jw, cam, width, height, a.scale)
            part_img, depth_img = z_raster(xyz, tris, part, cam, width, height, a.scale)
            png = a.out / f"{pose_id:05d}_{view}.png"
            npz = a.out / f"{pose_id:05d}_{view}_labels.npz"
            scene.camera = cam
            scene.render.resolution_x, scene.render.resolution_y = width, height
            scene.render.filepath = str(png)
            png.unlink(missing_ok=True)
            bpy.ops.render.render(write_still=True)
            if not png.is_file() or png_dimensions(png) != (width, height):
                raise RuntimeError("Invalid rendered image " + str(png))
            atomic_npz(npz, part_id=part_img, depth_m=depth_img,
                       joints_uv=uv.astype(np.float32),
                       joints_depth_m=joint_depth.astype(np.float32))
            artifact_hashes[png.name] = sha256_file(png)
            artifact_hashes[npz.name] = sha256_file(npz)
            view_info[view] = {
                "image_size": [width, height], "joints_uv": np.round(uv, 5).tolist(),
                "surface_pixels": int(np.isfinite(depth_img).sum()),
                "triangle_count": len(tris),
            }
        for left, right, width in (
            ("front", "back", dims["front"][0]),
            ("side", "opposite_side", dims["side"][0]),
        ):
            u = np.asarray(view_info[left]["joints_uv"])
            v = np.asarray(view_info[right]["joints_uv"])
            err = max(float(np.abs(u[:, 0] + v[:, 0] - width).max()),
                      float(np.abs(u[:, 1] - v[:, 1]).max()))
            if err > 0.02:
                raise ValueError(f"Opposite camera mirror QA failed {left} vs {right}: {err:.4f}px")
        audit = {
            "schema": PROFILE, "sample_index": pose_id, "source_meta_path": str(meta_path),
            "source_pose_sha256": meta_sha, "render_contract_sha256": sign,
            "render_engine": engine, "appearance": "prepainted_skinmax41",
            "source_blur": bool(meta.get("blur")), "rendered_as_static": True,
            "max_fk_vs_source_gt_error_m": fk_error,
            "joints_world": np.round(jw, 6).tolist(), "views": view_info,
            "artifacts": artifact_hashes,
        }
        atomic_json(a.out / f"{pose_id:05d}_visibility_meta.json", audit)
        done += 1
        print("FOUR_VIEW_POSE_COMPLETE", pose_id, "done", done, "skipped", skipped,
              "elapsed_seconds", round(time.monotonic() - started, 1), flush=True)
    print("COLOR41_FOURVIEW_DONE", json.dumps({
        "requested": len(to_do), "rendered": done, "skipped": skipped,
        "engine": engine, "output": str(a.out),
    }), flush=True)


if __name__ == "__main__":
    main()
