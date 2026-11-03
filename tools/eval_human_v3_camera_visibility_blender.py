"""Blender V3 SkinMax53 surface-region camera-visibility comparison.

Uses original V3 mesh, its dominant bone weights/unique vertex colors, and
replays EXISTING human V6 GT motion. For sampled surface vertices, tests
first-hit visibility with a world-space BVH per view, no render or AI video.
Outputs approximate color-region visibility, NOT learned keypoint confidence.
"""
import json
import math
import os
import pathlib
import sys
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = pathlib.Path("F:/anymove")
VARIANT = os.environ.get("ANYMOVE_HUMAN_VARIANT", "v3")
if VARIANT not in ("v3", "nose"):
    raise RuntimeError("ANYMOVE_HUMAN_VARIANT must be v3 or nose")
SCENE = ROOT / ("runs/human/human_skinmax_multicolor_v3_nose.blend" if VARIANT == "nose" else "runs/human/human_skinmax_multicolor_v3.blend")
BASE = ROOT / "runs/human/lora_synth_A/v6_reach_source"
OUT = ROOT / "runs/human/camera_angle_p0_v3"
PAIRS = {"baseline_0_90": (0, 90), "symmetric_315_45": (315, 45),
         "oblique_45_135": (45, 135)}
CLIPS = (
    "full_train_reach_forward_left_a44_s7301",
    "full_train_reach_forward_right_a44_s7301",
    "full_train_reach_backward_left_a40_s7301",
    "full_train_reach_backward_right_a40_s7301",
)
FRAMES = (0, 32, 53, 80)
WIDTH, HEIGHT = 384, 448
METRES_PER_PIXEL = 2.432802200317383 / 448
Z_CENTER = 0.8688581585884094
MAX_PER_REGION = int(os.environ.get('ANYMOVE_VIS_SAMPLES_PER_REGION', '40'))
SAMPLE_SEED = int(os.environ.get('ANYMOVE_VIS_SEED', '20261103'))
BVH_EPSILON = 1e-5
HIT_TOLERANCE_METRES = 0.003

if pathlib.Path(bpy.data.filepath).resolve() != SCENE.resolve():
    raise RuntimeError("This test must load the exact Human SkinMax V3 blend")
arm = bpy.data.objects["Armature"]
mesh = bpy.data.objects["Dummy"]
groups = {g.index: g.name for g in mesh.vertex_groups}
regions = defaultdict(list)
for v in mesh.data.vertices:
    if v.groups:
        strongest = max(v.groups, key=lambda g: (g.weight, -g.group))
        regions[groups[strongest.group]].append(v.index)
if len(regions) != 53:
    raise RuntimeError(f"Expected 53 V3 color regions, got {len(regions)}")

rng = np.random.default_rng(SAMPLE_SEED)
samples = {}
for region, indices in sorted(regions.items()):
    select = rng.choice(indices, size=min(len(indices), MAX_PER_REGION), replace=False)
    samples[region] = np.sort(select)
sample_index = np.concatenate(list(samples.values()))
sample_region = [name for name, ids in samples.items() for _ in ids]
region_index = {name: np.where(np.asarray(sample_region) == name)[0] for name in samples}
unique_angles = sorted(set(angle for pair in PAIRS.values() for angle in pair))
# Angles 0/90/315/45/135 = 5 views
assert unique_angles == [0, 45, 90, 135, 315]
observations = {key: {"surface_sample_first_visible": [], "surface_sample_second_visible": [],
                      "surface_sample_either_visible": [], "surface_sample_both_visible": [],
                      "region_first_visible": [], "region_second_visible": [],
                      "region_either_visible": [], "region_both_visible": [],
                      "finger_region_both_visible": [], "hand_region_both_visible": [],
                      "per_region_first": defaultdict(list), "per_region_second": defaultdict(list)}
                for key in PAIRS}

def project(points, angle):
    a = math.radians(angle)
    u = WIDTH / 2 + (math.cos(a) * points[:, 0] - math.sin(a) * points[:, 1]) / METRES_PER_PIXEL
    v = HEIGHT / 2 - (points[:, 2] - Z_CENTER) / METRES_PER_PIXEL
    return (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)

for clip in CLIPS:
    src = BASE / clip
    motion = json.loads((src / "motion.json").read_text())
    gt = json.loads((src / "joints.json").read_text())
    assert gt["names"] == motion["bones"] and len(motion["bones"]) == 55
    for frame in FRAMES:
        assert frame < len(gt["joint_position_m"])
        for name, euler in zip(motion["bones"], motion["pose_euler_xyz_rad"][frame]):
            pb = arm.pose.bones[name]
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = euler
        from mathutils import Matrix
        arm.matrix_world = Matrix(motion["rig_world_matrices"][frame])
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = mesh.evaluated_get(dg)
        emesh = eval_obj.to_mesh()
        try:
            obj_matrix = eval_obj.matrix_world.copy()
            world_verts = [obj_matrix @ v.co for v in emesh.vertices]
            polys = [tuple(poly.vertices) for poly in emesh.polygons]
            bvh = BVHTree.FromPolygons(world_verts, polys, all_triangles=False, epsilon=BVH_EPSILON)
            pts = np.array([world_verts[int(i)] for i in sample_index], dtype=np.float64)
            visible = {}
            for angle in unique_angles:
                a = math.radians(angle)
                direction = Vector((math.sin(a), math.cos(a), 0))
                frame_mask = project(pts, angle)
                mask = np.zeros(len(sample_index), dtype=bool)
                for i in np.flatnonzero(frame_mask):
                    vertex = world_verts[int(sample_index[i])]
                    origin = vertex - 10.0 * direction
                    hit, normal, index, distance = bvh.ray_cast(origin, direction, 11.0)
                    if hit is None or distance is None:
                        mask[i] = True  # tangent silhouette, no triangle hit
                    else:
                        mask[i] = distance >= 10.0 - HIT_TOLERANCE_METRES
                visible[angle] = mask
            for label, pair in PAIRS.items():
                left, right = visible[pair[0]], visible[pair[1]]
                rec = observations[label]
                for field, mask in (("surface_sample_first_visible", left),
                                    ("surface_sample_second_visible", right),
                                    ("surface_sample_either_visible", left | right),
                                    ("surface_sample_both_visible", left & right)):
                    rec[field].append(float(mask.mean()))
                anyleft = {name: float(left[ids].mean()) for name, ids in region_index.items()}
                anyright = {name: float(right[ids].mean()) for name, ids in region_index.items()}
                # A region with >= 8% sampled visible surface is deemed exposed
                # in a view. This is a proxy, not image-detection probability.
                f = {name: x >= .08 for name, x in anyleft.items()}
                s = {name: x >= .08 for name, x in anyright.items()}
                rec["region_first_visible"].append(float(np.mean(list(f.values()))))
                rec["region_second_visible"].append(float(np.mean(list(s.values()))))
                rec["region_either_visible"].append(float(np.mean([f[n] or s[n] for n in samples])))
                rec["region_both_visible"].append(float(np.mean([f[n] and s[n] for n in samples])))
                finger_names = [n for n in samples if any(t in n.lower() for t in ("pinky", "ring", "mid", "index", "thumb"))]
                hand_names = [n for n in samples if any(t in n.lower() for t in ("hand", "pinky", "ring", "mid", "index", "thumb"))]
                rec["finger_region_both_visible"].append(float(np.mean([f[n] and s[n] for n in finger_names])))
                rec["hand_region_both_visible"].append(float(np.mean([f[n] and s[n] for n in hand_names])))
                for n in samples:
                    rec["per_region_first"][n].append(anyleft[n])
                    rec["per_region_second"][n].append(anyright[n])
        finally:
            eval_obj.to_mesh_clear()
        print("SAMPLE_DONE", clip, frame, flush=True)

report = {"schema":"anymove.human_v3_mesh_surface_visibility_ab/1",
          "source_blend": str(SCENE),
          "source_motion_gt_dir": str(BASE),
          "provenance":"Blender mesh BVH raycast, sampled dominant-color surface vertices (not joint detector)",
          "clips": list(CLIPS), "frames_per_clip": list(FRAMES), "frames_evaluated":len(CLIPS)*len(FRAMES),
          "v3_color_regions":len(regions), "sampled_surface_vertices":len(sample_index),
          "seed":SAMPLE_SEED,"samples_per_region_cap":MAX_PER_REGION,
          "source_variant": VARIANT,
          "region_visible_threshold":0.08, "raycast_tolerance_m":HIT_TOLERANCE_METRES,
          "pairs":{}}
for label, rec in observations.items():
    report["pairs"][label] = {
        "angles_deg": list(PAIRS[label]),
        "metrics": {name:round(float(np.mean(values)), 6) for name, values in rec.items()
                    if not name.startswith("per_region")},
        "region_surface_visibility": {
            name: {"first_fraction":round(float(np.mean(rec["per_region_first"][name])),5),
                   "second_fraction":round(float(np.mean(rec["per_region_second"][name])),5)}
            for name in samples}
    }
OUT.mkdir(parents=True, exist_ok=True)
prefix = "human_v3_nose_surface_visibility" if VARIANT == "nose" else "human_v3_surface_visibility"
path = OUT / (f"{prefix}.json" if SAMPLE_SEED == 20261103 and MAX_PER_REGION == 40
              else f"{prefix}_seed{SAMPLE_SEED}_samples{MAX_PER_REGION}.json")
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("RESULT_SUMMARY",json.dumps({k:v["metrics"] for k,v in report["pairs"].items()},indent=2))
print("RESULT_PATH",path)
