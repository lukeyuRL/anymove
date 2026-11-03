"""Render human SkinMax53 V3 identical GT pose at 3 two-view camera pairs.

No AI video. Existing V3 shader and Blender motion source, no source .blend save.
"""
import json
import math
import os
import pathlib
import sys

import bpy
from mathutils import Matrix

ROOT = pathlib.Path("F:/anymove")
OUT = ROOT / "runs/human/camera_angle_p0_v3"
SRC = ROOT / "runs/human/lora_synth_A/v6_reach_source/full_train_reach_forward_left_a44_s7301"
motion = json.loads((SRC / "motion.json").read_text(encoding="utf-8"))
variant = os.environ.get("ANYMOVE_HUMAN_VARIANT", "v3")
if variant not in ("v3", "nose"):
    raise RuntimeError("ANYMOVE_HUMAN_VARIANT must be v3 or nose")
expected_scene = ROOT / ("runs/human/human_skinmax_multicolor_v3_nose.blend" if variant == "nose" else "runs/human/human_skinmax_multicolor_v3.blend")
if pathlib.Path(bpy.data.filepath).resolve() != expected_scene.resolve():
    raise RuntimeError("This preview must load the selected Human SkinMax V3 variant")
frame = 53
arm = bpy.data.objects["Armature"]
scene = bpy.context.scene
for name, euler in zip(motion["bones"], motion["pose_euler_xyz_rad"][frame]):
    bone = arm.pose.bones[name]
    bone.rotation_mode = "XYZ"
    bone.rotation_euler = euler
arm.matrix_world = Matrix(motion["rig_world_matrices"][frame])
bpy.context.view_layer.update()

scene.render.engine = "BLENDER_EEVEE"
scene.eevee.taa_render_samples = 20
scene.render.film_transparent = True
scene.render.resolution_percentage = 100
scene.render.resolution_x = 384
scene.render.resolution_y = 448
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
world = scene.world or bpy.data.worlds.new("HumanCameraABWorld")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.75
for name, rot, power in [
    ("AngleABKey", (math.radians(45), 0, math.radians(-35)), 3),
    ("AngleABFill", (math.radians(60), 0, math.radians(110)), 1.3),
    ("AngleABRim", (math.radians(35), 0, math.radians(200)), 1.7)
]:
    light = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    scene.collection.objects.link(light)
    light.rotation_euler = rot
    light.data.energy = power
    light.data.angle = math.radians(8)
camera = bpy.data.objects.new("HumanV3AngleABCamera", bpy.data.cameras.new("HumanV3AngleABCamera"))
scene.collection.objects.link(camera)
camera.data.type = "ORTHO"
camera.data.ortho_scale = 2.432802200317383
camera.data.clip_end = 100
scene.camera = camera
pairs = {"baseline_0_90": [0, 90], "symmetric_315_45": [315, 45], "oblique_45_135": [45, 135]}
OUT.mkdir(parents=True, exist_ok=True)
for label, angles in pairs.items():
    for n, angle in enumerate(angles):
        theta = math.radians(angle)
        camera.location = (-20 * math.sin(theta), -20 * math.cos(theta), 0.8688581585884094)
        camera.rotation_euler = (math.pi / 2, 0, -theta)
        scene.render.filepath = str(OUT / f"human_v3_{'nose_' if variant == 'nose' else ''}{label}_cam{n+1}_{angle:03d}.png")
        bpy.ops.render.render(write_still=True)
        print("HUMAN_V3_CAMERA_RENDER", label, angle, scene.render.filepath, flush=True)
print("HUMAN_V3_CAMERA_RENDER_COMPLETE",str(OUT),flush=True)
