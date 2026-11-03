"""Render the rest pose from both layout cameras (RGBA), for the video model's first frame.

blender -b -P blender/render_ref.py -- rigged.blend out_dir [samples]
"""
import json, pathlib, sys

import bpy

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
out = pathlib.Path(args[1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=args[0])
for pb in bpy.data.objects["rig"].pose.bones:
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0, 0, 0)
    pb.location = (0, 0, 0)
common.studio(bpy.context.scene, samples=int(args[2]) if len(args) > 2 else 128)
common.render_views(out, "ref")
(out / "render_camera_angles.json").write_text(json.dumps({
    "camera_azimuth_deg": [common.FRONT_AZIMUTH_DEG, common.SIDE_AZIMUTH_DEG],
    "widths": [common.W_FRONT, common.W_SIDE], "height": common.H,
    "metres_per_pixel": common.M_PER_PX
}, indent=2), encoding="utf-8")
