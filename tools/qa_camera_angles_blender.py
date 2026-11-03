"""Geometry-only Blender QA: verify camera matrices reproduce Reader/Solver UV.

Blender 5, no render or model file required:
"C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe" -b -P tools/qa_camera_angles_blender.py

Tests original 0/90 and alternate 315/45, 45/135. Exits nonzero on
matrix/local-image projection mismatch or incorrect camera direction.
"""
import json
import pathlib
import sys

import bpy
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender"))
import common  # noqa: E402

scene = bpy.context.scene
points = np.array([[0.0, 0.0, common.Z_CENTER], [0.5, -1.0, 0.7],
                   [-0.6, -0.4, 1.3], [0.8, 1.1, 0.25]], dtype=np.float64)
checks = []

for angles in ((0, 90), (315, 45), (45, 135), (20, 100)):
    common.FRONT_AZIMUTH_DEG, common.SIDE_AZIMUTH_DEG = angles
    cams = common.cameras(scene)
    targets = common.project_joints(points)
    for ix, (name, (camera, width)) in enumerate(cams.items()):
        cal = common.camera_json(camera, width, common.H)
        R = np.asarray(cal["R"])
        t = np.asarray(cal["t"])
        pc = points @ R.T + t
        observed = np.stack((cal["cx"] + cal["f"] * pc[:, 0],
                             cal["cy"] - cal["f"] * pc[:, 1]), axis=-1)
        max_diff = float(np.max(np.abs(observed - targets[ix])))
        checks.append({"angles_deg": list(angles), "slot": name,
                       "max_project_difference_px": max_diff})
        # Blender object transforms are float32; sub-millipixel differences are expected.
        assert max_diff < 2e-3, (angles, name, max_diff)
print(json.dumps({"status": "PASS", "checks": checks}, indent=2))
