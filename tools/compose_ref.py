"""Compose the two RGBA renders from blender/render_ref.py into the two-camera reference frame for the video model.

MOCAP_CREATURE=assets/dragon python tools/compose_ref.py runs/dragon/ref       ref_front.png + ref_side.png -> ref.png
The frame is the layout's: grey studio wall, lighter floor band, front view left, side view right, a dark divider.
"""
import hashlib, json, pathlib, sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mocap import layout  # noqa: E402
from mocap.camera_geometry import verify_record_angles

d = pathlib.Path(sys.argv[1])
render_camera_manifest = d / "render_camera_angles.json"
angles = (layout.FRONT_AZIMUTH_DEG, layout.SIDE_AZIMUTH_DEG)
render_metadata = json.loads(render_camera_manifest.read_text(encoding="utf-8")) if render_camera_manifest.exists() else {}
verify_record_angles(render_metadata.get("camera_azimuth_deg"), angles, "Blender reference renders")
if render_metadata:
    if render_metadata.get("widths") != [layout.W_FRONT, layout.W_SIDE] or render_metadata.get("height") != layout.H:
        raise ValueError("CAMERA_REFERENCE_LAYOUT_MISMATCH: reference image dimensions changed")
f = np.array(Image.open(d / "ref_front.png").convert("RGBA"))
s = np.array(Image.open(d / "ref_side.png").convert("RGBA"))
ref_path = d / "ref.png"
Image.fromarray(layout.compose(f, s)).save(ref_path)
ref_metadata = {"camera_azimuth_deg": list(angles),
                "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                "widths": [layout.W_FRONT, layout.W_SIDE], "height": layout.H}
ref_path.with_suffix(".camera.json").write_text(json.dumps(ref_metadata, indent=2), encoding="utf-8")
print("wrote", ref_path, f"{layout.W} x {layout.H}", "angles", angles)
