"""Compose dataset samples into the two-camera frame and draw their true joints, as one contact sheet.

python tools/show_samples.py DATASET_DIR out.jpg [n]
"""
import json, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mocap import layout

d = pathlib.Path(sys.argv[1])
metas = sorted(d.glob("*_meta.json"))[: int(sys.argv[3]) if len(sys.argv) > 3 else 8]
tiles = []
for m in metas:
    meta = json.loads(m.read_text())
    s = meta["scale"]
    f = Image.open(str(m).replace("_meta.json", "_front.png")).convert("RGBA").resize((layout.W_FRONT, layout.H))
    si = Image.open(str(m).replace("_meta.json", "_side.png")).convert("RGBA").resize((layout.W_SIDE, layout.H))
    img = Image.fromarray(layout.compose(np.array(f), np.array(si)))
    dr = ImageDraw.Draw(img)
    for (u, v), (us, vs) in zip(meta["front_uv"], meta["side_uv"]):
        dr.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(255, 255, 0))
        dr.ellipse([us + layout.W_FRONT - 3, vs - 3, us + layout.W_FRONT + 3, vs + 3], fill=(255, 255, 0))
    shutter = meta.get("blur_shutter")
    blur_label = f"blur={shutter:.2f}" if shutter is not None else "sharp"
    dr.text((8, 8), f"{m.stem} {blur_label}", fill=(255, 255, 255))
    tiles.append(img.resize((640, 360)))
cols = 2
sheet = Image.new("RGB", (640 * cols, 360 * ((len(tiles) + cols - 1) // cols)))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 640, (i // cols) * 360))
sheet.save(sys.argv[2], quality=88)
