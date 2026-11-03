"""Compose V3 camera-angle Eevee comparisons from existing local renders."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, os

root=Path("runs/human/camera_angle_p0_v3")
variant=os.environ.get("ANYMOVE_HUMAN_VARIANT","v3")
if variant not in ("v3","nose"):
    raise ValueError("ANYMOVE_HUMAN_VARIANT must be v3 or nose")
choices=[("0° / 90° (Baseline)","human_v3_baseline_0_90_cam1_000.png","human_v3_baseline_0_90_cam2_090.png"),
("315° / 45° (Symmetric front)","human_v3_symmetric_315_45_cam1_315.png","human_v3_symmetric_315_45_cam2_045.png"),
("45° / 135° (Front/back oblique)","human_v3_oblique_45_135_cam1_045.png","human_v3_oblique_45_135_cam2_135.png")]
w,h=384,448
pad=22
rowh=526
canvas=Image.new("RGB",(w*2+pad*3,rowh*3+pad),(44,46,52))
draw=ImageDraw.Draw(canvas)
try:
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",21)
except OSError:
    font=ImageFont.load_default()
for i,(title,a,b) in enumerate(choices):
    y=pad+i*rowh
    draw.text((pad,y),title,font=font,fill=(245,247,250))
    for k,fname in enumerate((a,b)):
        if variant=="nose":
            fname=fname.replace("human_v3_","human_v3_nose_",1)
        img=Image.open(root/fname).convert("RGBA")
        if img.size!=(w,h):
            raise ValueError(f"{fname} resolution changed {img.size}")
        x=pad+k*(w+pad)
        background=Image.new("RGBA",(w,h),(90,90,94,255))
        background.alpha_composite(img)
        canvas.paste(background.convert("RGB"),(x,y+42))
out=root/("human_v3_nose_camera_angles_contact.png" if variant=="nose" else "human_v3_camera_angles_contact.png")
canvas.save(out)
print(json.dumps({"file":str(out),"bytes":out.stat().st_size,"size":canvas.size,"frames":len(choices)*2}))
