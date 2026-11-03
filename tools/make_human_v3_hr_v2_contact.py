"""Make 12-family HR 315/45 QA contact sheet (JPEG, does not count as PNG dataset).

Read-only: uses first distinct Train clip of every family at frame 50.
"""
import pathlib
from PIL import Image,ImageDraw,ImageFont
from tools import human_v3_fullbody_motion_v2 as motion

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DATA=ROOT/"dataset"
FRAME=50
W=340
H=340
GAP=16
HEADER=54
COLS=4
ROWS=6
canvas=Image.new("RGB",(GAP+COLS*(W+GAP),GAP+ROWS*(H+HEADER+GAP)),(32,34,39))
draw=ImageDraw.Draw(canvas)
try:font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",18)
except OSError:font=ImageFont.load_default()
for family_idx,family in enumerate(motion.FAMILIES):
    seed=72000+family_idx*1901
    clip=f"hrv2_train_{family}_s{seed}"
    for view_idx,(slot,angle) in enumerate((("front",315),("side",45))):
        col=(family_idx%2)*2+view_idx
        row=family_idx//2
        src=DATA/"train"/slot/(clip+"_f"+str(FRAME).zfill(3)+".png")
        if not src.is_file():raise FileNotFoundError("Family render incomplete "+str(src))
        with Image.open(src) as image:
            if image.size!=(672,672):
                raise RuntimeError("Contact image does not use native 672 source")
            rgba=image.convert("RGBA")
            bg=Image.new("RGBA",rgba.size,(72,73,77,255))
            bg.alpha_composite(rgba)
            im=bg.convert("RGB").resize((W,H),Image.Resampling.LANCZOS)
        x=GAP+col*(W+GAP)
        y=GAP+row*(H+HEADER+GAP)
        draw.text((x+3,y+7),f"{family} | {angle} deg",fill=(245,245,247),font=font)
        canvas.paste(im,(x,y+HEADER))
out=ROOT/"qa_contact_12_motion_families_315_45.jpg"
canvas.save(out,quality=88,optimize=True)
print("HRV2_VISUAL_QA_CONTACT",out,canvas.size,out.stat().st_size)
