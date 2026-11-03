"""Compose V3 nose before / after close-up QA contact sheet."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path("F:/anymove/runs/human/camera_angle_p0_v3/nose_preview")
angles=(0,45,90,315)
sz=520
header=55
pad=14
canvas=Image.new("RGB",(pad+4*(sz+pad),pad+2*(sz+header+pad)),(32,35,40))
draw=ImageDraw.Draw(canvas)
try:
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",23)
except OSError:
    font=ImageFont.load_default()
for row,variant in enumerate(("before","nose")):
    for col,a in enumerate(angles):
        src=ROOT/f"v3_{variant}_face_{a:03d}.png"
        rgba=Image.open(src).convert("RGBA").resize((sz,sz),Image.Resampling.LANCZOS)
        bg=Image.new("RGBA",(sz,sz),(95,96,101,255))
        bg.alpha_composite(rgba)
        x=pad+col*(sz+pad);y=pad+row*(sz+header+pad)
        draw.text((x+4,y+9),f"{variant.upper()}  |  {a:03d} deg",fill=(242,242,248),font=font)
        canvas.paste(bg.convert("RGB"),(x,y+header))
out=ROOT/"human_v3_before_after_nose_contact.png"
canvas.save(out,optimize=True)
print("CONTACT_OK",out,out.stat().st_size,canvas.size)
