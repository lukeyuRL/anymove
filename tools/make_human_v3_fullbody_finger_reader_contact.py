"""Build human fullbody + finger Reader A/B visual QA contact sheet."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path("runs/human/fullbody_finger_v3_nose/reader_models_strict")
PAIRS=[("baseline_0_90","0 / 90"),("symmetric_315_45","315 / 45")]
SCENES=[("fb_test_lunge_reach_8672",48,"Lunge / reach"),
        ("fb_test_balance_point_8801",48,"Balance / point")]
W=H=448
HEADER=56
PAD=16
canvas=Image.new("RGB",(4*(W+PAD)+PAD,2*(H+HEADER+PAD)+PAD),(31,34,40))
draw=ImageDraw.Draw(canvas)
try:
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",21)
except OSError:
    font=ImageFont.load_default()
for row,(clip,frame,caption) in enumerate(SCENES):
    for ix,(pair,deg) in enumerate(PAIRS):
        for vi,view in enumerate(("front","side")):
            col=ix*2+vi
            path=ROOT/pair/f"overlay_{clip}_f{frame}_{view}.png"
            if not path.is_file():raise FileNotFoundError(path)
            image=Image.open(path).convert("RGB")
            if image.size!=(W,H):raise ValueError("Unexpected Reader overlay size")
            x=PAD+col*(W+PAD)
            y=PAD+row*(H+HEADER+PAD)
            draw.text((x+2,y+6),f"{deg} | {view} | {caption}",fill=(244,244,246),font=font)
            canvas.paste(image,(x,y+HEADER))
out=ROOT/"human_v3_nose_fullbody_finger_reader_ab_contact.png"
canvas.save(out,optimize=True)
print("FULLBODY_FINGER_READER_AB_CONTACT",str(out),out.stat().st_size,canvas.size)
