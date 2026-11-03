"""Draw posthoc HR V2 worst-joint overlays, GT green vs Reader red.

Reads frozen diagnostic json and already-generated 672px Test PNGs.
Never retrains or selects thresholds based on Test.
"""
import json, pathlib
from PIL import Image, ImageDraw, ImageFont

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DATA=ROOT/"dataset"
REPORT=ROOT/"reader_v2/full/outlier_diagnostic_posthoc.json"
OUT=ROOT/"reader_v2/full/outliers_contact_posthoc.jpg"
TOP=4

def render():
    diag=json.loads(REPORT.read_text(encoding="utf-8"))
    mf=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    names=mf["joint_names"]
    image_rows={(r["clip"],r["frame"],r["slot"]):r for r in mf["samples"] if r["split"]=="test"}
    records=diag["splits"]["test"]["worst_outliers"]
    choices=[]
    chosen=set()
    for record in records:
        key=(record["clip"],record["frame"],record["joint"])
        if key not in chosen:
            choices.append(record)
            chosen.add(key)
        if len(choices)>=TOP:break
    if len(choices)<TOP:raise RuntimeError("Not enough reported outlier examples")
    padding=14
    tile_w=410
    tile_h=410
    title_h=70
    canvas=Image.new("RGB",(2*(tile_w+padding)+padding,TOP*(tile_h+title_h+padding)+padding),(27,31,38))
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",17)
    except OSError:font=ImageFont.load_default()
    for ri,record in enumerate(choices):
        clip,frame,name=record["clip"],record["frame"],record["joint"]
        idx=names.index(name)
        for ci,(slot,az,tag) in enumerate((("front",315,"315_pred_uv"),("side",45,"45_pred_uv"))):
            row=image_rows[(clip,frame,slot)]
            with Image.open(DATA/row["file"]) as base:
                rgba=base.convert("RGBA")
                back=Image.new("RGBA",rgba.size,(100,100,105,255))
                back.alpha_composite(rgba)
                image=back.convert("RGB")
            paint=ImageDraw.Draw(image)
            gt=row["uv"][idx]
            pred=record[tag]
            gx,gy=[int(round(x)) for x in gt]
            px,py=[int(round(x)) for x in pred]
            paint.line((gx-11,gy,gx+11,gy),fill=(35,250,50),width=3)
            paint.line((gx,gy-11,gx,gy+11),fill=(35,250,50),width=3)
            paint.ellipse((px-9,py-9,px+9,py+9),outline=(230,60,70),width=4)
            paint.line((gx,gy,px,py),fill=(250,215,80),width=2)
            image=image.resize((tile_w,tile_h),Image.Resampling.LANCZOS)
            x=padding+ci*(tile_w+padding)
            y=padding+ri*(tile_h+title_h+padding)
            draw.text((x+4,y+5),f"{name} | {az}deg | {record['error_cm']:.1f}cm 3D",font=font,fill="white")
            draw.text((x+4,y+30),f"{clip[-30:]} frame {frame}",font=font,fill="#b9c3d1")
            canvas.paste(image,(x,y+title_h))
    canvas.save(OUT,quality=90,optimize=True)
    print("HRV2_POSTHOC_OVERLAY",str(OUT),"bytes",OUT.stat().st_size)
if __name__=="__main__":
    render()
