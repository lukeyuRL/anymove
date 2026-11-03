"""Compose original Dino front/side reference + Blender-rest Top image as 16:9 three-panel fal input.

Original fronts and Top retain individual image aspect ratios. Panel
boundary offsets and exact crop-to-original transforms are saved for
objective post-generation comparison; do not treat crop pixels as 3D.
"""
import argparse,hashlib,json,pathlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont

PANELS=(("front",320),("side",480),("top",480))
CANVAS_W,CANVAS_H=1280,720
SOURCE_SIZES={"front":(400,720),"side":(880,720),"top":(880,720)}
BG=(128,130,134)


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as fd:
        for chunk in iter(lambda:fd.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def compose(front,side,top,outdir):
    imgs={"front":pathlib.Path(front),"side":pathlib.Path(side),
          "top":pathlib.Path(top)}
    for view,path in imgs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
    root=pathlib.Path(outdir).resolve()
    if not root.is_dir():
        raise FileNotFoundError("create separate fal reference directory first "+str(root))
    output=root/"ref_3view_1280x720.png"
    manifest=root/"ref_3view_layout.json"
    if output.exists() or manifest.exists():
        raise FileExistsError("refusing overwrite of 3-view reference")
    canvas=Image.new("RGB",(CANVAS_W,CANVAS_H),BG)
    details={}
    cursor=0
    for view,w in PANELS:
        with Image.open(imgs[view]) as src:
            rgba=src.convert("RGBA")
        if rgba.size!=SOURCE_SIZES[view]:
            raise ValueError(f"{view} reference resolution differs from Dino contract: {rgba.size}")
        s=min(w/rgba.width,CANVAS_H/rgba.height)
        nw,nh=int(round(rgba.width*s)),int(round(rgba.height*s))
        dx=cursor+(w-nw)//2
        dy=(CANVAS_H-nh)//2
        resized=rgba.resize((nw,nh),Image.Resampling.LANCZOS)
        # Preserve RGBA transparency with same studio backdrop used by
        # original Front+Side reference frame.
        canvas.paste(resized,(dx,dy),resized)
        details[view]={
            "panel_left":cursor,"panel_width":w,
            "render_left":dx,"render_top":dy,
            "render_width":nw,"render_height":nh,
            "source_width":rgba.width,"source_height":rgba.height,
            "scale_x":nw/rgba.width,"scale_y":nh/rgba.height,
            "alpha_composited_with_studio_bg":True,
            "source_sha256":sha(imgs[view]),"source_filename":imgs[view].name,
        }
        cursor+=w
    d=ImageDraw.Draw(canvas)
    for x in (320,800):
        d.rectangle([x-2,0,x+2,CANVAS_H-1],fill=(34,34,38))
    # Header labels stay fixed panels; acceptable for smoke test.
    # No watermark/independent generated 3D content added.
    canvas.save(output,optimize=True)
    info={
        "profile":"anymove.h3max_dino_front_side_top_reference_smoke_v1",
        "canvas":[CANVAS_W,CANVAS_H],"panels":details,
        "divider_x":[320,800],
        "synchronized_source_pose":"same Dino rig rest pose for all three cameras",
        "geometry":"Front X/Z, Side Y/Z, Top Y/X with +90deg camera rotation",
        "video_validation":"generated video must keep panel boundaries and time-coherent 3D motion; source geometry alone does not ensure that",
        "output_sha256":sha(output),
    }
    manifest.write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps({"reference":str(output),"manifest":str(manifest),
                      "dimensions":[CANVAS_W,CANVAS_H],
                      "panels":{key:[v["panel_left"],v["panel_width"],
                                           v["render_top"],v["render_height"]]
                                for key,v in details.items()}},indent=2))
    return output,manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front")
    parser.add_argument("side")
    parser.add_argument("top")
    parser.add_argument("outdir")
    args=parser.parse_args()
    compose(args.front,args.side,args.top,args.outdir)
