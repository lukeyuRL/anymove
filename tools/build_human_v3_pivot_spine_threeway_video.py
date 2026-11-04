"""Build synchronized 24FPS Blender visual comparison for Human V3 turning.

Panels: 315° GT/RootSO3/Train-neutral and 45° GT/RootSO3/Train-neutral.
All 107 frames rendered from independent root-static Blender .blend Actions,
not from just joint-projection drawings. This visual benchmark is Validation,
not sealed Test or iClone acceptance.
"""
import json,pathlib
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DIR=ROOT/"solver/spine_so3_train_neutral_init_research/blender_qa"
CLIP="spine_validation_pivot_turn_left_s405752"
MODES=("gt","root_so3","solver")
LABELS=("Blender GT","Root+SO3 600","Train-neutral 600")
SIZE=248
HEADER=55

def load_png(mode,slot,frame):
    f=DIR/"video_frames"/mode/slot/(f"frame_{frame:04d}.png")
    rgba=cv2.imread(str(f),cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape!=(672,672,4):
        raise FileNotFoundError("Incomplete rendered Blender movie "+str(f))
    return rgba

def composited(im):
    alpha=im[:,:,3:4].astype(np.float32)/255
    bgr=im[:,:,:3].astype(np.float32)
    pix=np.clip(bgr*alpha+np.array([100.,99.,96.],np.float32)*(1-alpha),0,255)
    return cv2.resize(np.uint8(pix),(SIZE,SIZE),interpolation=cv2.INTER_AREA)

def iou(a,b):
    av=a[:,:,3]>127
    bv=b[:,:,3]>127
    return float(np.logical_and(av,bv).sum()/max(1,np.logical_or(av,bv).sum()))

def build_video():
    out=DIR/(CLIP+"_GT_RootSO3_TrainNeutral_24fps.mp4")
    temp=out.with_suffix(".incomplete.mp4")
    vid=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*"mp4v"),24,(SIZE*6,SIZE+HEADER))
    if not vid.isOpened():raise RuntimeError("OpenCV MP4v encoder unavailable")
    stats=[]
    for frame in range(107):
        canvas=np.full((SIZE+HEADER,SIZE*6,3),43,dtype=np.uint8)
        for si,(slot,angle) in enumerate((("front",315),("side",45))):
            source=load_png("gt",slot,frame)
            images=[source,load_png("root_so3",slot,frame),load_png("solver",slot,frame)]
            for mi,im in enumerate(images):
                position=si*3+mi
                x=position*SIZE
                agreement=1.0 if mi==0 else iou(source,im)
                canvas[HEADER:,x:x+SIZE]=composited(im)
                cv2.putText(canvas,f"{angle}deg | {LABELS[mi]}",(x+6,22),
                            cv2.FONT_HERSHEY_SIMPLEX,.48,(238,238,239),1,cv2.LINE_AA)
                cv2.putText(canvas,f"Frame {frame:03d} | IoU {agreement:.3f}",(x+6,41),
                            cv2.FONT_HERSHEY_SIMPLEX,.43,(191,211,228),1,cv2.LINE_AA)
                if mi:
                    stats.append({"frame":frame,"camera":angle,"mode":MODES[mi],
                                  "silhouette_iou":agreement})
        vid.write(canvas)
    vid.release()
    if not temp.is_file() or temp.stat().st_size<40000:
        raise RuntimeError("Empty Blender turn comparison video")
    temp.replace(out)
    cap=cv2.VideoCapture(str(out))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if count!=107 or abs(fps-24)>.1:
        raise RuntimeError("Wrong recorded 107 source frames at 24FPS")
    grouped={}
    for mode in MODES[1:]:
        entries=np.asarray([r["silhouette_iou"] for r in stats if r["mode"]==mode])
        grouped[mode]={"mean_iou":float(entries.mean()),
                       "p05_iou":float(np.percentile(entries,5)),
                       "min_iou":float(entries.min())}
    result={
        "status":"VALIDATION_VISUAL_COMPARISON_GENERATED_NOT_PRODUCTION_ACCEPTANCE",
        "clip":CLIP,"fps":fps,"frames":count,"views":[315,45],
        "panels":["315 GT","315 Root SO3 600","315 Train-neutral 600",
                  "45 GT","45 Root SO3 600","45 Train-neutral 600"],
        "image_source":"actual Blender 5 reopened Action render at native 672px per view",
        "RL_BoneRoot_local_animation":"NO; static rest",
        "output":str(out),
        "scores":grouped,
        "limitations":"Procedural validation. More root-diverse families and independent synthetic test necessary."
    }
    (DIR/(CLIP+"_visual_ab.json")).write_text(json.dumps(result,indent=2),encoding="utf8")
    print("ROOT_STATIC_SPINE_THREEWAY_VIDEO",json.dumps(result),flush=True)
    return result

def make_contact():
    selection=(0,17,35,53,70,88,106)
    mini=194
    pad=10
    label_height=38
    ncols=6
    canvas=Image.new("RGB",(pad+ncols*(mini+pad),pad+len(selection)*(mini+label_height+pad)),(33,35,40))
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",15)
    except OSError:font=ImageFont.load_default()
    for row,frame in enumerate(selection):
        for si,(slot,angle) in enumerate((("front",315),("side",45))):
            for i,mode in enumerate(MODES):
                png=DIR/"video_frames"/mode/slot/(f"frame_{frame:04d}.png")
                with Image.open(png) as im:
                    rgba=im.convert("RGBA")
                    bg=Image.new("RGBA",rgba.size,(100,99,96,255))
                    bg.alpha_composite(rgba)
                    tile=bg.convert("RGB").resize((mini,mini),Image.Resampling.LANCZOS)
                col=si*3+i
                x=pad+col*(mini+pad)
                y=pad+row*(mini+label_height+pad)
                draw.text((x+3,y+3),f"{angle}deg | {LABELS[i]}",font=font,fill=(238,238,238))
                draw.text((x+3,y+21),f"Frame {frame}",font=font,fill=(191,211,220))
                canvas.paste(tile,(x,y+label_height))
    dest=DIR/(CLIP+"_spine_threeway_contact.jpg")
    canvas.save(dest,quality=92,optimize=True)
    print("SPINE_THREEWAY_CONTACT",str(dest),dest.stat().st_size,flush=True)

if __name__=="__main__":
    make_contact()
    build_video()
