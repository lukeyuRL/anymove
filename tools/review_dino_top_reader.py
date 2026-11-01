"""Render true image-inferred Top Tail keypoint QA video.

Red cross = independent Blender 2D Ground Truth, green circle = model
observation; connecting segment shows actual Top 2D localization residual.
Ground truth is used ONLY FOR REVIEW, NEVER passed into the Solver.

python -m tools.review_dino_top_reader TOP_RENDER_DIR TOP_READER.npz \
    RIG_INFO.json GT_JOINTS.json NEW_VIDEO.mp4
"""
import argparse
import json
import pathlib

import cv2
import numpy as np

from mocap import layout
from mocap.dino_top_camera import camera_from_json,project_top_numpy
from tools.evaluate_dino_full_body import classify_joint

PANEL=375
GREEN=(80,230,90)
RED=(70,75,225)
WHITE=(220,220,220)


def draw(frame,reader_uv,reader_peak,gt_uv,labels,t,fps):
    H,W=frame.shape[:2]
    if (H,W)!=(720,880):
        raise ValueError("Dino Top preview requires 880x720 BGR frames")
    display=np.zeros((H,W+PANEL,3),np.uint8)
    display[:,:W]=frame.copy()
    display[:,W:]=(35,35,38)
    def text(msg,y,color=WHITE,size=.49):
        cv2.putText(display,str(msg),(W+9,int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
    text(f"ACTUAL Top Reader | frame {t} | {t/fps:.2f} sec",28,WHITE,.52)
    text("GREEN: Reader; RED X: Blender GT QA",56,GREEN,.45)
    text("Only 9 Tail joint observations are reviewed",78,WHITE,.43)
    text("GT is NOT used by Solver",102,RED,.48)
    errors=np.linalg.norm(reader_uv-gt_uv,axis=-1)
    order=np.argsort(-errors,kind="stable")
    y=137
    for j in order:
        text(f"{labels[j][-24:]:24s} {errors[j]:5.1f}px  peak {reader_peak[j]:.2f}",y,
             RED if errors[j]>10 else WHITE,.4)
        y+=24
    for a,b in zip(reader_uv,gt_uv):
        if not np.isfinite([*a,*b]).all():
            continue
        pa=(int(round(a[0])),int(round(a[1])))
        pb=(int(round(b[0])),int(round(b[1])))
        if all(0<=pt[0]<W and 0<=pt[1]<H for pt in (pa,pb)):
            cv2.line(display,pa,pb,(100,150,210),1,cv2.LINE_AA)
            cv2.circle(display,pa,4,GREEN,-1,cv2.LINE_AA)
            cv2.line(display,(pb[0]-3,pb[1]-3),(pb[0]+3,pb[1]+3),RED,2,cv2.LINE_AA)
            cv2.line(display,(pb[0]-3,pb[1]+3),(pb[0]+3,pb[1]-3),RED,2,cv2.LINE_AA)
    text(f"Tail mean 2D error: {errors.mean():.2f} px",H-35,WHITE,.57)
    return display


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("top_render")
    p.add_argument("top_reader")
    p.add_argument("rig_info")
    p.add_argument("blender_gt")
    p.add_argument("output_video")
    args=p.parse_args(argv)
    render=pathlib.Path(args.top_render)
    target=pathlib.Path(args.output_video).resolve()
    sources=[pathlib.Path(args.top_reader),pathlib.Path(args.rig_info),
             pathlib.Path(args.blender_gt),render/"cam_top.json"]
    if target.exists() or target in [path.resolve() for path in sources]:
        raise FileExistsError("refuse overwrite of earlier Top review artifacts")
    rig=json.loads(pathlib.Path(args.rig_info).read_text(encoding="utf-8"))
    gt=json.loads(pathlib.Path(args.blender_gt).read_text(encoding="utf-8"))
    names=[kp["name"] for kp in rig["keypoints"]]
    if names!=gt["names"] or len(names)!=92:
        raise ValueError("Top Reader and GT keypoints mismatched")
    with np.load(args.top_reader,allow_pickle=False) as z:
        reader={key:z[key].copy() for key in z.files}
    T=len(gt["joints"])
    fps=float(reader["fps"])
    if (T<3 or abs(fps-float(gt["fps"]))>1e-5
            or names!=[str(x) for x in reader["names"]]):
        raise ValueError("Top Reader frame/FPS/names mismatch")
    uv=np.asarray(reader["top_uv"],float)
    peak=np.asarray(reader["top_conf"],float)
    if uv.shape!=(T,92,2) or peak.shape!=(T,92):
        raise ValueError("Top Reader 2D observation array shape mismatch")
    camera=camera_from_json(render/"cam_top.json")
    gtuv=project_top_numpy(np.asarray(gt["joints"],float),camera)
    tail=[j for j,name in enumerate(names) if classify_joint(name)=="tail"]
    if len(tail)!=9:
        raise ValueError("expected exactly nine Dino Tail joint points")
    target.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*"mp4v"),fps,
                           (880+PANEL,720))
    if not writer.isOpened():
        raise RuntimeError("unable to open Top QA MP4 output")
    count=0
    try:
        for t in range(T):
            image=cv2.imread(str(render/"top"/f"f_{t:04d}.png"),
                             cv2.IMREAD_UNCHANGED)
            if image is None or image.shape!=(720,880,4):
                raise FileNotFoundError("missing exact Top frame "+str(t))
            a=image[...,3:4].astype(np.float32)/255
            background=np.asarray(layout.BG,dtype=np.float32).reshape(1,1,3)
            img=(image[...,:3].astype(np.float32)*a+background*(1-a)).astype(np.uint8)
            writer.write(draw(img,uv[t,tail],peak[t,tail],gtuv[t,tail],
                              [names[j].split(":")[1] for j in tail],t,fps))
            count+=1
    except Exception:
        writer.release()
        target.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
    print("Top QA video:",target,"frames",count)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
