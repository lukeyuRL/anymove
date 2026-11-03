"""Video QA for per-view excluded observations, NOT animation key deletion.

python -m tools.review_dino_masked_observations \
  GT_VIDEO.mp4 MASKED_READER.npz NEW_MASK_REVIEW.mp4 [--geometry GEOMETRY.npz]

Green dot: Reader observation retained as Solver constraint.
Red X: per-view 2D observation excluded from Solver data loss.
Purple ring: Blender mesh support unknown (if geometry supplied).

All original 2D UV points remain in the NPZ and all frames remain keyed.
Never interpret this as verified natural-video occlusion ground truth.
"""
import argparse
import pathlib
import cv2
import numpy as np
from mocap import layout

GREEN=(95,210,110)
RED=(60,65,235)
UNKNOWN=(216,116,190)
WHITE=(224,224,224)
PANEL=420


def draw(frame, det, t, geometry=None):
    H,W=layout.H,layout.W
    if frame.shape[:2] != (H,W):
        raise ValueError("Dino review requires exactly calibrated 2-view layout video")
    out=np.zeros((H,W+PANEL,3),np.uint8)
    out[:,:W]=frame.copy()
    out[:,W:]=(35,35,40)
    def label(msg,y,color=WHITE,scale=.54):
        cv2.putText(out,str(msg),(W+10,int(y)),cv2.FONT_HERSHEY_SIMPLEX,
                    scale,color,1,cv2.LINE_AA)
    label("Dino 2D Observation Mask | Frame %03d"%t,26,WHITE,.57)
    label("GREEN = USE 2D JOINT / RED X = DROP OBS",52,GREEN,.47)
    label("PURPLE RING = mesh support UNKNOWN",75,UNKNOWN,.47)
    label("This does NOT delete animation keys.",98,WHITE,.46)
    names=[str(n) for n in det["names"]]
    h=130
    for view,offset,width in (("front",0,layout.W_FRONT),
                              ("side",layout.W_FRONT,layout.W_SIDE)):
        mask=np.asarray(det[view+"_visible_mask"][t],bool)
        coords=np.asarray(det[view+"_uv"][t],float)
        label("%s:  %d excluded / %d joints"%(view.upper(),(~mask).sum(),len(names)),h)
        h+=25
        missing=np.flatnonzero(~mask)
        # Long Dino names use the bone name rather than the ":tail" suffix.
        for j in missing[:8]:
            label("X "+names[j].split(":")[1][-32:],h,RED,.42)
            h+=20
        h+=18
        for j,(u,v) in enumerate(coords):
            if not np.isfinite([u,v]).all() or not(0<=u<width and 0<=v<H):
                continue
            x,y=int(round(u))+offset,int(round(v))
            if mask[j]:
                cv2.circle(out,(x,y),2,GREEN,-1,cv2.LINE_AA)
            else:
                cv2.line(out,(x-4,y-4),(x+4,y+4),RED,2,cv2.LINE_AA)
                cv2.line(out,(x-4,y+4),(x+4,y-4),RED,2,cv2.LINE_AA)
            if geometry is not None and not bool(geometry[view+"_mesh_support"][t,j]):
                cv2.circle(out,(x,y),8,UNKNOWN,1,cv2.LINE_AA)
    label("Mesh raycast is only a geometry proxy.",H-18,WHITE,.42)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("masked_detection")
    parser.add_argument("output")
    parser.add_argument("--geometry")
    a=parser.parse_args(argv)
    out=pathlib.Path(a.output).resolve()
    sources=[pathlib.Path(a.video).resolve(),
             pathlib.Path(a.masked_detection).resolve()]
    if a.geometry:sources.append(pathlib.Path(a.geometry).resolve())
    if out.exists() or out in sources:
        raise FileExistsError("refusing overwrite of masked-observation review")
    with np.load(a.masked_detection,allow_pickle=False) as data:
        det={k:data[k].copy() for k in data.files}
    geo=None
    if a.geometry:
        with np.load(a.geometry,allow_pickle=False) as data:
            geo={k:data[k].copy() for k in data.files}
    T,J=np.asarray(det["front_uv"]).shape[:2]
    if J!=92 or not all(
        np.asarray(det[view+"_visible_mask"]).shape==(T,J)
        and np.asarray(det[view+"_visible_mask"]).dtype==bool
        for view in ("front","side")):
        raise ValueError("missing valid Dino (T,92) per-view observation masks")
    if geo is not None and any(
        geo[view+"_mesh_support"].shape!=(T,J) for view in ("front","side")):
        raise ValueError("mesh support does not match reader mask")
    fps=float(det["fps"])
    if not np.isfinite(fps) or fps<=0:
        raise ValueError("invalid Dino masked detection FPS")
    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise ValueError("source Dino video cannot be decoded")
    if abs(cap.get(cv2.CAP_PROP_FPS)-fps)>.05:
        cap.release()
        raise ValueError("source Dino video FPS mismatch")
    out.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),
                           fps,(int(layout.W)+PANEL,int(layout.H)))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("cannot create masked-observation video")
    count=0
    try:
        for t in range(T):
            ok,frame=cap.read()
            if not ok:
                raise ValueError("masked Reader has more frames than input video")
            writer.write(draw(frame,det,t,geo))
            count+=1
        ok,_=cap.read()
        if ok:
            raise ValueError("masked Reader shorter than the source video")
    except Exception:
        writer.release()
        cap.release()
        out.unlink(missing_ok=True)
        raise
    finally:
        writer.release()
        cap.release()
    print("masked-view diagnostic video:",out,"frames:",count)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
