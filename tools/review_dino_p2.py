"""Review a live Dino P2 Reader on the original two-camera video.

python -m tools.review_dino_p2 VIDEO.mp4 P2_DET.npz OUTPUT.mp4 [--report REPORT.json]

Draws each of the 92 Reader joints with colored overlays and per-view
localization-error ranking. This is only a predicted reliability visualization:
a red/purple keypoint does NOT prove true occlusion or 3D error.

Optional --raw-det baseline.npz checks frame/FPS/joint/calibration consistency.
Never overwrites an existing MP4/report.
"""
import argparse
import json
import math
import pathlib

import cv2
import numpy as np

from mocap import layout

GREEN = (100, 225, 120)       # predicted error <=5px
YELLOW = (50, 200, 255)       # 5-12px
RED = (65, 70, 245)          # >12px
PURPLE = (225, 105, 215)     # low synthetic-cutout score (not real visibility)
WHITE = (225, 225, 225)
PANEL_W = 540


def _short(name):
    s = str(name).replace("joint:", "").replace("CC_Dinosaur_", "")
    return s.split(":")[0][:36]


def quality_report(det):
    names = [str(n) for n in det["names"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("P2 joint names/order must be unique")
    T,J = np.asarray(det["front_uv"]).shape[:2]
    if T < 2 or J != len(names):
        raise ValueError("P2 invalid frames/joint shape")
    views={}
    for view in ("front","side"):
        uv = np.asarray(det[view+"_uv"],dtype=float)
        err = np.asarray(det[view+"_expected_error_px"],dtype=float)
        score = np.asarray(det[view+"_synthetic_visibility_score"],dtype=float)
        if (uv.shape != (T,J,2) or err.shape!=(T,J) or score.shape!=(T,J)
            or not all(np.isfinite(z).all() for z in (uv,err,score))
            or (err<0).any() or ((score<0)|(score>1)).any()):
            raise ValueError("P2 missing/invalid quality fields for "+view)
        mean_joint=np.mean(err,axis=0)
        order=np.argsort(-mean_joint)[:12]
        views[view]={
            "mean_predicted_error_px":round(float(err.mean()),4),
            "p95_predicted_error_px":round(float(np.percentile(err,95)),4),
            "max_predicted_error_px":round(float(err.max()),4),
            "joint_frames_predicted_error_over_5px":int((err>5).sum()),
            "joint_frames_predicted_error_over_12px":int((err>12).sum()),
            "joint_frames_synthetic_cutout_score_below_0_35":int((score<.35).sum()),
            "top_mean_predicted_error_joints":[{
                "name":names[j],"mean_predicted_error_px":round(float(mean_joint[j]),4)}
                for j in order],
            "worst_frames_by_high_error_count":[
                {"frame":int(t),"flagged_joints":int((err[t]>12).sum())}
                for t in np.argsort(-np.count_nonzero(err>12,axis=1))[:10]]
        }
    return {
        "profile":"anymove.dino_p2_video_visual_review/1",
        "frames":T,"keypoints":J,"views":views,
        "warning":"Colors represent predicted synthetic-trained reliability; NOT true anatomy/visibility or 3D ground truth."
    }


def _draw_panel(canvas, det, t, panel_x):
    cv2.rectangle(canvas,(panel_x,0),(canvas.shape[1],canvas.shape[0]),(32,32,37),-1)
    def line(msg,y,col=WHITE,scale=.47):
        cv2.putText(canvas,str(msg),(panel_x+12,int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,scale,col,1,cv2.LINE_AA)
    line(f"DINO P2 | frame {t} | {t/float(det['fps']):.2f}s",29,WHITE,.64)
    line("Orange = estimated error >5px; red >12px",56,YELLOW,.48)
    line("Purple ring = LOW synthetic cutout score",79,PURPLE,.46)
    line("NOT true mesh visibility probability",104,PURPLE,.44)
    y=137
    for view in ("front","side"):
        err=np.asarray(det[view+"_expected_error_px"][t])
        sc=np.asarray(det[view+"_synthetic_visibility_score"][t])
        line(f"{view.upper()} | err avg {err.mean():.2f}px | >12px {(err>12).sum()}",y,WHITE,.54)
        y+=23
        names = [str(n) for n in det["names"]]
        for j in np.argsort(-err)[:6]:
            line(f"{_short(names[j])[:27]}   err {err[j]:.1f}px  synth {sc[j]:.2f}",
                 y,RED if err[j]>12 else WHITE,.42)
            y+=20
        y+=25
    line("Inspect toes / tail / root at frames 12, 82",canvas.shape[0]-37,WHITE,.46)


def draw_one(raw_bgr, det, t):
    W,H=int(layout.W),int(layout.H)
    out=np.zeros((H,W+PANEL_W,3),np.uint8)
    out[:,:W]=cv2.warpAffine(raw_bgr,np.asarray(det["calib"],np.float32),
                             (W,H),flags=cv2.INTER_AREA)
    _draw_panel(out,det,t,W)
    for view,xoffset,width in (("front",0,int(layout.W_FRONT)),
                               ("side",int(layout.W_FRONT),int(layout.W_SIDE))):
        uv=np.asarray(det[view+"_uv"][t])
        err=np.asarray(det[view+"_expected_error_px"][t])
        score=np.asarray(det[view+"_synthetic_visibility_score"][t])
        for j,(u,v) in enumerate(uv):
            if not (np.isfinite(u) and np.isfinite(v) and 0<=u<width and 0<=v<H):
                continue
            x,y=int(round(u))+xoffset,int(round(v))
            if err[j]>12:
                color=RED
            elif err[j]>5:
                color=YELLOW
            else:
                color=GREEN
            cv2.circle(out,(x,y),3 if err[j]<=5 else 5,color,-1,cv2.LINE_AA)
            if score[j]<.35:
                cv2.circle(out,(x,y),8,PURPLE,2,cv2.LINE_AA)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("quality_detection")
    parser.add_argument("output")
    parser.add_argument("--report",help="optional JSON summary path")
    parser.add_argument("--raw-det",help="original Reader detection for alignment validation")
    args=parser.parse_args(argv)
    video=pathlib.Path(args.video).resolve()
    quality=pathlib.Path(args.quality_detection).resolve()
    output=pathlib.Path(args.output).resolve()
    report_path=pathlib.Path(args.report).resolve() if args.report else output.with_suffix(".json")
    if (output.exists() or report_path.exists() or output in (video,quality)
            or report_path in (video,quality) or output==report_path):
        raise FileExistsError("P2 preview/report may not overwrite earlier outputs")
    with np.load(quality,allow_pickle=False) as src:
        det={key:src[key].copy() for key in src.files}
    if ("fps" not in det or "calib" not in det or
            np.asarray(det["calib"]).shape!=(2,3)):
        raise ValueError("missing P2 video calibration/FPS")
    fps=float(np.asarray(det["fps"]).item())
    if not math.isfinite(fps) or fps<=0:
        raise ValueError("invalid P2 video FPS")
    summary=quality_report(det)
    if args.raw_det:
        with np.load(args.raw_det,allow_pickle=False) as x:
            raw={key:x[key].copy() for key in x.files}
        from mocap.quality_weighting import _identical_reader
        summary["same_raw_reader_observations"]=_identical_reader(raw,det)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("cannot decode source video")
    source_fps=float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or abs(source_fps-fps)>.05:
        cap.release()
        raise ValueError("video and detection FPS mismatch")
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*"mp4v"),
                          fps,(int(layout.W)+PANEL_W,int(layout.H)))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("cannot open quality preview output video writer")
    frames=0
    try:
        while frames < summary["frames"]:
            ok,bgr=cap.read()
            if not ok:
                raise ValueError("source video shorter than P2 detections")
            writer.write(draw_one(bgr,det,frames))
            frames+=1
        extra,_=cap.read()
        if extra:
            raise ValueError("source video longer than P2 detections")
    except Exception:
        writer.release()
        output.unlink(missing_ok=True)
        cap.release()
        raise
    finally:
        writer.release()
        cap.release()
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("quality video:",output)
    print("quality summary:",report_path)
    print(json.dumps(summary,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
