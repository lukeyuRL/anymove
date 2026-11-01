"""Evaluate an actual Joint Reader on rendered Top images, NOT GT Oracle.

python -m tools.eval_dino_top_reader TOP_RENDER_DIR MODEL.pt     RIG.json GT_JOINTS.json NEW_OUTPUT_DIR

Uses the actual rendered RGB pixels, processes each view via the existing
672x672 Reader + 192 heatmap, reports held-out 2D error against independent
Blender Top projection. The original Front/Side Reader has never been
trained on this camera, so its results are a negative-control transfer test
and MUST NOT be mistaken for a trained Top Reader.

No train data, checkpoint, or existing motion is modified.
"""
import argparse
import hashlib
import json
import pathlib

import cv2
import numpy as np

from mocap import layout
from mocap.dino_top_camera import camera_from_json,project_top_numpy
from tools.evaluate_dino_full_body import classify_joint


def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as fd:
        for c in iter(lambda:fd.read(1048576),b""):
            h.update(c)
    return h.hexdigest()


def load_frames(directory,T):
    frames=[]
    for t in range(T):
        path=directory/"top"/f"f_{t:04d}.png"
        rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape!=(720,880,4):
            raise ValueError("missing/wrong-size synchronized Top RGBA render "+str(path))
        raw=rgba[..., [2,1,0,3]].astype(np.float32)
        a=raw[...,3:]/255
        rgb=(raw[...,:3]*a+np.asarray(layout.BG,dtype=np.float32)*(1-a)).clip(0,255)
        frames.append(rgb.astype(np.uint8))
    return frames


def evaluate(uv,conf,truth_uv,names):
    if uv.shape!=truth_uv.shape or conf.shape!=truth_uv.shape[:-1]:
        raise ValueError("Top Reader detections not aligned with GT")
    if (not np.isfinite(uv).all() or not np.isfinite(conf).all()):
        raise ValueError("invalid Reader Top UV/peak data")
    e=np.linalg.norm(uv-truth_uv,axis=-1)
    result={
        "joint_count":len(names),"frames":len(uv),
        "mean_2d_gt_error_px":round(float(e.mean()),5),
        "p95_2d_gt_error_px":round(float(np.percentile(e,95)),5),
        "max_2d_gt_error_px":round(float(e.max()),5),
        "mean_raw_heatmap_peak":round(float(conf.mean()),5),
        "per_group":{},
    }
    for group in ("head_neck","torso_root","forelimbs","hindlimbs","feet_toes","tail"):
        ids=[j for j,n in enumerate(names) if classify_joint(n)==group]
        if ids:
            result["per_group"][group]={
                "joints":len(ids),
                "mean_2d_gt_error_px":round(float(e[:,ids].mean()),5),
                "p95_2d_gt_error_px":round(float(np.percentile(e[:,ids],95)),5),
            }
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("top_render")
    p.add_argument("reader_checkpoint")
    p.add_argument("rig_info")
    p.add_argument("blender_truth")
    p.add_argument("out_dir")
    p.add_argument("--label",default="unverified_top_reader")
    a=p.parse_args(argv)
    render=pathlib.Path(a.top_render)
    ck=pathlib.Path(a.reader_checkpoint)
    rigpath=pathlib.Path(a.rig_info)
    gtpath=pathlib.Path(a.blender_truth)
    out=pathlib.Path(a.out_dir)
    if out.exists():
        raise FileExistsError("refuse overwrite of any Top Reader test results")
    for path in (ck,rigpath,gtpath,render/"cam_top.json",render/"top_render_audit.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    from mocap.detect import load,run
    cam=camera_from_json(render/"cam_top.json")
    gt=json.loads(gtpath.read_text(encoding="utf-8"))
    rig=json.loads(rigpath.read_text(encoding="utf-8"))
    report=json.loads((render/"top_render_audit.json").read_text(encoding="utf-8"))
    names=[k["name"] for k in rig["keypoints"]]
    T=len(gt["joints"])
    if names!=gt["names"] or len(names)!=92 or T!=report["frames"] or (
            report["blender_truth_sha256"]!=digest(gtpath) or
            report["rig_info_sha256"]!=digest(rigpath)):
        raise ValueError("Top Reader GT/rig/render camera provenance mismatch")
    frames=load_frames(render,T)
    model,reader_names=load(ck)
    if [str(k) for k in reader_names]!=names:
        raise ValueError("Reader checkpoint 92-joint order differs from Dino rig")
    pred=run(model,frames)
    uv,conf=np.asarray(pred[0],float),np.asarray(pred[1],float)
    truth_uv=project_top_numpy(np.asarray(gt["joints"],float),cam)
    quality=evaluate(uv,conf,truth_uv,names)
    quality["profile"]="anymove.dino_real_rendered_top_reader_2d_audit_v1"
    quality["source_model_label"]=a.label
    quality["warning"]="Evaluation against Blender GT, not model training. Existing Front/Side checkpoint is NOT trained for Top and cannot be used for production Top Motion."
    quality["provenance_sha256"]={
        "checkpoint":digest(ck),"rig":digest(rigpath),
        "gt":digest(gtpath),"cam_top":digest(render/"cam_top.json")}
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"top_reader_detections.npz",
                        top_uv=uv.astype(np.float32),
                        top_conf=conf.astype(np.float32),
                        names=np.array(names),
                        fps=float(gt["fps"]))
    (out/"top_reader_2d_gt_report.json").write_text(
        json.dumps(quality,indent=2),encoding="utf-8")
    print(json.dumps(quality,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
