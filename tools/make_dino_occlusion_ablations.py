"""Build opt-in Dino missing-2D-observation masks for controlled A/B.

Modes are strictly distinct:
  mesh       : ACTUAL evaluated Blender triangle intersections; only supported
               2+ crossings are dropped, unknown mesh proximity kept.
  p2_error   : P2 Reader's predicted 2D error >= threshold; no GT truth.
  mesh_p2    : union of mesh supported-blocked and P2 predicted high error.
  oracle_gt  : BLENDER 2D TRUTH error >= threshold; LEAKS ground truth,
               DIAGNOSTIC UPPER BOUND ONLY, never a production pipeline.

All preserve exact raw Reader coordinates, peaks, calibration and FPS;
add boolean front_visible_mask/side_visible_mask without overwriting.

python -m tools.make_dino_occlusion_ablations \
    RAW.npz RIG_INFO.json OUT_MASKED.npz --mode mesh --geometry GEOMETRY.npz

For predicted error:
    ... --mode p2_error --quality DET_P2.npz --threshold-px 10

For a strict oracle upper bound:
    ... --mode oracle_gt --truth BLENDER_GT/joints.json --threshold-px 10
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
from mocap import layout
from mocap.layout_guard import verify_layout_for_rig
from mocap.quality_weighting import _identical_reader
from tools.evaluate_dino_full_body import classify_joint

PROFILE="anymove.dino_per_view_observation_mask_ablation_v1"
MODES=("mesh","p2_error","mesh_p2","oracle_gt")


def sha256(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as fd:
        for block in iter(lambda:fd.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def _read_npz(path):
    with np.load(path,allow_pickle=False) as f:
        return {key:f[key].copy() for key in f.files}


def build(raw,rig,mode,*,geometry=None,quality=None,truth=None,
          threshold_px=10.0):
    if mode not in MODES:
        raise ValueError("unsupported Dino masked observation mode")
    names=[k["name"] for k in rig["keypoints"]]
    if len(names)!=92 or "names" not in raw or names != [str(n) for n in raw["names"]]:
        raise ValueError("Dino 92-joint Reader/Rig name order mismatch")
    T,J=np.asarray(raw["front_conf"]).shape
    if T<3 or J!=len(names):
        raise ValueError("invalid Dino Reader frame/joint count")
    if any(k in raw for k in ("front_track_weight","side_track_weight",
                              "front_visible_mask","side_visible_mask")):
        raise ValueError("masked ablation requires unmodified original Reader detections")
    if not np.isfinite(threshold_px) or threshold_px <= 0:
        raise ValueError("invalid threshold")
    original={}
    for view in ("front","side"):
        uv=np.asarray(raw[view+"_uv"],float)
        conf=np.asarray(raw[view+"_conf"],float)
        if (uv.shape!=(T,J,2) or conf.shape!=(T,J) or
                not np.isfinite(uv).all() or not np.isfinite(conf).all()):
            raise ValueError("invalid original Reader observations")
        original[view]=uv
    need_mesh=mode in ("mesh","mesh_p2")
    need_p2=mode in ("p2_error","mesh_p2")
    need_gt=mode=="oracle_gt"
    if (need_mesh and geometry is None or need_p2 and quality is None or
            need_gt and truth is None):
        raise ValueError("required geometry/P2/GT source missing for requested mode")
    if geometry is not None and not need_mesh or quality is not None and not need_p2 or truth is not None and not need_gt:
        raise ValueError("do not pass unused data sources for this mode")
    if need_p2:
        _identical_reader(raw,quality)  # same video, same frozen Reader UV/peaks
    if need_mesh:
        if [str(n) for n in geometry.get("names",[])]!=names or int(geometry["frame_count"])!=T:
            raise ValueError("mesh geometry joint order/frame count mismatch")
    if need_gt:
        if (truth.get("names")!=names or
                abs(float(truth["fps"])-float(raw["fps"]))>.00001):
            raise ValueError("oracle GT joint order or FPS mismatch")
        points=np.asarray(truth["joints"],float)
        if points.shape!=(T,J,3) or not np.isfinite(points).all():
            raise ValueError("oracle GT shape/finite mismatch")
        front_gt,side_gt=layout.project(points)
        gt_views={"front":front_gt,"side":side_gt}
    masks={}
    group_stats={}
    for view in ("front","side"):
        mask=np.ones((T,J),dtype=np.bool_)
        if need_mesh:
            blocker=np.asarray(geometry[view+"_mesh_blocked_proxy"])
            support=np.asarray(geometry[view+"_mesh_support"])
            crosses=np.asarray(geometry[view+"_surface_crossings"])
            if (blocker.shape!=(T,J) or support.shape!=(T,J) or crosses.shape!=(T,J) or
                    blocker.dtype!=np.bool_ or support.dtype!=np.bool_ or
                    np.any(blocker & ~support) or np.any(blocker & (crosses<2))):
                raise ValueError("mesh geometry proxy/support is invalid")
            mask &= ~blocker
        if need_p2:
            prediction=np.asarray(quality[view+"_expected_error_px"],float)
            if (prediction.shape!=(T,J) or not np.isfinite(prediction).all() or
                    np.any(prediction<0)):
                raise ValueError("invalid P2 predicted error source")
            mask &= prediction < threshold_px
        if need_gt:
            err=np.linalg.norm(original[view]-gt_views[view],axis=-1)
            mask &= err < threshold_px
        masks[view]=mask
        group_stats[view]={
            "masked":int((~mask).sum()),
            "total":int(T*J),
            "by_anatomy":{
                group:int((~mask[:,[i for i,n in enumerate(names) if classify_joint(n)==group]]).sum())
                for group in ("head_neck","torso_root","forelimbs",
                              "hindlimbs","feet_toes","tail")
            },
        }
    output={k:np.array(v,copy=True) for k,v in raw.items()}
    for view in ("front","side"):
        output[view+"_visible_mask"]=masks[view]
    audit={
        "profile":PROFILE,"mode":mode,"frames":T,"keypoints":J,
        "threshold_px":float(threshold_px) if need_p2 or need_gt else None,
        "views":group_stats,
        "joint_frames_both_views_masked":int((~(masks["front"]|masks["side"])).sum()),
        "no_2d_reader_observations_changed":True,
        "no_animation_keys_removed":True,
        "not_production":{
            "mesh":"Blender mesh line-of-sight 2+ intersection PROXY, not ground-truth anatomical visibility. Unknown skin proximity kept.",
            "p2_error":"Predicted localization error threshold is NOT learned true occlusion; outlier rejection only.",
            "mesh_p2":"Hybrid of geometric proxy and uncalibrated prediction; experimental, not a validated real-video gate.",
            "oracle_gt":"Uses Blender joint ground truth to choose bad observations. STRICT ORACLE / leakage; diagnostic potential only, never deploy.",
        }[mode],
    }
    return output,audit


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw_reader")
    p.add_argument("rig_info")
    p.add_argument("output_detection")
    p.add_argument("--mode",required=True,choices=MODES)
    p.add_argument("--geometry")
    p.add_argument("--quality")
    p.add_argument("--truth")
    p.add_argument("--threshold-px",type=float,default=10.0)
    a=p.parse_args(argv)
    inputs=[pathlib.Path(a.raw_reader),pathlib.Path(a.rig_info)]
    for value in (a.geometry,a.quality,a.truth):
        if value:
            inputs.append(pathlib.Path(value))
    out=pathlib.Path(a.output_detection).resolve()
    sidecar=out.with_suffix(".occlusion_input.json")
    if out.exists() or sidecar.exists() or out in {f.resolve() for f in inputs}:
        raise FileExistsError("refuse overwrite in occlusion experiment")
    for path in inputs:
        if not path.is_file():raise FileNotFoundError(path)
    cam=verify_layout_for_rig(a.rig_info,layout)
    if cam.get("creature")!="dino" or cam.get("status")!="camera_layout_verified_for_rig":
        raise ValueError("Dino occlusion ablation requires verified camera/rig layout")
    rig=json.loads(pathlib.Path(a.rig_info).read_text(encoding="utf-8"))
    raw=_read_npz(a.raw_reader)
    geom=_read_npz(a.geometry) if a.geometry else None
    qual=_read_npz(a.quality) if a.quality else None
    gt=json.loads(pathlib.Path(a.truth).read_text(encoding="utf-8")) if a.truth else None
    if a.geometry:
        meta_path=pathlib.Path(a.geometry).with_suffix(".mesh_geometry.json")
        if not meta_path.is_file():
            raise FileNotFoundError("missing Blender mesh geometry provenance audit")
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta["geometry_npz_sha256"]!=sha256(a.geometry)
                or meta["blender_truth_sha256"]!=sha256(a.truth or
                    pathlib.Path(a.geometry).parent/"gt_render/joints.json")
                or meta["rig_info_sha256"]!=sha256(a.rig_info)):
            raise ValueError("Blender geometry SHA validation failed")
    output,audit=build(raw,rig,a.mode,geometry=geom,quality=qual,
                       truth=gt,threshold_px=a.threshold_px)
    audit["input_sha256"]={str(path):sha256(path) for path in inputs}
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**output)
    audit["masked_detection_sha256"]=sha256(out)
    try:
        sidecar.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    except Exception:
        out.unlink(missing_ok=True)
        raise
    print(json.dumps(audit,indent=2))
    print("masked original Reader detection:",out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
