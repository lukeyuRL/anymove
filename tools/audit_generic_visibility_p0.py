"""Compare opt-in generic visibility Solver to same-raw-Reader baseline.

The original Reader observations are the common reprojection target.
This is NOT a 3D ground-truth accuracy test and cannot promote the proxy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def _load_motion(path, names, frames, fps):
    m=json.loads(Path(path).read_text(encoding="utf-8"))
    if [str(x) for x in m.get("joint_names",[])] != names:
        raise ValueError("baseline/variant joint names or order differ")
    if len(m.get("joints",[])) != frames or abs(float(m["fps"])-fps)>.01:
        raise ValueError("baseline/variant FPS or frame count mismatch")
    p=np.asarray(m["joints"],dtype=np.float64)
    root=np.asarray(m["trans"],dtype=np.float64)
    if p.shape!=(frames,len(names),3) or root.shape!=(frames,3) or not np.isfinite(p).all():
        raise ValueError("invalid motion joint shape or nonfinite data")
    return m,p,root


def _score(joints,d,layout):
    scale=float(layout["M_PER_PX"])
    z=layout["H"]/2-(joints[...,2]-layout["Z_CENTER"])/scale
    projections={
        "front":np.stack((layout["W_FRONT"]/2+joints[...,0]/scale,z),axis=-1),
        "side":np.stack((layout["W_SIDE"]/2-joints[...,1]/scale,z),axis=-1),
    }
    return {v:{
        "mean_px":round(float(np.linalg.norm(proj-d[v+"_uv"],axis=-1).mean()),5),
        "p95_px":round(float(np.percentile(np.linalg.norm(proj-d[v+"_uv"],axis=-1),95)),5)
    } for v,proj in projections.items()}


def report(det_path,rig_path,base_path,variant_path):
    raw=Path(det_path).read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    with np.load(det_path) as z:
        d={k:z[k].copy() for k in ("front_uv","side_uv","names","fps")}
    names=[str(s) for s in d["names"]]
    T=len(d["front_uv"])
    fps=float(d["fps"])
    info=json.loads(Path(rig_path).read_text(encoding="utf-8"))
    if [k["name"] for k in info["keypoints"]] != names:
        raise ValueError("Reader and Rig names differ")
    base,jbase,rbase=_load_motion(base_path,names,T,fps)
    variant,jvariant,rvariant=_load_motion(variant_path,names,T,fps)
    for tag,m in (("baseline",base),("variant",variant)):
        if m.get("source_detections_sha256")!=digest:
            raise ValueError(tag+" source detection hash mismatches original")
        if m.get("solver_camera_layout")!=base.get("solver_camera_layout"):
            raise ValueError(tag+" camera layout changed across runs")
    if any(float(base["solver_camera_layout"]["active_layout"][key]) != float(info["layout"][key])
           for key in ("W_FRONT","W_SIDE","M_PER_PX","GROUND_V")):
        raise ValueError("camera layout/rig mismatch")
    metrics={"baseline":_score(jbase,d,info["layout"]),
             "generic":_score(jvariant,d,info["layout"])}
    diff=np.linalg.norm(jvariant-jbase,axis=-1)
    rootdiff=np.linalg.norm(rvariant-rbase,axis=-1)
    one=Path(variant_path).with_suffix(".visibility.json")
    if not one.is_file():
        raise ValueError("variant missing Visibility sidecar")
    visibility=json.loads(one.read_text(encoding="utf-8"))
    if visibility.get("schema_version")!="anymove.generic_two_view_reliability_proxy/1":
        raise ValueError("variant Visibility profile mismatches generic P0")
    if visibility.get("source_detections_sha256")!=digest:
        raise ValueError("visibility sidecar source hash mismatch")
    result={
        "profile":"anymove.generic_visibility_P0_comparison/1",
        "not_3d_ground_truth":True,
        "not_true_occlusion_ground_truth":True,
        "baseline":str(base_path),"variant":str(variant_path),
        "original_reader_sha256":digest,"fps":fps,"frames":T,"keypoints":len(names),
        "common_raw_reader_reprojection_px":metrics,
        "generic_minus_baseline_mean_px":{
            v:round(metrics["generic"][v]["mean_px"]-metrics["baseline"][v]["mean_px"],5)
            for v in ("front","side")},
        "motion_between_solution_difference_cm":{
            "mean":round(float(diff.mean())*100,5),
            "p95":round(float(np.percentile(diff,95))*100,5),
            "max":round(float(diff.max())*100,5)},
        "root_between_solution_difference_cm":{
            "mean":round(float(rootdiff.mean())*100,5),
            "max":round(float(rootdiff.max())*100,5)},
        "reliability_summary":{v:{
            key:visibility["views"][v][key] for key in
                ("mean_weight","min_weight","joint_frames_downweighted","joints_downweighted")}
            for v in ("front","side")},
        "acceptance":"P0_INTEGRATION_ONLY_NOT_3D_ACCURACY",
        "do_not_enable_by_default_without_independent_3d_gt":True,
    }
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--detections",required=True)
    parser.add_argument("--rig",required=True)
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--variant",required=True)
    parser.add_argument("--out",required=True)
    args=parser.parse_args()
    dst=Path(args.out)
    if dst.exists():raise FileExistsError("report already exists: "+str(dst))
    result=report(args.detections,args.rig,args.baseline,args.variant)
    dst.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
