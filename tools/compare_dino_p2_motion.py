"""Dino P2 Reader weighting: actual Solver A/B on IDENTICAL raw 2D observations.

python -m tools.compare_dino_p2_motion   RAW_DET.npz P2_WEIGHTED_DET.npz BASE_MOTION.json P2_MOTION.json   RIG_INFO.json --out P2_MOTION_AUDIT.json

This is NOT ground truth; a lower raw-Reader reprojection might simply
favor the original Reader's mistakes. Requires the P2 weighted NPZ to
retain the original Reader 2D coordinates and confidence peaks exactly.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from tools.diagnose_dino_tracking import _project, _root_dynamics, _vecstats


def sha256_file(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def compare(raw, weighted, base, variant, camera):
    names=[str(x) for x in raw["names"]]
    if (not names or len(set(names))!=len(names) or
            names!=[str(x) for x in weighted["names"]]):
        raise ValueError("mismatched Dino Reader joint names/order")
    fps=float(raw["fps"])
    T,J=np.asarray(raw["front_conf"]).shape
    if T<2 or J!=len(names) or abs(fps-float(weighted["fps"]))>1e-6:
        raise ValueError("mismatched Reader frames or FPS")
    for key in ("bones","joint_names","fps"):
        if base.get(key)!=variant.get(key):
            raise ValueError("mismatched Motion JSON "+key)
    if (list(base["joint_names"])!=names or abs(float(base["fps"])-fps)>1e-6):
        raise ValueError("Motion names/FPS do not match original Reader")
    diffs={}
    for view in ("front","side"):
        for suffix in ("_uv","_conf"):
            k=view+suffix
            a,b=np.asarray(raw[k]),np.asarray(weighted[k])
            if a.shape!=b.shape or not np.array_equal(a,b):
                raise ValueError("P2 weighting changed original Reader observations: "+k)
        k=view+"_track_weight"
        if k not in weighted:
            raise ValueError("P2 weight-only detection lacks "+k)
        w=np.asarray(weighted[k],dtype=float)
        if (w.shape!=(T,J) or not np.isfinite(w).all() or
                np.any(w<=0) or np.any(w>1)):
            raise ValueError("invalid P2 Solver multiplier "+k)
        diffs[view]={
            "mean_multiplier":round(float(w.mean()),6),
            "downweighted_count":int((w<1).sum()),
            "strongly_downweighted_count":int((w<=.400001).sum()),
        }
    if ("calib" not in raw or "calib" not in weighted or
            not np.array_equal(raw["calib"],weighted["calib"])):
        raise ValueError("P2 ablation camera calibration modified")
    j0,j1=[np.asarray(m["joints"],float) for m in (base,variant)]
    r0,r1=[np.asarray(m["trans"],float) for m in (base,variant)]
    if (j0.shape!=(T,J,3) or j1.shape!=j0.shape or
            r0.shape!=(T,3) or r1.shape!=r0.shape or
            not all(np.isfinite(x).all() for x in (j0,j1,r0,r1))):
        raise ValueError("invalid Dino motion data")
    same_ref={}
    for name,joints in (("baseline",j0),("p2_weighted",j1)):
        front,side=_project(joints,camera)
        same_ref[name]={
            "front_px":_vecstats(np.linalg.norm(
                front-np.asarray(raw["front_uv"],float),axis=-1),"_px"),
            "side_px":_vecstats(np.linalg.norm(
                side-np.asarray(raw["side_uv"],float),axis=-1),"_px"),
        }
    joint_delta=np.linalg.norm(j1-j0,axis=-1)*100.
    root_delta=np.linalg.norm(r1-r0,axis=-1)*100.
    order=np.argsort(-joint_delta.mean(axis=0),kind="stable")[:12]
    return {
        "profile":"anymove.dino_p2_motion_ablation/1",
        "comparison_type":"same raw Reader UV/peaks; P2 error-only solver weighting; NOT 3D ground truth",
        "frames":T,"fps":fps,"keypoints":J,
        "p2_solver_multipliers":diffs,
        "shared_raw_reader_reprojection":same_ref,
        "between_solution_joint_cm":_vecstats(joint_delta,"_cm"),
        "between_solution_root_cm":_vecstats(root_delta,"_cm"),
        "max_root_difference_frame":int(np.argmax(root_delta)),
        "root_dynamics":{
            "baseline":_root_dynamics(r0,fps),
            "p2_weighted":_root_dynamics(r1,fps)},
        "worst_affected_joints":[
            {"name":names[int(j)],
             "mean_delta_cm":round(float(joint_delta[:,j].mean()),5),
             "max_delta_cm":round(float(joint_delta[:,j].max()),5)}
            for j in order],
        "warnings":[
            "The Reader's expected error was trained on synthetic poses; "
            "the heuristic is not a calibrated 3D observation model.",
            "Common-reader reprojection and between-solution differences "
            "are NOT independent 3D ground-truth errors.",
            "Check toe/foot sliding, tail and Root in actual rendered Blender/iClone scenes.",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw_detection")
    p.add_argument("p2_weighted_detection")
    p.add_argument("baseline_motion")
    p.add_argument("p2_motion")
    p.add_argument("rig_info")
    p.add_argument("--out")
    a=p.parse_args(argv)
    from mocap import layout
    from mocap.layout_guard import verify_layout_for_rig
    rig=verify_layout_for_rig(a.rig_info,layout)
    if rig.get("status")!="camera_layout_verified_for_rig" or rig.get("creature")!="dino":
        raise ValueError("P2 ablation requires verified Dino camera/rig")
    with np.load(a.raw_detection,allow_pickle=False) as z:
        raw={k:z[k].copy() for k in z.files}
    with np.load(a.p2_weighted_detection,allow_pickle=False) as z:
        weighted={k:z[k].copy() for k in z.files}
    motions=[
        json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        for path in (a.baseline_motion,a.p2_motion)]
    for motion,path in zip(motions,(a.raw_detection,a.p2_weighted_detection)):
        if motion.get("source_detections_sha256")!=sha256_file(path):
            raise ValueError("Motion source detection SHA-256 mismatch")
        source=motion.get("solver_camera_layout") or {}
        if (source.get("creature")!=rig["creature"] or
                source.get("active_layout")!=rig["active_layout"]):
            raise ValueError("Motion camera layout provenance mismatch")
    cam={k:getattr(layout,k) for k in ("H","W_FRONT","W_SIDE","M_PER_PX","Z_CENTER")}
    report=compare(raw,weighted,*motions,cam)
    report["source_detection_sha256"]=sha256_file(a.raw_detection)
    report["p2_detection_sha256"]=sha256_file(a.p2_weighted_detection)
    sidecar=pathlib.Path(a.p2_weighted_detection).with_suffix(".p2_weights.json")
    if not sidecar.is_file():
        raise ValueError("P2 detection lacks provenance sidecar")
    meta=json.loads(sidecar.read_text(encoding="utf-8"))
    if (meta.get("source_detection_sha256")!=report["source_detection_sha256"] or
            meta.get("weighted_detection_sha256")!=report["p2_detection_sha256"]):
        raise ValueError("P2 weighting sidecar source/output checksum mismatch")
    report["quality_weighting"]=meta
    if a.out:
        output=pathlib.Path(a.out).resolve()
        forbidden={pathlib.Path(p).resolve() for p in (
            a.raw_detection,a.p2_weighted_detection,
            a.baseline_motion,a.p2_motion,a.rig_info)}
        if output.exists() or output in forbidden:
            raise FileExistsError("cannot overwrite previous P2/baseline data")
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print("Saved P2 A/B report:",output)
    print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
