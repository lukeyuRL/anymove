"""Create a provably GT-LEAKING Top observation for synthetic upper-bound QA.

Only for same keyed Blender GT animation and independently rendered Top.
NEVER use GT-based Top observations in production, AI-video, or model claims.

python -m tools.make_dino_top_oracle RAW.npz RIG.json GT_JOINTS.json \
   TOP_RENDER_DIR NEW_TOP_ORACLE.npz --scope tail_only|all_joints

Camera exported by real Blender cam_top.json is validated as orthographic
and independently cross-checked against fixed Dino X/Y geometry. The
original Front/Side UV and peaks are preserved byte-for-byte as arrays.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
from mocap import layout
from mocap.dino_top_camera import camera_from_json,project_top_numpy,TOP_HEIGHT,TOP_WIDTH
from tools.evaluate_dino_full_body import classify_joint

PROFILE="anymove.dino_top_truth_leakage_oracle/1"


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda: f.read(1<<20),b""):
            h.update(x)
    return h.hexdigest()


def make(raw,rig,truth,camera,scope):
    if scope not in ("tail_only","all_joints"):
        raise ValueError("Top GT leakage scope must be tail_only or all_joints")
    names=[x["name"] for x in rig["keypoints"]]
    if (len(names)!=92 or truth["names"]!=names or
            [str(x) for x in raw["names"]]!=names):
        raise ValueError("Dino rig, Blender GT, original Reader names/order mismatch")
    T=len(truth["joints"])
    if T<3 or abs(float(raw["fps"])-float(truth["fps"]))>1e-5:
        raise ValueError("Dino TOP GT frame-count/FPS mismatch")
    xyz=np.asarray(truth["joints"],np.float64)
    if xyz.shape!=(T,92,3) or not np.isfinite(xyz).all():
        raise ValueError("Blender GT frame/joint dimensions invalid")
    for v in ("front","side"):
        if (np.asarray(raw[v+"_uv"]).shape!=(T,92,2)
                or np.asarray(raw[v+"_conf"]).shape!=(T,92)):
            raise ValueError("original two-view Reader detections do not match GT")
    camera_uv=project_top_numpy(xyz,camera)
    simple=np.stack([
        TOP_WIDTH/2+xyz[...,0]/layout.M_PER_PX,
        TOP_HEIGHT/2-xyz[...,1]/layout.M_PER_PX],axis=-1)
    err=np.linalg.norm(camera_uv-simple,axis=-1)
    if err.max()>.001:
        raise ValueError("Blender Top camera coordinates inconsistent with independent layout geometry")
    selected=np.array([
        (classify_joint(n)=="tail" if scope=="tail_only" else True)
        for n in names],bool)
    if not selected.any():
        raise ValueError("Top Oracle selection contains no Dino joints")
    uv=np.zeros((T,92,2),np.float32)
    uv[:,selected,:]=camera_uv[:,selected,:].astype(np.float32)
    confidence=np.zeros((T,92),np.float32)
    confidence[:,selected]=1.0
    out={k:np.array(val,copy=True) for k,val in raw.items()}
    if any(k.startswith("top_") for k in out):
        raise ValueError("original Reader source already contains Top data")
    out["top_uv"]=uv
    out["top_conf"]=confidence
    audit={
        "profile":PROFILE,"scope":scope,"frames":T,
        "keypoints":92,"selected_joint_count":int(selected.sum()),
        "top_valid_joint_frames":int(confidence.sum()),
        "blender_top_vs_geometry_max_error_px":float(err.max()),
        "leakage_warning":("Uses ACTUAL known 3D Blender truth to populate top_uv; "
           "Oracle only. Completely invalid for assessing a trainable Top Reader."),
        "raw_reader_coordinates_unchanged":True,
        "top_uv_for_unselected_joints":"set zero and top_conf=0",
    }
    return out,audit


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("original_reader")
    p.add_argument("rig_info")
    p.add_argument("blender_truth")
    p.add_argument("top_render_dir")
    p.add_argument("output_detection")
    p.add_argument("--scope",required=True,choices=("tail_only","all_joints"))
    a=p.parse_args(argv)
    rawpath=pathlib.Path(a.original_reader)
    rigpath=pathlib.Path(a.rig_info)
    truthpath=pathlib.Path(a.blender_truth)
    topdir=pathlib.Path(a.top_render_dir)
    cam_path=topdir/"cam_top.json"
    render_audit=topdir/"top_render_audit.json"
    sources=[rawpath,rigpath,truthpath,cam_path,render_audit]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    outpath=pathlib.Path(a.output_detection).resolve()
    sidecar=outpath.with_suffix(".top_oracle.json")
    if outpath.exists() or sidecar.exists() or outpath in [p.resolve() for p in sources]:
        raise FileExistsError("refuse overwriting original Dino GT or detection source")
    render=json.loads(render_audit.read_text(encoding="utf-8"))
    if (render.get("profile")!="anymove.dino_synced_top_blender_render_v1"
            or render["rig_info_sha256"]!=digest(rigpath)
            or render["blender_truth_sha256"]!=digest(truthpath)):
        raise ValueError("Blender Top render source SHA mismatch")
    camera=camera_from_json(cam_path)
    gt=json.loads(truthpath.read_text(encoding="utf-8"))
    rig=json.loads(rigpath.read_text(encoding="utf-8"))
    with np.load(rawpath,allow_pickle=False) as d:
        raw={k:d[k].copy() for k in d.files}
    if (render["frames"]!=len(gt["joints"]) or
            render["keypoints"]!=len(gt["names"]) or
            abs(float(render["fps"])-float(gt["fps"]))>1e-5):
        raise ValueError("Top rendered frame truth contract mismatch")
    # If any Top image is absent, do not pass off geometry-only GT as an
    # evaluated synchronized camera experiment.
    for t in range(render["frames"]):
        if not (topdir/"top"/f"f_{t:04d}.png").is_file():
            raise FileNotFoundError("Top render missing frame "+str(t))
    result,audit=make(raw,rig,gt,camera,a.scope)
    audit["input_sha256"]={str(p):digest(p) for p in sources}
    outpath.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outpath,**result)
    audit["top_detection_sha256"]=digest(outpath)
    sidecar.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
