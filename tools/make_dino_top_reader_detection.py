"""Bridge actual rendered Dino Top Reader predictions into an experimental 3-view Solver.

This script NEVER opens Blender 3D GT or uses true Top keypoint coordinates.
It combines original TWO-VIEW Reader detections with saved actual Top
image-inferred UV and bounded raw heatmap confidence peaks, restricting
Top constraints to the 9 Dino Tail joints. This remains non-production.

python -m tools.make_dino_top_reader_detection \
  BASE/det_baseline.npz TOP_READER_DIR/top_reader_detections.npz \
  assets/dino/rigged/rig_info.json TOP_RENDER_DIR NEW_DET.npz
"""
import argparse
import hashlib
import json
import pathlib
import numpy as np

from mocap.dino_top_camera import camera_from_json,TOP_HEIGHT,TOP_WIDTH
from tools.evaluate_dino_full_body import classify_joint

PROFILE="anymove.dino_top_reader_joint_observations_v1"


def sha256(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):
            h.update(c)
    return h.hexdigest()


def build(raw,top,rig):
    names=[k["name"] for k in rig["keypoints"]]
    if (len(names)!=92 or
            [str(x) for x in raw["names"]]!=names or
            [str(x) for x in top["names"]]!=names):
        raise ValueError("2-view/Top checkpoint keypoint names/order mismatch")
    T,J=np.asarray(raw["front_conf"]).shape
    if (T<3 or J!=92 or
            len(top["top_uv"])!=T or
            abs(float(raw["fps"])-float(top["fps"]))>1e-5):
        raise ValueError("Top Reader frame/FPS mismatch")
    if any(key.startswith("top_") for key in raw):
        raise ValueError("raw source already contains Top data")
    original={k:np.array(v,copy=True) for k,v in raw.items()}
    for v in ("front","side"):
        if (np.asarray(raw[v+"_uv"]).shape!=(T,J,2) or
                not np.isfinite(raw[v+"_uv"]).all() or
                not np.isfinite(raw[v+"_conf"]).all()):
            raise ValueError("invalid original source view")
    uv=np.asarray(top["top_uv"],np.float32)
    peaks=np.asarray(top["top_conf"],np.float32)
    if (uv.shape!=(T,J,2) or peaks.shape!=(T,J)
            or not np.isfinite(uv).all() or not np.isfinite(peaks).all()):
        raise ValueError("invalid original Top Reader prediction")
    tail=np.array([classify_joint(n)=="tail" for n in names],dtype=bool)
    if int(tail.sum())!=9:
        raise ValueError("expected 9 Dino tail keypoints")
    # Exclude outside-frame detections; must retain missing metadata.
    inbounds=(uv[...,0]>=0)&(uv[...,0]<TOP_WIDTH)&(
             uv[...,1]>=0)&(uv[...,1]<TOP_HEIGHT)
    out=original
    out["top_uv"]=np.zeros((T,J,2),np.float32)
    out["top_uv"][:,tail,:]=uv[:,tail,:]
    conf=np.zeros((T,J),np.float32)
    conf[:,tail]=np.clip(peaks[:,tail],0,1)
    conf[~inbounds]=0
    out["top_conf"]=conf
    audit={
        "profile":PROFILE,
        "source":"actual rendered Top images -> frozen JointReader heatmaps; NOT Blender 3D truth",
        "joint_scope":"tail_only",
        "tail_joint_count":int(tail.sum()),
        "frames":int(T),
        "active_top_joint_frames":int(np.count_nonzero(conf)),
        "out_of_frame_tail_joint_frames":int(np.count_nonzero(~inbounds[:,tail])),
        "mean_active_top_reader_peak":float(conf[conf>0].mean()) if np.any(conf>0) else None,
        "model_uncalibrated":"heatmap peak bounded [0,1] is NOT an anatomical visibility probability",
        "raw_front_and_side_unchanged":True,
        "experimental_not_production":True,
    }
    return out,audit


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw_reader")
    p.add_argument("top_reader_npz")
    p.add_argument("rig_info")
    p.add_argument("top_render_dir")
    p.add_argument("output_detection")
    a=p.parse_args(argv)
    source=[pathlib.Path(a.raw_reader),pathlib.Path(a.top_reader_npz),
            pathlib.Path(a.rig_info)]
    render=pathlib.Path(a.top_render_dir)
    camera=render/"cam_top.json"
    render_audit=render/"top_render_audit.json"
    evidence=pathlib.Path(a.top_reader_npz).parent/"top_reader_2d_gt_report.json"
    source.extend([camera,render_audit,evidence])
    out=pathlib.Path(a.output_detection).resolve()
    sidecar=out.with_suffix(".top_reader.json")
    if out.exists() or sidecar.exists() or out in {p.resolve() for p in source}:
        raise FileExistsError("refuse overwrite of original Dino observations")
    for path in source:
        if not path.is_file():
            raise FileNotFoundError(path)
    camera_from_json(camera)
    report=json.loads(evidence.read_text(encoding="utf-8"))
    render_meta=json.loads(render_audit.read_text(encoding="utf-8"))
    if (report.get("profile")!="anymove.dino_real_rendered_top_reader_2d_audit_v1" or
            render_meta.get("profile")!="anymove.dino_synced_top_blender_render_v1" or
            report["provenance_sha256"]["cam_top"]!=sha256(camera) or
            report["provenance_sha256"]["rig"]!=sha256(a.rig_info) or
            render_meta["rig_info_sha256"]!=sha256(a.rig_info) or
            report["provenance_sha256"]["gt"]!=render_meta["blender_truth_sha256"] or
            report["frames"]!=render_meta["frames"]):
        raise ValueError("actual Top Reader/synchronized Blender render provenance mismatch")
    # Older research snapshots lack this additional NPZ digest; accept them
    # only inside this explicitly experimental path, not as production proof.
    prediction_digest=report.get("top_reader_detections_sha256")
    if prediction_digest is not None and prediction_digest!=sha256(a.top_reader_npz):
        raise ValueError("Top Reader inference NPZ differs from independent audit SHA")
    rig=json.loads(pathlib.Path(a.rig_info).read_text(encoding="utf-8"))
    with np.load(a.raw_reader,allow_pickle=False) as z:
        raw={k:z[k].copy() for k in z.files}
    with np.load(a.top_reader_npz,allow_pickle=False) as z:
        top={k:z[k].copy() for k in z.files}
    obs,audit=build(raw,top,rig)
    audit["input_sha256"]={str(p):sha256(p) for p in source}
    audit["top_camera_sha256"]=sha256(camera)
    audit["reader_checkpoint_sha256"]=report["provenance_sha256"]["checkpoint"]
    audit["source_model_label"]=report["source_model_label"]
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**obs)
    audit["top_detection_sha256"]=sha256(out)
    sidecar.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
