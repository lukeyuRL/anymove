"""Official validation QA for frozen 672px Reader + TRAIN-calibrated P1.

Uses precomputed Validation Reader UV cache. Geometry is tracked without
GT. Blender validation GT is used only after predictions for scoring.
"""
import json, pathlib
from collections import defaultdict
import numpy as np

from mocap.camera_geometry import triangulate_orthographic
from mocap.temporal_skeleton_tracker import track_two_view
from tools.eval_human_v3_hr_v2_reader import body_groups
from tools.train_human_v3_hr_v2_reader import DATA, OUT, SHA, load_manifest

def summary(xs):
    a=np.asarray(xs,dtype=float)
    return {"count":int(a.size),"mean":round(float(a.mean()),5),
            "p95":round(float(np.percentile(a,95)),5),
            "p99":round(float(np.percentile(a,99)),5),
            "max":round(float(a.max()),5),
            "gt5cm":int((a>5).sum()),
            "gt10cm":int((a>10).sum()),
            "gt50cm":int((a>50).sum())}

def score():
    manifest=load_manifest()
    cache=np.load(OUT/"full"/"validation_raw_uv_frozen.npz")
    cal=json.loads((OUT/"full"/"temporal_skeleton_train_calibration.json").read_text(encoding="utf-8"))
    names=manifest["joint_names"]
    if cache["names"].tolist()!=names or cache["source_sha"].tolist()!=[SHA] or cal["names"]!=names:
        raise RuntimeError("Validation source/Reader/calibration mismatch")
    clips=cache["clips"].tolist()
    if set(clips)!=set(manifest["validation_clips"]) or len(clips)!=24:
        raise RuntimeError("Validation split contract mismatch")
    groups=body_groups(names)
    errors=defaultdict(list)
    frame_counts=defaultdict(int)
    per_clip={}
    for i,clip in enumerate(clips):
        frames=cache["frames"][i].astype(int)
        f=cache["front_uv"][i]
        s=cache["side_uv"][i]
        result=track_two_view(f,s,frames,cal,names)
        src=json.loads((DATA/"motion_gt"/(clip+".json")).read_text(encoding="utf-8"))
        if src["clip"]["split"]!="validation":raise RuntimeError("Non Validation GT")
        true=np.asarray(src["joint_position_m"],dtype=np.float64)[frames]
        ones=np.ones((len(frames),67))
        raw=triangulate_orthographic(f,s,ones,ones,
            front_width=672,side_width=672,height=672,
            metres_per_pixel=cal["metres_per_pixel"],z_center=cal["z_center"],
            front_angle_deg=315,side_angle_deg=45,front_height_weight=1.)
        tracked=result["joints_world_m"]
        original_err=np.linalg.norm(raw-true,axis=-1)*100
        tracked_err=np.linalg.norm(tracked-true,axis=-1)*100
        for label,perframe in (("raw",original_err),("tracked",tracked_err)):
            for group,ids in groups.items():
                errors[label+"_"+group].extend(perframe[:,ids].ravel())
            delta=raw if label=="raw" else tracked
            tempor=np.linalg.norm(np.diff(delta,axis=0)-np.diff(true,axis=0),axis=-1)*100
            errors[label+"_temporal"].extend(tempor.ravel())
        per_clip[clip]={
            "raw_3d_mean_cm":float(original_err.mean()),
            "tracked_3d_mean_cm":float(tracked_err.mean()),
            "inferred_front":result["diagnostics"]["rejected_front"],
            "inferred_side":result["diagnostics"]["rejected_side"],
            "skeleton_untrusted":result["diagnostics"]["skeleton_untrusted"],
        }
        for key,value in result["diagnostics"].items():
            if isinstance(value,int):frame_counts[key]+=value
    out={
       "schema":"anymove.human_p1_conservative_temporal_validation/1",
       "status":"VALIDATION_TUNING_NOT_UNSEEN_TEST",
       "source_sha":SHA,
       "calibration_source":"TRAIN GT only",
       "reader_predictions":"Validation-only frozen Reader UV",
       "clips":24,"frames":336,"points":67,
       "diagnostic_counts":dict(frame_counts),
       "raw":{g:summary(errors["raw_"+g]) for g in groups},
       "tracked":{g:summary(errors["tracked_"+g]) for g in groups},
       "raw_temporal_cm":summary(errors["raw_temporal"]),
       "tracked_temporal_cm":summary(errors["tracked_temporal"]),
       "per_clip":per_clip,
       "caution":"Validation used during development. Fresh never-viewed clips are required for acceptance."
    }
    path=OUT/"full"/"temporal_conservative_validation.json"
    path.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("CONSERVATIVE_P1_VALIDATION",json.dumps({
        "raw_3d":out["raw"]["all_67"],
        "tracked_3d":out["tracked"]["all_67"],
        "raw_temporal":out["raw_temporal_cm"],
        "tracked_temporal":out["tracked_temporal_cm"],
        "diagnostics":out["diagnostic_counts"],
        "saved":str(path)
    }),flush=True)
if __name__=="__main__":
    score()
