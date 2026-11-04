"""Compare full 14-frame Human V3 HR V2 Reader-to-FK motion A/B.

Only after Solver outputs exist; Blender procedural GT is used exclusively to
score completed motion. Does not tune the Reader, P1 guard, optimizer or rig.
This clip was part of earlier fresh P1 testing; results are integration QA.
"""
import json,pathlib
from collections import defaultdict

import numpy as np

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
BASE=ROOT/"solver/reader_p1_fresh"
CLIP="hrv2_p1_fresh_finger_independent_roll_s208392"
GT_ROOT=ROOT/"p1_fresh_test/motion_gt"

def describe(values):
    a=np.asarray(values,dtype=np.float64)
    return {
      "observations":int(a.size),
      "mean_cm":round(float(np.mean(a)),5),
      "p95_cm":round(float(np.percentile(a,95)),5),
      "max_cm":round(float(np.max(a)),5),
      "over_5cm":int((a>5).sum()),
      "over_10cm":int((a>10).sum()),
      "over_50cm":int((a>50).sum()),
    }

def run():
    with np.load(BASE/(CLIP+".npz")) as det:
        frames=det["frame_indices"].astype(int).tolist()
        names=det["names"].tolist()
        f=det["front_uv"]
        s=det["side_uv"]
    record=json.loads((GT_ROOT/(CLIP+".json")).read_text(encoding="utf-8"))
    if record["joint_names"]!=names or len(frames)!=14:
        raise RuntimeError("GT marker/frame identities differ from detections")
    gt=np.asarray(record["joint_position_m"],dtype=np.float64)[frames]
    categories={
        "all_67":list(range(67)),
        "fingers_and_fingertips":[i for i,n in enumerate(names) if any(q in n for q in ("Pinky","Ring","Mid","Index","Thumb"))],
        "fingertips_10":[i for i,n in enumerate(names) if n.endswith("Tip") and "Toe" not in n],
        "toe_tips_2":[i for i,n in enumerate(names) if n.endswith("ToeTip")],
        "legs_feet_toes":[i for i,n in enumerate(names) if any(q in n for q in ("Thigh","Calf","Foot","ToeBase","ToeTip"))],
        "arms_hands_clavicles":[i for i,n in enumerate(names) if any(q in n for q in ("Clavicle","Upperarm","Forearm","_Hand"))],
        "torso_root":[i for i,n in enumerate(names) if not any(q in n for q in ("Thigh","Calf","Foot","ToeBase","ToeTip","Pinky","Ring","Mid","Index","Thumb","Clavicle","Upperarm","Forearm","_Hand","Head","NeckTwist"))],
        "head_neck":[i for i,n in enumerate(names) if any(q in n for q in ("Head","NeckTwist"))],
    }
    if list(map(len,[categories[k] for k in ("fingers_and_fingertips","legs_feet_toes","arms_hands_clavicles","torso_root","head_neck")]))!=[40,10,8,6,3]:
        raise RuntimeError("Human anatomical group partition mismatch")
    outputs={}
    for tag,filename in (("raw","motion_reader_raw.json"),("p1_guarded","motion_reader_p1_guarded.json")):
        m=json.loads((BASE/filename).read_text(encoding="utf-8"))
        if (m["joint_names"]!=names or m["solver_camera_azimuth_deg"]!=[315,45]
            or m.get("source_frame_indices")!=frames or len(m["bones"])!=55):
            raise RuntimeError("Motion JSON missing correct provenance/timing")
        xyz=np.asarray(m["joints"],dtype=np.float64)
        if xyz.shape!=gt.shape or not np.isfinite(xyz).all():
            raise RuntimeError("Invalid 3D motion joints")
        if not np.isfinite(np.asarray(m["root_quat"])).all():
            raise RuntimeError("Non-finite root rotation in motion")
        err_cm=np.linalg.norm(xyz-gt,axis=-1)*100
        velocity=np.linalg.norm(np.diff(xyz,axis=0)-np.diff(gt,axis=0),axis=-1)*100
        by={name:describe(err_cm[:,idx].ravel()) for name,idx in categories.items()}
        whole={
            "3d_cm":by,
            "temporal_delta_error_cm":describe(velocity),
            "reprojection_px":m["reproj_px"],
            "guard_diagnostics":(m.get("temporal_joint_tracking") or {}).get("tracker_diagnostics"),
        }
        outputs[tag]=whole
    result={
        "schema":"anymove.human_p1_reader_to_fk_cli_ab/1",
        "status":"DIAGNOSTIC_FK_INTEGRATION_NOT_ICLONE_ACCEPTANCE",
        "source_blend_sha256":record["source_blend_sha256"],
        "clip":CLIP,
        "source_frame_indices":frames,
        "frames":len(frames),
        "rig_bones":55,"3d_markers":67,
        "inputs":"Frozen native Reader 672/192, P1 new-seed Blender images",
        "gt_use":"Evaluation only, no GT inputs to Reader or Solver",
        "scores":outputs,
        "warning":"One synthetic clip and provisional human Euler joint limits; not validated in iClone UI.",
    }
    out=BASE/"reader_to_fk_qa.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf8")
    print("P1_READER_TO_FK_AB",json.dumps({
        k:{"3d_cm":v["3d_cm"]["all_67"],
           "fingertip_cm":v["3d_cm"]["fingertips_10"],
           "toe_cm":v["3d_cm"]["toe_tips_2"],
           "temporal_cm":v["temporal_delta_error_cm"],
           "reproj":v["reprojection_px"],
           "guard":v["guard_diagnostics"]}
        for k,v in outputs.items()}),flush=True)
if __name__=="__main__":
    run()
