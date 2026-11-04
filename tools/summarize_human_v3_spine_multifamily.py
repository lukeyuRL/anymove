"""Release read-only report of 4-family Human V3 Spine SO(3) Validation.

Applies development gates written BEFORE results in
docs/HUMAN_V3_SPINE_MULTIFAMILY_VALIDATION_GATE.md.
Does not tune the Solver, retrain Reader or read any TEST clip.
"""
import json
import pathlib
from collections import defaultdict
import numpy as np

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DIR=ROOT/"solver/spine_so3_research/multifamily_validation"
SOURCE=DIR/"multifamily_1takes_400iter_validation.json"
DEST=DIR/"multifamily_4families_decision_report.json"
FAMILIES=("walk_turn_right","waist_twist_right",
          "lateral_bend_left","reach_counterrotate")
MODES=("baseline","root_so3","neutral_so3")

def main():
    data=json.loads(SOURCE.read_text(encoding="utf8"))
    if (data["status"]!="VALIDATION_COMPLETE_TEST_24_CLIPS_UNTOUCHED" or
        data["iterations_per_mode"]!=400 or
        data["model_modes"]!=list(MODES) or
        data["clips"]!=4 or
        set(data["families"])!=set(FAMILIES) or
        data["test_clips_loaded"]!=0 or
        not data["all_samples_bonerooot_static"]):
        raise RuntimeError("Validation result incomplete or not predeclared")
    rows=data["scores_per_clip"]
    if len(rows)!=4 or {row["family"] for row in rows.values()}!=set(FAMILIES):
        raise RuntimeError("Validation missing one of four independent action families")
    agg={}
    for mode in MODES:
        group=[r["metrics"][mode] for r in rows.values()]
        def avg(field):
            return float(np.mean([float(v) for v in field]))
        def choose(fn):
            return avg([fn(m) for m in group])
        agg[mode]={
          "body_mpjpe_cm":choose(lambda m:m["body_mpjpe_cm"]["mean"]),
          "body_p95_cm":choose(lambda m:m["body_mpjpe_cm"]["p95"]),
          "root_orientation_error_deg":choose(lambda m:m["root_orientation_error_deg"]["mean"]),
          "waist_angular_acc_deg_per_frame2":choose(lambda m:m["spine_bones"]["CC_Base_Waist"]["angular_acc_deg_frame2"]["mean"]),
          "pelvis_angular_acc_deg_per_frame2":choose(lambda m:m["spine_bones"]["CC_Base_Pelvis"]["angular_acc_deg_frame2"]["mean"]),
          "spine_angular_acc_deg_per_frame2":choose(lambda m:m["spine_acc_mean_deg_frame2"]),
          "spine_gt_acc_deg_per_frame2":choose(lambda m:np.mean([m["spine_bones"][bone]["gt_angular_acc_deg_frame2"]["mean"] for bone in m["spine_bones"]])),
          "ground_proxy_foot_slide_cm_per_frame":choose(lambda m:m["contact_proxy_foot_horizontal_speed_cm_per_frame"]["mean"] if m["contact_proxy_foot_horizontal_speed_cm_per_frame"] is not None else float("nan")),
          "max_perclip_mpjpe_regression_cm":max(
            float(r["metrics"][mode]["body_mpjpe_cm"]["mean"])
            -float(r["metrics"]["baseline"]["body_mpjpe_cm"]["mean"])
            for r in rows.values()),
        }
    baseline=agg["baseline"]
    status={}
    for mode in MODES[1:]:
        v=agg[mode]
        gates={
          "waist_acc_30pct_below_baseline":v["waist_angular_acc_deg_per_frame2"]<=.70*baseline["waist_angular_acc_deg_per_frame2"],
          "pelvis_acc_30pct_below_baseline":v["pelvis_angular_acc_deg_per_frame2"]<=.70*baseline["pelvis_angular_acc_deg_per_frame2"],
          "mean_mpjpe_regression_le_0p10_cm":v["body_mpjpe_cm"]-baseline["body_mpjpe_cm"]<=.10,
          "each_family_mpjpe_regression_le_0p20_cm":v["max_perclip_mpjpe_regression_cm"]<=.20,
          "mean_root_error_below_50pct_baseline":v["root_orientation_error_deg"]<=.5*baseline["root_orientation_error_deg"],
          "root_static_and_test_sealed":True,
        }
        status[mode]={
            "gates":gates,
            "eligible_for_more_validation":all(gates.values()),
            "mpjpe_regression_cm":v["body_mpjpe_cm"]-baseline["body_mpjpe_cm"],
            "waist_acc_reduction_fraction":1-v["waist_angular_acc_deg_per_frame2"]/max(baseline["waist_angular_acc_deg_per_frame2"],1e-9),
            "pelvis_acc_reduction_fraction":1-v["pelvis_angular_acc_deg_per_frame2"]/max(baseline["pelvis_angular_acc_deg_per_frame2"],1e-9),
            "root_error_reduction_fraction":1-v["root_orientation_error_deg"]/max(baseline["root_orientation_error_deg"],1e-9),
        }
    overall="VALIDATION_RESEARCH_ELIGIBLE" if any(v["eligible_for_more_validation"] for v in status.values()) else "NO_NEW_SOLVER_PASS__KEEP_BASELINE"
    report={
       "schema":"anymove.human_v3_spine_multifamily_400iter_decision/1",
       "status":overall,
       "predeclared_gates_document":"docs/HUMAN_V3_SPINE_MULTIFAMILY_VALIDATION_GATE.md",
       "test_clips_loaded":0,
       "families":list(FAMILIES),"validation_clips":len(rows),
       "frames_per_clip":107,"iterations":400,
       "mean_metrics":agg,
       "candidate_gates":status,
       "per_family":{
            row["family"]:{name:{
               "body_mpjpe_cm":row["metrics"][name]["body_mpjpe_cm"]["mean"],
               "root_orientation_error_deg":row["metrics"][name]["root_orientation_error_deg"]["mean"],
               "waist_acc_deg_frame2":row["metrics"][name]["spine_bones"]["CC_Base_Waist"]["angular_acc_deg_frame2"]["mean"],
               "pelvis_acc_deg_frame2":row["metrics"][name]["spine_bones"]["CC_Base_Pelvis"]["angular_acc_deg_frame2"]["mean"],
            } for name in MODES}
            for row in rows.values()
       },
       "cautions":[
          "Result is procedural synthetic Validation only, not an unseen Test result.",
          "Scores are from noisy rendered-GT projections (no Reader or video generator).",
          "Train-derived SO3 prior and median local pose are experimental research only.",
          "Bone Root RL_BoneRoot has zero local animation; Root turn belongs to Armature Object.",
          "No motion should be imported directly to iClone: Blender Euler is not FK-channel contract.",
       ]
    }
    DEST.write_text(json.dumps(report,indent=2),encoding="utf8")
    print("SPINE_MULTIFAMILY_DECISION",json.dumps({
        "status":overall,
        "metrics":agg,
        "candidate_gates":status,
        "report":str(DEST)
    }),flush=True)

if __name__=="__main__":
    main()
