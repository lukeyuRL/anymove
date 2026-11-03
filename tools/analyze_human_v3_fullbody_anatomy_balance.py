"""Disjoint five-region equal-weight fullbody camera A/B audit.

A 67-marker mean is heavily dominated by 40 finger markers. This report
calculates one independent mean for each anatomical region, then gives the
five regions equal weight as a second *noncanonical* score.
No test-driven retraining or checkpoint selection.
"""
import json,pathlib
import numpy as np

ROOT=pathlib.Path("runs/human/fullbody_finger_v3_nose/reader_models_strict")
FILES={name:ROOT/name/"unseen_test.json" for name in ("baseline_0_90","symmetric_315_45")}
records={p:json.loads(path.read_text(encoding="utf-8")) for p,path in FILES.items()}
names=list(records["baseline_0_90"]["per_joint"])
if set(records["symmetric_315_45"]["per_joint"])!=set(names) or len(names)!=67:
    raise RuntimeError("Expected same 67 body markers")
regions={
    "fingers_and_fingertips":[],
    "legs_feet_and_toes":[],
    "upper_arms_hands_clavicles":[],
    "torso_and_root":[],
    "head_and_neck":[],
}
for name in names:
    if any(s in name for s in ("Pinky","Ring","Mid","Index","Thumb")):
        group="fingers_and_fingertips"
    elif any(s in name for s in ("Thigh","Calf","Foot","ToeBase","ToeTip")):
        group="legs_feet_and_toes"
    elif any(s in name for s in ("Clavicle","Upperarm","Forearm","_Hand")):
        group="upper_arms_hands_clavicles"
    elif any(s in name for s in ("Head","NeckTwist")):
        group="head_and_neck"
    else:
        group="torso_and_root"
    regions[group].append(name)
expected=(40,10,8,6,3)
observed=tuple(map(len,regions.values()))
if observed!=expected or len(set(n for v in regions.values() for n in v))!=67:
    raise RuntimeError("Non-disjoint or missing body category markers: "+str(observed))
summary={"schema":"anymove.human_fullbody_equal_region_ab/1",
         "provenance":"Independent heldout Reader predictions; no test-driven training",
         "equal_anatomy_regions":True,"region_counts":{k:len(v) for k,v in regions.items()},
         "total_markers":len(names),"pairs":{}}
for pair,record in records.items():
    by={}
    for name,selected in regions.items():
        v=np.asarray([record["per_joint"][n]["3d_cm"]["mean"] for n in selected],dtype=float)
        p=np.asarray([record["per_joint"][n]["2d_px"]["mean"] for n in selected],dtype=float)
        by[name]={"3d_mpjpe_cm":float(v.mean()),"2d_mean_px":float(p.mean())}
    equal_score=float(np.mean([x["3d_mpjpe_cm"] for x in by.values()]))
    canonical=float(record["groups"]["full_67"]["3d_cm"]["mean"])
    from_perjoint=float(np.mean([record["per_joint"][n]["3d_cm"]["mean"] for n in names]))
    if abs(canonical-from_perjoint)>.0002:
        raise RuntimeError("Individual joints do not reproduce canonical MPJPE")
    summary["pairs"][pair]={"angles_deg":record["camera_angles_deg"],
                            "canonical_67_mpjpe_cm":canonical,
                            "equal_5_region_score_cm":equal_score,
                            "regions":by}
a=summary["pairs"]["baseline_0_90"]
b=summary["pairs"]["symmetric_315_45"]
summary["difference"]={
    "canonical_67_delta_cm":b["canonical_67_mpjpe_cm"]-a["canonical_67_mpjpe_cm"],
    "equal_5_region_delta_cm":b["equal_5_region_score_cm"]-a["equal_5_region_score_cm"],
}
dest=ROOT/"anatomy_balanced_ab.json"
dest.write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
