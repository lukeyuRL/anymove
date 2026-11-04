"""One-time opt-in Human V3 SO(3) prior + Root Kabsch research rig.

Preserves the baseline rig and the SO3-only experiment unchanged, enabling
controlled A/B comparison of (a) original, (b) SO3 prior alone and (c)
Root-axis anatomical initialization + SO3 prior.
"""
import json,pathlib
ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/solver")
SOURCE=ROOT/"spine_so3_research/rigged/rig_info.json"
TARGET=ROOT/"spine_so3_root_research"

def run():
    source=json.loads(SOURCE.read_text(encoding="utf8"))
    if (source.get("limits_status")!="PROVISIONAL_TRAIN_SO3_MOTION_PRIOR_RESEARCH_ONLY"
        or source["experimental_motion_prior"]["schema"]!="anymove.human_v3_so3_prior/1"
        or source["experimental_motion_prior"]["train_clips"]!=96):
        raise RuntimeError("SO3 profile not trained from strict independent Train set")
    profile=source["experimental_motion_prior"]
    if "root_anchor_initializer" in profile:
        raise RuntimeError("Source SO3-only profile already modified")
    profile["root_anchor_initializer"]="KABSCH_HIP_WAIST_THIGHS"
    profile["root_anchor_joint_names"]=[
        "CC_Base_Hip","CC_Base_Waist","CC_Base_L_Thigh","CC_Base_R_Thigh"]
    profile["root_anchor_gt_used_at_inference"]=False
    source["prior_scope"]="PROVISIONAL Human V3 Nose SO(3) Train prior with four-joint observed Root initializer, not general human"
    filename=TARGET/"rigged/rig_info.json"
    filename.parent.mkdir(parents=True,exist_ok=True)
    if filename.exists():
        raise FileExistsError("Refuse to overwrite frozen Root-anchored research profile")
    filename.write_text(json.dumps(source,indent=2),encoding="utf8")
    creature=json.loads((ROOT/"spine_so3_research/creature.json").read_text())
    creature["name"]="human_v3_nose_spine_so3_root_research"
    (TARGET/"creature.json").write_text(json.dumps(creature,indent=2),encoding="utf8")
    report={"status":"NEW_RIG_CREATED",
        "source_prior":"96 train clips, 24 Validation, 24 sealed Test",
        "root_initializer":profile["root_anchor_initializer"],
        "camera_angles":[315,45],
        "new_rig":str(filename),
        "underlying_so3_motion_prior_unchanged":True,
        "root_gt_not_used_at_inference":True}
    (TARGET/"profile_report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("SO3_ROOT_INIT_RIG_READY",json.dumps(report),flush=True)
if __name__=="__main__":
    run()
