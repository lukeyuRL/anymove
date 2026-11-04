"""Two explicitly opt-in Root evidence / SO(3) angular-acceleration ablations.

The base Train profile and both previous experimental rigs remain untouched.
Only Validation may be used to choose between these variants. Test untouched.
"""
import json,pathlib

BASE=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/solver")
SOURCE=BASE/"spine_so3_root_research/rigged/rig_info.json"
SETTINGS=(
 ("root_geom_evidence",800.0,4000.0),
 ("root_geom_evidence_strong",3200.0,4000.0),
)
def run():
    original=json.loads(SOURCE.read_text(encoding="utf8"))
    profile=original["experimental_motion_prior"]
    if (profile["root_anchor_initializer"]!="KABSCH_HIP_WAIST_THIGHS" or
        profile["schema"]!="anymove.human_v3_so3_prior/1" or
        profile["train_clips"]!=96):
        raise RuntimeError("Only frozen Human Train SO3 prior is permitted")
    for name,bone_acc,rootweight in SETTINGS:
        x=json.loads(json.dumps(original))
        x["experimental_motion_prior"]["weights"]["bone_acceleration"]=bone_acc
        x["experimental_motion_prior"]["root_orientation_observation_weight"]=rootweight
        x["prior_scope"]="Root Kabsch 3D-observed orientation plus SO3 rotational constraints, research only"
        folder=BASE/("spine_so3_"+name+"_research")
        rig=folder/"rigged/rig_info.json"
        rig.parent.mkdir(parents=True,exist_ok=True)
        if rig.exists():
            raise FileExistsError("Refusing to replace frozen study "+str(rig))
        rig.write_text(json.dumps(x,indent=2),encoding="utf8")
        creature=json.loads((BASE/"spine_so3_root_research/creature.json").read_text(encoding="utf8"))
        creature["name"]="human_v3_spine_so3_"+name+"_research"
        (folder/"creature.json").write_text(json.dumps(creature,indent=2),encoding="utf8")
        print("SO3_ROOT_EVIDENCE_RESEARCH_VARIANT",name,
              json.dumps({"bone_acc_weight":bone_acc,"root_observation_weight":rootweight,
                          "path":str(rig)}),flush=True)
if __name__=="__main__":
    run()
