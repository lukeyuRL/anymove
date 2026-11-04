"""Freeze a separate SO(3) Human V3 rig with TRAIN-only Spine neutral initializer.

A Kabsch Root initializer already improves global yaw. The per-frame
Analytic IK can assign inconsistent local torso twists. This variant uses
TRAIN 96 clips' *median* local Euler starting orientation (not Test/Val GT)
for the 9 spine/neck/clavicle bones, but allows all permissible rotations
to move freely during the differentiable Solver fit.

Existing research/source rigs are never modified. No trained neural model
or physical contact behavior implied by this initialization.
"""
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DATA=ROOT/"spine_motion_prior_v1"
PRIOR_ROOT=ROOT/"solver/spine_so3_root_geom_evidence_research"
RIG_SOURCE=PRIOR_ROOT/"rigged/rig_info.json"
OUT=ROOT/"solver/spine_so3_train_neutral_init_research"
SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"

def main():
    original=json.loads(RIG_SOURCE.read_text(encoding="utf8"))
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf8"))
    if (manifest.get("status")!="PASS" or manifest.get("source_blend_sha256")!=SHA or
        original.get("source_blend_sha256")!=SHA):
        raise RuntimeError("Train Spine source or rig provenance mismatch")
    profile=original["experimental_motion_prior"]
    if (profile.get("train_clips")!=96 or
        profile.get("root_anchor_initializer")!="KABSCH_HIP_WAIST_THIGHS" or
        profile.get("root_orientation_observation_weight")!=4000.0):
        raise RuntimeError("Expected frozen root/Spine SO3 profile")
    names=profile["spine_bone_names"]
    indices=profile["spine_bone_indices"]
    medians=[]
    rows=[r for r in manifest["rows"] if r["split"]=="train"]
    if len(rows)!=96:
        raise RuntimeError("Train clip count mismatch")
    for row in rows:
        if not row["npz"].startswith("clips/train/"):
            raise ValueError("Attempt to access non-Train GT")
        with np.load(DATA/row["npz"],allow_pickle=False) as f:
            e=np.asarray(f["bone_euler_xyz_rad"][:,indices,:],dtype=np.float64)
        if e.shape!=(107,len(names),3):
            raise RuntimeError("Wrong rotation prior shape")
        medians.append(e)
    median=np.median(np.concatenate(medians,axis=0),axis=0)
    if not np.isfinite(median).all() or median.shape!=(len(names),3):
        raise RuntimeError("Invalid Train median Euler")
    original["experimental_motion_prior"]["train_spine_neutral_init"]={
        "source_train_clips":96,
        "bone_names":names,
        "median_euler_xyz_rad":median.tolist(),
        "weight":1.0,
        "root_static_contract":"RL_BoneRoot local identity, global turn via ArmatureObject",
        "purpose":"Initialize unobservable local rotations near the full Train motion median before 2D observation optimization",
    }
    original["prior_scope"]="Train-derived SO3 prior, Root Kabsch + static RL_BoneRoot + median local Spine IK initializer; research only"
    out=OUT/"rigged/rig_info.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        raise FileExistsError("Refuse overwrite existing frozen median-Spine experiment")
    out.write_text(json.dumps(original,indent=2),encoding="utf8")
    creature=json.loads((PRIOR_ROOT/"creature.json").read_text(encoding="utf8"))
    creature["name"]="human_v3_nose_spine_so3_train_neutral_init_research"
    (OUT/"creature.json").write_text(json.dumps(creature,indent=2),encoding="utf8")
    print("SO3_NEUTRAL_INIT_TRAIN_PROFILE",json.dumps({
        "output":str(out),"source_blend_sha256":SHA,
        "train_clips":96,"heldout_test_not_loaded":True,
        "target_bones":names,"median_euler_xyz_deg":np.rad2deg(median).round(2).tolist(),
        "initialization_weight":1.0,"RL_BoneRoot_static":True
    }),flush=True)

if __name__=="__main__":
    main()
