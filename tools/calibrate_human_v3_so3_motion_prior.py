"""Calibrate research SO(3) Root/Spine Motion Prior from TRAIN GT ONLY.

Input is the audited 144-clip Human V3 Nose Spine Motion Dataset:
Train 96 clips x 107 frames, Validation 24 for read-only coverage checks,
Test 24 completely excluded from calibration and threshold/weight selection.

Outputs a separate opt-in 315/45 55-bone Human rig. No source/model/Solver
defaults are overwritten. SO(3) angular acceleration calibrated in source
frames; Root yaw freedom is preserved with no zero-heading prior.
"""
import hashlib,json,math,pathlib
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
DATA=ROOT/"spine_motion_prior_v1"
RIG=ROOT/"solver/rigged/rig_info_trainlimits.json"
DEST=ROOT/"solver/spine_so3_research"
SRC_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
SPINE=("CC_Base_Pelvis","CC_Base_Waist","CC_Base_Spine01","CC_Base_Spine02",
       "CC_Base_NeckTwist01","CC_Base_NeckTwist02","CC_Base_Head",
       "CC_Base_L_Clavicle","CC_Base_R_Clavicle")
WEIGHTS={"bone_velocity":120.0,"bone_acceleration":800.0,
         "root_velocity":60.0,"root_acceleration":1000.0,
         "bone_axis_range":150.0}

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):
            h.update(buf)
    return h.hexdigest()

def read_splits():
    audit=json.loads((DATA/"audit.json").read_text(encoding="utf8"))
    manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf8"))
    if (audit["status"]!="PASS" or manifest["status"]!="PASS"
        or manifest["source_blend_sha256"]!=SRC_SHA or
        manifest["train_val_test_clips"]!={"train":96,"validation":24,"test":24}):
        raise RuntimeError("Motion GT release has not passed 144-clip independent audit")
    rig=json.loads(RIG.read_text(encoding="utf8"))
    if rig.get("source_blend_sha256")!=SRC_SHA or len(rig["bones"])!=55:
        raise RuntimeError("Expected verified V3 Nose skeleton")
    bones=[b["name"] for b in rig["bones"]]
    if bones!=manifest["source_rig_bones"]:
        raise RuntimeError("Train prior bones disagree with released skeleton")
    return manifest,rig,bones

def angular_norms(xyz_eul,root_wxyz,indices):
    rotations=Rotation.from_euler("xyz",np.asarray(xyz_eul[:,indices,:],dtype=np.float64).reshape(-1,3))
    mats=rotations.as_matrix().reshape(len(xyz_eul),len(indices),3,3)
    # SciPy Rotation supports broadcasting with flattened selected bones.
    rel=np.transpose(mats[:-1],(0,1,3,2))@mats[1:]
    local_v=Rotation.from_matrix(rel.reshape(-1,3,3)).as_rotvec().reshape(len(xyz_eul)-1,len(indices),3)
    local_a=np.diff(local_v,axis=0)
    root=np.asarray(root_wxyz,dtype=float)
    rootrot=Rotation.from_quat(root[:,[1,2,3,0]])
    root_v=(rootrot[:-1].inv()*rootrot[1:]).as_rotvec()
    root_a=np.diff(root_v,axis=0)
    return (np.linalg.norm(local_v,axis=-1),
            np.linalg.norm(local_a,axis=-1),
            np.linalg.norm(root_v,axis=-1),
            np.linalg.norm(root_a,axis=-1))

def read_npz(entry,bones,ind):
    if "clips/test/" in entry["npz"]:
        raise RuntimeError("TEST GT must NEVER be available to rotation calibration")
    with np.load(DATA/entry["npz"],allow_pickle=False) as f:
        e=np.asarray(f["bone_euler_xyz_rad"],dtype=np.float64)
        root=np.asarray(f["root_rotation_wxyz"],dtype=np.float64)
    if e.shape!=(107,55,3) or root.shape!=(107,4):
        raise RuntimeError("Invalid source clip")
    v,a,rv,ra=angular_norms(e,root,ind)
    return e[:,ind],v,a,rv,ra

def calibrate():
    manifest,rig,bones=read_splits()
    indices=[bones.index(n) for n in SPINE]
    gt_train=[r for r in manifest["rows"] if r["split"]=="train"]
    gt_val=[r for r in manifest["rows"] if r["split"]=="validation"]
    if len(gt_train)!=96 or len(gt_val)!=24:
        raise RuntimeError("Clip count error")
    metrics=[]
    for row in gt_train:
        if row["split"]!="train":raise RuntimeError("Test leak")
        metrics.append(read_npz(row,bones,indices))
    el=np.concatenate([m[0] for m in metrics])
    lv=np.concatenate([m[1] for m in metrics])
    la=np.concatenate([m[2] for m in metrics])
    rv=np.concatenate([m[3] for m in metrics])
    ra=np.concatenate([m[4] for m in metrics])
    # A 99.5% TRAIN quantile with generous motion allowance. Minimum
    # thresholds avoid 0^2 traps for static axes. Nothing here is chosen
    # based on the previously inspected Reader TEST.
    bone_v=np.maximum(np.quantile(lv,.995,axis=0)*1.4,np.deg2rad(.30))
    bone_a=np.maximum(np.quantile(la,.995,axis=0)*1.4+np.deg2rad(.08),np.deg2rad(.24))
    root_v=max(float(np.quantile(rv,.995)*1.4),math.radians(.35))
    root_a=max(float(np.quantile(ra,.995)*1.4+math.radians(.09)),math.radians(.25))
    # The train-only range on each local Euler axis. This is an inactivity
    # safety envelope; active turning axes receive their full observed range.
    low=np.min(el,axis=0)-np.deg2rad(6)
    high=np.max(el,axis=0)+np.deg2rad(6)
    if np.any(high<=low):
        raise RuntimeError("Train prior invalid local axis range")
    profile={
        "schema":"anymove.human_v3_so3_prior/1",
        "source_blend_sha256":SRC_SHA,
        "dataset_manifest_sha256":sha(DATA/"manifest.json"),
        "train_clips":96,
        "validation_clips_not_used_for_fit":24,
        "test_clips_not_used_for_fit":24,
        "camera_azimuth_deg":[315,45],
        "source_fps":24,
        "rig_bone_names":bones,
        "spine_bone_names":list(SPINE),
        "spine_bone_indices":indices,
        "bone_velocity_threshold_rad_per_frame":bone_v.tolist(),
        "bone_acc_threshold_rad_per_frame2":bone_a.tolist(),
        "root_velocity_threshold_rad_per_frame":root_v,
        "root_acc_threshold_rad_per_frame2":root_a,
        "bone_axis_lower_rad":low.tolist(),
        "bone_axis_upper_rad":high.tolist(),
        "train_median_local_euler_xyz_rad":np.median(el,axis=0).tolist(),
        "weights":WEIGHTS,
        "absolute_root_heading_constraint":False,
        "physical_contact_assertion":False,
        "type":"Empirical TRAIN GT SO(3) velocity/acceleration safety prior, not learned neural model nor true biomechanical human ROM",
    }
    # Validation data is ONLY inspected for coverage, not used in param fit.
    val_metrics=[read_npz(row,bones,indices) for row in gt_val]
    v_lv=np.concatenate([m[1] for m in val_metrics])
    v_la=np.concatenate([m[2] for m in val_metrics])
    v_rv=np.concatenate([m[3] for m in val_metrics])
    v_ra=np.concatenate([m[4] for m in val_metrics])
    v_el=np.concatenate([m[0] for m in val_metrics])
    validation={
        "validation_source_clips":24,
        "validation_samples":len(v_el),
        "bone_velocity_exceed_rate":float(np.mean(v_lv>bone_v)),
        "bone_acceleration_exceed_rate":float(np.mean(v_la>bone_a)),
        "root_velocity_exceed_rate":float(np.mean(v_rv>root_v)),
        "root_acceleration_exceed_rate":float(np.mean(v_ra>root_a)),
        "axis_bounds_exceed_rate":float(np.mean((v_el<low)|(v_el>high))),
        "maximum_root_turn_angular_speed_deg_per_frame":float(np.rad2deg(v_rv.max())),
    }
    output_rig=json.loads(json.dumps(rig))
    output_rig["experimental_motion_prior"]=profile
    output_rig["limits_status"]="PROVISIONAL_TRAIN_SO3_MOTION_PRIOR_RESEARCH_ONLY"
    output_rig["prior_scope"]="Human V3 Nose 315/45; prior fitted to synthetic full-body Root turns and Spine; not iClone default"
    dst=DEST/"rigged/rig_info.json"
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        raise FileExistsError("Refuse to overwrite frozen existing SO3 research rig "+str(dst))
    dst.write_text(json.dumps(output_rig,indent=2),encoding="utf8")
    creature=json.loads((ROOT/"solver/creature.json").read_text(encoding="utf8"))
    creature["name"]="human_v3_nose_spine_so3_research"
    (DEST/"creature.json").write_text(json.dumps(creature,indent=2),encoding="utf8")
    report={
        "status":"TRAIN_CALIBRATED_VALIDATION_AUDITED",
        "source_blend_sha256":SRC_SHA,
        "prior_schema":profile["schema"],
        "training_clips":96,
        "validation_clips":24,
        "test_clips_sealed":24,
        "bone_names":list(SPINE),
        "per_bone_train_angular_acc_limit_deg":np.rad2deg(bone_a).round(3).tolist(),
        "root_train_angular_speed_limit_deg_per_frame":float(np.rad2deg(root_v)),
        "root_train_angular_acc_limit_deg_per_frame2":float(np.rad2deg(root_a)),
        "weights":WEIGHTS,
        "validation_coverage":validation,
        "research_rig":str(dst),
        "limitations":["No neural model fit; thresholds are empirical Train quantiles only",
            "Root yaw can turn freely; no zero-root heading prior",
            "Same single stylized human source and synthetic motions; natural Mocap unavailable",
            "Not automatically promoted to iClone"],
    }
    (DEST/"calibration_report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("HUMAN_V3_SO3_PRIOR_TRAIN_CALIBRATED",json.dumps({
        "rig":str(dst),"train":96,"val":24,"test_sealed":24,
        "bone_acc_max_deg":float(np.rad2deg(bone_a).max()),
        "bone_acc_median_deg":float(np.median(np.rad2deg(bone_a))),
        "root_vel_deg":round(float(np.rad2deg(root_v)),3),
        "root_acc_deg":round(float(np.rad2deg(root_a)),3),
        "validation":validation,
    }),flush=True)

if __name__=="__main__":
    calibrate()
