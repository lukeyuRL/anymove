"""Independent release audit for Human V3 Nose SPINE MOTION Dataset v1.

144 clip-disjoint procedural Blender trajectories, 107 frames @24 FPS,
55 local XYZ rotations, world Root quaternions and translations, 67 world
markers and explicit foot-contact PROXIES (not physics labels).
No Reader retraining, no image generation and no test-selected thresholds.
"""
import hashlib
import json
import pathlib
from collections import Counter,defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

from tools import human_v3_spine_motion_prior_v1 as motion

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
OUT=ROOT/"spine_motion_prior_v1"
SOURCE=pathlib.Path("runs/human/human_skinmax_multicolor_v3_nose.blend")
SOURCE_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
SPLITS={"train":96,"validation":24,"test":24}
EXPECTED_CLIPS=144
EXPECTED_FRAMES=144*107
RIG=ROOT/"solver/rigged/rig_info.json"

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for p in iter(lambda:f.read(1<<20),b""):
            h.update(p)
    return h.hexdigest()

def fail(text):
    raise RuntimeError("SPINE_DATASET_AUDIT_FAILED: "+str(text))

def atomic_json(path,payload):
    t=path.with_suffix(".tmp")
    t.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf8")
    t.replace(path)

def audit():
    if not SOURCE.is_file() or sha(SOURCE)!=SOURCE_SHA:
        fail("Human V3 original source model SHA256 changed")
    rig=json.loads(RIG.read_text(encoding="utf8"))
    names=[r["name"] for r in rig["bones"]]
    marks=[r["name"] for r in rig["keypoints"]]
    if len(names)!=55 or len(marks)!=67 or len(set(marks))!=67:
        fail("Blender rig schema not 55 bones + 67 markers")
    definition=motion.all_unique_clips()
    clip_set={x["clip"] for x in definition}
    if len(clip_set)!=EXPECTED_CLIPS:
        fail("Duplicate train/validation/test procedural clip")
    gt_rows=[]
    family_splits=defaultdict(Counter)
    splits=Counter()
    all_clip_shas={}
    all_gt_yaw=[]
    world_rotation_steps=[]
    samples_by_split=Counter()
    total_points=0
    contact_values=[]
    for family_index,family in enumerate(motion.FAMILIES):
        file=OUT/f"family_{family_index:02d}_manifest.json"
        if not file.is_file():fail("Missing completed motion family "+str(file))
        batch=json.loads(file.read_text(encoding="utf8"))
        if (batch["state"]!="COMPLETED" or batch["clip_count"]!=12
            or batch["source_model_sha256"]!=SOURCE_SHA
            or {r["family"] for r in batch["rows"]}!={family}):
            fail("Wrong motion family generated "+family)
    for desc in definition:
        data=OUT/"clips"/desc["split"]/(desc["clip"]+".npz")
        meta=OUT/"clips"/desc["split"]/(desc["clip"]+".json")
        if not data.is_file() or not meta.is_file():
            fail("Missing 107-frame motion clip "+desc["clip"])
        info=json.loads(meta.read_text(encoding="utf8"))
        if (info["schema"]!="anymove.human_v3_spine_motion_gt_clip/1"
            or info["clip"]!=desc
            or info["source_model_sha256"]!=SOURCE_SHA
            or info["bone_names"]!=names
            or info["marker_names"]!=marks
            or info["source_frames"]!=107
            or info["fps"]!=24):
            fail("Motion source/bone/marker metadata mismatch "+desc["clip"])
        with np.load(data,allow_pickle=False) as npz:
            expected={
                "frames":(107,),"bone_euler_xyz_rad":(107,55,3),
                "root_rotation_wxyz":(107,4),"root_translation_m":(107,3),
                "world_joints_m":(107,67,3),
                "toe_height_m":(107,2),"contact_proxy":(107,2),
            }
            for key,shape in expected.items():
                if key not in npz or npz[key].shape!=shape:
                    fail(f"GT clip wrong array {desc['clip']}.{key}")
                if not np.isfinite(npz[key]).all():
                    fail(f"Non-finite motion GT {desc['clip']}.{key}")
            if not np.array_equal(npz["frames"],np.arange(107)):
                fail("Non-contiguous 24FPS times")
            root_id=names.index("RL_BoneRoot")
            if not np.array_equal(npz["bone_euler_xyz_rad"][:,root_id,:],
                                  np.zeros((107,3),dtype=np.float32)):
                fail("RL_BoneRoot must have exactly zero local animation across all 107 frames")
            if not np.all((npz["contact_proxy"]==0)|(npz["contact_proxy"]==1)):
                fail("Invalid contact flag")
            q=np.asarray(npz["root_rotation_wxyz"],dtype=float)
            if np.max(np.abs(np.linalg.norm(q,axis=-1)-1))>5e-5:
                fail("Bad Root quaternion normalization")
            root_rotation=Rotation.from_quat(q[:,[1,2,3,0]])
            dv=(root_rotation[:-1].inv()*root_rotation[1:]).magnitude()
            max_ang=float(np.rad2deg(dv.max()))
            if max_ang>16:
                fail("Excessive adjacent Root rotation (>16deg/frame): "+desc["clip"])
            positions=np.asarray(npz["world_joints_m"],dtype=np.float64)
            for bone_idx,parent in enumerate(rig["bones"]):
                par=parent["parent"]
                if par<0:continue
                dist=np.linalg.norm(positions[:,bone_idx]-positions[:,par],axis=-1)
                if np.ptp(dist)>0.00002:
                    fail("GT violates constant inter-bone head offset for "+parent["name"])
            contact_values.append(np.asarray(npz["contact_proxy"],dtype=float).mean(axis=0))
            total_points+=npz["world_joints_m"].shape[0]*67
            all_clip_shas[str(data.relative_to(OUT)).replace("\\","/")]=sha(data)
            all_gt_yaw.append(info["quality"]["root_yaw_range_deg"])
            world_rotation_steps.append(max_ang)
            if (info["quality"]["root_yaw_range_deg"]<4
                or info["quality"]["waist_axis_range_max_deg"]<4
                or info["quality"]["max_leg_world_travel_m"]<.025
                or info["quality"]["max_fingertip_world_travel_m"]<.025):
                fail("One motion family lacks wholebody/Root/spine variance "+desc["clip"])
        splits[desc["split"]]+=1
        family_splits[desc["family"]][desc["split"]]+=1
        samples_by_split[desc["split"]]+=107
        gt_rows.append({"clip":desc["clip"],"family":desc["family"],"seed":desc["seed"],
                        "split":desc["split"],
                        "fps":24,"frames":107,
                        "npz":str(data.relative_to(OUT)).replace("\\","/"),
                        "root_yaw_range_deg":info["quality"]["root_yaw_range_deg"],
                        "waist_axis_range_max_deg":info["quality"]["waist_axis_range_max_deg"]})
    if len(gt_rows)!=EXPECTED_CLIPS or dict(splits)!=SPLITS:
        fail("Total clip count/splits wrong")
    if dict(samples_by_split)!={"train":10272,"validation":2568,"test":2568}:
        fail("Frame count per split wrong")
    if any(family_splits[family]!={"train":8,"validation":2,"test":2} for family in motion.FAMILIES):
        fail("Family split not balanced")
    tracked=list(OUT.rglob("*.npz"))
    if len(tracked)!=EXPECTED_CLIPS+3:
        # smoke has 3 clips saved under a different directory, not part of
        # released Train/Val/Test data; allow only those 3 extras.
        fail("Unexpected extra NPZ files in frozen Motion GT root")
    final={
        "schema":"anymove.human_v3_spine_motion_gt_dataset_v1/1",
        "status":"PASS",
        "source_blend_sha256":SOURCE_SHA,
        "source_rig_bones":names,
        "tracked_markers":marks,
        "fps":24,"frames_per_clip":107,
        "train_val_test_clips":dict(splits),
        "train_val_test_frames":dict(samples_by_split),
        "motion_family_count":len(motion.FAMILIES),
        "motion_families":list(motion.FAMILIES),
        "unique_clip_count":len(gt_rows),
        "observed_frames_total":EXPECTED_FRAMES,
        "total_joint_xyz_observations":total_points,
        "distinct_root_yaw_range_deg":{
            "min":float(np.min(all_gt_yaw)),
            "median":float(np.median(all_gt_yaw)),
            "max":float(np.max(all_gt_yaw))},
        "root_adjacent_rotation_max_deg":float(max(world_rotation_steps)),
        "contact_proxy_mean":np.mean(contact_values,axis=0).tolist(),
        "foot_contact_warning":"Proxy uses toe height+horizontal displacement only; NOT physical contact or validated natural gait",
        "dataset_role":"TRAIN V3 statistical rotation/temporal prior; NOT Reader images, NOT natural mocap",
        "dataset_independent_from_prior_reader_test":True,
        "rows":gt_rows,
        "clip_sha256":all_clip_shas,
    }
    atomic_json(OUT/"manifest.json",final)
    atomic_json(OUT/"audit.json",{k:v for k,v in final.items() if k not in ("rows","clip_sha256","source_rig_bones","tracked_markers")})
    print("SPINE_PRIOR_V1_DATASET_RELEASE_PASS",json.dumps({
        "clips":len(gt_rows),"frames":EXPECTED_FRAMES,
        "split_clips":dict(splits),"split_frames":dict(samples_by_split),
        "root_yaw_deg_range":final["distinct_root_yaw_range_deg"],
        "root_adjacent_max_deg":final["root_adjacent_rotation_max_deg"],
        "contact_proxy_mean":final["contact_proxy_mean"],
        "output":str(OUT/"manifest.json")}),flush=True)


if __name__=="__main__":
    audit()
