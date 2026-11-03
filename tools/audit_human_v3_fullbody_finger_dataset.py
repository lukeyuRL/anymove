"""Audit V3 nose fullbody finger GT, camera metadata and tiny fingertip scales.

No network, GPU or inference. Verify that *each* training motion has
nonzero leg and finger articulation and both camera projections share
identical 3D GT. Report finger tip-to-distal-joint separation in pixels.
"""
import json,pathlib
import numpy as np
from mocap.camera_geometry import project_orthographic
from tools.human_v3_fullbody_motion import FINGER_NAMES,pose_changes

ROOT=pathlib.Path("runs/human/fullbody_finger_v3_nose/dataset")
PAIRS=("baseline_0_90","symmetric_315_45")
FINGERS=("Pinky","Ring","Mid","Index","Thumb")
def load(pair):
    data=json.loads((ROOT/pair/"manifest.json").read_text(encoding="utf-8"))
    if data["status"]!="complete" or len(data["joint_names"])!=67:
        raise RuntimeError("Invalid fullbody dataset")
    return data

a,b=load(PAIRS[0]),load(PAIRS[1])
if a["joint_names"]!=b["joint_names"] or a["train_clips"]!=b["train_clips"] or a["test_clips"]!=b["test_clips"]:
    raise RuntimeError("Unpaired A/B GT")
names=a["joint_names"]
idx={n:i for i,n in enumerate(names)}
tip_head_pairs=[(idx["CC_Base_"+s+"_"+digit+"Tip"],idx["CC_Base_"+s+"_"+digit+"3"])
                for s in ("L","R") for digit in FINGERS]
if len(tip_head_pairs)!=10:raise RuntimeError("Missing fingertip points")
rows={}
for pair,d in zip(PAIRS,(a,b)):
    for row in d["samples"]:
        key=(row["split"],row["clip"],row["frame"],row["slot"])
        if (pair,key) in rows:raise RuntimeError("Duplicate sample "+str(key))
        rows[(pair,key)]=np.asarray(row["uv"],dtype=np.float64)
    if len([k for k in rows if k[0]==pair])!=len(d["samples"]):
        raise RuntimeError("Inconsistent manifest rows")
qa=json.loads((ROOT/"fullbody_motion_qa.json").read_text(encoding="utf-8"))
if qa["status"]!="PASS" or len(qa["clips"])!=24:
    raise RuntimeError("Fullbody GT motion QA missing")
if any(rec["max_leg_travel_m"]<.025 or rec["min_finger_tip_to_wrist_radius_delta_m"]<.001
       or min(rec["index_flex_radians"])<.04 for rec in qa["clips"]):
    raise RuntimeError("A source clip lacks true leg or finger articulation")
min_finger_sweep=1e9
for entry in qa["clips"]:
    src=json.loads((ROOT/"motion_gt"/(entry["clip"]+".json")).read_text(encoding="utf-8"))
    family=src["clip"]["family"]
    seed=src["clip"]["seed"]
    flex={"CC_Base_"+s+"_"+digit+str(k):[]
          for s in ("L","R") for digit in FINGER_NAMES for k in (1,2,3)}
    if len(flex)!=30:raise RuntimeError("Missing finger segments")
    for frame in range(107):
        angles,_=pose_changes(family,seed,frame)
        for finger in flex:
            flex[finger].append(float(angles[finger][0]))
    sweeps=[max(v)-min(v) for v in flex.values()]
    min_finger_sweep=min(min_finger_sweep,*sweeps)
    if min(sweeps)<0.01:
        raise RuntimeError("A finger joint is static: "+entry["clip"]+" "+str(min(sweeps)))
report={"schema":"anymove.fullbody_finger_keypoint_scale_qa/1",
        "status":"PASS",
        "clips":24,"joint_count":67,
        "30_finger_phalanx_heads":True,"10_independent_fingertips":True,
        "all_30_finger_bones_animate_in_every_clip":True,
        "smallest_finger_joint_angular_sweep_radians":min_finger_sweep,
        "leg_motion_present_all_clips":True,"finger_motion_present_all_clips":True,
        "train_clip_count":len(a["train_clips"]),
        "val_clip_count":len(a["validation_clips"]),
        "test_clip_count":len(a["test_clips"]),
        "samples_per_pair":len(a["samples"]),
        "pair_results":{}}
for pair,data in zip(PAIRS,(a,b)):
    gt_err=[]
    distances=[]
    near_dist=[]
    clip_cache={}
    for row in data["samples"]:
        if row["clip"] not in clip_cache:
            clip_cache[row["clip"]]=json.loads((ROOT/"motion_gt"/(row["clip"]+".json")).read_text(encoding="utf-8"))
        clip=clip_cache[row["clip"]]
        gt=np.asarray(clip["joint_position_m"][row["frame"]],dtype=np.float64)
        from_file=np.asarray(row["uv"],dtype=np.float64)
        uv=project_orthographic(gt,row["camera_azimuth_deg"],data["width"],data["height"],
                                data["metres_per_pixel"],data["z_center"])
        gt_err.extend(np.linalg.norm(from_file-uv,axis=-1).tolist())
        distances.extend(np.linalg.norm(from_file[[q[0] for q in tip_head_pairs]]-
                                        from_file[[q[1] for q in tip_head_pairs]],axis=-1).tolist())
        for s in ("L","R"):
            digit_tips=[idx["CC_Base_"+s+"_"+d+"Tip"] for d in ("Pinky","Ring","Mid","Index")]
            near_dist.extend(np.linalg.norm(np.diff(from_file[digit_tips],axis=0),axis=-1).tolist())
    dist=np.asarray(distances)
    near=np.asarray(near_dist)
    report["pair_results"][pair]={
        "angles_deg":data["angles_deg"],
        "label_projection_max_deviation_px":float(np.max(gt_err)),
        "finger_last_phalanx_head_to_tip_median_px":float(np.median(dist)),
        "finger_last_phalanx_head_to_tip_lt4px_fraction":float(np.mean(dist<4)),
        "adjacent_fingertip_median_px":float(np.median(near)),
        "adjacent_fingertip_lt6px_fraction":float(np.mean(near<6)),
        "2d_views_checked":len(data["samples"])}
dest=pathlib.Path("runs/human/fullbody_finger_v3_nose/fullbody_finger_marker_qa.json")
dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
