"""Heldout quantitative Reader V2 672/192 test, after validation-only selection.

Asserts source/checkpoint/camera/rig, test split disjointness. Reports native
2D pixels, 448-equivalent pixels, raw 3D MPJPE cm, per-region and temporal
displacement consistency. No test-tuned model or thresholds.
"""
import hashlib, json, pathlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mocap.reader_hr_v2 import HumanReaderHRV2, decode_hr
from mocap.camera_geometry import triangulate_orthographic
from tools.train_human_v3_hr_v2_reader import DATA, OUT, SHA, V2Images, load_manifest

def filehash(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):
            h.update(buf)
    return h.hexdigest()

def report_stats(values):
    x=np.asarray(values,dtype=np.float64)
    if x.size==0:raise RuntimeError("Empty test metric")
    return {
        "observations":int(x.size),
        "mean":round(float(np.mean(x)),5),
        "p95":round(float(np.percentile(x,95)),5),
        "max":round(float(np.max(x)),5),
    }

def body_groups(names):
    groups={
        "fingers_and_fingertips":[],
        "legs_feet_and_toes":[],
        "arms_hands_clavicles":[],
        "torso_root":[],
        "head_neck":[],
    }
    for idx,name in enumerate(names):
        if any(x in name for x in ("Pinky","Ring","Mid","Index","Thumb")):
            group="fingers_and_fingertips"
        elif any(x in name for x in ("Thigh","Calf","Foot","ToeBase","ToeTip")):
            group="legs_feet_and_toes"
        elif any(x in name for x in ("Clavicle","Upperarm","Forearm","_Hand")):
            group="arms_hands_clavicles"
        elif any(x in name for x in ("Head","NeckTwist")):
            group="head_neck"
        else:
            group="torso_root"
        groups[group].append(idx)
    if list(map(len,groups.values()))!=[40,10,8,6,3]:
        raise RuntimeError("Disjoint five-region keypoint partition failed")
    groups["all_67"]=list(range(len(names)))
    groups["finger_tip_10"]=[i for i,n in enumerate(names) if n.endswith("Tip") and "Toe" not in n]
    groups["toe_tip_2"]=[i for i,n in enumerate(names) if n.endswith("ToeTip")]
    if len(groups["finger_tip_10"])!=10 or len(groups["toe_tip_2"])!=2:
        raise RuntimeError("Missing fingertip or toe-tip markers")
    return groups

@torch.inference_mode()
def evaluate():
    mf=load_manifest()
    checkpoint=OUT/"full"/"best.pt"
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (
        ck["names"]!=mf["joint_names"] or
        ck["source_blend_sha256"]!=SHA or
        ck["camera_azimuth_deg"]!=[315,45] or
        ck["input_size"]!=672 or ck["heatmap_size"]!=192 or
        set(ck["train_clips"])!=set(mf["train_clips"]) or
        set(ck["validation_clips"])!=set(mf["validation_clips"]) or
        set(ck["test_clips"])!=set(mf["test_clips"])
    ):
        raise RuntimeError("Reader checkpoint/source/camera provenance mismatch")
    if any(set(ck[a+"_clips"])&set(ck[b+"_clips"])
           for a,b in (("train","validation"),("train","test"),("validation","test"))):
        raise RuntimeError("Clip leakage into checkpoint")
    ds=V2Images(mf,"test",augment=False)
    if len(ds)!=672:raise RuntimeError("Unexpected heldout image count")
    loader=DataLoader(ds,batch_size=8,shuffle=False,num_workers=0,pin_memory=True)
    model=HumanReaderHRV2(67).cuda().eval()
    model.load_state_dict(ck["state_dict"],strict=True)
    groups=body_groups(mf["joint_names"])
    twoD=defaultdict(list)
    preds=defaultdict(dict)
    for x,gt,indices in loader:
        with torch.autocast("cuda",dtype=torch.float16):
            heat=model(x.cuda(non_blocking=True))
        uv,peak=decode_hr(heat.float())
        points=uv.cpu().numpy()
        truth=gt.numpy()
        for i,index in enumerate(indices.tolist()):
            row=ds.samples[index]
            slot=row["slot"]
            if row["camera_azimuth_deg"]!=(315 if slot=="front" else 45):
                raise RuntimeError("Test camera angle mismatch")
            err=np.linalg.norm(points[i]-truth[i],axis=-1)
            for group,members in groups.items():
                twoD[group].extend(err[members])
                twoD[slot+"_"+group].extend(err[members])
            key=(row["clip"],row["frame"])
            if slot in preds[key]:raise RuntimeError("Duplicated view")
            preds[key][slot]=np.asarray(points[i],dtype=np.float64)
    if len(preds)!=336:raise RuntimeError("Expected 336 paired heldout frames")
    gt_cache={}
    errors=defaultdict(list)
    clipped={}
    frame_results=defaultdict(dict)
    for (clip,frame),obs in sorted(preds.items()):
        if set(obs)!={"front","side"}:
            raise RuntimeError("Orphan view for clip/frame")
        if clip not in gt_cache:
            record=json.loads((DATA/"motion_gt"/(clip+".json")).read_text(encoding="utf-8"))
            if record["source_blend_sha256"]!=SHA or record["joint_names"]!=mf["joint_names"]:
                raise RuntimeError("Unknown Blender GT provenance")
            gt_cache[clip]=np.asarray(record["joint_position_m"],dtype=np.float64)
        world=gt_cache[clip][frame]
        joints=triangulate_orthographic(
            obs["front"],obs["side"],np.ones(67),np.ones(67),
            front_width=672,side_width=672,height=672,
            metres_per_pixel=mf["metres_per_pixel"],z_center=mf["z_center"],
            front_angle_deg=315,side_angle_deg=45,front_height_weight=1.0,
        )
        diff=np.linalg.norm(joints-world,axis=-1)*100
        for group,members in groups.items():
            errors[group].extend(diff[members])
            errors[clip+"_"+group].extend(diff[members])
        frame_results[clip][frame]=(joints,world)
    temporal=defaultdict(list)
    for clip,frames in frame_results.items():
        frames_sorted=sorted(frames)
        for prev,cur in zip(frames_sorted,frames_sorted[1:]):
            a,ag=frames[prev]
            b,bg=frames[cur]
            displacement_error=np.linalg.norm((b-a)-(bg-ag),axis=-1)*100
            for group,members in groups.items():
                temporal[group].extend(displacement_error[members])
    metric_groups={}
    for group in groups:
        metric_groups[group]={
            "2d_px":report_stats(twoD[group]),
            "2d_equivalent_448_px":round(float(np.mean(twoD[group]))*448/672,5),
            "2d_315_px":report_stats(twoD["front_"+group]),
            "2d_45_px":report_stats(twoD["side_"+group]),
            "3d_cm":report_stats(errors[group]),
            "temporal_displacement_error_cm":report_stats(temporal[group]),
        }
    result={
        "schema":"anymove.human_v3_nose_672_reader_heldout_v2/1",
        "status":"COMPLETE_READER_HELDOUT",
        "training_checkpoint_sha256":filehash(checkpoint),
        "checkpoint_selected_epoch":ck["epoch"],
        "source_blend_sha256":SHA,
        "camera_angles_deg":[315,45],
        "native_input_size":672,
        "heatmap_size":192,
        "test_clips":mf["test_clips"],
        "heldout_images":len(ds),
        "heldout_stereo_frames":len(preds),
        "source":"Independent procedural Blender V3 Nose GT, no test-directed training",
        "regions":metric_groups,
        "five_anatomy_regions_equal_score_cm":round(float(np.mean([
            metric_groups[k]["3d_cm"]["mean"] for k in (
                "fingers_and_fingertips","legs_feet_and_toes","arms_hands_clavicles","torso_root","head_neck")
        ])),5),
        "per_clip_3d_cm":{
            clip:report_stats(errors[clip+"_all_67"]) for clip in mf["test_clips"]
        },
        "limitations":[
            "Procedural synthetic motion, single human rig; no H3 videos",
            "No temporal filtering or iClone FK fitting",
            "Visibility annotations are surface ray proxies, not calibrated confidence",
            "672px metrics are not directly comparable in pixels to old 448px Reader; use centimetres or normalized pixels",
            "Only one backbone/head training seed",
        ],
    }
    dest=OUT/"full"/"heldout_test.json"
    dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("HRV2_READER_HELDOUT",json.dumps({
        "checkpoint_epoch":ck["epoch"],
        "2d_mean_px":metric_groups["all_67"]["2d_px"]["mean"],
        "3d_mpjpe_cm":metric_groups["all_67"]["3d_cm"]["mean"],
        "fingertip_cm":metric_groups["finger_tip_10"]["3d_cm"]["mean"],
        "toe_tip_cm":metric_groups["toe_tip_2"]["3d_cm"]["mean"],
        "five_region_cm":result["five_anatomy_regions_equal_score_cm"],
        "frames":len(preds),
    }),flush=True)

if __name__=="__main__":
    if not torch.cuda.is_available():raise RuntimeError("Needs local GPU")
    torch.set_num_threads(4)
    evaluate()
