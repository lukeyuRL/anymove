"""Validation-only P1 temporal/skeleton association study on frozen HR V2 Reader.

No Test samples used. The frozen model was selected using Validation 2D
mean; this stage is engineering/calibration evaluation, not unbiased
generalization proof. GT is read exclusively *after* tracker inference,
and no GT reaches track_two_view.
"""
import json, pathlib
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mocap.camera_geometry import triangulate_orthographic
from mocap.reader_hr_v2 import HumanReaderHRV2,decode_hr
from mocap.temporal_skeleton_tracker import track_two_view
from tools.train_human_v3_hr_v2_reader import DATA,OUT,SHA,V2Images,load_manifest
from tools.eval_human_v3_hr_v2_reader import body_groups

def stat(x):
    v=np.asarray(x,dtype=float)
    return {"n":int(v.size),"mean":float(v.mean()),
            "p95":float(np.percentile(v,95)),
            "p99":float(np.percentile(v,99)),
            "max":float(v.max()),
            "over_5cm":int((v>5).sum()),"over_10cm":int((v>10).sum()),
            "over_50cm":int((v>50).sum())}

def eval_split(split="validation"):
    if split!="validation":
        raise RuntimeError("Validation-only study must not inspect Test")
    config=load_manifest()
    calibration=json.loads((OUT/"full"/"temporal_skeleton_train_calibration.json").read_text(encoding="utf-8"))
    checkpoint=torch.load(OUT/"full"/"best.pt",map_location="cpu",weights_only=True)
    if (checkpoint["names"]!=calibration["names"] or checkpoint["camera_azimuth_deg"]!=[315,45] or
        checkpoint["source_blend_sha256"]!=SHA):
        raise RuntimeError("Calibrated rig/checkpoint mismatch")
    ds=V2Images(config,split,augment=False)
    loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    model=HumanReaderHRV2(len(config["joint_names"])).cuda().eval()
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    allpred=defaultdict(dict)
    view2d=defaultdict(list)
    with torch.inference_mode():
        for images,truth,ids in loader:
            with torch.autocast("cuda",dtype=torch.float16):
                heat=model(images.cuda(non_blocking=True))
            xy,_=decode_hr(heat.float())
            points=xy.cpu().numpy()
            for i,idx in enumerate(ids.tolist()):
                row=ds.samples[idx]
                key=(row["clip"],int(row["frame"]))
                slot=row["slot"]
                if slot in allpred[key]:
                    raise RuntimeError("Repeated camera view")
                allpred[key][slot]=np.asarray(points[i],dtype=np.float64)
    if len(allpred)!=24*14:raise RuntimeError("Unexpected Validation frames")
    clip_list=sorted(config["validation_clips"])
    cached_frames=np.asarray([sorted(f for c,f in allpred if c==clip) for clip in clip_list],dtype=np.int32)
    cached_f=np.asarray([[allpred[(clip,int(frame))]["front"] for frame in frames]
                         for clip,frames in zip(clip_list,cached_frames)],dtype=np.float64)
    cached_s=np.asarray([[allpred[(clip,int(frame))]["side"] for frame in frames]
                         for clip,frames in zip(clip_list,cached_frames)],dtype=np.float64)
    np.savez_compressed(OUT/"full"/"validation_raw_uv_frozen.npz",
                        clips=np.asarray(clip_list),frames=cached_frames,
                        front_uv=cached_f,side_uv=cached_s,
                        names=np.asarray(config["joint_names"]),
                        camera_angles=np.asarray([315,45]),
                        source_sha=np.asarray([SHA]),
                        checkpoint_epoch=np.asarray([checkpoint["epoch"]]))
    print("VALIDATION_PREDICTIONS_CACHED",len(clip_list),cached_f.shape,flush=True)
    groups=body_groups(config["joint_names"])
    groups_3d=defaultdict(list)
    group_2d=defaultdict(list)
    temporal=defaultdict(list)
    counter=defaultdict(int)
    per_clip=defaultdict(lambda:defaultdict(list))
    calibration_frames=[]
    for clip in sorted(config["validation_clips"]):
        frames=sorted(f for c,f in allpred if c==clip)
        if len(frames)!=14:raise RuntimeError("Clip frame count mismatch")
        raw_f=np.stack([allpred[clip,f]["front"] for f in frames])
        raw_s=np.stack([allpred[clip,f]["side"] for f in frames])
        tracked=track_two_view(raw_f,raw_s,frames,calibration,config["joint_names"])
        gt_json=json.loads((DATA/"motion_gt"/(clip+".json")).read_text(encoding="utf-8"))
        if gt_json["clip"]["split"]!="validation" or gt_json["source_blend_sha256"]!=SHA:
            raise RuntimeError("GT cross split leakage")
        xyz=np.asarray(gt_json["joint_position_m"],dtype=np.float64)[frames]
        ones=np.ones((len(frames),len(config["joint_names"])),dtype=np.float64)
        raw3=triangulate_orthographic(raw_f,raw_s,ones,ones,
            front_width=672,side_width=672,height=672,
            metres_per_pixel=config["metres_per_pixel"],z_center=config["z_center"],
            front_angle_deg=315,side_angle_deg=45,front_height_weight=1.0)
        new3=tracked["joints_world_m"]
        err_raw=np.linalg.norm(raw3-xyz,axis=-1)*100
        err_track=np.linalg.norm(new3-xyz,axis=-1)*100
        for label,val in (("raw",err_raw),("tracked",err_track)):
            for group,inds in groups.items():
                groups_3d[label+"_"+group].extend(val[:,inds].ravel())
                per_clip[clip][label+"_"+group].extend(val[:,inds].ravel())
        for label,obs_f,obs_s in (("raw",raw_f,raw_s),("tracked",tracked["front_uv"],tracked["side_uv"])):
            for slot,pred in (("front",obs_f),("side",obs_s)):
                gt_stack=np.stack([np.asarray(r["uv"]) for r in ds.samples if
                       r["clip"]==clip and r["slot"]==slot])
                # dataloader originally comes in manifest row order which
                # is not guaranteed to be time sorted; use manifest mapping.
                mapping={int(r["frame"]):np.asarray(r["uv"]) for r in ds.samples if
                       r["clip"]==clip and r["slot"]==slot}
                reference=np.stack([mapping[f] for f in frames])
                err=np.linalg.norm(pred-reference,axis=-1)
                for group,inds in groups.items():
                    group_2d[label+"_"+slot+"_"+group].extend(err[:,inds].ravel())
        for label,pred in (("raw",raw3),("tracked",new3)):
            difference=np.diff(pred,axis=0)-np.diff(xyz,axis=0)
            perstep=np.linalg.norm(difference,axis=-1)*100
            for group,inds in groups.items():
                temporal[label+"_"+group].extend(perstep[:,inds].ravel())
        for k,value in tracked["diagnostics"].items():
            counter[k]+=int(value) if "fraction" not in k else 0
        calibration_frames.append((clip,frames,raw_f,raw_s,tracked))
    metrics={}
    for label in ("raw","tracked"):
        metrics[label]={
            "3d_cm":{k:stat(groups_3d[label+"_"+k]) for k in groups},
            "2d_px":{slot:{k:stat(group_2d[label+"_"+slot+"_"+k]) for k in groups}
                     for slot in ("front","side")},
            "temporal_cm":{k:stat(temporal[label+"_"+k]) for k in groups},
        }
    report={
        "schema":"anymove.human_hrv2_validation_temporal_skeleton_ab/1",
        "status":"VALIDATION_ONLY_COMPLETED",
        "source_blend_sha256":SHA,
        "train_calibration_clips":calibration["calibration_clip_count"],
        "validation_clips":len(config["validation_clips"]),
        "validation_frames":len(allpred),
        "keypoints":len(config["joint_names"]),
        "tracker_has_no_access_to_GroundTruth":True,
        "counts":dict(counter),
        "metrics":metrics,
        "per_clip_mean3d_cm":{
            clip:{label:float(np.mean(per_clip[clip][label+"_all_67"])) for label in ("raw","tracked")}
            for clip in config["validation_clips"]},
        "limitations":"Validation was used to choose Reader weights; not fresh Test. No FK optimization or H3 video.",
    }
    out=OUT/"full"/"temporal_skeleton_validation.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    a=metrics["raw"]["3d_cm"]["all_67"]
    b=metrics["tracked"]["3d_cm"]["all_67"]
    print("TEMPORAL_VALIDATION_RESULT",json.dumps({
        "raw_3d_cm":a,"tracked_3d_cm":b,
        "raw_fingertip_3d_cm":metrics["raw"]["3d_cm"]["finger_tip_10"],
        "tracked_fingertip_3d_cm":metrics["tracked"]["3d_cm"]["finger_tip_10"],
        "raw_toe_3d_cm":metrics["raw"]["3d_cm"]["toe_tip_2"],
        "tracked_toe_3d_cm":metrics["tracked"]["3d_cm"]["toe_tip_2"],
        "raw_temporal_cm":metrics["raw"]["temporal_cm"]["all_67"],
        "tracked_temporal_cm":metrics["tracked"]["temporal_cm"]["all_67"],
        "counts":dict(counter),"report":str(out)
    }),flush=True)

if __name__=="__main__":
    torch.set_num_threads(4)
    eval_split()
