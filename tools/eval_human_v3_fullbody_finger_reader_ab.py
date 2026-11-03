"""Final unseen TEST on 67-point Human V3 Nose fullbody+finger Readers.

Includes fingers (30 phalange head joints), ten fingertip ends, both toe ends,
legs, arms, torso, whole-body 2D/3D MPJPE, per-clip errors, and temporal
displacement mismatch. Never used by checkpoint selection.
"""
import hashlib,json,pathlib
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from mocap.model import JointReader,SIZE,HM,decode
from mocap.camera_geometry import triangulate_orthographic
from tools.train_human_v3_fullbody_finger_reader_ab import (
    ROOT,PAIR_NAMES,SOURCE_SHA,manifest,FullBodyImages
)

GT=ROOT/"dataset"/"motion_gt"
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):
            h.update(x)
    return h.hexdigest()

def stat(vals):
    a=np.asarray(vals,dtype=np.float64)
    return {"n":int(a.size),
            "mean":round(float(a.mean()),4),
            "median":round(float(np.median(a)),4),
            "p95":round(float(np.percentile(a,95)),4)}
def pair_indices(names):
    ix={name:i for i,name in enumerate(names)}
    pairs=[]
    for name in names:
        if "_L_" in name:
            other=name.replace("_L_","_R_",1)
            if other in ix:pairs.append((ix[name],ix[other]))
    return pairs

@torch.inference_mode()
def evaluate(pair):
    data=manifest(pair)
    path=ROOT/"reader_models_strict"/pair/"best.pt"
    if not path.is_file():raise FileNotFoundError("Best validation checkpoint missing: "+str(path))
    ck=torch.load(path,map_location="cpu",weights_only=True)
    if (ck["names"]!=data["joint_names"] or ck["camera_azimuth_deg"]!=data["angles_deg"] or
        ck["source_blend_sha256"]!=SOURCE_SHA or
        len(ck["train_clips"])!=16 or len(ck["validation_clips"])!=4 or len(ck["test_clips"])!=4 or
        set(ck["test_clips"])!=set(data["test_clips"])):
        raise RuntimeError("Checkpoint source/provenance mismatch")
    if (set(ck["train_clips"]) & set(ck["test_clips"]) or
        set(ck["validation_clips"]) & set(ck["test_clips"])):
        raise RuntimeError("Train/validation leakage into unseen test")
    ds=FullBodyImages(pair,data,"test",False)
    loader=DataLoader(ds,batch_size=4,num_workers=0,pin_memory=True)
    names=data["joint_names"]
    groups=data["joint_groups"]
    groups["full_67"]=names
    groups["head_neck"]=[n for n in names if "Head" in n or "Neck" in n or "Jaw" in n]
    if len(names)!=67 or len(groups["finger_30_bone_heads"])!=30 or len(groups["finger_10_tips"])!=10:
        raise RuntimeError("Finger/fullbody marker definitions incorrect")
    grouped_indices={k:[names.index(x) for x in group] for k,group in groups.items()}
    model=JointReader(len(names)).cuda().eval()
    model.load_state_dict(ck["state_dict"],strict=True)
    err2=defaultdict(list)
    view_errors=defaultdict(list)
    preds={}
    swaps=defaultdict(lambda:{"eligible":0,"flipped":0,"close_gt":0})
    pair_idx=pair_indices(names)
    print("FULLBODY_TEST_START",pair,"images",len(ds),
        "markers",len(names),"checkpoint_epoch",ck["epoch"],flush=True)
    for x,gt,ids in loader:
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            logits=model(x.cuda(non_blocking=True))
        uv,_=decode(logits.float())
        uv=uv.cpu().numpy()*(SIZE/HM)
        true=gt.numpy()
        for i,idx in enumerate(ids.tolist()):
            row=ds.rows[idx]
            if row["camera_azimuth_deg"]!=data["angles_deg"][0 if row["slot"]=="front" else 1]:
                raise RuntimeError("Angle mismatch on TEST row")
            diff=np.linalg.norm(uv[i]-true[i],axis=-1)
            for category,indices in grouped_indices.items():
                err2[category].extend(diff[indices].tolist())
                view_errors[row["slot"]+"_"+category].extend(diff[indices].tolist())
            for j,name in enumerate(names):
                err2["individual_"+name].append(float(diff[j]))
            s=swaps[row["slot"]]
            for left,right in pair_idx:
                if np.linalg.norm(true[i,left]-true[i,right])<8:
                    s["close_gt"]+=1
                    continue
                s["eligible"]+=1
                regular=np.linalg.norm(uv[i,left]-true[i,left])+np.linalg.norm(uv[i,right]-true[i,right])
                flipped=np.linalg.norm(uv[i,left]-true[i,right])+np.linalg.norm(uv[i,right]-true[i,left])
                if flipped+2<regular:s["flipped"]+=1
            framekey=(row["clip"],int(row["frame"]))
            predictions=preds.setdefault(framekey,{})
            predictions[row["slot"]]=np.asarray(uv[i],dtype=np.float64)
            if (row["frame"]==48 and ("lunge" in row["clip"] or "balance" in row["clip"])):
                overlay=cv2.imread(str(ROOT/"dataset"/row["path"]),cv2.IMREAD_UNCHANGED)
                bg=np.empty(overlay.shape[:2]+(3,),dtype=np.float32);bg[:]=(88,92,95)
                alpha=overlay[...,3,None].astype(np.float32)/255.
                im=np.clip(overlay[...,:3]*alpha+bg*(1-alpha),0,255).astype(np.uint8)
                selected=["CC_Base_L_Hand","CC_Base_R_Hand","CC_Base_L_Foot","CC_Base_R_Foot",
                          "CC_Base_L_IndexTip","CC_Base_R_IndexTip","CC_Base_L_ThumbTip",
                          "CC_Base_R_ThumbTip","CC_Base_Hip","CC_Base_Head"]
                for n in selected:
                    if n not in names:continue
                    j=names.index(n)
                    pos=tuple(np.rint(true[i,j]).astype(int))
                    pred=tuple(np.rint(uv[i,j]).astype(int))
                    cv2.circle(im,pos,3,(15,250,40),-1,cv2.LINE_AA)
                    cv2.circle(im,pred,3,(35,45,250),1,cv2.LINE_AA)
                target=ROOT/"reader_models_strict"/pair/("overlay_"+row["clip"]+"_f"+str(row["frame"])+"_"+row["slot"]+".png")
                cv2.imwrite(str(target),im)
    gt_data={name:json.loads((GT/(name+".json")).read_text(encoding="utf-8"))
             for name in data["test_clips"]}
    err3=defaultdict(list)
    per_clip={}
    reconstructed=defaultdict(dict)
    for (clip,frame),views in sorted(preds.items()):
        if set(views)!={"front","side"}:raise RuntimeError("Unpaired test frame")
        world=np.asarray(gt_data[clip]["joint_position_m"][frame],dtype=np.float64)
        # All test predictions are paired by exact clip and frame and use
        # their manifest's calibrated two-camera projection parameters.
        ones=np.ones(len(names),dtype=np.float64)
        result=triangulate_orthographic(
           views["front"],views["side"],ones,ones,
           front_width=data["width"],side_width=data["width"],
           height=data["height"],metres_per_pixel=data["metres_per_pixel"],
           z_center=data["z_center"],
           front_angle_deg=data["angles_deg"][0],
           side_angle_deg=data["angles_deg"][1],front_height_weight=1.)
        delta_cm=np.linalg.norm(result-world,axis=-1)*100
        for k,indices in grouped_indices.items():
            err3[k].extend(delta_cm[indices].tolist())
        for j,name in enumerate(names):
            err3["individual_"+name].append(float(delta_cm[j]))
        per_clip.setdefault(clip,[]).extend(delta_cm.tolist())
        reconstructed[clip][frame]=(result,world)
    temporal_error=defaultdict(list)
    for clip,frames in reconstructed.items():
        sequence=sorted(frames)
        for early,later in zip(sequence,sequence[1:]):
            p0,g0=frames[early]
            p1,g1=frames[later]
            diff=(p1-p0)-(g1-g0)
            for group,ids in grouped_indices.items():
                temporal_error[group].extend((np.linalg.norm(diff[ids],axis=-1)*100).tolist())
    report={
       "schema":"anymove.human_v3_nose_fullbody_finger_unseen_test/1",
       "status":"UNSEEN_TEST_COMPLETE_SYNTHETIC",
       "source_blend_sha256":SOURCE_SHA,
       "pair":pair,"camera_angles_deg":data["angles_deg"],
       "checkpoint_sha256":digest(path),
       "checkpoint_selected_on_validation_epoch":ck["epoch"],
       "train_clips":ck["train_clips"],
       "validation_clips":ck["validation_clips"],
       "test_clips":ck["test_clips"],
       "test_images":len(ds),
       "test_joint_observations":len(ds)*len(names),
       "keypoints":len(names),
       "groups":{group:{"keypoints":len(ids),
           "2d_px":stat(err2[group]),"3d_cm":stat(err3[group]),
           "temporal_displacement_error_cm":stat(temporal_error[group]),
           "per_view_2d_px":{
               slot:stat(view_errors[slot+"_"+group]) for slot in ("front","side")
           }} for group,ids in grouped_indices.items()},
       "per_test_clip_3d_cm":{c:stat(e) for c,e in per_clip.items()},
       "per_joint":{name:{"2d_px":stat(err2["individual_"+name]),
                           "3d_cm":stat(err3["individual_"+name])} for name in names},
       "swap_proxy":{slot:{"eligible_pairs":v["eligible"],"flipped_pairs":v["flipped"],
           "gt_close_lt8px":v["close_gt"],
           "proxy_rate":round(v["flipped"]/max(v["eligible"],1),6)}
           for slot,v in swaps.items()},
       "limitations":["Procedural animation, not natural human mocap",
           "Single DINOv2 training seed and short training schedule",
           "No actual H3 video or iClone FK motion fitting",
           "Unsupervised/self-occluded joint locations included in heatmap GT",
           "Sparse dynamic evaluation frames, no subframe temporal accuracy"]
    }
    dest=ROOT/"reader_models_strict"/pair/"unseen_test.json"
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("FULLBODY_UNSEEN_RESULT",pair,json.dumps({
         k:{"2d_px":v["2d_px"]["mean"],"3d_cm":v["3d_cm"]["mean"],
            "temporal_cm":v["temporal_displacement_error_cm"]["mean"]}
         for k,v in report["groups"].items()}),flush=True)
    del model
    torch.cuda.empty_cache()
    return report

def run():
    reports={pair:evaluate(pair) for pair in PAIR_NAMES}
    a=reports["baseline_0_90"]
    b=reports["symmetric_315_45"]
    if a["test_clips"]!=b["test_clips"] or a["keypoints"]!=b["keypoints"]:
        raise RuntimeError("Paired A/B test mismatch")
    def pct(b,a):return round(100*(b-a)/a,2) if a else None
    groups={}
    for k in a["groups"]:
        baseline=a["groups"][k]
        chosen=b["groups"][k]
        groups[k]={
            "0_90_2d_px":baseline["2d_px"]["mean"],
            "315_45_2d_px":chosen["2d_px"]["mean"],
            "2d_change_percent":pct(chosen["2d_px"]["mean"],baseline["2d_px"]["mean"]),
            "0_90_3d_cm":baseline["3d_cm"]["mean"],
            "315_45_3d_cm":chosen["3d_cm"]["mean"],
            "3d_change_percent":pct(chosen["3d_cm"]["mean"],baseline["3d_cm"]["mean"]),
            "0_90_temporal_cm":baseline["temporal_displacement_error_cm"]["mean"],
            "315_45_temporal_cm":chosen["temporal_displacement_error_cm"]["mean"],
        }
    comparison={"schema":"anymove.human_v3_nose_fullbody_finger_camera_ab/1",
                "status":"FINAL_CONTROLLED_SYNTHETIC_SINGLE_SEED",
                "clip_disjoint_splits":True,
                "test_clips":a["test_clips"],
                "test_images_each":a["test_images"],
                "keypoints":a["keypoints"],"groups":groups,
                "per_clip_3d":{clip:{"0_90":a["per_test_clip_3d_cm"][clip]["mean"],
                                    "315_45":b["per_test_clip_3d_cm"][clip]["mean"]}
                               for clip in a["test_clips"]},
                "limitations":a["limitations"]}
    dest=ROOT/"reader_models_strict"/"fullbody_finger_final_ab.json"
    dest.write_text(json.dumps(comparison,indent=2),encoding="utf-8")
    print("FULLBODY_FINAL_AB",json.dumps(comparison),flush=True)

if __name__=="__main__":
    if not torch.cuda.is_available():raise RuntimeError("Requires local CUDA")
    torch.set_num_threads(4)
    run()
