"""Independent AFTER-FREEZE P1 temporal/skeleton Test using new 12 seed clips.

Use exactly the unchanged frozen Reader and TRAIN-only calibration. No Test GT
can enter the Tracker or candidate selection; GT is consumed only after
processing to produce raw/processed 2D and 3D errors, tail rates and temporal
displacement error. This is still synthetic Blender data, not H3.
"""
import hashlib,json,pathlib
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mocap.reader_hr_v2 import HumanReaderHRV2,decode_hr
from mocap.camera_geometry import triangulate_orthographic
from mocap.temporal_skeleton_tracker import track_two_view
from tools.eval_human_v3_hr_v2_reader import body_groups
from tools.train_human_v3_hr_v2_reader import DATA,OUT,SHA,load_manifest

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/p1_fresh_test")

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()

class FreshImages(Dataset):
    def __init__(self,meta):
        self.rows=meta["samples"]
        if len(self.rows)!=336:raise RuntimeError("Expected 336 native 672 Test images")
    def __len__(self):return len(self.rows)
    def __getitem__(self,idx):
        row=self.rows[idx]
        rgba=cv2.imread(str(ROOT/row["file"]),cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape!=(672,672,4):
            raise RuntimeError("Test RGBA image missing/invalid")
        alpha=rgba[...,3,None].astype(np.float32)/255
        rgb=rgba[...,[2,1,0]].astype(np.float32)
        img=(rgb*alpha+(1-alpha)*np.array([105.,107.,111.],np.float32))
        x=torch.from_numpy(np.ascontiguousarray(img)).permute(2,0,1).float()/255
        return x,idx

def score(arr):
    x=np.asarray(arr,dtype=np.float64)
    return {
        "n":int(x.size),"mean":round(float(x.mean()),5),
        "p95":round(float(np.percentile(x,95)),5),
        "p99":round(float(np.percentile(x,99)),5),
        "max":round(float(x.max()),5),
        "over_5cm":int((x>5).sum()),
        "over_10cm":int((x>10).sum()),
        "over_50cm":int((x>50).sum()),
    }

@torch.inference_mode()
def evaluate():
    data=load_manifest()
    meta=json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))
    ckpt=OUT/"full"/"best.pt"
    cal_path=OUT/"full"/"temporal_skeleton_train_calibration.json"
    if meta.get("status")!="complete" or meta.get("schema")!="anymove.human_p1_hrv2_fresh_test_manifest/1":
        raise RuntimeError("Fresh test was not completed")
    if (sha(ckpt)!=meta["checkpoint_sha256"] or sha(cal_path)!=meta["calibration_sha256"]
        or meta["source_blend_sha256"]!=SHA or meta["joint_names"]!=data["joint_names"]
        or meta["camera_angles_deg"]!=[315,45]):
        raise RuntimeError("Frozen checkpoint/calibration Test provenance mismatch")
    if set(c["clip"] for c in meta["clips"]) & set(data["train_clips"]+data["validation_clips"]+data["test_clips"]):
        raise RuntimeError("Fresh test identities overlap old samples")
    ck=torch.load(ckpt,map_location="cpu",weights_only=True)
    calibration=json.loads(cal_path.read_text(encoding="utf-8"))
    model=HumanReaderHRV2(67).cuda().eval()
    model.load_state_dict(ck["state_dict"],strict=True)
    ds=FreshImages(meta)
    loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
    names=meta["joint_names"]
    groups=body_groups(names)
    observations=defaultdict(dict)
    for image,indices in loader:
        with torch.autocast("cuda",dtype=torch.float16):
            heat=model(image.cuda(non_blocking=True))
        coords,_=decode_hr(heat.float())
        pred=coords.cpu().numpy()
        for b,index in enumerate(indices.tolist()):
            row=ds.rows[index]
            key=(row["clip"],int(row["frame"]))
            if row["slot"] in observations[key]:
                raise RuntimeError("Duplicate camera pair slot")
            observations[key][row["slot"]]=np.asarray(pred[b],dtype=float)
    if len(observations)!=12*14:raise RuntimeError("Missing paired test frames")
    gt_cache={c["clip"]:json.loads((ROOT/"motion_gt"/(c["clip"]+".json")).read_text(encoding="utf-8")) for c in meta["clips"]}
    metrics=defaultdict(list)
    counts=defaultdict(int)
    per_clip={}
    worst=[]
    frames_processed=0
    for c in meta["clips"]:
        clip=c["clip"]
        frames=sorted(f for name,f in observations if name==clip)
        if len(frames)!=14:raise RuntimeError("Incomplete clip")
        f=np.stack([observations[clip,fr]["front"] for fr in frames])
        s=np.stack([observations[clip,fr]["side"] for fr in frames])
        # No ground truth passed into track_two_view.
        filtered=track_two_view(f,s,frames,calibration,names)
        gt=gt_cache[clip]
        if gt["clip"]["split"]!="fresh_test" or gt["source_blend_sha256"]!=SHA or gt["joint_names"]!=names:
            raise RuntimeError("Invalid fresh GT provenance")
        xyz=np.asarray(gt["joint_position_m"],dtype=float)[frames]
        ones=np.ones((len(frames),67),dtype=np.float64)
        raw=triangulate_orthographic(f,s,ones,ones,
            front_width=672,side_width=672,height=672,
            metres_per_pixel=meta["metres_per_pixel"],z_center=meta["z_center"],
            front_angle_deg=315,side_angle_deg=45,front_height_weight=1.)
        estimates={"raw":raw,"tracked":filtered["joints_world_m"]}
        clip_stats={}
        for method,position in estimates.items():
            error=np.linalg.norm(position-xyz,axis=-1)*100
            for group,idx in groups.items():
                metrics[method+"_"+group].extend(error[:,idx].ravel().tolist())
            local=np.linalg.norm(np.diff(position,axis=0)-np.diff(xyz,axis=0),axis=-1)*100
            metrics[method+"_temporal"].extend(local.ravel().tolist())
            clip_stats[method]=float(error.mean())
            for ti,j in np.argwhere(error>10.0):
                worst.append({"clip":clip,"frame":int(frames[ti]),"joint":names[j],
                    "method":method,"error_cm":float(error[ti,j])})
        per_clip[clip]=clip_stats
        for key,value in filtered["diagnostics"].items():
            if isinstance(value,int):counts[key]+=value
        frames_processed+=len(frames)
    scores={}
    for method in ("raw","tracked"):
        scores[method]={
            "3d_cm":{group:score(metrics[method+"_"+group]) for group in groups},
            "temporal_cm":score(metrics[method+"_temporal"]),
            "five_region_score_cm":round(float(np.mean([
                np.mean(metrics[method+"_"+part]) for part in (
                    "fingers_and_fingertips","legs_feet_and_toes",
                    "arms_hands_clavicles","torso_root","head_neck"
                )])),5)
        }
    result={
        "schema":"anymove.human_p1_hrv2_fresh_12_seed_ab/1",
        "status":"FRESH_TEST_COMPLETE_NO_RETRAINING",
        "source_blend_sha256":SHA,
        "frozen_checkpoint_sha256":meta["checkpoint_sha256"],
        "train_calibration_sha256":meta["calibration_sha256"],
        "camera_angles_deg":[315,45],
        "new_seed_clip_count":len(meta["clips"]),
        "images":len(ds),"paired_frames":frames_processed,
        "joint_names":names,
        "counts":dict(counts),
        "scores":scores,
        "per_clip":per_clip,
        "worst_10cm_cases":sorted(worst,key=lambda r:-r["error_cm"])[:40],
        "limitations":[
            "Blender procedural motion, same model/rig and 12 motion families as training",
            "Single fixed Reader checkpoint and Train-only threshold calibration",
            "3D output is triangulation after safety-gated view positions, NOT full FK",
            "Natural MoCap, H3 generated video and iClone human rig are not yet validated",
        ]
    }
    target=ROOT/"p1_fresh_results.json"
    target.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("P1_FRESH_UNSEEN_RESULT",json.dumps({
        "clips":len(meta["clips"]),"frames":frames_processed,
        "raw_3d_cm":scores["raw"]["3d_cm"]["all_67"],
        "tracked_3d_cm":scores["tracked"]["3d_cm"]["all_67"],
        "raw_temporal_cm":scores["raw"]["temporal_cm"],
        "tracked_temporal_cm":scores["tracked"]["temporal_cm"],
        "counts":dict(counts),
        "report":str(target),
    }),flush=True)
if __name__=="__main__":
    if not torch.cuda.is_available():raise RuntimeError("Needs CUDA")
    torch.set_num_threads(4)
    evaluate()
