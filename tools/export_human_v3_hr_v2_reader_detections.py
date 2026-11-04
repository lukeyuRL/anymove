"""Freeze Reader HR V2 predictions for a P1 fullbody human fresh clip.

No GT used at inference. Input: audited P1 fresh render PNGs.
Output: unchanged NPZ Reader detections with true sparse frame indices,
67 names, 315/45 cameras, uncalibrated heatmap logits separately retained.
Reader reliability is initialized to 1.0, NOT claimed calibrated.
"""
import argparse
import hashlib
import json
import pathlib

import cv2
import numpy as np
import torch

from mocap.reader_hr_v2 import HumanReaderHRV2, decode_hr
from tools.train_human_v3_hr_v2_reader import OUT, SHA

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
FRESH=ROOT/"p1_fresh_test"
OUTPUT=ROOT/"solver/reader_p1_fresh"
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()

@torch.inference_mode()
def export(clip_index=0):
    manifest=json.loads((FRESH/"manifest.json").read_text(encoding="utf8"))
    checkpoint=OUT/"full"/"best.pt"
    ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (manifest["source_blend_sha256"]!=SHA or manifest["checkpoint_sha256"]!=digest(checkpoint)
         or ck["camera_azimuth_deg"]!=[315,45] or ck["names"]!=manifest["joint_names"]):
        raise RuntimeError("Checkpoint/Blender/angles/marker provenance mismatch")
    clips=manifest["clips"]
    if not 0<=clip_index<len(clips):raise ValueError("Clip index invalid")
    clip=clips[clip_index]["clip"]
    rows=[r for r in manifest["samples"] if r["clip"]==clip]
    frames=sorted({int(r["frame"]) for r in rows})
    if len(frames)!=14 or len(rows)!=28:
        raise RuntimeError("Expected 14 synchronized camera pairs")
    lookup={(int(r["frame"]),r["slot"]):r for r in rows}
    model=HumanReaderHRV2(67).cuda().eval()
    model.load_state_dict(ck["state_dict"],strict=True)
    points={}
    peaks={}
    for slot in ("front","side"):
        all_imgs=[]
        for frame in frames:
            row=lookup[(frame,slot)]
            if row["camera_azimuth_deg"]!=(315 if slot=="front" else 45):
                raise RuntimeError("Camera angle mismatch")
            img=cv2.imread(str(FRESH/row["file"]),cv2.IMREAD_UNCHANGED)
            if img is None or img.shape!=(672,672,4):
                raise RuntimeError("Invalid saved image")
            alpha=img[...,3,None].astype(np.float32)/255.
            rgb=img[...,[2,1,0]].astype(np.float32)
            pixels=np.clip(rgb*alpha+(1-alpha)*np.array([105.,107.,111.],dtype=np.float32),0,255)
            all_imgs.append(torch.from_numpy(np.ascontiguousarray(pixels)).permute(2,0,1).float()/255.)
        batch=torch.stack(all_imgs)
        predicted=[]
        raw=[]
        for start in range(0,len(batch),8):
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                heat=model(batch[start:start+8].cuda())
            uv,peak=decode_hr(heat.float())
            predicted.append(uv.cpu().numpy())
            raw.append(peak.cpu().numpy())
        points[slot]=np.concatenate(predicted)
        peaks[slot]=np.concatenate(raw)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    detpath=OUTPUT/(clip+".npz")
    np.savez_compressed(detpath,
        front_uv=points["front"],side_uv=points["side"],
        front_conf=np.ones((len(frames),67),dtype=np.float32),
        side_conf=np.ones((len(frames),67),dtype=np.float32),
        front_raw_peak_logit=peaks["front"],side_raw_peak_logit=peaks["side"],
        fps=np.asarray(24.),
        frame_indices=np.asarray(frames,dtype=np.int32),
        names=np.asarray(ck["names"]),
        camera_azimuth_deg=np.asarray([315.,45.]))
    report={
        "status":"READER_PREDICTION_COMPLETE_NO_GT_AT_INFERENCE",
        "clip":clip,"frames":frames,
        "source_blend_sha256":SHA,
        "frozen_checkpoint_sha256":digest(checkpoint),
        "detections_npz_sha256":digest(detpath),
        "camera_angles":[315,45],
        "confidence_note":"Uncalibrated Reader heatmap peaks are saved separately. Solver base evidence is 1.0 until P1 safety modifiers.",
        "source_dataset":"Blender P1 fresh test; previously used for P1 aggregate assessment",
        "predicted_point_count":67,
    }
    (OUTPUT/(clip+".detection_report.json")).write_text(json.dumps(report,indent=2),encoding="utf8")
    print("READER_P1_FRESH_EXPORT",json.dumps({"clip":clip,"file":str(detpath),
        "frames":len(frames),"checksum":report["detections_npz_sha256"],
        "peak_range":[float(min(peaks['front'].min(),peaks['side'].min())),
                      float(max(peaks['front'].max(),peaks['side'].max()))]}),flush=True)
    del model
    torch.cuda.empty_cache()
    return detpath
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--clip-index",type=int,default=8)
    args=p.parse_args()
    export(args.clip_index)
