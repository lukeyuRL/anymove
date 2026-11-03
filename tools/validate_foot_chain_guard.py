"""Training-only foot chain calibration and validation-only QA.

The last four heldout Test clips are never read here.
Input Reader checkpoints were selected BEFORE this guard.
"""
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from mocap.model import JointReader, HM, SIZE, decode
from mocap.toe_tip_guard import (
    calibrate_from_train, calibrate_base_from_train, guard_foot_chain
)
from tools.train_human_v3_fullbody_finger_reader_ab import (
    ROOT, PAIR_NAMES, manifest, FullBodyImages, SOURCE_SHA,
)

def hash_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

@torch.inference_mode()
def validate(pair):
    config=manifest(pair)
    tip=calibrate_from_train(config)
    base=calibrate_base_from_train(config)
    reader_path=ROOT/"reader_models_strict"/pair/"best.pt"
    ck=torch.load(reader_path,map_location="cpu",weights_only=True)
    if ck["names"]!=config["joint_names"] or ck["camera_azimuth_deg"]!=config["angles_deg"]:
        raise RuntimeError("Reader checkpoint metadata mismatch")
    if set(ck["validation_clips"]) != set(config["validation_clips"]) or set(ck["test_clips"])!=set(config["test_clips"]):
        raise RuntimeError("No true clip-disjoint validation set")
    model=JointReader(len(ck["names"])).cuda().eval()
    model.load_state_dict(ck["state_dict"],strict=True)
    ds=FullBodyImages(pair,config,"validation",False)
    loader=DataLoader(ds,batch_size=4,shuffle=False,num_workers=0,pin_memory=True)
    names=config["joint_names"]
    errors=defaultdict(lambda:{"raw":[],"corrected":[],"replaced":0,"failed":0})
    cases=[]
    for x,kp,indices in loader:
        with torch.autocast("cuda",dtype=torch.float16):
            logits=model(x.cuda(non_blocking=True))
        uv,_=decode(logits.float())
        uv=uv.cpu().numpy()*(SIZE/HM)
        truth=kp.numpy()
        for bi,j in enumerate(indices.tolist()):
            row=ds.rows[j]
            slot=row["slot"]
            after,flags=guard_foot_chain(uv[bi],names,slot,base,tip,config["angles_deg"])
            for side in ("L","R"):
                for jointtype in ("ToeBase","ToeTip"):
                    joint="CC_Base_"+side+"_"+jointtype
                    typekey="toe_base_flags" if jointtype=="ToeBase" else "toe_tip_flags"
                    flag=flags[typekey][joint]
                    name=joint+"_"+slot
                    k=names.index(joint)
                    t=errors[name]
                    t["raw"].append(float(np.linalg.norm(uv[bi,k]-truth[bi,k])))
                    t["corrected"].append(float(np.linalg.norm(after[k]-truth[bi,k])))
                    t["replaced"]+=int(flag["repaired"] if jointtype=="ToeBase" else flag["replaced"])
                    t["failed"]+=int(flag["untrusted_no_repair"])
                    if ((flag["repaired"] if jointtype=="ToeBase" else flag["replaced"])
                         or flag["untrusted_no_repair"]):
                        cases.append({"clip":row["clip"],"frame":row["frame"],
                                     "slot":slot,"joint":joint,
                                     "raw_err":round(t["raw"][-1],3),
                                     "guard_err":round(t["corrected"][-1],3),
                                     "flags":flag})
    result={
        "schema":"anymove.human_train_calibrated_foot_chain_validation_v1",
        "status":"VALIDATION_ONLY",
        "source_blend_sha256":SOURCE_SHA,
        "checkpoint_sha256":hash_file(reader_path),
        "pair":pair,"angles_deg":config["angles_deg"],
        "train_clips":config["train_clips"],
        "validation_clips":config["validation_clips"],
        "test_clips_NOT_READ":config["test_clips"],
        "base_calibration":{k:v.to_dict() for k,v in base.items()},
        "tip_calibration":{k:v.to_dict() for k,v in tip.items()},
        "validation":{},
        "flagged_cases":cases}
    for name,detail in sorted(errors.items()):
        before=np.asarray(detail["raw"],dtype=float)
        after=np.asarray(detail["corrected"],dtype=float)
        result["validation"][name]={
            "n":len(before),"replaced":detail["replaced"],
            "untrusted":detail["failed"],
            "raw_mean_px":round(float(before.mean()),4),
            "corrected_mean_px":round(float(after.mean()),4),
            "raw_p95_px":round(float(np.percentile(before,95)),4),
            "corrected_p95_px":round(float(np.percentile(after,95)),4),
        }
    path=ROOT/"reader_models_strict"/pair/"foot_chain_guard_train_val.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("FOOT_CHAIN_VAL",pair,json.dumps(result["validation"]),flush=True)
    print("FOOT_CHAIN_FLAGGED",pair,len(cases),flush=True)
    del model
    torch.cuda.empty_cache()
    return result

if __name__=="__main__":
    torch.set_num_threads(4)
    for p in PAIR_NAMES:
        validate(p)
