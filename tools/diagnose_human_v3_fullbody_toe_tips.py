"""Posthoc toe-tip failure diagnosis on already tested models.

READ-ONLY model inference. Does not retrain or choose checkpoints using test.
Inspect individual toe-tip 2D errors and whether predictions collapse onto
nearby foot/toe-base landmarks under each camera.
"""
import json
from collections import defaultdict
import pathlib

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from mocap.model import JointReader,SIZE,HM,decode
from tools.train_human_v3_fullbody_finger_reader_ab import (
    ROOT,PAIR_NAMES,manifest,FullBodyImages)

@torch.inference_mode()
def diagnose(pair):
    config=manifest(pair)
    names=config["joint_names"]
    data=FullBodyImages(pair,config,"test")
    loader=DataLoader(data,batch_size=4,shuffle=False,num_workers=0)
    model=JointReader(67).cuda().eval()
    ck=torch.load(ROOT/"reader_models_strict"/pair/"best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(ck["state_dict"])
    toe_names=["CC_Base_L_ToeTip","CC_Base_R_ToeTip"]
    toeidx=[names.index(n) for n in toe_names]
    footidx=[names.index("CC_Base_"+s+"_Foot") for s in ("L","R")]
    baseidx=[names.index("CC_Base_"+s+"_ToeBase") for s in ("L","R")]
    rows=[]
    for x,gt,indices in loader:
        with torch.autocast("cuda",dtype=torch.float16):
            result=model(x.cuda())
        uv,_=decode(result.float())
        uv=uv.cpu().numpy()*(SIZE/HM)
        truth=gt.numpy()
        for i,j in enumerate(indices):
            row=data.rows[int(j)]
            for t,label in enumerate(toe_names):
                k=toeidx[t]
                d=float(np.linalg.norm(uv[i,k]-truth[i,k]))
                fb=float(np.linalg.norm(uv[i,k]-truth[i,footidx[t]]))
                tb=float(np.linalg.norm(uv[i,k]-truth[i,baseidx[t]]))
                rows.append({"clip":row["clip"],"frame":row["frame"],"view":row["slot"],
                    "toe":label,"error_px":round(d,4),"closer_to_foot_head":fb<d,
                    "closer_to_toebase_head":tb<d,
                    "gt_px":np.round(truth[i,k],2).tolist(),
                    "pred_px":np.round(uv[i,k],2).tolist(),
                    "foot_head_gt_px":np.round(truth[i,footidx[t]],2).tolist()})
    report={"pair":pair,"source":"Unseen synthetic test; posthoc only, no retraining",
        "toe_tip_images_evaluated":len(rows),"error_gt10px":sum(r["error_px"]>10 for r in rows),
        "error_gt20px":sum(r["error_px"]>20 for r in rows),
        "closer_to_foot_head_count":sum(r["closer_to_foot_head"] for r in rows),
        "closer_to_toebase_head_count":sum(r["closer_to_toebase_head"] for r in rows),
        "by_view":{v:{"mean_px":round(float(np.mean([r["error_px"] for r in rows if r["view"]==v])),3)}
            for v in ("front","side")},
        "by_toe":{v:{"mean_px":round(float(np.mean([r["error_px"] for r in rows if r["toe"]==v])),3)}
            for v in toe_names},
        "worst":sorted(rows,key=lambda r:-r["error_px"])[:20]}
    dest=ROOT/"reader_models_strict"/pair/"toe_tip_diagnostic.json"
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("TOE_TIP_DIAG",pair,json.dumps({
        "gt10":report["error_gt10px"],"gt20":report["error_gt20px"],
        "to_foot":report["closer_to_foot_head_count"],"to_toebase":report["closer_to_toebase_head_count"],
        "by_view":report["by_view"],"by_toe":report["by_toe"],
        "worst":report["worst"][:7]}),flush=True)
    del model
    torch.cuda.empty_cache()
    return report

if __name__=="__main__":
    torch.set_num_threads(4)
    for p in PAIR_NAMES:diagnose(p)
