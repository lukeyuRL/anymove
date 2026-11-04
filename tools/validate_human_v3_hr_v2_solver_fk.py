"""Compare exported human Blender FK to mocap.solve.Rig.forward at 3 poses.

This checks scale/basis/keypoint mapping numerically and rejects any layout
or rig-geometry discrepancy before optional iClone Solver integration.
"""
import json,pathlib
import numpy as np
import torch
from mocap.solve import Rig

ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/solver")
def run():
    info=json.loads((ROOT/"rigged/rig_info.json").read_text(encoding="utf-8"))
    ref=json.loads((ROOT/"fk_probe.json").read_text(encoding="utf-8"))
    names=[k["name"] for k in info["keypoints"]]
    assert len(info["bones"])==55 and len(names)==67
    assert names==ref["joint_names"]
    rig=Rig(info,"cpu")
    results=[]
    for probe in ref["probes"]:
        trans=torch.tensor([probe["root_translation_m"]],dtype=torch.float64)
        rot=torch.eye(3,dtype=torch.float64).unsqueeze(0)
        eul=torch.tensor([probe["euler_xyz_rad"]],dtype=torch.float64)
        pred,_=rig.forward(trans,rot,eul)
        gt=np.asarray(probe["joints_world_m"],dtype=np.float64)
        err=np.linalg.norm(pred.detach().numpy()[0]-gt,axis=-1)
        results.append({"frame":probe["frame"],
                        "mean_cm":float(err.mean()*100),
                        "max_cm":float(err.max()*100),
                        "worst_joint":names[int(err.argmax())]})
    status="PASS" if max(x["max_cm"] for x in results)<.05 else "FAIL"
    result={"status":status,"source":"Independent Blender FK / Torch FK agreement",
            "probe_frames":results,"blender_world_scale":0.01,
            "camera_angles":[315,45],"marker_count":67,
            "provisional_rig_limits":True}
    (ROOT/"fk_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("HUMAN_FK_VALIDATION",json.dumps(result),flush=True)
    if status!="PASS":raise RuntimeError("Human FK contract not aligned")
if __name__=="__main__":
    run()
