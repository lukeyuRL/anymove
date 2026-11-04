"""Compare GT-independent human FK initializers for 315/45 rig.

Zero-Euler rest vs analytic IK against triangulated view observations, on
TRAIN clip only. No Test GT enters model/threshold selection.
"""
import json,os,pathlib
import numpy as np
import torch
ROOT=pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45")
os.environ["MOCAP_CREATURE"]=str((ROOT/"solver").resolve())
os.environ["MOCAP_CAMERA_AZIMUTHS"]="315,45"
from mocap import layout,solve

def run():
    info=json.loads((ROOT/"solver/rigged/rig_info.json").read_text())
    manifest=json.loads((ROOT/"dataset/manifest.json").read_text())
    clip=next(c for c in manifest["train_clips"] if "finger_independent_roll" in c)
    record=json.loads((ROOT/"dataset/motion_gt"/(clip+".json")).read_text())
    frames=[8,15,22,29,36]
    world=np.asarray(record["joint_position_m"])[frames]
    rig=solve.Rig(info,"cpu")
    d={"front_uv":layout.project(world)[0],"side_uv":layout.project(world)[1],
       "front_conf":np.ones(world.shape[:2]),"side_conf":np.ones(world.shape[:2])}
    tri=solve.triangulate(d)
    conf=np.ones(tri.shape[:2])
    root=solve.root_from_triangulation(tri,conf,rig,rig.kp_names,info)
    init_ik=solve.ik_init(tri,root,info,rig)
    init_rest=np.zeros_like(init_ik)
    trans=tri[:,rig.kp_names.index(info["bones"][0]["head_joint"])]-rig.rest_kp[0].numpy()
    mat=solve.axis_angle(torch.tensor(root,dtype=torch.float64))
    for label,euler in (("ik",init_ik),("rest",init_rest)):
        pred,_=rig.forward(torch.tensor(trans),mat,torch.tensor(euler))
        err=np.linalg.norm(pred.detach().numpy()-tri,axis=-1)
        print("HUMAN_FK_INIT",label,"mean_cm",err.mean()*100,"p95_cm",np.percentile(err,95)*100,
              "worst_cm",err.max()*100,
              "extreme_euler_count",int((np.abs(euler)>2.3).sum()),flush=True)
    # Source BASE euler pose gives an upper-bound achievable fit accuracy
    from tools.human_v3_fullbody_motion import BASE
    e=np.zeros_like(init_ik)
    for joint,rot in BASE.items():
        e[:,rig.names.index(joint)]=rot
    pred,_=rig.forward(torch.tensor(trans),mat,torch.tensor(e))
    print("HUMAN_FK_INIT_BASE",float(np.mean(np.linalg.norm(pred.detach().numpy()-tri,axis=-1))*100))
if __name__=="__main__":
    run()
