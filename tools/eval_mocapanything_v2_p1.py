"""Post-hoc P1 QA: only reads the held-out Side after the Front-only runs."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

A=Path(r"F:\anymove")
M=Path(r"F:\mocapanything_v2")
O=Path(r"F:\mocapanything_v2_test\p1_front_only_160")
sys.path[:0]=[str(A),str(M)]
from utils import animation,bvh
from tools.mocapanything_v2_front_solver import eval_world

root=json.loads((Path(r"F:\mocapanything_v2_test")/"Dino_V2_front_root_124f.report.json").read_text(encoding="utf-8"))
basis=np.array(root["basis_rotation_bvh_to_world"])
scale=root["basis_scale"]
det=np.load(A/"runs/dino/clips/test_walk/det.npz")
info=json.loads((A/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
bone_names=[b["name"] for b in info["bones"]]
obs_names=[str(x) for x in det["names"]]
obsidx=[obs_names.index(b["head_joint"]) for b in info["bones"]]
observations={k:np.asarray(det[k])[:,obsidx] for k in ("front_uv","side_uv")}
obs_centroid=np.asarray(det["front_uv"][:,obsidx[0]],float)
layout=info["layout"]
B={name:Path(p) for name,p in {
    "original_V2":r"F:\mocapanything_v2_test\Dino_V2_front_root_124f.bvh",
    "front_only_rest":str(O/"Dino_P1_rest_front_only.bvh"),
    "front_only_V2_prior":str(O/"Dino_P1_v2_front_only.bvh"),
}.items()}
motion=json.loads((A/"runs/dino/clips/test_walk/motion.json").read_text())
Afull=np.array(motion["joints"])[:,obsidx]

groups={}
for i,n in enumerate(bone_names):
    if "Tail" in n:tag="tail"
    elif any(z in n for z in ("Head","Neck","Mouth","Tongue","Eye")):tag="head_neck"
    elif any(z in n for z in ("Thigh","Calf","Foot","Toe")):tag="hind_legs_toes"
    elif any(z in n for z in ("Upperarm","Forearm","Calvicle","Hand","Index","Mid")):tag="forelimbs"
    else:tag="body_root"
    groups.setdefault(tag,[]).append(i)

def split_errors(world):
    p=np.asarray(world); m=layout["M_PER_PX"]
    fu=np.stack([layout["W_FRONT"]/2+p[...,0]/m,
                 layout["H"]/2-(p[...,2]-layout["Z_CENTER"])/m],-1)
    su=np.stack([layout["W_SIDE"]/2-p[...,1]/m,
                 layout["H"]/2-(p[...,2]-layout["Z_CENTER"])/m],-1)
    return np.linalg.norm(fu-observations["front_uv"],axis=-1),np.linalg.norm(su-observations["side_uv"],axis=-1)

def metrics(world,names,rot=None):
    out={}
    front,side=split_errors(world)
    def fmt(x):return {"mean_px":round(float(x.mean()),3),"p95_px":round(float(np.percentile(x,95)),3)}
    out["reprojection"]={"front":fmt(front),"heldout_side":fmt(side)}
    out["regional"]={g:{"bone_count":len(v),"front":fmt(front[:,v]),"side":fmt(side[:,v])} for g,v in groups.items()}
    world=np.asarray(world)
    rel=world-world[:,0:1]
    jerk=np.diff(rel,n=2,axis=0)
    speed=np.diff(rel,axis=0)
    out["motion"]={
        "mean_root_relative_speed_m_per_frame":round(float(np.linalg.norm(speed,axis=-1).mean()),5),
        "p95_root_relative_acceleration_m_per_frame2":round(float(np.percentile(np.linalg.norm(jerk,axis=-1),95)),5),
        "max_root_relative_acceleration_m_per_frame2":round(float(np.linalg.norm(jerk,axis=-1).max()),5)}
    # Zero mesh deformation not evaluated; head-to-head bone lengths are enforced by BVH FK.
    bones=info["bones"]
    links=[]
    for i,b in enumerate(bones):
        if b["parent"]<0:continue
        p=b["parent"]
        now=np.linalg.norm(world[:,i]-world[:,p],axis=1)
        ref=np.linalg.norm(np.array(b["matrix_local"])[:3,3]-np.array(bones[p]["matrix_local"])[:3,3])
        links.extend(np.abs(now-ref).tolist())
    out["bone_length"]={"max_abs_error_m":round(float(np.max(links)),6),"mean_abs_error_m":round(float(np.mean(links)),6)}
    feet=[i for i,n in enumerate(names) if any(tok in n for tok in ("_Foot","_Toe","_BackToe"))]
    # Contact heuristic from predicted feet. May be wrong for the current walking-in-place H3 video.
    foot_ok=0; contact_vel=[]
    for i in feet:
        z=world[:,i,2]
        contact=(z[:-1]<0.12)&(z[1:]<0.12)
        if np.any(contact):
            foot_ok+=int(np.sum(contact))
            contact_vel.extend(np.linalg.norm(np.diff(world[:,i,:2],axis=0)[contact],axis=-1).tolist())
    out["contact_proxy"]={"joint_frame_pairs":foot_ok,
                          "mean_horizontal_displacement_m_per_contact_frame":
                           round(float(np.mean(contact_vel)),5) if contact_vel else None,
                          "status":"NO_CONTACT_EVIDENCE" if not contact_vel else "HEURISTIC_ONLY_NOT_GT"}
    if rot is not None:
        Euler=Rotation.from_quat(np.asarray(rot)[...,[1,2,3,0]].reshape(-1,4)).as_euler("xyz",degrees=True)
        out["draft_joint_limit_audit"]={"angles_over_20deg_percent":round(float(np.mean(np.abs(Euler)>20)*100),3),
            "is_verified_anatomical_limit":False}
    return out

results={"evaluation":"posthoc only; raw source H3 Reader, same 67 bone-head subset",
 "front_only_solver_side_used_during_optimization":False,
 "warning":"Heldout Side is not 3D ground truth. 20deg axis limits are imported draft defaults, not trusted anatomy.",
 "frames":len(observations["front_uv"]),"models":{}}
for tag,path in B.items():
    a,n,_=bvh.load(str(path))
    world=(animation.positions_global(a)[:,[n.index(x) for x in bone_names]]@basis)*scale
    if len(world)!=len(Afull):raise RuntimeError("frame mismatch")
    results["models"][tag]=metrics(world,bone_names,rot=np.asarray(a.rotations))
results["models"]["AnyMove_FrontSide_solver"]=metrics(Afull,bone_names)
O.mkdir(parents=True,exist_ok=True)
(O/"p1_quality_diagnostics.json").write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf8")
print(json.dumps({k:{"front":v["reprojection"]["front"]["mean_px"],
                "side":v["reprojection"]["heldout_side"]["mean_px"],
                "hind_side":v["regional"]["hind_legs_toes"]["side"]["mean_px"],
                "foot_contact_proxy":v["contact_proxy"],
                "bone_length":v["bone_length"]} for k,v in results["models"].items()},indent=2))
print("saved",O/"p1_quality_diagnostics.json")
