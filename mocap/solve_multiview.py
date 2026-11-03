"""Experimental N-camera orthographic FK optimizer (2,3,4... views).

Compatible with imported arbitrary Creature rig_info.json, not its bone names.
Observations are view-indexed NPZ from mocap.multiview; original P0 two-view
Solver and iClone motion.json remain untouched.

Important:
- Never reads 3D Ground Truth.
- Never reads a surface-part visibility oracle as an input weight.
- Separate independently sampled still poses are NOT a temporal sequence.
- Rotation-limit defaults in imported Rigs are draft, not anatomical truth.
- Saves experiment .npz + manifest, not a production motion.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mocap.multiview import check_packet,load_packet,linear_cameras,triangulate
from mocap.solve import Rig,axis_angle


def torch_project(world, matrix, image_size, meters_per_px):
    """World coordinates [T,J,3] -> pixels [V,T,J,2]; camera is rigid W2C."""
    local=torch.einsum("vij,tkj->vtki",matrix[:,:3,:3],world)+matrix[:,None,None,:3,3]
    return torch.stack((
        image_size[:,None,None,0]*.5+local[...,0]/meters_per_px[:,None,None],
        image_size[:,None,None,1]*.5-local[...,1]/meters_per_px[:,None,None]
    ),dim=-1)


def rest_kabsch_initialization(tri,available,info,rig):
    """Root initial rotation using only TRIANGULATED body points, never 3D GT.

    This is independently computed per pose; no temporal convolution
    on a collection of unrelated Blender still poses.
    """
    names=[k["name"] for k in info["keypoints"]]
    bones=info["bones"]
    ride={bones[0]["head_joint"],bones[0]["tail_joint"]}
    ride.update(b["head_joint"] for b in bones if b["parent"]==0)
    indices=np.array([names.index(n) for n in sorted(ride) if n in names],dtype=int)
    hips=names.index(bones[0]["head_joint"])
    rest=rig.rest_kp.cpu().numpy()
    base=rig.B[0,:3,:3].cpu().numpy()
    result=[]
    for i in range(len(tri)):
        take=indices[available[i,indices]]
        if len(take)<3:
            result.append(np.zeros(3))
            continue
        reference=rest[take]-rest[hips]
        observed=tri[i,take]-tri[i,hips]
        u,s,vt=np.linalg.svd(reference.T@observed)
        D=np.diag([1,1,np.linalg.det(vt.T@u.T)])
        world=vt.T@D@u.T
        local=base.T@world@base
        result.append(Rotation.from_matrix(local).as_rotvec())
    return np.stack(result)


def make_initial_rig(info,tri,available, *, device):
    """Translate 3D root observations, initialize root rotation from body."""
    rig=Rig(info,dev=device)
    ref=rig.rest_kp.detach().cpu().numpy()
    name=[k["name"] for k in info["keypoints"]]
    root_index=name.index(info["bones"][0]["head_joint"])
    root=np.asarray(tri[:,root_index],float)
    if not np.all(available[:,root_index]):
        raise ValueError("root joint not triangulable with supplied camera views")
    initial_translation=root-ref[root_index]
    root_initial=rest_kabsch_initialization(tri,available,info,rig)
    return rig,initial_translation,root_initial


def _serialize_world_metrics(joints,packet,selected):
    world=torch.as_tensor(joints,dtype=torch.float64)
    cams=torch.as_tensor(np.asarray(packet["world_to_camera"])[selected],dtype=torch.float64)
    size=torch.as_tensor(np.asarray(packet["image_size_px"])[selected],dtype=torch.float64)
    mpp=torch.as_tensor(np.asarray(packet["meters_per_px"])[selected],dtype=torch.float64)
    project=torch_project(world,cams,size,mpp).numpy()
    target=np.asarray(packet["uv"])[selected]
    conf=np.asarray(packet["confidence"])[selected].clip(0,1)
    valid=np.asarray(packet["valid"])[selected]
    error=np.linalg.norm(project-target,axis=-1)
    summary={}
    for i,v in enumerate(selected):
        has=valid[i]&(conf[i]>.001)
        if not np.any(has):
            raise ValueError("no observed joints in active camera "+str(v))
        summary[str(packet["view_ids"][v])]={
            "mean_px":float(np.mean(error[i][has])),
            "median_px":float(np.median(error[i][has])),
            "p95_px":float(np.percentile(error[i][has],95)),
            "weighted_mean_px":float(np.sum(error[i]*conf[i]*has)/
                                     max(np.sum(conf[i]*has),1e-8))
        }
    return summary


def fit_multiview(packet,info,*,view_ids,iterations=400,device="cuda",
                  lr_trans=.02,lr_pose=.035,prior_strength=.8,
                  temporal_strength=.03,verbose=True):
    """Optimize articulated FK using ONLY selected camera 2D observations.

    2-view and 4-view runs use identical 2D data, initial principles,
    priors and budget. Unselected camera data is not accessed in the loss.
    """
    names=[k["name"] for k in info["keypoints"]]
    meta=check_packet(packet,rig_names=names)
    views=meta["view_ids"]
    ids=np.array([views.index(v) for v in view_ids],dtype=int)
    if len(ids)<2 or len(set(ids.tolist()))!=len(ids):
        raise ValueError("at least 2 different Camera IDs required")
    if iterations<=0:raise ValueError("iterations must be positive")
    if any(not math.isfinite(x) or x<0 for x in
           (lr_trans,lr_pose,prior_strength,temporal_strength)) or lr_pose<=0 or lr_trans<=0:
        raise ValueError("invalid solver learning rates/prior")
    # Independent 2D least-squares triangulation is from chosen cameras only.
    tri,geom=triangulate(packet,index_views=ids)
    # Occluded/out-of-frame joints may be supported by only one view. They
    # cannot be triangulated, but FK can still place them from the Rig's rest
    # offsets and optimize against whatever 2D observations remain. Never
    # hallucinate a '3D GT' or use a hidden evaluation Side view as fallback.
    available=geom["triangulable"]
    root_name=info["bones"][0]["head_joint"]
    root_index=names.index(root_name)
    if not available[:,root_index].all() or available.sum()<.60*available.size:
        raise ValueError("too few rank-valid 2D observations to initialize Root / Rig")
    if not available.all():
        rest=np.asarray([k["rest"] for k in info["keypoints"]],dtype=float)
        translation=tri[:,root_index:root_index+1,:]-rest[root_index:root_index+1][None]
        rest_guess=rest[None,:,:]+translation
        tri=np.where(available[:,:,None],tri,rest_guess)
    device=torch.device(device)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    rig,translation0,rotation0=make_initial_rig(info,tri,geom["triangulable"],
                                                 device=device)
    T=meta["T"];B=len(info["bones"])
    dtype=torch.float64
    camera=torch.tensor(np.asarray(packet["world_to_camera"])[ids],
                        dtype=dtype,device=device)
    size=torch.tensor(np.asarray(packet["image_size_px"])[ids],dtype=dtype,device=device)
    meter=torch.tensor(np.asarray(packet["meters_per_px"])[ids],dtype=dtype,device=device)
    observed=torch.tensor(np.asarray(packet["uv"])[ids],dtype=dtype,device=device)
    raw_conf=torch.tensor(np.asarray(packet["confidence"])[ids],
                          dtype=dtype,device=device)
    valid=torch.tensor(np.asarray(packet["valid"])[ids],dtype=torch.bool,device=device)
    score=(raw_conf.clamp(0,1)**2)*valid.to(dtype)
    if score.sum()<10:raise ValueError("not enough trusted 2D joint observations")
    # Root observations are already triangulated and only mildly regularized.
    trans=torch.nn.Parameter(torch.as_tensor(translation0,dtype=dtype,device=device))
    root=torch.nn.Parameter(torch.as_tensor(rotation0,dtype=dtype,device=device))
    lo=rig.lo.clone();hi=rig.hi.clone()
    if not bool(torch.all(hi[1:]-lo[1:]>1e-6)):
        raise ValueError("invalid imported joint limits")
    mid=(lo+hi)*.5
    radius=(hi-lo)*.5
    # Euler angles limited to the imported rig's declared intervals;
    # for current Dino/Dragon these remain draft/approximate constraints.
    phi=torch.nn.Parameter(torch.zeros((T,B-1,3),dtype=dtype,device=device))
    zero_root=torch.zeros((T,1,3),dtype=dtype,device=device)
    optim=torch.optim.Adam([{"params":[trans],"lr":lr_trans},
                            {"params":[root],"lr":lr_pose},
                            {"params":[phi],"lr":lr_pose}])
    independent=meta["independent_stills"]
    tstart=time.perf_counter()
    best=float("inf")
    best_params=None
    grad_norms=[]
    for step in range(iterations):
        eul=torch.cat((zero_root,mid[1:]+radius[1:]*phi.tanh()),dim=1)
        joints,_=rig.forward(trans,axis_angle(root),eul)
        pred=torch_project(joints,camera,size,meter)
        residual=torch.linalg.vector_norm(pred-observed,dim=-1)
        # Huber-like robust loss, stable under occluded and OOD Reader peaks.
        rho=(torch.sqrt(36.0+residual.square())-6.)
        data=(rho*score).sum()/score.sum()
        pose_prior=eul[:,1:].square().mean()
        root_prior=root.square().mean()
        # P2 independent Blender poses are not consecutive video frames.
        if independent or T<3:
            temporal=torch.zeros((),dtype=dtype,device=device)
        else:
            acceleration=(joints[2:]-2*joints[1:-1]+joints[:-2])
            temporal=(acceleration.square().sum(-1).mean()/
                      meter.mean().square())
        loss=data + prior_strength*pose_prior + .03*root_prior + temporal_strength*temporal
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite N-view loss iter={step}")
        value=float(loss.detach())
        if value<best:
            best=value
            best_params=(trans.detach().clone(),root.detach().clone(),phi.detach().clone())
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(
            (trans,root,phi),max_norm=15.).detach()))
        optim.step()
        if verbose and (step==0 or (step+1)%100==0 or step==iterations-1):
            print("NVIEW",",".join(view_ids),"step",step+1,
                  "loss",round(value,4),"data",round(float(data.detach()),4),
                  "prior",round(float(pose_prior.detach()),5),
                  "seconds",round(time.perf_counter()-tstart,1),flush=True)
    with torch.no_grad():
        trans.copy_(best_params[0]);root.copy_(best_params[1]);phi.copy_(best_params[2])
        eul=torch.cat((zero_root,mid[1:]+radius[1:]*phi.tanh()),dim=1)
        J,_=rig.forward(trans,axis_angle(root),eul)
        output=J.detach().cpu().numpy()
        rootmat=axis_angle(root).detach().cpu().numpy()
        es=eul.detach().cpu().numpy()
        transout=trans.detach().cpu().numpy()
    if not np.isfinite(output).all():raise FloatingPointError("nonfinite optimized world")
    # Independent bone lengths are exact by using Rig.forward; verify no scale drift.
    tri_eval=np.linalg.norm(output-tri,axis=-1)
    obs_metrics=_serialize_world_metrics(output,packet,ids)
    # Output must have physically meaningful reprojectability and accurate Root FK.
    return {
        "world":output,"trans":transout,"euler":es,"root_rotmat":rootmat,
        "view_ids":view_ids,"best_loss":best,"iterations":iterations,
        "triangulation":tri,"triangulation_count":geom["triangulable_count"],
        "triangulation_vs_fit_mean_m":float(tri_eval.mean()),
        "per_view_reprojection":obs_metrics,
        "elapsed_seconds":time.perf_counter()-tstart,
        "gradient_norm_p95":float(np.percentile(grad_norms,95)),
        "mean_root_initial_rot_deg":float(np.mean(np.linalg.norm(rotation0,axis=-1))*180/np.pi),
        "independent_stills":independent
    }


def save_solution(prefix,sol,packet, *, source_packet):
    dest=Path(prefix)
    npz=dest.with_suffix(".npz");report=dest.with_suffix(".json")
    if npz.exists() or report.exists():
        raise FileExistsError("solver experiment output already exists: "+str(dest))
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(npz,
        world_joints=np.asarray(sol["world"],np.float64),
        trans=np.asarray(sol["trans"],np.float64),
        root_rotmat=np.asarray(sol["root_rotmat"],np.float64),
        euler=np.asarray(sol["euler"],np.float64),
        view_ids=np.asarray(sol["view_ids"]),
        names=np.asarray(packet["names"]),frame_ids=np.asarray(packet["frame_ids"]))
    manifest={
        "schema_version":"anymove.multiview_fk_solver_experimental/1",
        "source_packet":str(Path(source_packet).resolve()),
        "source_packet_sha256":hashlib.sha256(Path(source_packet).read_bytes()).hexdigest(),
        "active_views":sol["view_ids"],"frames":sol["world"].shape[0],
        "joint_count":sol["world"].shape[1],"iterations":sol["iterations"],
        "best_loss":sol["best_loss"],
        "per_view_reprojection":sol["per_view_reprojection"],
        "triangulation_vs_fit_mean_m":sol["triangulation_vs_fit_mean_m"],
        "triangulation_count":sol["triangulation_count"],
        "elapsed_seconds":sol["elapsed_seconds"],
        "independent_stills":sol["independent_stills"],
        "not_a_motion_clip":bool(sol["independent_stills"]),
        "not_3d_ground_truth":True,
        "does_not_read_ground_truth_or_surface_oracle":True,
        "no_change_to_production_2view_solver":True}
    report.write_text(json.dumps(manifest,indent=2),encoding="utf8")
    return npz,report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("packet")
    ap.add_argument("rig_info")
    ap.add_argument("--views",required=True)
    ap.add_argument("--out-prefix",required=True)
    ap.add_argument("--iterations",type=int,default=400)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr-trans",type=float,default=.02)
    ap.add_argument("--lr-pose",type=float,default=.035)
    ap.add_argument("--prior-strength",type=float,default=.8)
    ap.add_argument("--temporal-strength",type=float,default=.03)
    args=ap.parse_args()
    rig=json.loads(Path(args.rig_info).read_text(encoding="utf8"))
    p,meta=load_packet(args.packet,
        rig_names=[k["name"] for k in rig["keypoints"]])
    views=[v.strip() for v in args.views.split(",")]
    sol=fit_multiview(p,rig,view_ids=views,device=args.device,
            iterations=args.iterations,lr_trans=args.lr_trans,
            lr_pose=args.lr_pose,prior_strength=args.prior_strength,
            temporal_strength=args.temporal_strength)
    npz,report=save_solution(args.out_prefix,sol,p,source_packet=args.packet)
    print("SAVED",npz,report,flush=True)


if __name__=="__main__":
    main()
