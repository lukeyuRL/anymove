"""P1 experimental Front-only IK on V2 BVH with AnyMove calibrated camera.

This is an ablation, not a replacement for mocap.solve production.
- Fit ONLY original front_uv/front_conf. Side is read *after* optimization for evaluation.
- Shared 67-bone BVH topology, frame count and Root track for both arms.
- Arm 1: static rest-pose rotation prior (no V2 motion).
- Arm 2: pretrained V2 rotation prior (uses same optimization, weights, and root).
- Exact FK bone lengths, explicit rotation limits and temporal regularization.
- No video generation. Export experimental BVHs separately.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot


def matrices_from_wxyz(quaternions):
    q=np.asarray(quaternions,dtype=np.float64)
    return SciRot.from_quat(q[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(*q.shape[:-1],3,3)


def finite_and_shape(x,shape,label):
    if tuple(x.shape)!=tuple(shape) or not np.isfinite(x).all():
        raise ValueError(f"{label} expected {shape} finite, got {x.shape}")


def euler_xyz(theta):
    """Intrinsic Blender XYZ in source-bone LOCAL coordinates: Rz @ Ry @ Rx."""
    x,y,z=theta.unbind(-1)
    cx,sx,cy,sy,cz,sz=x.cos(),x.sin(),y.cos(),y.sin(),z.cos(),z.sin()
    zero,one=torch.zeros_like(x),torch.ones_like(x)
    rx=torch.stack((one,zero,zero,zero,cx,-sx,zero,sx,cx),-1).reshape(*x.shape,3,3)
    ry=torch.stack((cy,zero,sy,zero,one,zero,-sy,zero,cy),-1).reshape(*x.shape,3,3)
    rz=torch.stack((cz,-sz,zero,sz,cz,zero,zero,zero,one),-1).reshape(*x.shape,3,3)
    return rz @ ry @ rx


def fk_torch(root_position,local_positions,parents,local_rotation):
    """BVH hierarchy: T_parent @ T_local; local translation BEFORE local rotation."""
    n=local_rotation.shape[1]
    rs=[None]*n
    ps=[None]*n
    for j,p in enumerate(parents):
        if p<0:
            rs[j]=local_rotation[:,j]
            ps[j]=root_position
        else:
            if p>=j:
                raise ValueError(f"BVH requires topological bone ordering (bone {j})")
            rs[j]=rs[p] @ local_rotation[:,j]
            ps[j]=ps[p] + (rs[p] @ local_positions[j].reshape(1,3,1)).squeeze(-1)
    return torch.stack(ps,dim=1)


def project_world(pos,layout):
    scale=layout["M_PER_PX"]
    return torch.stack((layout["W_FRONT"]/2+pos[...,0]/scale,
                        layout["H"]/2-(pos[...,2]-layout["Z_CENTER"])/scale),-1)


def eval_world(world,observation,layout,subset):
    from numpy import linalg
    p=world[:,subset,:]
    h=layout["H"];m=layout["M_PER_PX"]
    fp=np.stack((layout["W_FRONT"]/2+p[...,0]/m,
                 h/2-(p[...,2]-layout["Z_CENTER"])/m),-1)
    sp=np.stack((layout["W_SIDE"]/2-p[...,1]/m,
                 h/2-(p[...,2]-layout["Z_CENTER"])/m),-1)
    f=linalg.norm(fp-observation["front_uv"],axis=-1)
    s=linalg.norm(sp-observation["side_uv"],axis=-1)
    valid=np.isfinite(f)&np.isfinite(s)
    if not valid.all():raise ValueError("non-finite reprojection")
    def metric(x):
        return {"mean_px":round(float(x.mean()),4),
                "median_px":round(float(np.median(x)),4),
                "p95_px":round(float(np.percentile(x,95)),4),
                "max_px":round(float(x.max()),4)}
    return {"front":metric(f),"heldout_side":metric(s)}, f, s


def prepare(args):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,str(Path(args.v2_repo).resolve()))
    from utils import bvh as BVH,animation
    from tools.mocapanything_v2_root_bridge import similarity_from_rest
    info=json.loads(Path(args.rig_info).read_text(encoding="utf-8"))
    bones=info["bones"];names=[x["name"] for x in bones]
    layout=info["layout"]
    assert names[0]=="CC_Dinosaur_Hip", "P1 requires known Dino Root for this test"
    orig,original_names,dt=BVH.load(args.root_bvh)
    ref,ref_names,_=BVH.load(args.reference_bvh)
    if set(original_names)!=set(names) or set(ref_names)!=set(names):
        raise ValueError("rig/BVH joint set mismatch")
    if not np.array_equal(orig.parents,ref.parents):
        raise ValueError("V2 BVH and rest hierarchy differ")
    if not np.allclose(orig.offsets,ref.offsets,atol=2e-3):
        raise ValueError("V2 BVH offset differs from reference")
    T=len(orig); J=len(names)
    reader=np.load(args.detections)
    if len(reader["front_uv"])!=T or abs(float(reader["fps"])-1/dt)>.02:
        raise ValueError("video/detection frame or FPS mismatch")
    keypoint_names=[str(x) for x in reader["names"]]
    join=np.array([original_names.index(n) for n in names])
    obs_idx=[keypoint_names.index(b["head_joint"]) for b in bones]
    front=np.asarray(reader["front_uv"][:,obs_idx,:],dtype=np.float64)
    side=np.asarray(reader["side_uv"][:,obs_idx,:],dtype=np.float64)
    fw=np.clip(np.asarray(reader["front_conf"][:,obs_idx],dtype=np.float64),0,1)
    sw=np.clip(np.asarray(reader["side_conf"][:,obs_idx],dtype=np.float64),0,1)
    if not np.isfinite(front).all() or not np.isfinite(fw).all():
        raise ValueError("invalid Front observations")
    # Side is stored for posthoc metrics; optimize() never receives or reads those values.
    restworld=np.stack([np.asarray(b["matrix_local"])[:3,3] for b in bones])
    ref_pose=animation.positions_global(ref)[0, [ref_names.index(n) for n in names]]
    scale,basis,offset,err=similarity_from_rest(ref_pose,restworld)
    if np.max(err)>0.002 or not (0.98<=scale<=1.02):
        raise ValueError("reference Rest pose mismatch")
    v2_rot=matrices_from_wxyz(np.asarray(orig.rotations))
    rest_rot=matrices_from_wxyz(np.asarray(ref.rotations)[0])
    localpos=np.asarray(orig.positions,dtype=np.float64)
    if not np.allclose(localpos[:,1:],orig.offsets[None,1:],atol=2e-5):
        raise ValueError("unsupported animated nonroot translations")
    active=np.array([i for i in range(J) if np.any(orig.parents==i)],dtype=int)
    # All 67 bones have named bone-head observations; leaves have no observable rotation
    # (their head moves with parents), and are frozen at each arm's anchor.
    result=dict(original_bvh=orig,original_names=original_names,rig_names=names,
      T=T,J=J,frame_interval=dt,layout=layout,active=active,
      root_positions=localpos[:,0],joint_offsets=orig.offsets,
      v2_rot=v2_rot,rest_rot=rest_rot,scale=scale,basis=basis,reference_error=float(err.max()),
      initial_world_fk=np.asarray(animation.positions_global(orig)),frames=np.arange(T),
      obs={"front_uv":front,"side_uv":side,"front_conf":fw,"side_conf":sw},
      mapping=join,rig=info,reader_path=Path(args.detections))
    return result


def optimize(case,anchor_mode,steps,prior_strength,temporal_strength,lr,device):
    """All gradient inputs originate only from Front. Side not accessed."""
    T,J=case["T"],case["J"]
    dtype=torch.float32
    dev=torch.device(device)
    anchor_rot=(case["v2_rot"] if anchor_mode=="v2" else
                np.broadcast_to(case["rest_rot"],(T,J,3,3))).copy()
    base=torch.as_tensor(anchor_rot,dtype=dtype,device=dev)
    offsets=torch.as_tensor(case["joint_offsets"],dtype=dtype,device=dev)
    root=torch.as_tensor(case["root_positions"],dtype=dtype,device=dev)
    basis=torch.as_tensor(case["basis"],dtype=dtype,device=dev)
    obs=torch.as_tensor(case["obs"]["front_uv"],dtype=dtype,device=dev)
    conf=torch.as_tensor(case["obs"]["front_conf"],dtype=dtype,device=dev)
    conf=(conf.clamp(0,1)**2)
    front_scale=float(case["layout"]["M_PER_PX"])
    act=torch.as_tensor(case["active"],dtype=torch.long,device=dev)
    parents=case["original_bvh"].parents.astype(int).tolist()
    delta_max=float(np.deg2rad(48.0))
    theta=torch.nn.Parameter(torch.zeros(T,len(act),3,device=dev,dtype=dtype))
    optimizer=torch.optim.Adam([theta],lr=lr)
    last_loss=float("nan")
    best=float("inf")
    best_theta=None
    t0=time.perf_counter()
    # lambda values are exploratory; checked on held-out Side later.
    def forward():
        corr=delta_max*theta.tanh()
        raw_delta=torch.zeros((T,J,3),dtype=dtype,device=dev)
        raw_delta[:,act,:]=corr
        rr=base@euler_xyz(raw_delta)
        pk=fk_torch(root,offsets,parents,rr)
        world=(pk@basis)*float(case["scale"])
        return rr,world,corr
    for i in range(steps):
        rr,world,corr=forward()
        fitted=project_world(world[:,case["mapping"]],case["layout"])
        error=torch.sqrt(((fitted-obs)**2).sum(-1)+1e-5)
        # sqrt(6^2+r^2)-6 robust to 2D outliers; use raw Front confidences.
        robust=(torch.sqrt(36.0+error.square())-6)
        data_loss=(robust*conf).sum()/conf.sum().clamp(min=1)
        prior_loss=corr.square().sum(dim=-1).mean()
        # Temporal regularization directly on predicted joint acceleration (layout px).
        rel_world=world-world[:,0:1,:]
        accel=((rel_world[2:]-2*rel_world[1:-1]+rel_world[:-2])/front_scale)
        acc_loss=accel.square().sum(dim=-1).mean()
        angle_velocity=(corr[1:]-corr[:-1]).square().sum(dim=-1).mean()
        # Limits are draft +/-20deg, so do not treat them as anatomical hard truth.
        # Penalize angular corrections exceeding 20deg without forcing to this loose draft limit.
        max_local=float(np.deg2rad(20))
        limit_loss=(corr.abs()-max_local).clamp(min=0).square().mean()
        loss=data_loss + prior_strength*prior_loss + temporal_strength*acc_loss + 3.0*angle_velocity + 3.0*limit_loss
        if not torch.isfinite(loss):raise FloatingPointError(f"nonfinite loss at step {i}")
        value=float(loss.detach().cpu())
        if value<best:
            best=value;best_theta=theta.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta],max_norm=10.0)
        optimizer.step()
        if i==0 or (i+1)%40==0 or i+1==steps:
            print(f"[{anchor_mode}] {i+1}/{steps} loss={value:.4f} front_huber={data_loss.detach().item():.3f} "
                  f"prior={prior_loss.detach().item():.4f} accel={acc_loss.detach().item():.4f} elapsed={time.perf_counter()-t0:.1f}s",
                  flush=True)
    with torch.no_grad():
        if best_theta is not None:theta.copy_(best_theta)
        rr,world,corr=forward()
        out_rot=rr.detach().cpu().numpy().astype(np.float64)
        out_world=world.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(out_world).all():raise FloatingPointError("nonfinite 3D motion")
        # Bone-length contract: between parent and child head, no stretch from FK.
        parent=case["original_bvh"].parents.astype(int)
        bad=[]
        for child,p in enumerate(parent):
            if p<0:continue
            pred_dist=np.linalg.norm(out_world[:,child]-out_world[:,p],axis=-1)
            rig_length=case["scale"]*np.linalg.norm(case["joint_offsets"][child])
            worst=float(np.max(np.abs(pred_dist-rig_length)))
            if worst>0.00003:bad.append((child,worst))
        if bad:raise AssertionError(f"bone stretch in FK {bad[:5]}")
        # Correction absolute-angle bound from parameterization.
        max_bound=float(corr.abs().max().cpu())
        if max_bound>delta_max+1e-6:raise AssertionError("correction angle bound broken")
    return {"anchor":anchor_mode,"rot_matrices":out_rot,"world":out_world,
            "elapsed_s":time.perf_counter()-t0,"best_loss":best,
            "steps":steps,"prior_strength":prior_strength,"temporal_strength":temporal_strength,
            "max_abs_correction_deg":float(np.degrees(max_bound)),
            "rest_default_limit_note":"draft 20deg/axis, NOT a verified anatomical constraint"}


def bvh_export(case,result,path):
    """Export experiment BVH only, never source or AnyMove production motion.json."""
    from utils import bvh as BVH,animation
    orig=case["original_bvh"]
    rot=result["rot_matrices"]
    t,j=rot.shape[:2]
    angles=SciRot.from_matrix(rot.reshape(-1,3,3)).as_euler("ZYX",degrees=True).reshape(t,j,3)
    new=dict(rotations=angles,positions=np.asarray(orig.positions).copy(),
             offsets=np.asarray(orig.offsets).copy(),parents=np.asarray(orig.parents).copy(),
             names=list(case["original_names"]),order="zyx",frametime=case["frame_interval"])
    path.parent.mkdir(parents=True,exist_ok=True)
    BVH.save_dict(filename=str(path),data=new,fps=1.0/case["frame_interval"])
    saved,n,dt=BVH.load(str(path))
    if n!=list(case["original_names"]) or len(saved)!=t:raise AssertionError("BVH export topology error")
    if abs(dt-case["frame_interval"])>1e-7:raise AssertionError("BVH fps changed")
    expected=result["world"]
    actual=animation.positions_global(saved)
    actual_world=actual@case["basis"]*case["scale"]
    err=float(np.linalg.norm(actual_world-expected,axis=-1).max())
    if err>.001:raise AssertionError(f"BVH-FK roundtrip error {err}")
    return err


def world_fk_check(case):
    from utils import animation
    T,J=case["T"],case["J"]
    with torch.no_grad():
        R=torch.as_tensor(case["v2_rot"],dtype=torch.float64)
        position=fk_torch(torch.as_tensor(case["root_positions"],dtype=torch.float64),
                          torch.as_tensor(case["joint_offsets"],dtype=torch.float64),
                          case["original_bvh"].parents.astype(int),R).numpy()
    native=animation.positions_global(case["original_bvh"])
    error=float(np.abs(position-native).max())
    if error>1e-5:raise AssertionError(f"BVH FK mismatch {error}")
    return error


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--v2-repo",default="F:/mocapanything_v2")
    p.add_argument("--rig-info",default="F:/anymove/assets/dino/rigged/rig_info.json")
    p.add_argument("--reference-bvh",default="F:/mocapanything_v2/zoo/characters_face_zplus/Dino/Dino_ffs.bvh")
    p.add_argument("--root-bvh",default="F:/mocapanything_v2_test/Dino_V2_front_root_124f.bvh")
    p.add_argument("--detections",default="F:/anymove/runs/dino/clips/test_walk/det.npz")
    p.add_argument("--out-dir",default="F:/mocapanything_v2_test/p1_front_only")
    p.add_argument("--steps",type=int,default=160)
    p.add_argument("--prior",type=float,default=12.0)
    p.add_argument("--temporal",type=float,default=0.035)
    p.add_argument("--lr",type=float,default=.045)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke",action="store_true",help="Run FK and contracts without optimization")
    p.add_argument("--arm",choices=("both","v2","rest"),default="both")
    args=p.parse_args()
    if args.steps<0 or args.prior<0 or args.temporal<0:raise ValueError("invalid solver args")
    torch.set_num_threads(min(torch.get_num_threads(),4))
    case=prepare(args)
    error=world_fk_check(case)
    print("CASE",json.dumps({"frames":case["T"],"joints":case["J"],"active_joint_count":len(case["active"]),
        "reference_mismatch_m":case["reference_error"],"fk_numpy_torch_max_error_m":error,
        "front_observations_only":True,"device":args.device}))
    if args.smoke:return
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    subset=case["mapping"]
    # For strict phase separation, optimization has *no* Side array argument.
    obs_eval=case["obs"]
    baseline_world=case["initial_world_fk"]@case["basis"]*case["scale"]
    current_metrics,_,_=eval_world(baseline_world,obs_eval,case["layout"],subset)
    reports={"label":"AnyMove P1 V2 prior ablation (NOT production)","T":case["T"],"J":case["J"],
             "bones_fitted":len(case["active"]),"device":args.device,
             "front_only_optimization":True,"heldout_side_used_for_fitting":False,
             "original_v2_metrics":current_metrics,
             "inputs":{"bvh":args.root_bvh,"rig":args.rig_info,"detections":args.detections},
             "config":{"steps":args.steps,"lr":args.lr,"prior":args.prior,"temporal":args.temporal},
             "results":{}}
    branches=["rest","v2"] if args.arm=="both" else [args.arm]
    for branch in branches:
        sol=optimize(case,branch,args.steps,args.prior,args.temporal,args.lr,args.device)
        metrics,f,s=eval_world(sol["world"],obs_eval,case["layout"],subset)
        export=out/f"Dino_P1_{branch}_front_only.bvh"
        validation=bvh_export(case,sol,export)
        v2_accuracy={"front":metrics["front"]["mean_px"],
                     "side":metrics["heldout_side"]["mean_px"]}
        reports["results"][branch]={"metrics":metrics,"duration_s":round(sol["elapsed_s"],2),
                     "best_loss":round(sol["best_loss"],6),
                     "max_abs_correction_deg":round(sol["max_abs_correction_deg"],3),
                     "export_bvh":str(export),"bvh_roundtrip_fk_max_m":validation}
        print("BRANCH",branch,json.dumps(reports["results"][branch],indent=2),flush=True)
    if "rest" in reports["results"] and "v2" in reports["results"]:
        r=reports["results"]["rest"]["metrics"];v=reports["results"]["v2"]["metrics"]
        reports["comparison_vs_front_only_baseline"]={
            "front_mean_v2_minus_rest_px":round(v["front"]["mean_px"]-r["front"]["mean_px"],4),
            "heldout_side_mean_v2_minus_rest_px":round(v["heldout_side"]["mean_px"]-r["heldout_side"]["mean_px"],4),
            "improves_both":v["front"]["mean_px"]<r["front"]["mean_px"] and
                v["heldout_side"]["mean_px"]<r["heldout_side"]["mean_px"]}
    source_sha=hashlib.sha256(Path(args.detections).read_bytes()).hexdigest()
    reports["detections_sha256"]=source_sha
    (out/"p1_report.json").write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding="utf8")
    print("REPORT",out/"p1_report.json",flush=True)


if __name__=="__main__":main()
