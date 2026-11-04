"""Differentiable SO(3) angular-motion prior for AnyMove Human V3.

Preserve rotation-manifold topology: penalize excess local angular velocity
and acceleration, NOT raw XYZ Euler differences or absolute Root orientation.
All thresholds come from a train-only 24FPS Blender GT calibration.

Research-only; this implements calibrated statistical safety limits,
NOT a learned neural Motion Prior or human biomechanical motion capture.
"""
from __future__ import annotations

import math

import torch


def relative_rotvec(rotations:torch.Tensor)->torch.Tensor:
    """Rotation vectors of R[t].T @ R[t+1]; output (...,3).

    rotations: (T,...,3,3), T is the leading time dimension. Computation
    remains differentiable even near the identity; max 180deg motions aren't
    in this 24FPS procedural data and should be rejected upstream.
    """
    if rotations.ndim<3 or rotations.shape[-2:]!=(3,3):
        raise ValueError("Expected (T,...,3,3) rotations")
    if len(rotations)<2:
        return rotations.new_zeros((0,)+tuple(rotations.shape[1:-2])+(3,))
    rel=rotations[:-1].transpose(-1,-2) @ rotations[1:]
    vx=(rel[...,2,1]-rel[...,1,2])*.5
    vy=(rel[...,0,2]-rel[...,2,0])*.5
    vz=(rel[...,1,0]-rel[...,0,1])*.5
    vee=torch.stack((vx,vy,vz),dim=-1)
    sin=torch.sqrt(torch.sum(vee**2,dim=-1)+1e-16)
    cos=((torch.diagonal(rel,dim1=-2,dim2=-1).sum(-1)-1)*.5).clamp(-1,1)
    theta=torch.atan2(sin,cos)
    # Small-angle limit theta/sin(theta)->1; don't divide through zero.
    multiplier=torch.where(sin>1e-5,theta/sin,torch.ones_like(theta)+theta.square()/6)
    return vee*multiplier.unsqueeze(-1)


def angular_motion(rotations:torch.Tensor,dt:torch.Tensor):
    """Return angular velocity (T-1,...) and acceleration (T-2,...).

    Time intervals dt are measured in SOURCE FRAMES, not sample ordinal
    index. It is rotation-manifold aware, suitable for differentiable loss.
    """
    if dt.ndim!=1 or len(dt)!=max(0,len(rotations)-1) or bool(torch.any(~torch.isfinite(dt))) or bool(torch.any(dt<=0)):
        raise ValueError("Strictly positive finite source frame deltas required")
    relative=relative_rotvec(rotations)
    dt_shape=(len(dt),)+(1,)*(relative.ndim-1)
    vel=relative/dt.reshape(dt_shape)
    if len(rotations)<3:
        return vel,vel[:0]
    denom=(dt[:-1]+dt[1:]).reshape(
        (len(dt)-1,)+(1,)*(relative.ndim-1))
    acc=2*(vel[1:]-vel[:-1])/denom
    return vel,acc


def excess_angular_motion_loss(
        local_rotations:torch.Tensor,
        root_rotations:torch.Tensor,
        local_euler:torch.Tensor,
        source_frame_deltas:torch.Tensor,
        profile:dict):
    """Soft train-calibrated excess motion loss and its component breakdown.

    Input local_rotations shape (T, K, 3,3) with K the profile's selected
    anatomical spine bones. Root shape (T,3,3) in the Solver's local FK root
    basis. Euler (T,K,3) only for train-range violations, not for time diffs.

    The root may freely turn by 45-90deg: there is NO absolute heading prior.
    """
    if profile.get("schema")!="anymove.human_v3_so3_prior/1":
        raise ValueError("SO3 motion profile schema mismatch")
    if local_rotations.ndim!=4 or local_rotations.shape[-2:]!=(3,3):
        raise ValueError("Expected (T,K,3,3) selected bone rotations")
    T,K=local_rotations.shape[:2]
    if (root_rotations.shape!=(T,3,3)
        or local_euler.shape!=(T,K,3) or K!=len(profile["spine_bone_names"])):
        raise ValueError("SO3 prior joint channels/order mismatch")
    if T<3:raise ValueError("SO3 acceleration calibration requires 3 frames")
    if not (torch.isfinite(local_rotations).all() and
            torch.isfinite(root_rotations).all() and
            torch.isfinite(local_euler).all()):
        raise ValueError("Nonfinite rotation matrix/euler")

    v,a=angular_motion(local_rotations,source_frame_deltas)
    rv,ra=angular_motion(root_rotations,source_frame_deltas)
    dtype=local_euler.dtype
    dev=local_euler.device
    def limit(name,shape):
        x=torch.as_tensor(profile[name],dtype=dtype,device=dev)
        if x.shape!=shape or not torch.isfinite(x).all() or bool(torch.any(x<0)):
            raise ValueError("Invalid SO3 prior threshold "+name)
        return x
    def weight(name):
        v=float(profile["weights"].get(name,0))
        if not math.isfinite(v) or v<0 or v>1e6:
            raise ValueError("Invalid SO3 prior loss weight "+name)
        return v
    velocity_threshold=limit("bone_velocity_threshold_rad_per_frame",(K,))
    accel_threshold=limit("bone_acc_threshold_rad_per_frame2",(K,))
    root_v_limit=limit("root_velocity_threshold_rad_per_frame",())
    root_a_limit=limit("root_acc_threshold_rad_per_frame2",())
    axis_lower=torch.as_tensor(profile["bone_axis_lower_rad"],dtype=dtype,device=dev)
    axis_upper=torch.as_tensor(profile["bone_axis_upper_rad"],dtype=dtype,device=dev)
    if (axis_lower.shape!=(K,3) or axis_upper.shape!=(K,3)
        or torch.any(axis_lower>=axis_upper)
        or not torch.isfinite(axis_lower).all() or not torch.isfinite(axis_upper).all()):
        raise ValueError("Invalid train-derived local axis activity limits")
    bv=torch.linalg.norm(v,dim=-1)
    ba=torch.linalg.norm(a,dim=-1)
    rvel=torch.linalg.norm(rv,dim=-1)
    racc=torch.linalg.norm(ra,dim=-1)

    bone_vel=weight("bone_velocity")*torch.relu(bv-velocity_threshold[None]).square().sum()
    bone_acc=weight("bone_acceleration")*torch.relu(ba-accel_threshold[None]).square().sum()
    root_vel=weight("root_velocity")*torch.relu(rvel-root_v_limit).square().sum()
    root_acc=weight("root_acceleration")*torch.relu(racc-root_a_limit).square().sum()
    # This is an activity envelope, not a pose-matching loss: angles within
    # the Train-derived range incur NO penalty even during large turns.
    excess_axis=(torch.relu(axis_lower[None]-local_euler)+
                 torch.relu(local_euler-axis_upper[None]))
    axis_penalty=weight("bone_axis_range")*excess_axis.square().sum()
    terms={
        "bone_ang_vel":bone_vel,
        "bone_ang_acc":bone_acc,
        "root_ang_vel":root_vel,
        "root_ang_acc":root_acc,
        "bone_axis_range":axis_penalty,
    }
    total=sum(terms.values())
    return total,terms
