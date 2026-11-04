"""Opt-in Human V3 root orientation initialization from observable hip landmarks.

The original RL_BoneRoot has only ONE direct child, which makes the old
root Kabsch search degenerate and initializes all 107 frames at zero.
This code estimates the global Root orientation from observed hip/waist/
thigh-head positions. It converts the world rotation to the rig-root
local orientation used by mocap.solve.Rig.forward via rest-basis conjugacy.

Never uses reference GT at inference; only 3D triangulated Reader points,
the fixed rig rest positions and confidence. For Human V3 research only.
"""
import math

import numpy as np


ANCHORS=("CC_Base_Hip","CC_Base_Waist",
         "CC_Base_L_Thigh","CC_Base_R_Thigh")


def _orthonormal_factor(matrix):
    m=np.asarray(matrix,dtype=np.float64)
    if m.shape!=(3,3) or not np.isfinite(m).all():
        raise ValueError("Expected finite root rest basis (3x3)")
    U,S,Vt=np.linalg.svd(m)
    Q=U@np.diag([1.,1.,np.sign(np.linalg.det(U@Vt))])@Vt
    return Q


def rotation_log(R):
    """Stable SO(3) logarithm for rotations away from 180 degrees."""
    mat=np.asarray(R,dtype=np.float64)
    vee=np.array((mat[2,1]-mat[1,2],mat[0,2]-mat[2,0],
                  mat[1,0]-mat[0,1]))*.5
    sin=np.linalg.norm(vee)
    cos=np.clip((np.trace(mat)-1)*.5,-1.,1.)
    theta=np.arctan2(sin,cos)
    if sin<1e-8:
        return vee.copy()
    return vee*(theta/sin)


def estimate_human_v3_root_axis_angle(triangulated_xyz,rest_xyz,names,
                                      root_rest_basis,confidence=None,
                                      *,smoothing_window=3):
    """Output (T,3) axis-angle in ROOT-BONE-LOCAL reference coordinates.

    The input XYZ is the measured world-space triangulation. Neither
    reference motion GT nor future test labels are consulted.
    """
    xyz=np.asarray(triangulated_xyz,dtype=np.float64)
    rest=np.asarray(rest_xyz,dtype=np.float64)
    if (xyz.ndim!=3 or xyz.shape[-1]!=3 or rest.shape!=(len(names),3)
        or xyz.shape[1]!=len(names) or not np.isfinite(xyz).all()
        or not np.isfinite(rest).all()):
        raise ValueError("Invalid 3D observation/rest marker positions")
    if len(names)!=67 or len(set(names))!=67:
        raise ValueError("This is specifically the 67-marker Human V3 contract")
    indices=[names.index(n) for n in ANCHORS]
    if confidence is None:
        weights=np.ones((len(xyz),len(indices)),dtype=np.float64)
    else:
        c=np.asarray(confidence,dtype=np.float64)
        if c.shape!=(len(xyz),len(names)) or not np.isfinite(c).all():
            raise ValueError("Bad 2D confidence provenance")
        weights=c[:,indices].clip(.01,1.0)
    A=rest[indices]
    local_root_basis=_orthonormal_factor(root_rest_basis)
    raw=[]
    for t in range(len(xyz)):
        B=xyz[t,indices]
        w=weights[t]
        if np.sum(w)<.05:
            raw.append(raw[-1].copy() if raw else np.zeros(3))
            continue
        w=w/w.sum()
        ac=np.sum(A*w[:,None],axis=0)
        bc=np.sum(B*w[:,None],axis=0)
        pa=A-ac
        pb=B-bc
        H=(pa*w[:,None]).T@pb
        U,S,Vt=np.linalg.svd(H)
        if np.sum(S>1e-7)<2:
            raw.append(raw[-1].copy() if raw else np.zeros(3))
            continue
        world=Vt.T@np.diag([1.,1.,np.sign(np.linalg.det(Vt.T@U.T))])@U.T
        root_local=local_root_basis.T@world@local_root_basis
        axis=rotation_log(root_local)
        if not np.isfinite(axis).all():
            raise RuntimeError("Unstable Human root Kabsch result")
        raw.append(axis)
    raw=np.stack(raw)
    if smoothing_window not in (1,3,5):
        raise ValueError("Only 1/3/5-frame root smoothing is supported")
    if len(raw)<smoothing_window or smoothing_window==1:
        return raw
    radius=smoothing_window//2
    smooth=np.stack([
        np.median(raw[max(0,t-radius):min(len(raw),t+radius+1)],axis=0)
        for t in range(len(raw))])
    return smooth
