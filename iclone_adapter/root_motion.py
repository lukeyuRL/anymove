"""Experimental Blender AnyMove root_quat / trans -> iClone Pelvis FK layer.

Rotation candidate uses the same Rest Basis conjugation as the visually
validated 41-joint Dragon, but at the root (pelvis) Rest orientation.
Translation: AnyMove trans is armature/world-axis displacement, but
blender/apply_motion.py specifically sets the Pelvis Pose Bone location to
inverse(R_rest_pelvis) @ trans. iClone Pelvis Position semantics are NOT yet
verified; offer both world/armature and Blender-pelvis-local hypotheses
without altering visually validated root or joint rotations. The iClone
unit scale remains explicitly user-specified. Never guess it.
"""
import math

from .anymove_bridge import qnormalize, qmul, qinverse_unit, to_euler_xyz
from .rest_basis import root_rest_basis
from .schema import ContractError


def _unwrap(previous,current):
    if previous is None:
        return current
    result=[]
    for prior,angle in zip(previous,current):
        while angle-prior>math.pi:
            angle-=2*math.pi
        while angle-prior< -math.pi:
            angle+=2*math.pi
        result.append(angle)
    return result


def _rotate_vector(q, v):
    """Apply unit quaternion q to xyz as R(q) @ vector (RH column basis)."""
    w,x,y,z=qnormalize(q)
    qvec=(x,y,z)
    cross=(y*v[2]-z*v[1],z*v[0]-x*v[2],x*v[1]-y*v[0])
    t=[2*a for a in cross]
    cross2=(y*t[2]-z*t[1],z*t[0]-x*t[2],x*t[1]-y*t[0])
    return [float(v[i])+w*t[i]+cross2[i] for i in range(3)]


def convert_pelvis_root(motion,info,target_bone_id,mode="rotation",
                        position_scale=None,position_space="armature"):
    if mode not in ("rotation","full"):
        raise ContractError("root mode requires rotation or full")
    if position_space not in ("armature","pelvis_local"):
        raise ContractError("root position_space must be armature or pelvis_local")
    if mode!="full" and position_space!="armature":
        raise ContractError("root position_space is only relevant with root mode full")
    if not isinstance(target_bone_id,str) or not target_bone_id:
        raise ContractError("root target bone ID required")
    frames=motion.get("euler")
    count=len(frames) if isinstance(frames,list) else 0
    quats=motion.get("root_quat")
    if not isinstance(quats,list) or len(quats)!=count or count<2:
        raise ContractError("source root_quat frames missing/mismatched")
    if mode=="full" and (type(position_scale) not in (int,float) or
                        not math.isfinite(position_scale) or
                        position_scale<=0 or position_scale>10000):
        raise ContractError("root Translation requires explicit positive --root-position-scale (experimental)")
    basis=root_rest_basis(info)
    inv=qinverse_unit(basis)
    angles=[]
    previous=None
    for value in quats:
        quat=qnormalize(value)
        converted=qmul(qmul(basis,quat),inv)
        xyz=_unwrap(previous,to_euler_xyz(converted))
        if not all(math.isfinite(v) for v in xyz):
            raise ContractError("nonfinite converted root Euler")
        angles.append(xyz)
        previous=xyz
    positions=None
    if mode=="full":
        trans=motion.get("trans")
        if not isinstance(trans,list) or len(trans)!=count:
            raise ContractError("source trans count must match root Quaternion count")
        positions=[]
        for values in trans:
            if not isinstance(values,list) or len(values)!=3 or not all(
                    type(v) in (int,float) and math.isfinite(float(v))
                    for v in values):
                raise ContractError("source root translation must be 3 finite values")
            local=(list(map(float,values)) if position_space=="armature" else
                   _rotate_vector(inv,list(map(float,values))))
            row=[float(v)*float(position_scale) for v in local]
            if any(abs(x)>1000000 for x in row):
                raise ContractError("root translation outside safe experimental magnitude")
            positions.append(row)
    metadata={
        "mode":"pelvis_layer",
        "components":("rotation_and_translation" if mode=="full" else
                      "rotation_only"),
        "source_root":"pelvis",
        "target_bone_id":target_bone_id,
        "root_rest_basis_wxyz":basis,
        "root_position_scale":(float(position_scale) if mode=="full" else None),
        "root_position_space":(position_space if mode=="full" else None),
        "rotation_rule":"Q_rest * Q_source_root * Q_rest^-1",
        "translation_rule":(
            ("source_armature_xyz * explicitly_configured_scale"
             if position_space=="armature" else
             "transpose(R_rest_pelvis) @ source_armature_xyz * "
             "explicitly_configured_scale")
            if mode=="full" else "excluded"),
        "calibrated":False,
        "warning":"EXPERIMENT ONLY: root Layer axis/position units are NOT runtime calibrated.",
    }
    return angles,positions,metadata
