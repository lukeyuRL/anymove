"""Strict Blender Bone rest quaternion conversion, relative to its parent.

Used by the *experiment-only* Parent/Child test and full Dragon preview.
These quaternions are hypotheses for mapping Blender Pose Euler XYZ to
iClone FK channels; they do not by themselves certify an SDK convention.
"""
import math

from .anymove_bridge import qnormalize
from .schema import ContractError, matrix, relative


def _quat_from_rest_matrix(rest):
    """Quaternion for positive-handed, near-orthogonal 3x3 rest rotation."""
    r = [[float(rest[i*4+j]) for j in range(3)] for i in range(3)]
    # The input is from Blender Bone.matrix_local: SHOULD be pure rotation.
    # Reject shear/mirror/scale rather than constructing an invalid basis.
    for c in range(3):
        length=sum(r[i][c]**2 for i in range(3))
        if abs(length-1.)>0.01:
            raise ContractError("relative Blender Bone rest matrix is scaled")
        for c2 in range(c):
            dot=sum(r[i][c]*r[i][c2] for i in range(3))
            if abs(dot)>0.01:
                raise ContractError("relative Blender Bone rest matrix is sheared")
    det=(r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
         -r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
         +r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if abs(det-1.)>0.01:
        raise ContractError("relative Blender Bone rest matrix is mirrored")
    trace=r[0][0]+r[1][1]+r[2][2]
    if trace>0:
        s=2.*math.sqrt(max(1e-12,trace+1.))
        q=[.25*s,(r[2][1]-r[1][2])/s,(r[0][2]-r[2][0])/s,
           (r[1][0]-r[0][1])/s]
    else:
        i=max(range(3),key=lambda c:r[c][c])
        j=(i+1)%3
        k=(i+2)%3
        s=2.*math.sqrt(max(1e-12,1.+r[i][i]-r[j][j]-r[k][k]))
        v=[0.,0.,0.,0.]
        v[i+1]=.25*s
        v[0]=(r[k][j]-r[j][k])/s
        v[j+1]=(r[j][i]+r[i][j])/s
        v[k+1]=(r[k][i]+r[i][k])/s
        q=v
    return qnormalize(q)


def relative_rest_basis(info, bone_name):
    """Return parent-local Bone rest quaternion, not armature-world basis."""
    bones=info.get("bones")
    if not isinstance(bones,list):
        raise ContractError("invalid source rig_info bones")
    names=[b.get("name") for b in bones]
    if names.count(bone_name)!=1:
        raise ContractError("source rig Bone missing or duplicated: "+bone_name)
    b=bones[names.index(bone_name)]
    p=b.get("parent")
    if type(p) is not int or p<0 or p>=len(bones):
        raise ContractError("probe requires a non-root bone with parent")
    current=matrix([x for row in b["matrix_local"] for x in row])
    parent=matrix([x for row in bones[p]["matrix_local"] for x in row])
    return _quat_from_rest_matrix(relative(parent,current))



def root_rest_basis(info, name="pelvis"):
    """Root rest Bone quaternion relative to its armature object (no parent Bone)."""
    bones=info.get("bones")
    if not isinstance(bones,list):
        raise ContractError("invalid source root info")
    matches=[b for b in bones if b.get("name")==name]
    if len(matches)!=1 or matches[0].get("parent")!=-1:
        raise ContractError("cannot find unique top-level Blender root Bone "+name)
    return _quat_from_rest_matrix(
        matrix([x for row in matches[0]["matrix_local"] for x in row]))
