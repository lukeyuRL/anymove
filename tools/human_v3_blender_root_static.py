"""Human V3 Nose Blender pose writer: RL_BoneRoot must stay completely static.

The FK Solver models a global root rotation as B_root @ R_root. In Blender,
the old preview wrongly animated RL_BoneRoot.pose.rotation_quaternion. This
helper conjugates that same rotation through the root bone REST matrix and
bakes the resulting transform into the Armature OBJECT instead.

Invariant in world coordinates:
    T(delta) @ ArmatureRest @ RootBoneRest @ R
    ==
    [T(delta) @ ArmatureRest @ RootBoneRest @ R @ inverse(RootBoneRest)]
       @ RootBoneRest @ Identity

Every frame's world bone/joint positions are preserved. Root bone receives
NO pose or keyed channels; the Armature Object owns translation and rotation.
"""
import bpy
from mathutils import Matrix,Quaternion,Vector

ROOT_NAME="RL_BoneRoot"


def ensure_static_root(armature):
    roots=[bone for bone in armature.pose.bones if bone.parent is None]
    if len(roots)!=1 or roots[0].name!=ROOT_NAME:
        raise ValueError("Human V3 requires exactly one static RL_BoneRoot")
    root=roots[0]
    root.rotation_mode="XYZ"
    root.rotation_euler=(0.,0.,0.)
    root.location=(0.,0.,0.)
    root.scale=(1.,1.,1.)
    if any(abs(root.matrix_basis[i][j]-(1. if i==j else 0.))>1e-6
           for i in range(4) for j in range(4)):
        raise RuntimeError("RL_BoneRoot has non-rest transform")
    return root


def write_static_root_frame(armature,rest_world,trans_m,solver_root_wxyz,frame):
    """Key **Armature Object** root, not RL_BoneRoot pose bone.

    root quaternion is the Solver's *root-bone-local* rotation, which is
    conjugated by the Blender root bone's matrix_local. This is NOT a raw
    world quaternion; using it directly on the object can create an axis
    mismatch and bad Root translation.
    """
    root=ensure_static_root(armature)
    rest=root.bone.matrix_local.copy()
    q=Quaternion(tuple(float(v) for v in solver_root_wxyz))
    if sum(float(v)*float(v) for v in q)<1e-16:
        raise ValueError("Solver Root Quaternion cannot be zero")
    q.normalize()
    rotated=rest@q.to_matrix().to_4x4()@rest.inverted()
    armature.rotation_mode="QUATERNION"
    armature.matrix_world=Matrix.Translation(Vector(tuple(float(x) for x in trans_m)))@rest_world@rotated
    armature.keyframe_insert(data_path="location",frame=int(frame),group="AnyMove_ArmatureWorldRoot")
    armature.keyframe_insert(data_path="rotation_quaternion",frame=int(frame),group="AnyMove_ArmatureWorldRoot")
    # Do not key RL_BoneRoot. This bone exists in 55-bone FK hierarchy but
    # its pose remains identity at every source frame.
    ensure_static_root(armature)
    return root
