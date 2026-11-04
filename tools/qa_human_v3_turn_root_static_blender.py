"""Validate nontrivial 45-60° turns with a STATIC RL_BoneRoot in Blender.

Loads an existing Human V3 controlled-validation Solver motion with nonzero
root_quat and the original V3 Nose .blend. Reconstructs every world-space
joint through the Armature OBJECT, no root pose-bone animation. Saves a
separate 107-frame Blender Action for inspection. No H3, no iClone writes.
"""
import bpy, hashlib, json, pathlib, sys
import numpy as np
from mathutils import Matrix
ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools import render_human_v3_hr_v2_315_45 as scene_utils
from tools.human_v3_fullbody_motion import finger_tip_names,toe_tip_names
from tools.human_v3_blender_root_static import write_static_root_frame,ensure_static_root

SOURCE=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
RUN=ROOT/"runs/human/fullbody_finger_v2_hr_315_45"
MOTION=RUN/"solver/spine_so3_research/controlled_validation/spine_validation_pivot_turn_left_s405752.root_evidence_so3_600.motion.json"
DEST=RUN/"solver/spine_so3_research/root_static_blender_qa"/"pivot_turn_left_rootstatic_solver_107frames.blend"

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()

def world_joints(arm,names):
    result=[arm.matrix_world@arm.pose.bones[n].head for n in names[:55]]
    result.extend([arm.matrix_world@arm.pose.bones[n.replace("Tip","3")].tail for n in finger_tip_names()])
    result.extend([arm.matrix_world@arm.pose.bones[n.replace("ToeTip","ToeBase")].tail for n in toe_tip_names()])
    return np.asarray(result,dtype=np.float64)

def run():
    if pathlib.Path(bpy.data.filepath).resolve()!=SOURCE.resolve() or sha(SOURCE)!=scene_utils.SHA256:
        raise RuntimeError("Original Human V3 Nose asset/source mismatch")
    data=json.loads(MOTION.read_text(encoding="utf8"))
    if data.get("solver_camera_azimuth_deg")!=[315,45] or data["source_frame_indices"]!=list(range(107)):
        raise RuntimeError("Wrong turning rig motion/provenance")
    arm,mesh,rest,scene,cameras=scene_utils.studio()
    bones=[pb.name for pb in arm.pose.bones]
    names=bones+finger_tip_names()+toe_tip_names()
    if len(bones)!=55 or len(names)!=67 or data["joint_names"]!=names or data["bones"]!=bones:
        raise RuntimeError("Wrong Blender FK 55/67 marker rig")
    quat=np.asarray(data["root_quat"],dtype=np.float64)
    trans=np.asarray(data["trans"],dtype=np.float64)
    angles=np.asarray(data["euler"],dtype=np.float64)
    target=np.asarray(data["joints"],dtype=np.float64)
    if quat.shape!=(107,4) or trans.shape!=(107,3) or angles.shape!=(107,55,3) or target.shape!=(107,67,3):
        raise RuntimeError("Incorrect 107-frame motion channels")
    if np.max(np.linalg.norm(quat-quat[0:1],axis=-1))<.1:
        raise RuntimeError("No substantial world/root turning in this test clip")
    scene.frame_start=0
    scene.frame_end=106
    scene.render.fps=24
    scene["AnyMove_RL_BoneRoot_Rest_Pose"]=True
    scene["AnyMove_Root_Animation_Target"]="ArmatureObject"
    scene["AnyMove_StaticRoot_TurnQA_Source"]=str(MOTION)
    arm.animation_data_clear()
    maxerror=0.
    for f in range(107):
        scene.frame_set(f)
        write_static_root_frame(arm,rest,trans[f],quat[f],f)
        for j,bone in enumerate(arm.pose.bones):
            if bone.name=="RL_BoneRoot":
                continue
            bone.rotation_mode="XYZ"
            bone.rotation_euler=tuple(angles[f,j])
            bone.keyframe_insert(data_path="rotation_euler",frame=f,group=bone.name)
        bpy.context.view_layer.update()
        mismatch=float(np.linalg.norm(world_joints(arm,names)-target[f],axis=-1).max())
        maxerror=max(maxerror,mismatch)
        if maxerror>.0003:
            raise RuntimeError("Static bone Root object motion does NOT reproduce FK, m="+str(maxerror))
    scene.frame_set(0)
    ensure_static_root(arm)
    scene.camera=cameras["front"]
    DEST.parent.mkdir(parents=True,exist_ok=True)
    if DEST.exists():
        raise FileExistsError("Root-static turn scene exists, not overwritten")
    bpy.ops.wm.save_as_mainfile(filepath=str(DEST))
    print("TURNING_RL_BONEROOT_STATIC_BLENDER_EXPORT",json.dumps({
        "status":"PASS","frames":107,"fps":24,
        "file":str(DEST),"world_fk_max_delta_cm":maxerror*100,
        "root_quaternion_span":float(np.max(np.linalg.norm(quat-quat[0:1],axis=-1))),
        "RL_BoneRoot_has_pose_keys":False,
        "root_motion_owner":"ArmatureObject",
        "original_model_SHA_unchanged":sha(SOURCE)==scene_utils.SHA256
    }),flush=True)
if __name__=="__main__":
    run()
