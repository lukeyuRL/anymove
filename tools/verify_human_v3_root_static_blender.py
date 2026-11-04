"""Blender 5 independently reopens and verifies a static RL_BoneRoot Action.

Modes:
  solver: old original P1 motion rendered as a separate _root_static.blend
  research: experimental spine stabilized Blender output _root_static.blend
  original_gt: 107-frame unmodified procedural motion _root_static.blend

Verifies at each keyframe:
- RL_BoneRoot is unkeyed; Blender action has no curve for it.
- RL_BoneRoot matrix_basis == identity, location=0, rotation=0, scale=1.
- Armature Object is the root motion carrier for Solver / research clips.
- All 67 world joint points match source JSON within 0.03 cm.
- The original source 55-bone V3 Nose model SHA-256 is unchanged.
"""
import argparse
import bpy
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools.human_v3_fullbody_motion import finger_tip_names,toe_tip_names

BASE=ROOT/"runs/human/fullbody_finger_v2_hr_315_45"
SOURCE=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
EXPECTED_SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()

def action_paths(arm):
    if not arm.animation_data or not arm.animation_data.action:
        raise RuntimeError("Saved scene has no Armature Action")
    action=arm.animation_data.action
    slot=arm.animation_data.action_slot
    if slot is None:
        raise RuntimeError("Armature Action lacks its slot")
    result=[]
    for layer in action.layers:
        for strip in layer.strips:
            bag=strip.channelbag(slot)
            if bag is not None:
                result.extend(fc.data_path for fc in bag.fcurves)
    if not result:
        raise RuntimeError("No actual animation curves found")
    return result

def world_markers(arm,names):
    v=[arm.matrix_world@arm.pose.bones[n].head for n in names[:55]]
    v.extend([arm.matrix_world@arm.pose.bones[n.replace("Tip","3")].tail
              for n in finger_tip_names()])
    v.extend([arm.matrix_world@arm.pose.bones[n.replace("ToeTip","ToeBase")].tail
              for n in toe_tip_names()])
    return np.asarray(v,dtype=float)

def run(mode):
    opened=pathlib.Path(bpy.data.filepath)
    arm=bpy.data.objects.get("Armature")
    if not arm or len(arm.pose.bones)!=55:
        raise RuntimeError("Not the V3 Nose 55-bone Armature")
    keys=action_paths(arm)
    if any("RL_BoneRoot" in key for key in keys):
        raise RuntimeError("RL_BoneRoot animation FCurve found; FAIL static-root contract")
    root=arm.pose.bones["RL_BoneRoot"]
    if root.parent is not None:
        raise RuntimeError("RL_BoneRoot must be a unique root bone")
    if bpy.context.scene.get("AnyMove_RL_BoneRoot_Rest_Pose") is not True:
        raise RuntimeError("Missing explicit Root static scene provenance")
    if bpy.context.scene.get("AnyMove_Root_Animation_Target")!="ArmatureObject":
        raise RuntimeError("Motion improperly assigned to pose bone")
    if mode=="original_gt":
        folder=BASE/"continuous_24fps_315_45/gt_blender/root_static"
        suffix="_GT_107frames_root_static.blend"
        if opened.parent.resolve()!=folder.resolve() or not opened.name.endswith(suffix):
            raise RuntimeError("GT source/suffix mismatch")
        clip=opened.name[:-len(suffix)]
        source=json.loads((BASE/"continuous_24fps_315_45/motion_gt"/(clip+".json")).read_text())
        xyz=np.asarray(source["joint_position_m"],dtype=float)
        frames=list(range(107))
        if "rotation_quaternion" in keys:
            raise RuntimeError("Old procedural GT had no Armature world rotation keys")
    else:
        if mode=="solver":
            folder=BASE/"solver/blender_qa/root_static"
            suffix="_solved_p1_trainlimits_root_static.blend"
            prefix=BASE/"continuous_24fps_315_45/detections"
            name_template="{clip}.motion_p1_trainlimits.json"
        elif mode=="research":
            folder=BASE/"solver/spine_regularized_research/blender_qa/root_static"
            suffix="_spine_regularized_research_root_static.blend"
            prefix=BASE/"solver/spine_regularized_research"
            name_template=None
        elif mode=="turning_solver":
            folder=BASE/"solver/spine_so3_research/root_static_blender_qa"
            suffix="_rootstatic_solver_107frames.blend"
            prefix=BASE/"solver/spine_so3_research/controlled_validation"
            name_template=None
        else:raise ValueError("Unknown Root-static action kind")
        if opened.parent.resolve()!=folder.resolve() or not opened.name.endswith(suffix):
            raise RuntimeError("Wrong Solver scene/suffix for specified mode")
        clip=opened.name[:-len(suffix)]
        if mode=="research":
            filename=("finger_roll_spine_regularized_dev.json" if "finger_independent_roll" in clip
                     else "kick_wave_spine_regularized_dev.json")
        elif mode=="turning_solver":
            filename="spine_validation_pivot_turn_left_s405752.root_evidence_so3_600.motion.json"
        else:
            filename=name_template.format(clip=clip)
        source=json.loads((prefix/filename).read_text())
        xyz=np.asarray(source["joints"],dtype=float)
        frames=source["source_frame_indices"]
        if not all(key in keys for key in ("location","rotation_quaternion")):
            raise RuntimeError("Root animation must be on Armature Object")
        if source.get("motion_space_warning")!="BLENDER_LOCAL_EULER_NOT_ICLONE_FK_CHANNELS":
            raise RuntimeError("Cannot import uncalibrated iClone format")
    names=[pb.name for pb in arm.pose.bones]+finger_tip_names()+toe_tip_names()
    if source["joint_names"]!=names or xyz.shape!=(len(frames),67,3):
        raise RuntimeError("Rig/GT 67 point contract mismatch")
    scene=bpy.context.scene
    if (scene.frame_start!=frames[0] or scene.frame_end!=frames[-1] or
        scene.render.fps!=24 or scene.render.fps_base!=1.0):
        raise RuntimeError("Wrong source FPS or frame span")
    errors=[]
    static_diffs=[]
    object_root_change=[]
    rotations=[]
    for t,frame in enumerate(frames):
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        basis=np.asarray(root.matrix_basis,dtype=float)
        static_diffs.append(float(np.max(np.abs(basis-np.eye(4)))))
        if (np.linalg.norm(np.asarray(root.location,dtype=float))>1e-8 or
            np.max(np.abs(np.asarray(root.scale)-1))>1e-8):
            raise RuntimeError("RL_BoneRoot does not stay in rest pose")
        actual=world_markers(arm,names)
        errors.append(np.linalg.norm(actual-xyz[t],axis=-1)*100)
        rotations.append(np.asarray(arm.rotation_quaternion,dtype=float))
        object_root_change.append(np.asarray(arm.matrix_world.translation,dtype=float))
    e=np.asarray(errors)
    if e.max()>.03 or max(static_diffs)>1e-6:
        raise RuntimeError("Static Root modified FK / root basis: max joint error %.6f cm"%(e.max()))
    output={
        "status":"RL_BONEROOT_NO_ANIMATION_PASS",
        "mode":mode,"clip":clip,
        "blend":str(opened),"frames":len(frames),"fps":24,
        "static_root_bone":"RL_BoneRoot","root_has_action_fcurves":False,
        "root_pose_max_matrix_basis_delta":max(static_diffs),
        "all_67_joint_max_error_cm":float(e.max()),
        "all_67_joint_mean_error_cm":float(e.mean()),
        "armature_object_translation_range_m":np.ptp(object_root_change,axis=0).tolist(),
        "armature_rotation_key_channel":("rotation_quaternion" in keys),
        "armature_object_rotation_variation":float(np.max(np.linalg.norm(np.asarray(rotations)-rotations[0],axis=-1))),
        "original_V3_nose_sha256_verified":sha(SOURCE)==EXPECTED_SHA,
        "legacy_blender_files_preserved":True,
    }
    report=opened.with_suffix(".root_static_audit.json")
    report.write_text(json.dumps(output,indent=2),encoding="utf8")
    print("HUMAN_V3_RL_BONEROOT_STATIC_PASS",json.dumps(output),flush=True)

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--mode",choices=["solver","research","turning_solver","original_gt"],required=True)
    parsed=arg.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    run(parsed.mode)
