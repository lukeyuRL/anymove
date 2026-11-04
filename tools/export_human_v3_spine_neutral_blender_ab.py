"""Blender 5 ground-truth vs experimental SO(3) Solver pivot-turn compare.

Exports TWO distinct 107-frame 24FPS Blender animations for the same
Pivot Turn Validation clip with 55 bones / 67 markers:
- procedural GT from newly audited Spine Motion Dataset
- frozen train-median SO(3) Solver (600 iterations)

BOTH keep RL_BoneRoot pose STATIC and unkeyed. Root motion is written ONLY
as Armature Object transform: GT world quaternion, Solver root-rest basis
conjugated through the validated writer. Replays each frame against source
GT/Solver 3D coordinates, without altering original V3 Nose Blender scene.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools import render_human_v3_hr_v2_315_45 as renderer
from tools import human_v3_fullbody_motion as motion
from tools.human_v3_blender_root_static import write_static_root_frame,ensure_static_root

BASE=ROOT/"runs/human/fullbody_finger_v2_hr_315_45"
SOURCE=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
SCORE=BASE/"solver/spine_so3_train_neutral_init_research"
OUT=SCORE/"blender_qa"
CLIP="spine_validation_pivot_turn_left_s405752"
GT=BASE/"spine_motion_prior_v1/clips/validation"/(CLIP+".npz")
RIG=BASE/"solver/rigged/rig_info_trainlimits.json"
RECON=SCORE/"pivot_turn_left_neutral_init_600.motion.json"
ROOT_SO3=BASE/"solver/spine_so3_research/controlled_validation/spine_validation_pivot_turn_left_s405752.root_evidence_so3_600.motion.json"
FRAMES=tuple(range(107))
SHA=renderer.SHA256

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):h.update(part)
    return h.hexdigest()

def points(arm,names):
    xyz=[arm.matrix_world@arm.pose.bones[n].head for n in names[:55]]
    xyz.extend([arm.matrix_world@arm.pose.bones[n.replace("Tip","3")].tail for n in motion.finger_tip_names()])
    xyz.extend([arm.matrix_world@arm.pose.bones[n.replace("ToeTip","ToeBase")].tail for n in motion.toe_tip_names()])
    return np.asarray(xyz,dtype=np.float64)

def export(mode,render_all=True):
    if mode not in ("gt","solver","root_so3"):raise ValueError("Unknown GT/Solver comparison mode")
    if pathlib.Path(bpy.data.filepath).resolve()!=SOURCE.resolve() or sha(SOURCE)!=SHA:
        raise RuntimeError("Wrong original V3 source Blend")
    arm,mesh,rest,scene,cameras=renderer.studio()
    names=renderer.marker_names(arm)
    if len(names)!=67 or len(arm.pose.bones)!=55 or names[0]!="RL_BoneRoot":
        raise RuntimeError("Human V3 body contract mismatch")
    if mode=="gt":
        with np.load(GT,allow_pickle=False) as f:
            times=np.asarray(f["frames"],dtype=int)
            eul=np.asarray(f["bone_euler_xyz_rad"],dtype=np.float64)
            trans=np.asarray(f["root_translation_m"],dtype=np.float64)
            root=np.asarray(f["root_rotation_wxyz"],dtype=np.float64)
            joints=np.asarray(f["world_joints_m"],dtype=np.float64)
        srcfile=GT
    else:
        src=RECON if mode=="solver" else ROOT_SO3
        m=json.loads(src.read_text(encoding="utf8"))
        if (m["joint_names"]!=names or m["bones"]!=names[:55]
            or m.get("solver_camera_azimuth_deg")!=[315,45]):
            raise RuntimeError("Wrong experiment Solver Human V3 provenance")
        times=np.asarray(m["source_frame_indices"],dtype=int)
        eul=np.asarray(m["euler"],dtype=np.float64)
        trans=np.asarray(m["trans"],dtype=np.float64)
        root=np.asarray(m["root_quat"],dtype=np.float64)
        joints=np.asarray(m["joints"],dtype=np.float64)
        srcfile=src
    if (times.tolist()!=list(FRAMES) or eul.shape!=(107,55,3)
        or trans.shape!=(107,3) or root.shape!=(107,4)
        or joints.shape!=(107,67,3) or np.max(np.abs(eul[:,0,:]))>1e-9):
        raise RuntimeError("Animated RL_BoneRoot or GT/Solver wrong channel shapes")

    arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.rotation_mode="XYZ"
    scene.frame_start=0
    scene.frame_end=106
    scene.render.fps=24
    scene.render.fps_base=1.
    scene["AnyMove_RL_BoneRoot_Rest_Pose"]=True
    scene["AnyMove_Root_Animation_Target"]="ArmatureObject"
    scene["AnyMove_GroundTruth_Or_SO3_Experiment"]=mode
    scene["AnyMove_GT_Clip"]=CLIP
    scene["AnyMove_SourceSHA256"]=SHA
    worst=0.
    for f in FRAMES:
        scene.frame_set(f)
        if mode=="solver":
            write_static_root_frame(arm,rest,trans[f],root[f],f)
        else:
            ensure_static_root(arm)
            q=Quaternion(tuple(root[f]))
            q.normalize()
            arm.rotation_mode="QUATERNION"
            arm.matrix_world=Matrix.Translation(tuple(trans[f]))@q.to_matrix().to_4x4()@rest
            arm.keyframe_insert(data_path="location",frame=f,group="AnyMove_WorldMotion")
            arm.keyframe_insert(data_path="rotation_quaternion",frame=f,group="AnyMove_WorldMotion")
        for b,pb in enumerate(arm.pose.bones):
            if pb.name=="RL_BoneRoot":continue
            pb.rotation_mode="XYZ"
            pb.rotation_euler=tuple(eul[f,b])
            pb.keyframe_insert(data_path="rotation_euler",frame=f,group=pb.name)
        bpy.context.view_layer.update()
        worst=max(worst,float(np.linalg.norm(points(arm,names)-joints[f],axis=-1).max()))
        if worst>.0003:
            raise RuntimeError("Blender FK replay differs at frame "+str(f)+", max_m="+str(worst))
    scene.frame_set(0)
    ensure_static_root(arm)
    scene.camera=cameras["front"]
    scene.render.filepath=""
    OUT.mkdir(parents=True,exist_ok=True)
    dst=OUT/(CLIP+"_"+mode+"_rl_boneroot_static.blend")
    if dst.exists():raise FileExistsError("Refusing overwrite earlier GT or Solver QA Blend")
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    if dst.stat().st_size<100000:raise RuntimeError("Empty saved 107-frame Blender project")
    new_files=0
    if render_all:
        for f in FRAMES:
            scene.frame_set(f)
            for slot,cam in cameras.items():
                scene.camera=cam
                path=OUT/"video_frames"/mode/slot/(f"frame_{f:04d}.png")
                path.parent.mkdir(parents=True,exist_ok=True)
                if renderer.png_size(path)!=(672,672):
                    scene.render.filepath=str(path)
                    bpy.ops.render.render(write_still=True)
                    new_files+=1
                if renderer.png_size(path)!=(672,672):
                    raise RuntimeError("Missing actual Blender render "+str(path))
            if f in (0,26,53,80,106):
                print("SPINE_NEUTRAL_BLENDER_PROGRESS",mode,"frame",f,flush=True)
    result={
        "status":"STATIC_RL_BONEROOT_FK_REPLAY_PASS",
        "mode":mode,"clip":CLIP,
        "fps":24,"frames":107,"marker_count":67,
        "blender_file":str(dst),"blender_file_sha256":sha(dst),
        "source_sha256":sha(srcfile),
        "original_model_sha256":sha(SOURCE),
        "max_world_joint_error_cm":worst*100,
        "no_root_bone_local_keys":True,
        "root_motion_animation_target":"ArmatureObject",
        "newly_rendered_images":new_files,
        "expected_total_rendered_images":214 if render_all else 0,
        "note":"Validation only; original global yaw kept. New Solver is not production/iClone accepted."
    }
    (OUT/(CLIP+"_"+mode+"_blender_export.json")).write_text(json.dumps(result,indent=2),encoding="utf8")
    print("SPINE_NEUTRAL_BLENDER_GT_SOLVER",json.dumps(result),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("gt","solver","root_so3"),required=True)
    p.add_argument("--no-full-render",action="store_true")
    a=p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    export(a.mode,render_all=not a.no_full_render)
