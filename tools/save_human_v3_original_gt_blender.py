"""Save ORIGINAL procedural Human V3 Nose GT as animated Blender 5 files.

Uses the exact frozen V2 procedural motion and GT used to render the
315/45 continuous test. Creates a new .blend per clip; never overwrites
the unanimated source V3 Nose .blend or the Solver animations.
Includes *all 107* frames 0..106 at 24 FPS and all 55 pose-bone channels.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import bpy
import numpy as np

ROOT=pathlib.Path("F:/anymove")
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools import render_human_v3_hr_v2_315_45 as source
from tools import render_human_v3_fullbody_finger_ab as prior
from tools import human_v3_fullbody_motion_v2 as movement

BASE=ROOT/"runs/human/fullbody_finger_v2_hr_315_45/continuous_24fps_315_45"
ORIGINAL=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
OUTPUT=BASE/"gt_blender"
CLIPS=(
 "hrv2_continuous_finger_independent_roll_s305007",
 "hrv2_continuous_kick_and_wave_s306019",
)

def sha256(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()

def build_one(clip):
    if clip not in CLIPS:raise ValueError("Unknown continuous GT clip")
    if pathlib.Path(bpy.data.filepath).resolve()!=ORIGINAL.resolve():
        raise RuntimeError("Expected unmodified V3 Nose original Blender scene")
    if sha256(ORIGINAL)!=source.SHA256:
        raise RuntimeError("Original V3 Nose model SHA changed")
    manifest=json.loads((BASE/"manifest.json").read_text(encoding="utf8"))
    record=json.loads((BASE/"motion_gt"/(clip+".json")).read_text(encoding="utf8"))
    description=record["clip"]
    if (description["clip"]!=clip or
        description["split"]!="continuous_qa" or
        description["family"] not in movement.FAMILIES or
        record["source_blend_sha256"]!=source.SHA256 or
        record["joint_names"]!=manifest["joint_names"] or
        record["fps"]!=24 or record["frame_count"]!=107):
        raise RuntimeError("Existing GT motion provenance/clip mismatch")
    gt_xyz=np.asarray(record["joint_position_m"],dtype=np.float64)
    if gt_xyz.shape!=(107,67,3) or not np.isfinite(gt_xyz).all():
        raise RuntimeError("Invalid original GT 3D joint coordinates")
    arm, mesh, rest, scene, cameras=source.studio()
    names=source.marker_names(arm)
    if names!=manifest["joint_names"]:
        raise RuntimeError("Blender rig order differs from GT")
    scene.frame_start=0
    scene.frame_end=106
    scene.render.fps=24
    scene.render.fps_base=1.0
    scene.camera=cameras["front"]
    scene.render.filepath=""
    scene["AnyMove_motion_source"]="ORIGINAL_PROCEDURAL_GROUND_TRUTH"
    scene["AnyMove_GT_clip"]=clip
    scene["AnyMove_GT_family"]=description["family"]
    scene["AnyMove_GT_seed"]=int(description["seed"])
    scene["AnyMove_GT_source_sha256"]=source.SHA256
    scene["AnyMove_GT_frames"]=107
    scene["AnyMove_GT_joint_points"]=67
    scene["AnyMove_GT_not_solver_output"]=True
    scene["AnyMove_RL_BoneRoot_Rest_Pose"]=True
    scene["AnyMove_Root_Animation_Target"]="ArmatureObject"
    arm.animation_data_clear()
    for pb in arm.pose.bones:
        pb.rotation_mode="XYZ"
    source_err=[]
    for frame in range(107):
        scene.frame_set(frame)
        original=prior.pose_changes
        try:
            prior.pose_changes=movement.pose_changes
            prior.apply_pose(arm,rest,names,description["family"],int(description["seed"]),frame)
        finally:
            prior.pose_changes=original
        actual=prior.joints_world(arm,names)
        per_joint=np.linalg.norm(actual-gt_xyz[frame],axis=-1)
        source_err.append(per_joint)
        if per_joint.max()>0.00001:
            raise RuntimeError("Procedural pose source differs from GT at frame="+str(frame)+
                               " max_cm="+str(per_joint.max()*100))
        arm.keyframe_insert(data_path="location",frame=frame,group="AnyMove_GT_Root")
        for bone in arm.pose.bones:
            if bone.name=="RL_BoneRoot":
                if any(abs(float(v))>1e-8 for v in bone.rotation_euler):
                    raise RuntimeError("Original GT unexpectedly animated RL_BoneRoot")
                continue  # No RL_BoneRoot pose keys, even constant keys
            bone.keyframe_insert(data_path="rotation_euler",frame=frame,group=bone.name)
        if frame in (0,30,53,80,106):
            print("GT_FRAME_KEYED",clip,frame,"max_cm",float(per_joint.max()*100),flush=True)
    if not arm.animation_data or not arm.animation_data.action:
        raise RuntimeError("Source Blender action was not created")
    # The original ground-truth came from a procedural function; insertion
    # produced real Blender Action fcurves, not an external runtime dependency.
    scene.frame_set(0)
    bpy.context.view_layer.update()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    dst=OUTPUT/"root_static"/(clip+"_GT_107frames_root_static.blend")
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        raise FileExistsError("Refusing to overwrite existing GT Blender scene "+str(dst))
    bpy.ops.wm.save_as_mainfile(filepath=str(dst))
    if not dst.is_file() or dst.stat().st_size<100000:
        raise RuntimeError("Blender GT file not saved correctly")
    if sha256(ORIGINAL)!=source.SHA256:
        raise AssertionError("Source unanimated V3 Nose model was changed")
    maxerr=float(np.max(source_err)*100)
    result={
        "status":"EXPORT_COMPLETE_AWAITING_REOPEN_QA",
        "schema":"anymove.human_v3_nose_gt_blender_action/1",
        "clip":clip,"family":description["family"],"seed":description["seed"],
        "file":str(dst),"sha256":sha256(dst),
        "source_blend_sha256":source.SHA256,
        "source_motion_gt_sha256":sha256(BASE/"motion_gt"/(clip+".json")),
        "fps":24,"frame_start":0,"frame_end":106,"frames":107,
        "pose_bones":55,"keyframed_bones":54,"static_root_bone":"RL_BoneRoot",
        "root_animation_owner":"ArmatureObject","gt_points":67,"skinmax53_retained":True,
        "last_frame_with_gt":106,
        "max_source_pose_to_gt_delta_cm":maxerr,
        "source_model_not_overwritten":True,
        "actual_blender_action":True,
        "note":"ORIGINAL procedural GT with all source frames, NOT reconstructed Solver motion",
    }
    (OUTPUT/"root_static"/(clip+"_GT_export_report.json")).write_text(json.dumps(result,indent=2),encoding="utf8")
    print("ORIGINAL_HUMAN_GT_BLENDER_EXPORTED",json.dumps({
        "file":str(dst),"frames":107,"fps":24,
        "maximum_pose_gt_deviation_cm":maxerr,
        "bytes":dst.stat().st_size}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--clip",choices=CLIPS,required=True)
    a=ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    build_one(a.clip)
