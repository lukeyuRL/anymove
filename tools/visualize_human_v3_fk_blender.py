"""Author a separate Blender 5 visual QA scene of Human V3 Nose P1 FK motion.

Never alters the source Blend. Inserts the 91 actual 24-FPS body/hand/toe
poses into Blender pose bones and the armature Root, saves a DIFFERENT .blend
in blender_qa. Renders the SAME calibrated cameras/light as original GT
and outputs representative samples for image-level validation.
"""
import argparse, bpy, json, hashlib, math, pathlib, sys
import numpy as np
from mathutils import Matrix, Quaternion

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools import render_human_v3_hr_v2_315_45 as renderer
from tools.human_v3_fullbody_motion import finger_tip_names,toe_tip_names
from tools.human_v3_blender_root_static import write_static_root_frame,ensure_static_root

SRC=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
BASE=ROOT/"runs/human/fullbody_finger_v2_hr_315_45"
DEST=BASE/"solver/blender_qa"
CLIPS=(
 "hrv2_continuous_finger_independent_roll_s305007",
 "hrv2_continuous_kick_and_wave_s306019",
)
SELECTED=(8,26,44,53,62,80,98)

def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for c in iter(lambda:f.read(1048576),b""):h.update(c)
 return h.hexdigest()

def actual_joints(arm,names):
 bones=names[:55]
 vec=[arm.matrix_world @ arm.pose.bones[n].head for n in bones]
 vec.extend([arm.matrix_world @ arm.pose.bones[n.replace("Tip","3")].tail for n in finger_tip_names()])
 vec.extend([arm.matrix_world @ arm.pose.bones[n.replace("ToeTip","ToeBase")].tail for n in toe_tip_names()])
 return np.asarray(vec,dtype=float)

def execute(clip):
 if clip not in CLIPS:raise ValueError("Unknown complete Human V3 continuous clip")
 if pathlib.Path(bpy.data.filepath).resolve()!=SRC.resolve() or sha256(SRC)!=renderer.SHA256:
  raise RuntimeError("Wrong source V3 nose Blend or SHA")
 src_motion=BASE/"continuous_24fps_315_45/detections"/(clip+".motion_p1_trainlimits.json")
 motion=json.loads(src_motion.read_text(encoding="utf8"))
 scene=bpy.context.scene
 arm,mesh,rest,scene,cameras=renderer.studio()
 names=[p.name for p in arm.pose.bones]+finger_tip_names()+toe_tip_names()
 frames=np.asarray(motion["source_frame_indices"],dtype=int)
 if (motion["joint_names"]!=names or motion["bones"]!=names[:55] or
     frames.tolist()!=list(range(8,99)) or
     motion["motion_space_warning"]!="BLENDER_LOCAL_EULER_NOT_ICLONE_FK_CHANNELS" or
     motion["solver_camera_azimuth_deg"]!=[315,45]):
  raise RuntimeError("Source motion metadata does not match Blender model")
 trans=np.asarray(motion["trans"],dtype=float)
 euler=np.asarray(motion["euler"],dtype=float)
 quat=np.asarray(motion["root_quat"],dtype=float)
 joints=np.asarray(motion["joints"],dtype=float)
 if (trans.shape!=(91,3) or euler.shape!=(91,55,3) or
     quat.shape!=(91,4) or joints.shape!=(91,67,3)):
  raise RuntimeError("Wrong FK channel shape")
 scene["AnyMove_RL_BoneRoot_Rest_Pose"]=True
 scene["AnyMove_Root_Animation_Target"]="ArmatureObject"
 scene.frame_start=int(frames[0]);scene.frame_end=int(frames[-1])
 scene.render.fps=24
 scene.render.fps_base=1.0
 maxerr=0.
 for i,frame in enumerate(frames):
  scene.frame_set(int(frame))
  write_static_root_frame(arm,rest,trans[i],quat[i],int(frame))
  for b,pbone in enumerate(arm.pose.bones):
   if pbone.name=="RL_BoneRoot":
    continue  # no pose keyframes for the fixed root bone
   pbone.rotation_mode="XYZ"
   pbone.rotation_euler=tuple(euler[i,b])
   pbone.keyframe_insert(data_path="rotation_euler",frame=int(frame),group=pbone.name)
  bpy.context.view_layer.update()
  localerr=np.linalg.norm(actual_joints(arm,names)-joints[i],axis=-1).max()
  maxerr=max(maxerr,float(localerr))
 if maxerr>0.0003:
  raise RuntimeError("Blender FK mismatch after keyframing: "+str(maxerr))
 frames_out={}
 for frame in SELECTED:
  scene.frame_set(frame)
  for slot,camera in cameras.items():
   scene.camera=camera
   path=DEST/"root_static/images"/clip/(slot+"_"+str(frame).zfill(4)+".png")
   path.parent.mkdir(parents=True,exist_ok=True)
   scene.render.filepath=str(path)
   bpy.ops.render.render(write_still=True)
   if renderer.png_size(path)!=(672,672):
    raise RuntimeError("Could not render frame "+str(path))
   frames_out[f"{slot}_{frame}"]=str(path)
 scene.frame_set(8)
 scene.camera=cameras["front"]
 scene.render.filepath=""
 dst=DEST/"root_static"/(clip+"_solved_p1_trainlimits_root_static.blend")
 if dst.exists():raise FileExistsError("Refusing to overwrite QA Blender project "+str(dst))
 bpy.ops.wm.save_as_mainfile(filepath=str(dst))
 if not dst.exists() or dst.stat().st_size<100000:
  raise RuntimeError("Blender QA animation file save failed")
 info={
  "schema":"anymove.human_v3_blender_render_qa/1",
  "status":"PASS",
  "clip":clip,
  "source_blend_sha256":renderer.SHA256,
  "motion_json_sha256":sha256(src_motion),
  "saved_animation_blend":str(dst),
  "frames":91,"fps":24,"source_frame_start":8,"source_frame_end":98,
  "keyframed_bones":54,"static_root_bone":"RL_BoneRoot",
  "root_animation_owner":"ArmatureObject","world_markers":67,
  "max_replay_fk_mismatch_cm":maxerr*100.,
  "views":[315,45],
  "rendered":frames_out,
  "note":"Exact Blender FK animation and representative renders, not production retarget quality approval."
 }
 (DEST/"root_static"/(clip+"_blender_visual_qa.json")).write_text(json.dumps(info,indent=2),encoding="utf8")
 print("BLENDER_SOLVER_ANIMATION_SAVED",json.dumps({
  "file":str(dst),"frames":91,"max_fk_error_cm":round(maxerr*100,6),
  "preview_images":len(frames_out),
  "source_unchanged":True}),flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--clip",required=True,choices=CLIPS)
 args=ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
 execute(args.clip)
