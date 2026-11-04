"""Blender preview of experimental Human V3 Spine temporal regularizer.

Research-only and separate from original GT and validated Solver .blend.
Reopens the original nose source, keys every one of the 91 observed
24FPS frames from opt-in Spine-regularized FK, verifies 67 FK markers,
saves a distinct animated .blend and renders paired 315/45 preview PNGs.

No touch of source V3 Blend, old P1 animation or iClone.
"""
import argparse,bpy,json,hashlib,pathlib,sys
import numpy as np
from mathutils import Matrix,Quaternion

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools import render_human_v3_hr_v2_315_45 as render
from tools.human_v3_fullbody_motion import finger_tip_names,toe_tip_names
from tools.human_v3_blender_root_static import write_static_root_frame,ensure_static_root

BASE=ROOT/"runs/human/fullbody_finger_v2_hr_315_45"
EXPERIMENT=BASE/"solver/spine_regularized_research"
ORIGINAL=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
CLIPS={
 "hrv2_continuous_finger_independent_roll_s305007":"finger_roll_spine_regularized_dev.json",
 "hrv2_continuous_kick_and_wave_s306019":"kick_wave_spine_regularized_dev.json"
}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()

def replay_joints(arm,names):
 pts=[arm.matrix_world @ arm.pose.bones[name].head for name in names[:55]]
 pts.extend([arm.matrix_world @ arm.pose.bones[name.replace("Tip","3")].tail for name in finger_tip_names()])
 pts.extend([arm.matrix_world @ arm.pose.bones[name.replace("ToeTip","ToeBase")].tail for name in toe_tip_names()])
 return np.asarray(pts,dtype=np.float64)

def main(clip):
 if pathlib.Path(bpy.data.filepath).resolve()!=ORIGINAL.resolve() or sha(ORIGINAL)!=render.SHA256:
  raise RuntimeError("Wrong source V3 Nose model / SHA")
 arm,mesh,rest,scene,cameras=render.studio()
 names=render.marker_names(arm)
 path=EXPERIMENT/CLIPS[clip]
 outdir=EXPERIMENT/"blender_qa/root_static"
 if not path.is_file():raise FileNotFoundError(path)
 motion=json.loads(path.read_text(encoding="utf8"))
 frames=list(range(8,99))
 if (motion["source_frame_indices"]!=frames or
     motion["joint_names"]!=names or
     motion["bones"]!=names[:55] or
     motion["solver_camera_azimuth_deg"]!=[315,45] or
     motion["solver_rig_limits_status"]!="PROVISIONAL_TRAIN_DERIVED_SPINE_SOFT_REGULARIZATION_ONLY"):
  raise RuntimeError("Wrong research pose source/provenance")
 trs=np.asarray(motion["trans"],dtype=np.float64)
 roots=np.asarray(motion["root_quat"],dtype=np.float64)
 eul=np.asarray(motion["euler"],dtype=np.float64)
 world=np.asarray(motion["joints"],dtype=np.float64)
 if (trs.shape!=(91,3) or roots.shape!=(91,4) or
     eul.shape!=(91,55,3) or world.shape!=(91,67,3)):
  raise RuntimeError("Incomplete experimental fullbody motion")
 scene.frame_start=8
 scene.frame_end=98
 scene.render.fps=24
 scene.render.fps_base=1.0
 scene["AnyMove_RL_BoneRoot_Rest_Pose"]=True
 scene["AnyMove_Root_Animation_Target"]="ArmatureObject"
 scene["AnyMove_Spine_Research_Only"]=True
 scene["AnyMove_Spine_Clip"]=clip
 scene["AnyMove_Spine_Original_Solver_Motion_SHA256"]=sha(path)
 arm.animation_data_clear()
 maxerr=0.
 for i,frame in enumerate(frames):
  scene.frame_set(frame)
  write_static_root_frame(arm,rest,trs[i],roots[i],frame)
  for j,b in enumerate(arm.pose.bones):
   if b.name=="RL_BoneRoot":
    continue  # RL_BoneRoot remains at rest and has no keyframes
   b.rotation_mode="XYZ"
   b.rotation_euler=tuple(eul[i,j])
   b.keyframe_insert(data_path="rotation_euler",frame=frame,group=b.name)
  bpy.context.view_layer.update()
  discrepancy=np.max(np.linalg.norm(replay_joints(arm,names)-world[i],axis=-1))
  maxerr=max(maxerr,float(discrepancy))
 if maxerr>.0003:
  raise RuntimeError("Blender and experimental Solver FK mismatch, metres="+str(maxerr))
 outdir.mkdir(parents=True,exist_ok=True)
 target=outdir/(clip+"_spine_regularized_research_root_static.blend")
 if target.is_file():raise FileExistsError("Refuse overwrite existing comparison Blender Action")
 scene.frame_set(8)
 scene.camera=cameras["front"]
 scene.render.filepath=""
 bpy.ops.wm.save_as_mainfile(filepath=str(target))
 if not target.is_file() or target.stat().st_size<100000:
  raise RuntimeError("Research comparison .blend save failed")
 new=0
 for frame in frames:
  scene.frame_set(frame)
  for slot,cam in cameras.items():
   scene.camera=cam
   file=outdir/"video_frames"/clip/slot/f"frame_{frame:04d}.png"
   file.parent.mkdir(parents=True,exist_ok=True)
   if render.png_size(file)!=(672,672):
    scene.render.filepath=str(file)
    bpy.ops.render.render(write_still=True)
    new+=1
   if render.png_size(file)!=(672,672):
    raise RuntimeError("Research animation frame missing "+str(file))
  if frame in (8,44,98):
   print("SPINE_RESEARCH_FRAME",clip,frame,"new_images",new,flush=True)
 if sha(ORIGINAL)!=render.SHA256:raise RuntimeError("Original V3 Nose Blend changed")
 audit={"schema":"anymove.blender_spine_research_visual_qa/1",
        "clip":clip,"status":"SAVE_RENDER_AND_FK_PASS",
        "frames":91,"fps":24,"views":[315,45],"rendered_pngs":182,
        "max_fk_replay_error_cm":maxerr*100,
        "RL_BoneRoot_pose_static":True,
        "root_motion_owner":"ArmatureObject",
        "original_source_unmodified":True,
        "research_animation_blend":str(target),
        "experimental_motion_sha256":sha(path),
        "note":"Controlled experiment only; not iClone or anatomical motion acceptance"}
 (outdir/(clip+"_research_qa.json")).write_text(json.dumps(audit,indent=2),encoding="utf8")
 print("SPINE_RESEARCH_BLENDER_PASS",json.dumps(audit),flush=True)

if __name__=="__main__":
 ap=argparse.ArgumentParser()
 ap.add_argument("--clip",choices=tuple(CLIPS),required=True)
 args=ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
 main(args.clip)
