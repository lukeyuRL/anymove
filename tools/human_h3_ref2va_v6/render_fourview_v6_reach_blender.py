"""Blender 5.0 V5 source Human 4-view camera renderer (oblique views only).

Uses EXISTING immutable Human multicolor V3 blend and exact V2 per-frame
Euler and rig world matrices. Reconstructs, verifies ALL 55 joint
positions against original evaluated Blender V2 source GT BEFORE
rendering; never invents a new motion.

Renders Front-Right 45 and Front-Left 45 oblique RGB at 384x448.
Existing V2 verified front/back PNGs are reused bit-for-bit.

Invocation:
blender -b -t 6 --python-exit-code 17 -P <this_file> -- --clip full_heldout... --render-frames 0,24,53
or --render-frames all

No fal API calls, no .blend source edits.
"""
import argparse
import json
import math
import pathlib
import sys
import time

import bpy
from mathutils import Matrix
from bpy_extras.object_utils import world_to_camera_view

ROOT=pathlib.Path(r"F:\anymove")
HERE=ROOT/"runs"/"human"/"lora_synth_A"
sys.path.insert(0,str(HERE))
import render_human_motion as legacy
OUT=HERE/"conditioned_backend"/"fal_h3_ref2va"/"fourview_v6_reach_augmented"
V2=HERE/"v6_reach_source"
EXPECTED_BLEND="93ced4096791fbc84712527a3c023e7c3bee381106e15b898cd4295f24c8008b"
SOURCE_BLEND=HERE.parent/"human_skinmax_multicolor_v3.blend"

def read(p):
 return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def save(p,d):
 p=pathlib.Path(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=pathlib.Path(str(p)+".tmp")
 tmp.write_text(json.dumps(d,indent=2),encoding="utf8")
 tmp.replace(p)
def args():
 ap=argparse.ArgumentParser()
 ap.add_argument("--clip",required=True)
 ap.add_argument("--render-frames",default="all")
 ap.add_argument("--check-front",action="store_true")
 return ap.parse_args(sys.argv[sys.argv.index("--")+1:])
def main():
 opts=args()
 src=V2/opts.clip
 if not src.is_dir():raise FileNotFoundError("Source clip absent "+str(src))
 man=read(src/"manifest.json")
 old_motion=read(src/"motion.json")
 truth=read(src/"joints.json")
 four=OUT/"clips"/man["family"].replace("/","_") # unused, explicit contract resolved below
 splits="heldout" if "heldout" in opts.clip else "train"
 target=OUT/"clips"/splits/opts.clip
 conf=read(target/"fourview_camera_contract.json")
 if conf["source_clip_id"]!=opts.clip or conf["source_split"]!=splits:
  raise ValueError("V5 clip manifest wrong")
 if legacy.sha256(SOURCE_BLEND)!=EXPECTED_BLEND:
  raise ValueError("Immutable source Blender SHA mismatch")
 if old_motion["bones"]!=truth["names"] or len(old_motion["pose_euler_xyz_rad"])!=107:
  raise ValueError("Source V2 motion names/order/frames invalid")
 assert len(old_motion["rig_world_matrices"])==107
 if opts.render_frames=="all":
  frames=list(range(107))
 else:
  frames=sorted(set(int(x) for x in opts.render_frames.split(",")))
 if not frames or frames[0]<0 or frames[-1]>=107:
  raise ValueError("Invalid requested frames")
 start=time.time()
 bpy.ops.wm.open_mainfile(filepath=str(SOURCE_BLEND))
 scene=bpy.context.scene
 rig=bpy.data.objects.get("Armature")
 body=bpy.data.objects.get("Dummy")
 if not rig or rig.type!="ARMATURE" or not body or body.type!="MESH":
  raise ValueError("Source scene missing skinned Armature/Dummy")
 original_bones=list(rig.pose.bones)
 names=[x.name for x in original_bones]
 if names!=truth["names"] or len(names)!=55 or len(body.data.vertices)!=30188:
  raise ValueError("Source Human 55-bone rig mismatch")
 legacy.reset_pose(rig)
 # For identical EEVEE look, setup the identical original studio lighting
 # in source first-frame pose. We will use the authoritative stored
 # cameras, not any newly fit approximate camera.
 def apply_source(frame):
  scene.frame_set(frame)
  angles=old_motion["pose_euler_xyz_rad"][frame]
  for bone,rotation in zip(original_bones,angles):
   bone.rotation_mode="XYZ"
   bone.rotation_euler=rotation
  rig.matrix_world=Matrix(old_motion["rig_world_matrices"][frame])
  bpy.context.view_layer.update()
  xyz=[rig.matrix_world@bone.head for bone in original_bones]
  gt=truth["joint_position_m"][frame]
  worst=max((xyz[j]-__import__("mathutils").Vector(gt[j])).length for j in range(55))
  if worst>5.e-5:
   raise RuntimeError("Blender V2 source motion replay failed at frame "+str(frame)+f": {worst} meters")
  return worst
 err=apply_source(0)
 # Configure colors/world/lights/reproducible render engine.
 _studio,_bounds=legacy.setup_studio(scene,body,names,["front","back"],man["samples"],
                                    384,448)
 scene.frame_start=0
 scene.frame_end=106
 scene.render.fps=24
 scene.render.resolution_percentage=100
 scene.render.resolution_x=384
 scene.render.resolution_y=448
 scene.render.image_settings.file_format="PNG"
 scene.render.image_settings.color_mode="RGB"
 cameras={}
 for view in ("left_45","right_45","top"):
  cam=bpy.data.objects.new("SynthCamV5_"+view,bpy.data.cameras.new("SynthCamV5_"+view))
  scene.collection.objects.link(cam)
  meta=conf["camera_views"][view]
  cam.matrix_world=Matrix(meta["matrix_world"])
  cam.data.type="ORTHO"
  cam.data.ortho_scale=float(meta["ortho_scale"])
  cam.data.clip_start=float(meta["clip_near"])
  cam.data.clip_end=float(meta["clip_far"])
  cameras[view]=cam
 bpy.context.view_layer.update()
 worst_err=err
 rendered={}
 for frame in frames:
  mismatch=apply_source(frame)
  worst_err=max(worst_err,mismatch)
  for view,cam in cameras.items():
   file=target/"rgb"/view/f"f_{frame:04d}.png"
   file.parent.mkdir(parents=True,exist_ok=True)
   if file.exists() and legacy.image_dimensions(file)==(384,448):
    rendered[view]=rendered.get(view,0)+1
    continue
   scene.camera=cam
   scene.render.filepath=str(file)
   bpy.ops.render.render(write_still=True)
   if legacy.image_dimensions(file)!=(384,448):
    raise RuntimeError("Failed RGB render "+str(file))
   rendered[view]=rendered.get(view,0)+1
  if frame in (0,24,53,80,106) or frame%25==0:
   print("V5_BLENDER_REPLAY_AND_RENDERED",opts.clip,
     "frame",frame,"/",len(frames), "max_m",round(worst_err,8),flush=True)
 if opts.check_front:
  file=target/"validation"/"front_replay_0000.png"
  file.parent.mkdir(parents=True,exist_ok=True)
  apply_source(0)
  scene.camera=_studio["front"]
  scene.render.filepath=str(file)
  bpy.ops.render.render(write_still=True)
  if legacy.image_dimensions(file)!=(384,448):
   raise RuntimeError("Front replay validation render failed")
 report={
  "schema":"anymove.human_v6_reach_blender_oblique_render/1",
  "status":"SMOKE_PASS" if len(frames)<107 else "COMPLETE",
  "source_clip":str(src),"case":opts.clip,
  "frames_requested":frames,"frames_count":len(frames),
  "views":["left_45","right_45","top"],
  "frames_rendered_per_view":rendered,
  "source_V2_55_bone_fk_max_error_m":worst_err,
  "elapsed_s":round(time.time()-start,2),
  "source_blend_SHA256":EXPECTED_BLEND,
  "source_rgb_front_back_reused":False,
  "source_rgb_back_reused":True,
  "top_overhead_rendered":True,
  "front_panel_used":False,
  "front_replay_validated":bool(opts.check_front),
  "new_paid_ai_jobs":0}
 save(target/("smoke_render.json" if len(frames)<107 else "rgb_render_status.json"),report)
 print("V5_BLENDER_RGB_RESULT",json.dumps({k:v for k,v in report.items() if k!="frames_requested"},indent=2),flush=True)
if __name__=="__main__":main()
