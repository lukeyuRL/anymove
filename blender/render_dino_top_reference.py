"""Render same Dino rest pose as original Front/Side ref, orthographic Top.

Usage:
blender -b -P blender/render_dino_top_reference.py -- RIGGED.blend RIG_INFO.json NEW_OUTPUT_DIR
No original reference files or rig are modified.
"""
import json,math,pathlib,sys
import bpy,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import common
from mathutils import Vector
args=sys.argv[sys.argv.index("--")+1:]
if len(args)!=3:
    raise ValueError("expected BLEND RIG_INFO NEW_OUTPUT_DIR")
blend,info_path,out=[pathlib.Path(v).resolve() for v in args]
if out.exists():
    raise FileExistsError("refusing overwrite of Top reference directory")
info=json.loads(info_path.read_text(encoding="utf-8"))
if len(info["keypoints"])!=92:raise ValueError("expected original Dino 92 keypoints")
bpy.ops.wm.open_mainfile(filepath=str(blend))
rig,body=common.skinned_character_objects()
for pb in rig.pose.bones:
    pb.rotation_mode="XYZ"
    pb.rotation_euler=(0,0,0)
    pb.location=(0,0,0)
scene=bpy.context.scene
scene.frame_set(0)
common.studio(scene,samples=128)
width,height=880,720
top=bpy.data.objects.new("cam_top_ref",bpy.data.cameras.new("cam_top_ref"))
scene.collection.objects.link(top)
top.location=(0,0,30)
top.rotation_euler=(0,0,math.pi/2)
top.data.type="ORTHO"
top.data.ortho_scale=common.M_PER_PX*max(width,height)
top.data.clip_end=100.
scene.view_layers[0].update()
camera=common.camera_json(top,width,height)
R=np.asarray(camera["R"])
t=np.asarray(camera["t"])
names=[b["name"]for b in info["bones"]]
jw=[]
for kp in info["keypoints"]:
    pb=rig.pose.bones[names[kp["bone"]]]
    p=rig.matrix_world@(pb.head if kp["end"]=="head" else pb.tail)
    jw.append(list(p))
xyz=np.asarray(jw,float)
loc=xyz@R.T+t
uv=np.stack([camera["cx"]+camera["f"]*loc[:,0],
             camera["cy"]-camera["f"]*loc[:,1]],-1)
outside=(uv[:,0]<0)|(uv[:,0]>=width)|(uv[:,1]<0)|(uv[:,1]>=height)
if outside.any():
    raise ValueError("Top ref crops "+str(int(outside.sum()))+" keypoints")
out.mkdir(parents=True,exist_ok=False)
scene.camera=top
scene.render.resolution_x=width
scene.render.resolution_y=height
scene.render.resolution_percentage=100
scene.render.filepath=str(out/"ref_top.png")
bpy.ops.render.render(write_still=True)
if not (out/"ref_top.png").is_file():
    raise RuntimeError("Top ref PNG missing")
(out/"cam_top.json").write_text(json.dumps(camera,indent=2),encoding="utf-8")
(out/"ref_top_pose.json").write_text(json.dumps({
    "profile":"dino_rest_pose_top_reference_v1","keypoints":92,
    "source_blend":str(blend),"source_rig":str(info_path),
    "joint_names":[kp["name"]for kp in info["keypoints"]],
    "top_uv":uv.round(5).tolist(),
    "world_joints":xyz.round(5).tolist(),
    "note":"same zero local Euler pose as original blender/render_ref.py; independent third camera; cannot prove AI video synchronization"
},indent=2),encoding="utf-8")
print("TOP_REST_REFERENCE",out/"ref_top.png","joint_bounds_x",uv[:,0].min(),uv[:,0].max(),"joint_bounds_y",uv[:,1].min(),uv[:,1].max(),flush=True)
