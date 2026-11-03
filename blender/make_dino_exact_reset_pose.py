"""Create exact zero-transform Dino Blender Reset Pose metadata.

Unlike 9001 randomized P4 synthetic GT, this has no procedural joint
rotations, translations, root yaw, animation, or simulated motion.

Input original immutable rigged.blend; generates one 3D pose + front/side
orthographic projection metadata for the verified existing four-view renderer.
"""
import bpy,json,pathlib,sys,numpy as np
root=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(root/"blender"))
import common
from dino_skinmax_color20 import sha
BLEND=root/"assets/dino/rigged/rigged.blend"
INFO=root/"assets/dino/rigged/rig_info.json"
OUT=root/"runs/dino/fal_h3max_multiview_v1/reset_pose_source"
FILE=OUT/"12001_meta.json"
if FILE.exists():raise FileExistsError(FILE)
if not BLEND.exists() or not INFO.exists():raise FileNotFoundError("Dino reset source files missing")
info=json.loads(INFO.read_text(encoding="utf8"))
if len(info["bones"])!=67 or len(info["keypoints"])!=92:
    raise ValueError("Dino Rig bone/joint count unexpectedly changed")
src_hash=sha(BLEND);rig_hash=sha(INFO)
bpy.ops.wm.open_mainfile(filepath=str(BLEND))
rig,mesh=common.skinned_character_objects()
rig.animation_data_clear()
for pb in rig.pose.bones:
    pb.rotation_mode="XYZ"
    pb.rotation_euler=(0,0,0)
    pb.location=(0,0,0)
    pb.scale=(1,1,1)
rig.rotation_mode="XYZ"
rig.rotation_euler=(0,0,0)
rig.location=(0,0,0)
rig.scale=(1,1,1)
bpy.context.scene.frame_set(1)
bpy.context.view_layer.update()
bone_names=[b["name"] for b in info["bones"]]
if set(bone_names)!=set(b.name for b in rig.pose.bones):
    raise ValueError("Blender Dino Bone order mismatch")
max_rotation=max(max(abs(float(v)) for v in pb.rotation_euler) for pb in rig.pose.bones)
max_pos=max(pb.location.length for pb in rig.pose.bones)
max_scale_err=max(max(abs(float(v)-1) for v in pb.scale) for pb in rig.pose.bones)
if max_rotation or max_pos or max_scale_err:
    raise AssertionError("Blender Dino Reset Pose has nonzero pose transforms")
mat=rig.matrix_world
xyz=np.asarray([list(mat@(rig.pose.bones[bone_names[kp["bone"]]].head
                 if kp["end"]=="head" else
                 rig.pose.bones[bone_names[kp["bone"]]].tail))
                for kp in info["keypoints"]],dtype=np.float64)
if xyz.shape!=(92,3) or not np.isfinite(xyz).all():
    raise ValueError("Reset Pose keypoints invalid")
m=common.M_PER_PX
fu=np.stack((common.W_FRONT/2+xyz[:,0]/m,
            common.H/2-(xyz[:,2]-common.Z_CENTER)/m),axis=-1)
su=np.stack((common.W_SIDE/2-xyz[:,1]/m,
            common.H/2-(xyz[:,2]-common.Z_CENTER)/m),axis=-1)
out={
    "schema":"anymove.dino_zero_transform_reset_pose_reference/1",
    "pose_source":"Original neutral rest pose in rigged.blend, not a procedurally sampled or animated pose",
    "sample_seed":None,
    "joints_world":np.round(xyz,6).tolist(),
    "front_uv":np.round(fu,5).tolist(),
    "side_uv":np.round(su,5).tolist(),
    "scale":1.0,
    "blur":False,
    "yaw":0.0,
    "root":[0.0,0.0,0.0],
    "pose":{name:[0.0,0.0,0.0] for name in bone_names},
    "all_67_bones_zero_euler":True,
    "all_67_bones_zero_location":True,
    "all_67_bones_unit_scale":True,
    "root_identity":True,
    "armature_action":None,
    "source_blend":str(BLEND),
    "source_blend_sha256":src_hash,
    "rig_info_sha256":rig_hash,
    "no_animation_or_video_generated":True,
}
OUT.mkdir(parents=True,exist_ok=True)
FILE.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
print("DINO_RESET_POSE_CREATED",str(FILE),flush=True)
print("RESET_CONTRACT",json.dumps({"zero_euler":max_rotation,"zero_translation":max_pos,
 "scale_error":max_scale_err,"joint_count":len(xyz),"world_joint_min":np.round(xyz.min(axis=0),4).tolist(),
 "world_joint_max":np.round(xyz.max(axis=0),4).tolist(),
 "root":[0,0,0],"yaw":0,"source_hash":src_hash},indent=2),flush=True)
