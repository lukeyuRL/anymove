"""Inspect original untouched Dino rig current pose and exact neutral Reset Pose."""
import bpy,sys,math,numpy as np,pathlib
root=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(root/"blender"))
import common
p=root/"assets/dino/rigged/rigged.blend"
bpy.ops.wm.open_mainfile(filepath=str(p))
rig,body=common.skinned_character_objects()
scene=bpy.context.scene
print("SOURCE_INFO","rig",rig.name,"world_matrix",[[round(float(c),6) for c in row] for row in rig.matrix_world])
print("SOURCE_INFO","current_frame",scene.frame_current,"action",str(rig.animation_data.action.name) if rig.animation_data and rig.animation_data.action else "NONE")
print("SOURCE_INFO","bone_count",len(rig.pose.bones))
print("SOURCE_INFO","rotation_nonzeros",[(pb.name,list(np.round(pb.rotation_euler,4))) for pb in rig.pose.bones if np.linalg.norm(list(pb.rotation_euler))>1e-6][:10])
rig.animation_data_clear()
for pb in rig.pose.bones:
    pb.rotation_mode="XYZ"
    pb.location=(0,0,0)
    pb.rotation_euler=(0,0,0)
    pb.scale=(1,1,1)
rig.rotation_mode="XYZ";rig.rotation_euler=(0,0,0);rig.location=(0,0,0);rig.scale=(1,1,1)
scene.frame_set(1)
bpy.context.view_layer.update()
deformed=body.evaluated_get(bpy.context.evaluated_depsgraph_get())
mesh=deformed.to_mesh()
try:
    a=np.empty(len(mesh.vertices)*3,float)
    mesh.vertices.foreach_get("co",a)
    world=np.asarray(body.matrix_world,float)
    positions=a.reshape(-1,3)@world[:3,:3].T+world[:3,3]
    print("RESET_NEUTRAL","min_xyz",np.round(positions.min(axis=0),5).tolist(),
          "max_xyz",np.round(positions.max(axis=0),5).tolist(),"extents",np.round(np.ptp(positions,axis=0),5).tolist())
finally:
    deformed.to_mesh_clear()
print("RESET_NEUTRAL","bone_rot_max",max(np.linalg.norm(list(pb.rotation_euler)) for pb in rig.pose.bones),"bone_loc_max",max(pb.location.length for pb in rig.pose.bones),
      "bone_scale_maxdev",max(max(abs(v-1) for v in pb.scale) for pb in rig.pose.bones))
print("RESET_NEUTRAL","root_location",list(rig.location),"root_rot",list(rig.rotation_euler))
