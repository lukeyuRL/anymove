"""Inspect local Dino Blender 5.0 rig and exact dataset pose replay."""
import bpy,sys,json,numpy as np
from mathutils import Euler,Vector
from pathlib import Path
root=Path(r"F:\anymove")
sys.path.insert(0,str(root/"blender"))
import common
bpy.ops.wm.open_mainfile(filepath=str(root/"assets/dino/rigged/rigged.blend"))
rig,body=common.skinned_character_objects()
info=json.loads((root/"assets/dino/rigged/rig_info.json").read_text())
meta=json.loads((root/"runs/dino/dataset_hr_v1/00000_meta.json").read_text())
rig.animation_data_clear()
for pb in rig.pose.bones:
    pb.rotation_mode="XYZ"
    pb.location=(0,0,0)
for name,rot in meta["pose"].items():
    rig.pose.bones[name].rotation_euler=Euler(rot,"XYZ")
rig.rotation_euler=(0,0,meta["yaw"])
rig.location=meta["root"]
bpy.context.view_layer.update()
mw=rig.matrix_world
jw=np.array([list(mw@(rig.pose.bones[info["bones"][kp["bone"]]["name"]].head if kp["end"]=="head" else rig.pose.bones[info["bones"][kp["bone"]]["name"]].tail)) for kp in info["keypoints"]])
old=np.array(meta["joints_world"])
print("REPLAY","joint_max_abs_m",float(np.max(np.abs(jw-old))),"joint_mean_3d_m",float(np.mean(np.linalg.norm(jw-old,axis=1))))
print("OBJECTS",[(x.name,x.type) for x in bpy.data.objects],"RIGWORLD",list(rig.matrix_world[0]),"BODYWORLD",list(body.matrix_world[0]))
dg=bpy.context.evaluated_depsgraph_get()
bodyeval=body.evaluated_get(dg)
me=bodyeval.to_mesh()
me.calc_loop_triangles()
print("MESH","verts",len(me.vertices),"polys",len(me.polygons),"triangles",len(me.loop_triangles),
      "source verts",len(body.data.vertices),"vertex groups",len(body.vertex_groups),
      "evaluated vertex0 groups",[(int(g.group),round(float(g.weight),2)) for g in me.vertices[0].groups],
      "first tri",list(me.loop_triangles[0].vertices))
print("BONES",len(rig.pose.bones),"SOURCE_CAMERA",[(cam.name,tuple(cam.location),cam.data.type,cam.data.ortho_scale) for cam in bpy.data.objects if cam.type=="CAMERA"])
cams=common.cameras(bpy.context.scene)
for k,(cam,w) in cams.items():
    bpy.context.view_layer.update()
    inv=cam.matrix_world.inverted()
    pts=np.array([list(inv@Vector(p)) for p in jw])
    px=np.stack([w/2+pts[:,0]/common.M_PER_PX,common.H/2-pts[:,1]/common.M_PER_PX],1)
    tag=k.split("_")[1]
    delta=np.linalg.norm(px-np.array(meta[tag+"_uv"]),axis=1)
    print("PROJECTION",tag,"mean",float(delta.mean()),"max",float(delta.max()),"view first",px[:2])
bodyeval.to_mesh_clear()
