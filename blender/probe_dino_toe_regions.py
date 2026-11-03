import bpy,numpy as np,json
from pathlib import Path
bpy.ops.wm.open_mainfile(filepath=r"F:\anymove\assets\dino\rigged\rigged.blend")
arm=next(x for x in bpy.data.objects if x.type=="ARMATURE")
mesh=next(x for x in bpy.data.objects if x.type=="MESH")
info=json.loads(Path(r"F:\anymove\assets\dino\rigged\rig_info.json").read_text())
print("ARM:",arm.name,"WORLD",list(arm.matrix_world.translation),"MESH:",mesh.name,"world",list(mesh.matrix_world.translation))
print("SIZE:",len(mesh.data.vertices),"FACES:",len(mesh.data.polygons),"UV",[(u.name,len(u.data)) for u in mesh.data.uv_layers],"ATTRS",[(u.name,u.domain,u.data_type) for u in mesh.data.color_attributes])
print("MATERIALS",[(m.name if m else "-") for m in mesh.data.materials])
for b in info["bones"]:
 name=b["name"];low=name.lower()
 if any(w in low for w in ("foot","toe","calf","thigh")) and ("_l_" in low or "_r_" in low):
  bone=arm.data.bones.get(name);w=0
  vg=mesh.vertex_groups.get(name)
  if vg:
   group_idx=vg.index
   for v in mesh.data.vertices:
    for g in v.groups:
     if g.group==group_idx and g.weight>.01:w+=1
  head=list(arm.matrix_world@bone.head_local)
  tail=list(arm.matrix_world@bone.tail_local)
  print("BONE",name,"LEN",round(bone.length,4),"weighted_verts",w,"head",np.round(head,3),"tail",np.round(tail,3))
