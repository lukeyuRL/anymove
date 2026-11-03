import bpy,json
from pathlib import Path
bpy.ops.wm.open_mainfile(filepath=r"F:\anymove\assets\dino\rigged\rigged.blend")
mesh=next(o for o in bpy.data.objects if o.type=="MESH")
rig=json.loads(Path(r"F:\anymove\assets\dino\rigged\rig_info.json").read_text())
groups={g.name for g in mesh.vertex_groups}
names=[b["name"] for b in rig["bones"]]
missing=[n for n in names if n not in groups]
print("RIG_BONES",len(names),"MESH_SKIN_GROUPS",len(groups),
      "BONES_WITHOUT_WEIGHTS",len(missing))
print("NO_SKIN_GROUP:",missing)
