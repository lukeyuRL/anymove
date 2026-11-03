"""Save a non-destructive Dino 41-class view-aware colour Rig copy.

Keeps original Blender scene, Mesh, bone weights and images untouched.
Material nodes and one POINT vertex-color layer are edited only in COPY.
"""
import bpy,json,pathlib,sys
root=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(root/"blender"))
from dino_skinmax_camera import apply,ATTR_NAME
from dino_skinmax_color20 import sha
source=root/"assets/dino/rigged/rigged.blend"
palette=root/"runs/dino/skinmax_camera_aware_v1/optimized_palette_constrained_v4.json"
target=root/"runs/dino/skinmax_camera_aware_v1/dino_viewaware_skinmax_v4.blend"
if target.exists():raise FileExistsError(target)
digest=sha(source)
bpy.ops.wm.open_mainfile(filepath=str(source))
mesh=next(o for o in bpy.data.objects if o.type=="MESH" and o.vertex_groups)
rig=json.loads((root/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
info=apply(mesh,rig,palette,source_blend=source)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True,copy=True)
if sha(source)!=digest:raise AssertionError("original Dino source Blender file changed")
if not target.exists() or target.stat().st_size<1000000:
    raise RuntimeError("Blender preview output invalid")
print("CAMERA_AWARE_BLEND_SAVED",target,"bytes",target.stat().st_size,
      "colour_regions",info["colour_count"],"vertex_colours",info["coloured_vertex_count"],
      "unchanged_source_sha",digest,flush=True)
