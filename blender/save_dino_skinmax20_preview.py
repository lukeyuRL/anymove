"""Non-destructive editable Blender Dino Color20 skinmax preview copy."""
import bpy,json,pathlib,sys
root=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(root/"blender"))
from dino_skinmax_color20 import apply,dominant_weights,ATTR_NAME,sha
source=root/"assets/dino/rigged/rigged.blend"
rig=root/"assets/dino/rigged/rig_info.json"
palette=root/"runs/dino/color20_candidate_v1/skinmax_palette_v2.json"
target=root/"runs/dino/color20_candidate_v1/dino_skinmax20_preview.blend"
if target.exists():raise FileExistsError(target)
pre=sha(source)
bpy.ops.wm.open_mainfile(filepath=str(source))
body=next(x for x in bpy.data.objects if x.type=="MESH" and x.vertex_groups)
report=apply(body,json.loads(rig.read_text(encoding="utf8")),palette,original_blend_path=source)
if len(body.data.color_attributes.get(ATTR_NAME).data)!=len(body.data.vertices):
    raise RuntimeError("point vertex colors missing or mismatched")
# Pack the existing source albedo images so moving the separate preview
# Blender file does not silently turn its material textures black.
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True,copy=True)
if not target.is_file() or target.stat().st_size<10000:
    raise RuntimeError("Color20 Blender preview not written")
if sha(source)!=pre:
    raise RuntimeError("original rigged.blend was modified")
print("COLOR20_BLEND_WRITTEN",target,"bytes",target.stat().st_size)
print("SOURCE_SHA_UNCHANGED",pre)
print("SELECTED_VERTICES",report["selected_vertex_count"])
print("PACKED_IMAGES",[(x.name,bool(x.packed_file)) for x in bpy.data.images if x.source=="FILE"])
