"""Blender reload integrity test for Color20 experimental COPY.

No edits to either the original source .blend or to the preview.
"""
import bpy,sys,json,numpy as np,pathlib
root=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(root/"blender"))
from dino_skinmax_color20 import ATTR_NAME,dominant_weights,hex_to_linear,sha
file=root/"runs/dino/color20_candidate_v1/dino_skinmax20_preview.blend"
bpy.ops.wm.open_mainfile(filepath=str(file))
mesh=next(x for x in bpy.data.objects if x.type=="MESH" and x.vertex_groups)
palette=json.loads((root/"runs/dino/color20_candidate_v1/skinmax_palette_v2.json").read_text())
a=mesh.data.color_attributes.get(ATTR_NAME)
if a is None:raise AssertionError("vertex-color attribute not saved")
raw=np.zeros(len(mesh.data.vertices)*4,np.float32)
a.data.foreach_get("color",raw)
rgba=raw.reshape(-1,4)
owners,weights=dominant_weights(mesh)
expected=palette["bone_colors"]
count=0
for i,name in enumerate(owners):
    yes=name in expected
    if abs(float(rgba[i,3])-float(yes))>1e-5:
        raise AssertionError("alpha mask includes non-effective Bone")
    if yes:
        if np.max(np.abs(rgba[i,:3]-hex_to_linear(expected[name])))>1e-5:
            raise AssertionError("vertex color does not equal argmax Skin Bone")
        count+=1
if count!=2484:raise AssertionError("incorrect selected color Vertex count")
if len({c for c in expected.values()})!=20:raise AssertionError("duplicate Color20")
for material in mesh.data.materials:
    if material is None:raise AssertionError("empty source material")
    node=material.node_tree.nodes.get("AnyMoveOverlayOriginalAlbedo")
    if node is None:raise AssertionError("source material mix not saved")
if len([im for im in bpy.data.images if im.source=="FILE" and im.packed_file])<4:
    raise AssertionError("source albedo not packed in Color20 preview")
source=root/"assets/dino/rigged/rigged.blend"
print("SKINMAX20_COPY_REOPEN_PASS","colored",count,"uncolored",len(owners)-count,
      "attr",a.name,"meshverts",len(owners),"original_sha",sha(source))
