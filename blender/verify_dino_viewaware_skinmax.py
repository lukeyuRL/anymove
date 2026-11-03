"""Re-open view-aware skinmax Dino Blender copy and verify exact argmax-colour rule."""
import bpy,json,pathlib,sys
import numpy as np
root=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(root/"blender"))
from dino_skinmax_camera import ATTR_NAME
from dino_skinmax_color20 import dominant_weights,hex_to_linear,sha
folder=root/"runs/dino/skinmax_camera_aware_v1"
copy=folder/"dino_viewaware_skinmax_v4.blend"
palette=json.loads((folder/"optimized_palette_constrained_v4.json").read_text(encoding="utf8"))
original=root/"assets/dino/rigged/rigged.blend"
sha_before=sha(original)
bpy.ops.wm.open_mainfile(filepath=str(copy))
mesh=next(x for x in bpy.data.objects if x.type=="MESH" and x.vertex_groups)
layer=mesh.data.color_attributes.get(ATTR_NAME)
if layer is None:raise RuntimeError("missing viewaware color attribute")
owners,weights=dominant_weights(mesh)
values=np.empty((len(owners),4),np.float32)
layer.data.foreach_get("color",values.reshape(-1))
colourmap=palette["bone_colors"]
counts={name:0 for name in colourmap}
for i,name in enumerate(owners):
    assigned=name in colourmap
    if abs(float(values[i,3])-float(assigned))>1e-6:
        raise ValueError("unselected SkinMax vertex colored at index "+str(i))
    if assigned:
        if np.linalg.norm(values[i,:3]-hex_to_linear(colourmap[name]))>1e-4:
            raise ValueError("Color20 mismatch for SkinMax owner "+str(name))
        counts[name]+=1
if sum(counts.values())!=4909 or any(v<1 for v in counts.values()):
    raise AssertionError("skin regions lost vertices on reload")
if sum(owners.count(None) for _ in range(1))!=0:
    raise AssertionError("unexpected no-weights vertices")
for slot in mesh.material_slots:
    material=slot.material
    if material is None or "AnyMoveOriginalAlbedoOrViewAware" not in material.node_tree.nodes:
        raise AssertionError("original albedo preservation shader missing")
num_packed=sum(1 for im in bpy.data.images if im.source=="FILE" and im.packed_file is not None)
if num_packed<4:raise AssertionError("source textures not packed")
if sha(original)!=sha_before:
    raise RuntimeError("original Dino source file was overwritten")
print("VIEWAWARE_REOPEN_PASS",len(colourmap),"distinct classes",sum(counts.values()),"colored",
      len(owners)-sum(counts.values()),"original texture","packed",num_packed,
      "zero dominant excluded",len(palette["excluded_zero_weight_bones"]),
      "unchanged source",sha_before,flush=True)
