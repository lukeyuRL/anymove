"""Apply a camera-aware variable-size Dino Bone palette via true SkinWeight argmax.

Only bones with actual winning vertices, selected for motion visibility,
receive colours; no artificial surface classification for zero-weight ends.
Vertex colours are stored POINT domain and alpha=1 only on selected winners.
Original skin, rig transform, original albedo texture remain unchanged.
"""
from pathlib import Path
import json,numpy as np
from dino_skinmax_color20 import dominant_weights,hex_to_linear,sha
ATTR_NAME="AnyMoveDinoViewAware"
EXPECTED="anymove.dino_skinmax_camera_palette/2"

def apply(body,rig_info,palette_path,*,source_blend):
    palette_path=Path(palette_path).resolve()
    config=json.loads(palette_path.read_text(encoding="utf8"))
    if config.get("schema")!=EXPECTED:
        raise ValueError("not a verified view-aware SkinMax colour palette")
    colors=config["bone_colors"]
    names=set(colors)
    rig_names={b["name"] for b in rig_info["bones"]}
    if not names or names-rig_names or len(colors)!=config["class_count"]:
        raise ValueError("unknown Bone labels in camera-aware palette")
    if len(set(colors.values()))!=len(colors):
        raise ValueError("colours are not uniquely assigned to meaningful Bone groups")
    winner,weights=dominant_weights(body)
    from collections import Counter
    counts=Counter(x for x in winner if x is not None)
    expected={name:int(counts[name]) for name in colors}
    if any(k<1 for k in expected.values()) or expected!=config["counts_by_bone"]:
        raise ValueError("colour palette class counts do not agree with source Skin weights")
    n=len(winner)
    rgba=np.zeros((n,4),dtype=np.float32)
    rgb={name:hex_to_linear(hexcode) for name,hexcode in colors.items()}
    for i,name in enumerate(winner):
        if name in rgb:
            rgba[i,:3]=rgb[name]
            rgba[i,3]=1.
    if int(rgba[:,3].sum())!=config["selected_vertex_count"]:
        raise AssertionError("unexpected number of surface-coloured vertices")
    if body.data.color_attributes.get(ATTR_NAME):
        raise ValueError("already applied view-aware skin colour layer")
    layer=body.data.color_attributes.new(name=ATTR_NAME,type="FLOAT_COLOR",domain="POINT")
    layer.data.foreach_set("color",rgba.reshape(-1))
    copies=[]
    for slot in body.material_slots:
        if slot.material is None or not slot.material.use_nodes:
            raise ValueError("original texture-backed source Material missing")
        original=slot.material
        material=original.copy()
        material.name=original.name+"__AnyMoveViewAware"
        slot.material=material
        nt=material.node_tree
        bsdf=next((node for node in nt.nodes if node.type=="BSDF_PRINCIPLED"),None)
        if bsdf is None:raise ValueError("Dino material has no Principled BSDF")
        base=bsdf.inputs["Base Color"]
        initial_link=base.links[0].from_socket if base.links else None
        initial_rgba=base.default_value[:]
        vertex=nt.nodes.new("ShaderNodeVertexColor")
        vertex.name="AnyMoveDynamicSkinMaxColor"
        vertex.layer_name=ATTR_NAME
        blend=nt.nodes.new("ShaderNodeMix")
        blend.name="AnyMoveOriginalAlbedoOrViewAware"
        blend.data_type="RGBA"
        nt.links.new(vertex.outputs["Alpha"],blend.inputs["Factor"])
        if initial_link is not None:
            nt.links.new(initial_link,blend.inputs["A"])
        else:
            blend.inputs["A"].default_value=initial_rgba
        nt.links.new(vertex.outputs["Color"],blend.inputs["B"])
        nt.links.new(blend.outputs["Result"],base)
        copies.append(material.name)
    return {
        "schema":"anymove.dino_skinmax_camera_applied/1",
        "palette_file":str(palette_path),
        "palette_sha256":sha(palette_path),
        "source_blend":str(source_blend),
        "source_blend_sha256":sha(source_blend),
        "colour_count":len(colors),
        "colour_map":colors,
        "vertex_counts":expected,
        "coloured_vertex_count":int(rgba[:,3].sum()),
        "uncoloured_vertex_count":int(n-rgba[:,3].sum()),
        "all_skin_vertices":n,
        "zero_dominant_bones_coloured":[],
        "active_uncoloured_bones":config["active_unpainted_bones"],
        "vertex_color_domain":"POINT",
        "original_base_color_preserved_on_uncoloured_regions":True,
        "no_skin_weights_or_geometry_modified":True,
        "material_copies":copies,
        "note":"Neighbouring point-colour vertices interpolate across triangles; close camera-projection surfaces may partially mix, and occlusion remains a separate problem."
    }
