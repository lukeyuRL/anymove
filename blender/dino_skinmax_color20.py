"""Dino Color20: read exact vertex argmax(Skin Weight), paint only real weighted Bones.

Analyze mode exports real dominant-area counts, not imagined terminal Bone IDs.
Apply mode adds one POINT vertex-colour attribute and overlays only the 20
selected regions onto COPIES of material node trees. Other mesh regions keep
their ORIGINAL albedo texture. No changes to original rig file or textures.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import numpy as np
import bpy

SCHEMA="anymove.dino_color20_dominant_weight/1"
ATTR_NAME="AnyMoveDinoSkinMax20"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""):h.update(c)
    return h.hexdigest()

def dominant_weights(body):
    groupnames={g.index:g.name for g in body.vertex_groups}
    winners=[]
    winningweights=[]
    for vertex in body.data.vertices:
        # Skin Weight=0 is not an influence. Ties are assigned to the
        # lowest Vertex Group index consistently, not using geometry guesses.
        valid=[item for item in vertex.groups if item.weight>0]
        if not valid:
            winners.append(None);winningweights.append(0.)
            continue
        best=max(valid,key=lambda g:(float(g.weight),-int(g.group)))
        winners.append(groupnames[best.group])
        winningweights.append(float(best.weight))
    return winners,np.asarray(winningweights,dtype=float)

def region_bones(winners,rig_info):
    counts=Counter(x for x in winners if x is not None)
    rig_names={b["name"] for b in rig_info["bones"]}
    if any(x not in rig_names for x in counts):
        raise ValueError("a dominant Vertex Group does not match Rig bone name")
    region_names=[
        b["name"] for b in rig_info["bones"]
        if (("_L_" in b["name"] or "_R_" in b["name"]) and
            any(t in b["name"] for t in ("Calf","Foot","Toe")) and
            counts[b["name"]]>0)
    ]
    if len(region_names)!=20:
        raise ValueError(f"Dino expected exactly twenty nonempty hind-leg dominant Skinning regions, found {len(region_names)}: {region_names}")
    return region_names,counts

def analyze(body,rig_info):
    winners,weights=dominant_weights(body)
    names,counts=region_bones(winners,rig_info)
    allowed=set(names)
    i_to_name={i:n for i,n in enumerate(winners)}
    adjacency=Counter()
    for edge in body.data.edges:
        a,b=(i_to_name[int(i)] for i in edge.vertices)
        if a in allowed and b in allowed and a!=b:
            adjacency[tuple(sorted((a,b)))]+=1
    selected=sum(int(counts[name]) for name in names)
    return {
        "schema":SCHEMA,"selected_bone_names":names,
        "total_mesh_vertices":len(winners),
        "unweighted_vertex_count":int(sum(v is None for v in winners)),
        "selected_vertex_count":selected,
        "unselected_vertex_count":len(winners)-selected,
        "dominant_weight_min_selected":float(min(weights[i] for i,v in enumerate(winners) if v in allowed)),
        "counts_by_bone":{n:int(counts[n]) for n in names},
        "zero_color_classes_for_bones_without_dominant_weight":True,
        "mesh_adjacent_edges":[{"a":a,"b":b,"edge_count":int(c)}
                               for (a,b),c in sorted(adjacency.items())],
    }

def hex_to_linear(code):
    if len(code)!=7 or code[0]!="#":
        raise ValueError("Colour code requires #RRGGBB")
    c=np.array([int(code[i:i+2],16)/255 for i in (1,3,5)],dtype=np.float64)
    return np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)

def apply(body,rig_info,palette_path,*,original_blend_path=None):
    package=json.loads(Path(palette_path).read_text(encoding="utf8"))
    if package.get("schema")!="anymove.dino_color20_skinmax_palette/1":
        raise ValueError("expected validated 20-class SkinMax palette")
    palette=package["bone_colors"]
    detail=analyze(body,rig_info)
    if set(palette)!=set(detail["selected_bone_names"]) or len(set(palette.values()))!=20:
        raise ValueError("no-weight Bones may not be listed; 20 effective Bone-to-Color mapping required")
    winners,weights=dominant_weights(body)
    n=len(winners)
    arr=np.zeros((n,4),dtype=np.float32)
    packed={name:hex_to_linear(code) for name,code in palette.items()}
    for i,owner in enumerate(winners):
        if owner in packed:
            arr[i,:3]=packed[owner]
            arr[i,3]=1.
    existing=body.data.color_attributes.get(ATTR_NAME)
    if existing is not None:
        raise ValueError("colour20 already applied to this in-memory Mesh")
    attr=body.data.color_attributes.new(name=ATTR_NAME,type="FLOAT_COLOR",domain="POINT")
    attr.data.foreach_set("color",arr.ravel())
    # Blender 5 reads point colours interpolated across triangles.
    # This is exactly the requested per-VERTEX argmax rule; no fake
    # colours for non-skinning terminal Bones or extra bone-tip markers.
    material_names=[]
    for slot in body.material_slots:
        if slot.material is None or not slot.material.use_nodes:
            raise ValueError("source Dino materials must be node-based")
        original=slot.material
        material=original.copy()
        material.name=original.name+"__AnyMoveSkinMax20"
        slot.material=material
        nt=material.node_tree
        bsdf=next((node for node in nt.nodes if node.type=="BSDF_PRINCIPLED"),None)
        if bsdf is None:
            raise ValueError("missing Principled BSDF on Dino material")
        base=bsdf.inputs["Base Color"]
        src=base.links[0].from_socket if len(base.links) else None
        original_color=base.default_value[:]
        vertex=nt.nodes.new("ShaderNodeVertexColor")
        vertex.name="AnyMoveDominantWeightVertexColor"
        vertex.layer_name=ATTR_NAME
        mix=nt.nodes.new("ShaderNodeMix")
        mix.name="AnyMoveOverlayOriginalAlbedo"
        mix.data_type="RGBA"
        nt.links.new(vertex.outputs["Alpha"],mix.inputs["Factor"])
        if src is not None:nt.links.new(src,mix.inputs["A"])
        else:mix.inputs["A"].default_value=original_color
        nt.links.new(vertex.outputs["Color"],mix.inputs["B"])
        nt.links.new(mix.outputs["Result"],base)
        material_names.append(material.name)
    report={
        **detail,"palette_file":str(Path(palette_path).resolve()),
        "palette_sha256":sha(palette_path),
        "vertex_color_attribute_name":ATTR_NAME,
        "vertex_color_domain":"POINT",
        "color_assignment":"argmax of positive Skin Weight over all Mesh Vertex Groups; overlay only when winner belongs to 20 selected effective hind-leg groups",
        "original_materials_preserved":True,
        "material_copies":material_names,
        "source_blend_path":str(original_blend_path) if original_blend_path else None,
        "source_blend_sha256":sha(original_blend_path) if original_blend_path else None,
        "no_geometry_or_skin_weight_changed":True,
        "warning":"Vertex colors interpolate across face boundaries. No absent/weight-zero Bones are color classes."
    }
    return report
