"""Dino camera-aware colour coding: enumerate ALL useful dominant Skin Bones.

Automatically classifies effective hindlegs + forearms/fingers + tail by
real argmax positive Skin Weight. No arbitrary number-of-colours constraint.
Torso, neck and face original albedo remains intact to preserve appearance.
"""
import bpy,sys,json,numpy as np,pathlib
from collections import Counter
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"blender"))
from dino_skinmax_color20 import dominant_weights
orig=ROOT/"assets/dino/rigged/rigged.blend"
bpy.ops.wm.open_mainfile(filepath=str(orig))
mesh=next(x for x in bpy.data.objects if x.type=="MESH" and x.vertex_groups)
rig=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
owners,weights=dominant_weights(mesh)
counts=Counter(x for x in owners if x is not None)
selected=[]
groups={}
for b in rig["bones"]:
    name=b["name"]
    low=name.lower()
    category=None
    if ("_L_" in name or "_R_" in name):
        if any(x in low for x in ("thigh","calf","foot","toe")):
            category="hind_legs"
        elif any(x in low for x in ("calvicle","clavicle","upperarm","forearm","hand","index","mid")):
            category="forearms_hands"
    if category is None and "_Tail" in name:
        category="tail"
    if category and counts[name]>0:
        selected.append(name)
        groups[name]=category
    elif category and counts[name]==0:
        print("EXCLUDE_ZERO_DOMINANT_WEIGHT",name)
if len(selected)<20:raise AssertionError("too few meaningful active Skin Groups")
edges=Counter()
for e in mesh.data.edges:
    a=owners[e.vertices[0]]
    b=owners[e.vertices[1]]
    if a!=b and a in groups and b in groups:
        edges[tuple(sorted((a,b)))]+=1
data={
  "schema":"anymove.dino_camera_aware_skinmax_selection/1",
  "classes":len(selected),
  "selected_bone_names":selected,
  "categories":groups,
  "dominant_vertex_count":{k:int(counts[k]) for k in selected},
  "colorable_vertex_count":int(sum(counts[n] for n in selected)),
  "total_mesh_vertices":len(mesh.data.vertices),
  "noncolorable_vertex_count":len(mesh.data.vertices)-sum(counts[n] for n in selected),
  "unused_or_zero_dominant_weight_bones":[b["name"] for b in rig["bones"] if counts[b["name"]]==0],
  "active_unpainted_bones":[b["name"] for b in rig["bones"] if counts[b["name"]]>0 and b["name"] not in selected],
  "vertex_mesh_edge_adjacency":[{"a":a,"b":b,"edges":int(n)} for (a,b),n in sorted(edges.items())],
  "source_untouched":True
}
out=ROOT/"runs/dino/skinmax_camera_aware_v1"
out.mkdir(parents=True,exist_ok=True)
f=out/"skinmax_regions.json"
if f.exists():raise FileExistsError(f)
f.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
print("SUMMARY",len(selected),"classes","vertex",data["colorable_vertex_count"],"groups",Counter(groups.values()))
print("UNPAINTED_ACTIVES",data["active_unpainted_bones"])
print("EFFECTIVE_WITHOUT_WEIGHT",data["unused_or_zero_dominant_weight_bones"])
print("VERTEX_COUNTS",data["dominant_vertex_count"])
