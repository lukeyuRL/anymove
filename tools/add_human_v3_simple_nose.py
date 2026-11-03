"""Mesh-only, reversible Human SkinMax V3 simple-nose sculpt.

Loads the original audited 53-color V3 and saves a DIFFERENT blend. The face
surface is displaced with a smooth, localized bridge+tip Gaussian along -Y,
without changing topology, skin weights, color attributes, materials, rig or
rest basis. The original .blend is never overwritten.
"""
import bpy, hashlib, json, math, pathlib
import numpy as np
from mathutils import Vector

ROOT=pathlib.Path("F:/anymove/runs/human")
SOURCE=ROOT/"human_skinmax_multicolor_v3.blend"
TARGET=ROOT/"human_skinmax_multicolor_v3_nose.blend"
REPORT=ROOT/"human_skinmax_multicolor_v3_nose_audit.json"
EXPECTED_SOURCE_SHA="93ced4096791fbc84712527a3c023e7c3bee381106e15b898cd4295f24c8008b"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(1<<20),b""):
            h.update(buf)
    return h.hexdigest()

def mesh_attributes(mesh):
    return {a.name:{"data_type":a.data_type,"domain":a.domain,"length":len(a.data)}
            for a in mesh.color_attributes}

def vertex_skin_signature(mesh):
    h=hashlib.sha256()
    for v in mesh.vertices:
        for g in sorted(v.groups,key=lambda q:q.group):
            h.update(int(g.group).to_bytes(4,"little"))
            h.update(np.float32(g.weight).tobytes())
    return h.hexdigest()

def falloff(x,xmid,sigma):
    return math.exp(-.5*((x-xmid)/sigma)**2)

if not SOURCE.is_file():raise FileNotFoundError(SOURCE)
if TARGET.exists() or REPORT.exists():
    raise FileExistsError("Refuse to overwrite existing nose model or report")
initial_sha=sha(SOURCE)
if initial_sha!=EXPECTED_SOURCE_SHA:
    raise RuntimeError("V3 source SHA changed since audited: "+initial_sha)

if pathlib.Path(bpy.data.filepath).resolve()!=SOURCE.resolve():
    raise RuntimeError("Open audited Human V3 source before executing")

arm=bpy.data.objects.get("Armature")
obj=bpy.data.objects.get("Dummy")
if arm is None or obj is None or arm.type!="ARMATURE" or obj.type!="MESH":
    raise RuntimeError("Expected original Armature and skinned Dummy")
if len(arm.data.bones)!=55 or len(obj.data.vertices)!=30188:
    raise RuntimeError("Unexpected original Human V3 rig/mesh")
color=obj.data.color_attributes.get("AnyMoveHumanSkinMax53")
if color is None or color.domain!="POINT" or color.data_type!="FLOAT_COLOR":
    raise RuntimeError("Original V3 SkinMax53 vertex colors missing")

before_skin=vertex_skin_signature(obj.data)
before_attributes=mesh_attributes(obj.data)
before_material=[(slot.material.name if slot.material else None) for slot in obj.material_slots]
old=np.asarray([tuple(v.co) for v in obj.data.vertices],dtype=float)
forward_y=-1.0
head_group=obj.vertex_groups.get("CC_Base_Head")
if head_group is None:raise RuntimeError("CC_Base_Head vertex group not found")
inv=obj.matrix_world.inverted()
changes=[]
for v in obj.data.vertices:
    weights={p.group:p.weight for p in v.groups}
    if not weights or max(weights,key=weights.get)!=head_group.index:
        continue
    p=obj.matrix_world @ v.co
    x,y,z=float(p.x),float(p.y),float(p.z)
    if abs(x)>.065 or z<1.565 or z>1.691 or y>-.055:
        continue
    surface_gate=max(0.0,min(1.0,(-.055-y)/.028))
    # The bridge is narrow and extends upward toward the brow;
    # the tip is wider, rounded and approximately 3cm in front of the face.
    bridge=.012*falloff(x,.0002,.013)*falloff(z,1.642,.035)
    tip=.028*falloff(x,.0002,.022)*falloff(z,1.611,.021)
    # Very subtle rounded lower nose wings; no modeled nostril holes.
    wing=(.007*falloff(abs(x),.022,.011)*falloff(z,1.596,.013))
    extension=(bridge+tip+wing)*surface_gate
    if extension<.00005:
        continue
    new_world=Vector((x,y+forward_y*extension,z))
    v.co=inv @ new_world
    changes.append((v.index,extension))
obj.data.update()
bpy.context.view_layer.update()

local_delta=np.asarray([tuple(v.co) for v in obj.data.vertices])-old
world_basis=np.asarray(obj.matrix_world.to_3x3(),dtype=float)
delta=np.linalg.norm(local_delta @ world_basis.T,axis=1)
changed=int(np.count_nonzero(delta>1e-7))
max_change=float(delta.max())
if changed<40 or changed>2500 or not (.015<max_change<.05):
    raise AssertionError(f"Nose deformation safety bounds violated: {changed} verts, {max_change:.5f}m")
if vertex_skin_signature(obj.data)!=before_skin or mesh_attributes(obj.data)!=before_attributes:
    raise AssertionError("Skin group or vertex color attribute structure was altered")
if len(obj.data.polygons)!=30176 or [slot.material.name if slot.material else None for slot in obj.material_slots]!=before_material:
    raise AssertionError("Original mesh/material topology changed")
if len(arm.data.bones)!=55:
    raise AssertionError("Armature was changed")

bpy.ops.wm.save_as_mainfile(filepath=str(TARGET))
if not TARGET.exists() or TARGET.stat().st_size<200_000:
    raise RuntimeError("Could not save nose sculpt")
if sha(SOURCE)!=EXPECTED_SOURCE_SHA:
    raise AssertionError("Original V3 source blend unexpectedly mutated")
audit={
  "status":"PASS",
  "schema":"anymove.human_skinmax53_v3_nose_mesh_sculpt/1",
  "source_blend":str(SOURCE),
  "source_sha256":EXPECTED_SOURCE_SHA,
  "output_blend":str(TARGET),
  "output_sha256":sha(TARGET),
  "nose_approx_center_world_m":[0.0002,-.097,1.615],
  "forward_axis":"world -Y",
  "vertices_modified":changed,
  "max_vertex_displacement_m":max_change,
  "max_vertex_displacement_cm":max_change*100.0,
  "all_changed_vertices_are_CC_Base_Head":True,
  "mesh_verts_before_after":[30188,len(obj.data.vertices)],
  "mesh_faces_before_after":[30176,len(obj.data.polygons)],
  "bones_before_after":[55,len(arm.data.bones)],
  "color_attribute":"AnyMoveHumanSkinMax53",
  "color_regions":53,
  "skinning_sha256":before_skin,
  "skinning_unchanged":True,
  "material_unchanged":True,
  "source_untouched":True,
}
REPORT.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf8")
print("HUMAN_V3_NOSE_BUILD",json.dumps(audit,ensure_ascii=False),flush=True)
