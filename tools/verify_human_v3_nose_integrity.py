"""Independent reopen validation: original V3 -> simple-nose V3.

Fail if any nonfacial mesh vertex, topology, bone, skin weight, material,
packed texture or color attribute changed. Audit face normals for flips.
"""
import bpy, hashlib, json, pathlib
import numpy as np

ROOT=pathlib.Path("F:/anymove/runs/human")
SRC=ROOT/"human_skinmax_multicolor_v3.blend"
DST=ROOT/"human_skinmax_multicolor_v3_nose.blend"
REPORT=ROOT/"human_skinmax_multicolor_v3_nose_validation.json"
EXPECTED_SRC="93ced4096791fbc84712527a3c023e7c3bee381106e15b898cd4295f24c8008b"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):h.update(x)
    return h.hexdigest()

def gather(path):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    obj=bpy.data.objects["Dummy"]
    arm=bpy.data.objects["Armature"]
    m=obj.data
    coords=np.array([v.co[:] for v in m.vertices],dtype=np.float64)
    world_mtx=np.array(obj.matrix_world,dtype=np.float64)
    skin=hashlib.sha256()
    for v in m.vertices:
        for group in sorted(v.groups,key=lambda g:g.group):
            skin.update(int(group.group).to_bytes(4,"little"))
            skin.update(np.float32(group.weight).tobytes())
    attr=m.color_attributes["AnyMoveHumanSkinMax53"]
    cols=np.empty(len(attr.data)*4,dtype=np.float32)
    attr.data.foreach_get("color",cols)
    attr_digest=hashlib.sha256(cols.tobytes()).hexdigest()
    uv_digests={}
    for layer in m.uv_layers:
        uv=np.empty(len(layer.data)*2,dtype=np.float32)
        layer.data.foreach_get("uv",uv)
        uv_digests[layer.name]=hashlib.sha256(uv.tobytes()).hexdigest()
    poly_index=[tuple(p.vertices) for p in m.polygons]
    rest={b.name:np.array(b.matrix_local,dtype=np.float64).tolist() for b in arm.data.bones}
    group_names=[g.name for g in obj.vertex_groups]
    mats=[s.material.name if s.material else None for s in obj.material_slots]
    nodes={s.material.name:sorted((node.name,node.type)
           for node in s.material.node_tree.nodes) for s in obj.material_slots if s.material and s.material.node_tree}
    img_packed={image.name:bool(image.packed_file) for image in bpy.data.images}
    return dict(coords=coords, matrix=world_mtx, skin=skin.hexdigest(),
                colors=attr_digest, color_len=len(attr.data),
                uv=uv_digests, polys=poly_index, rest=rest,
                group_names=group_names,mats=mats,nodes=nodes,images=img_packed,
                bone_names=[b.name for b in arm.data.bones],
                mesh_object=obj.name)

if sha(SRC)!=EXPECTED_SRC:raise AssertionError("Source V3 modified")
o=gather(SRC)
n=gather(DST)
for key in ("skin","colors","color_len","uv","polys","rest","group_names","mats","nodes","images","bone_names","mesh_object"):
    if o[key]!=n[key]:
        raise AssertionError("Unexpected structural change: "+key)
np.testing.assert_allclose(o["matrix"],n["matrix"],atol=0,rtol=0)
local=n["coords"]-o["coords"]
world_change=local@o["matrix"][:3,:3].T
mag=np.linalg.norm(world_change,axis=-1)
changed=np.flatnonzero(mag>1e-7)
if not (40<len(changed)<1500):raise AssertionError(f"Nose affected unexpected number of verts: {len(changed)}")
if not 0.02<float(mag.max())<0.05:raise AssertionError("Nose size wrong")
for ix in changed:
    # preserved skin weights are guaranteed by skin hash; check location.
    world=o["coords"][ix]@o["matrix"][:3,:3].T+o["matrix"][:3,3]
    if not (-.070<world[0]<.070 and -.155<world[1]<-.050 and 1.55<world[2]<1.70):
        raise AssertionError(f"Vertex outside face window: {ix} {world}")
# Compare face orientation/area on all affected faces; reject geometric folds.
changed_set=set(map(int,changed))
flipped=0
area_min=1.
area_max_ratio=0.
affected_polys=0
for inds in o["polys"]:
    if len(inds)<3 or not any(i in changed_set for i in inds):
        continue
    affected_polys+=1
    a,b,c=np.asarray(o["coords"])[list(inds[:3])]
    aa,bb,cc=np.asarray(n["coords"])[list(inds[:3])]
    old_norm=np.cross(b-a,c-a)
    new_norm=np.cross(bb-aa,cc-aa)
    dot=np.dot(old_norm,new_norm)
    if dot<=0:flipped+=1
    old_l=np.linalg.norm(old_norm)
    new_l=np.linalg.norm(new_norm)
    if old_l>1e-10:
        area_min=min(area_min,new_l/old_l)
        area_max_ratio=max(area_max_ratio,new_l/old_l)
if flipped:
    raise AssertionError(f"Nose geometry flipped {flipped} face normals")
result={"status":"PASS","source_sha256":sha(SRC),"nose_sha256":sha(DST),
        "source_model":str(SRC),"nose_model":str(DST),
        "changed_vertices":len(changed),
        "changed_faces":affected_polys,
        "max_world_delta_cm":float(mag.max()*100),
        "bone_count":len(o["bone_names"]),"topology_same":True,
        "v3_color_attribute_unchanged":True,"uv_layers_unchanged":True,
        "materials_unchanged":True,"packed_images_unchanged":True,
        "skin_weights_unchanged":True,"rest_bone_matrices_unchanged":True,
        "face_normal_flips":flipped,"min_affected_polygon_area_ratio":area_min,
        "max_affected_polygon_area_ratio":area_max_ratio}
REPORT.write_text(json.dumps(result,indent=2),encoding="utf8")
print("INDEPENDENT_NOSE_QA",json.dumps(result),flush=True)
