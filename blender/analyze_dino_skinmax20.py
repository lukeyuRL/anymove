"""Inspect Dino dominant-weight regions from unmodified local .blend (no edits)."""
import bpy,sys,json,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"blender"))
from dino_skinmax_color20 import analyze
blend=root/"assets/dino/rigged/rigged.blend"
rig=root/"assets/dino/rigged/rig_info.json"
output=root/"runs/dino/color20_candidate_v1/skinmax_geometry.json"
if output.exists():raise FileExistsError(output)
bpy.ops.wm.open_mainfile(filepath=str(blend))
body=next(x for x in bpy.data.objects if x.type=="MESH" and x.vertex_groups)
report=analyze(body,json.loads(rig.read_text(encoding="utf8")))
report["source_blend"]=str(blend)
output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
print("SKINMAX_ANALYZED",report["selected_vertex_count"],"out of",report["total_mesh_vertices"])
print("REGIONS",report["counts_by_bone"])
print("ADJACENT_EDGES",len(report["mesh_adjacent_edges"]))
