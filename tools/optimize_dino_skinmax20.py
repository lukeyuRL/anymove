"""Optimize 20 high-contrast colours for Dino's 20 ACTUAL dominant-weight Bones.

Adjacency is measured from Mesh edges, Rig parent/child and siblings.
Zero-weight terminal Bones are NEVER assigned a class.
Optimizer uses seeded greedy swaps in Lab D65, heuristic rather than
a claimed universal/global optimum.
"""
import json,sys,pathlib
import numpy as np
import cv2
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.dino_color20_palette import PALETTE

def lab_palette(colors):
    pix=np.asarray([[int(x[k:k+2],16)/255. for k in (1,3,5)] for x in colors],dtype=np.float32)
    return cv2.cvtColor(pix[:,None,:],cv2.COLOR_RGB2Lab)[:,0,:]

def optimize(geometry,rig,*,seed=20261102):
    names=list(geometry["selected_bone_names"])
    if len(names)!=20 or len(set(names))!=20:
        raise ValueError("geometry must report exactly 20 nonempty Skin Max Bones")
    weights=dict(geometry["counts_by_bone"])
    if any(weights.get(n,0)<=0 for n in names) or sum(weights.values())!=geometry["selected_vertex_count"]:
        raise ValueError("cannot colour a bone without dominant-weight vertices")
    bmap={b["name"]:b for b in rig["bones"]}
    order=list(bmap)
    lookup={s:i for i,s in enumerate(names)}
    edges={}
    def add(name1,name2,weight):
        if name1 not in lookup or name2 not in lookup or name1==name2:
            return
        a,b=sorted((lookup[name1],lookup[name2]))
        edges[(a,b)]=edges.get((a,b),0)+weight
    # Real visible boundary edges are the most critical for interpolation.
    for entry in geometry["mesh_adjacent_edges"]:
        add(entry["a"],entry["b"],2+3*np.sqrt(entry["edge_count"]))
    for n in names:
        parent=bmap[n]["parent"]
        if parent>=0:
            add(n,order[parent],9)
        children=[m for m in names if bmap[m]["parent"]==parent and m!=n]
        for c in children:add(n,c,6)
        # Opposite limbs should never have identical looking labels.
        if "_L_" in n:add(n,n.replace("_L_","_R_"),12)
    for side in ("_L_","_R_"):
        same=[x for x in names if side in x]
        for i,x in enumerate(same):
            for y in same[i+1:]:
                add(x,y,1.4)
    observed=np.asarray([(a,b,w) for (a,b),w in edges.items()],float)
    aa=observed[:,0].astype(int)
    bb=observed[:,1].astype(int)
    ww=observed[:,2]
    labs=lab_palette(PALETTE)
    dist=np.linalg.norm(labs[:,None,:]-labs[None,:,:],axis=-1)
    left=lookup["CC_Dinosaur_L_Calf"]
    right=lookup["CC_Dinosaur_R_Calf"]
    blue=PALETTE.index("#2979FF")
    red=PALETTE.index("#E53935")
    rng=np.random.default_rng(seed)
    def score(mapping):
        dd=dist[mapping[aa],mapping[bb]]
        penalty=float(np.sum(ww*np.maximum(0,65-dd)**2))
        key=ww>7
        low=float(dd[key].min()) if key.any() else float(dd.min())
        return (-penalty+0.75*low+.022*float(np.dot(ww,np.sqrt(dd))),
                low,float(np.average(dd,weights=ww)))
    best=None
    all_nodes=[i for i in range(20) if i not in (left,right)]
    free_colors=[i for i in range(20) if i not in (blue,red)]
    for _ in range(18):
        mapping=np.empty(20,int)
        mapping[left]=blue
        mapping[right]=red
        mapping[all_nodes]=rng.permutation(free_colors)
        current=score(mapping)
        for _ in range(85):
            candidate=None
            candidate_score=current
            for i,x in enumerate(all_nodes):
                for y in all_nodes[i+1:]:
                    new=mapping.copy()
                    new[x],new[y]=new[y],new[x]
                    s=score(new)
                    if s[0]>candidate_score[0]+1e-7:
                        candidate_score=s;candidate=new
            if candidate is None:break
            mapping,current=candidate,candidate_score
        if best is None or current[0]>best[0][0]:
            best=(current,mapping.copy())
    current,mapping=best
    palette={name:PALETTE[mapping[i]] for i,name in enumerate(names)}
    if len(set(palette.values()))!=20 or palette["CC_Dinosaur_L_Calf"]!="#2979FF" or palette["CC_Dinosaur_R_Calf"]!="#E53935":
        raise AssertionError("colour optimizer violated uniqueness or side anchors")
    absent=[b["name"] for b in rig["bones"] if ("_L_" in b["name"] or "_R_" in b["name"])
            and any(k in b["name"] for k in ("Calf","Foot","Toe"))
            and b["name"] not in palette]
    return {
        "schema":"anymove.dino_color20_skinmax_palette/1",
        "seed":seed,
        "source_geometry_file":"runs/dino/color20_candidate_v1/skinmax_geometry.json",
        "selected_vertex_count":geometry["selected_vertex_count"],
        "total_mesh_vertices":geometry["total_mesh_vertices"],
        "color_class_count":len(palette),
        "bone_colors":palette,
        "skinmax_vertex_counts":{n:weights[n] for n in names},
        "excluded_bones_without_dominant_skin_weight":absent,
        "mesh_boundary_plus_rig_adjacency_count":len(edges),
        "priority_adjacency_min_deltaE76":round(current[1],2),
        "weighted_adjacency_mean_deltaE76":round(current[2],2),
        "original_thigh_skin_unchanged":True,
        "unknown_or_zero_weight_bone_is_not_colored":True,
        "color_science_caveat":"Seeded lab DeltaE76 heuristic; not validated colourblind-safe or optimal on all rendered lighting",
    }

if __name__=="__main__":
    geo=json.loads((ROOT/"runs/dino/color20_candidate_v1/skinmax_geometry.json").read_text(encoding="utf8"))
    rig=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
    path=ROOT/"runs/dino/color20_candidate_v1/skinmax_palette_v2.json"
    if path.exists():raise FileExistsError("color20 palette already exists")
    result=optimize(geo,rig)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(result,indent=2,ensure_ascii=False))
