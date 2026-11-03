"""Optimize Dino SkinMax palette using *actual* projected front/side conflicts.

No fixed colour-count limit. Number of colour classes is driven only by
effective argmax Skinning Weight bone regions useful to 3D Motion tracking.

Assignment is deterministic, seeded Lab-space heuristic. The colour
mapping is one fixed Bone->Color across all cameras and all poses.
Front/Side proximity constraints carry twice Back/Opposite weight.
"""
import json,sys,colorsys,pathlib
from collections import defaultdict
import numpy as np
import cv2

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.dino_color20_palette import PALETTE as INITIAL20

def lab_hex(colors):
    rgb=np.asarray([[int(code[k:k+2],16)/255. for k in (1,3,5)] for code in colors],np.float32)
    return cv2.cvtColor(rgb[:,None,:],cv2.COLOR_RGB2LAB)[:,0,:]

def candidates():
    colors=set(INITIAL20)
    for step in range(36):
        h=((step+.13)/36.)%1.
        for s,v in ((.95,1.),(.80,.9),(.96,.78)):
            rgb=colorsys.hsv_to_rgb(h,s,v)
            colors.add("#"+"".join(f"{round(255*c):02X}" for c in rgb))
    return sorted(colors)

def graph(region,geometry,rig):
    selected=region["selected_bone_names"]
    name_to_idx={n:i for i,n in enumerate(selected)}
    if len(selected)!=len(name_to_idx):raise ValueError("duplicate bone")
    bones={b["name"]:b for b in rig["bones"]}
    names=[b["name"] for b in rig["bones"]]
    weights=defaultdict(float)
    tags=defaultdict(set)
    def edge(a,b,w,tag):
        if a==b or a not in name_to_idx or b not in name_to_idx or w<=0:return
        u,v=sorted((name_to_idx[a],name_to_idx[b]))
        weights[(u,v)]+=float(w)
        tags[(u,v)].add(tag)
    for p in geometry["near_pairs"]:
        # Screen proximity: data-dependent view visibility, not arbitrary
        # anatomical naming. Clipped to avoid overfitting one static pose.
        val=min(float(p["weight"]),130.)
        edge(p["a"],p["b"],val,"pixel_near")
    for e in region["vertex_mesh_edge_adjacency"]:
        edge(e["a"],e["b"],9.*np.sqrt(e["edges"]),"mesh_adjacent")
    for b in selected:
        data=bones[b]
        parent=data["parent"]
        if parent>=0:
            edge(b,names[parent],45.,"rig_parent")
    for i,a in enumerate(selected):
        for b in selected[i+1:]:
            ia=bones[a]["parent"]; ib=bones[b]["parent"]
            if ia==ib and ia>=0:
                edge(a,b,30.,"rig_siblings")
            if "_L_" in a and a.replace("_L_","_R_")==b:
                edge(a,b,75.,"left_right_mirror")
            # Tail segments might fold over one another in any pose.
            if "Tail" in a and "Tail" in b:
                dist=abs(int(a.rsplit("Tail",1)[1])-int(b.rsplit("Tail",1)[1]))
                edge(a,b,26./max(dist,1),"tail_coils")
    u,v=np.asarray(sorted(weights),dtype=int).T
    ww=np.asarray([weights[p] for p in sorted(weights)],float)
    return u,v,ww,tags

def pick_palette(quantity,anchors,*,seed):
    pool=candidates()
    labs=lab_hex(pool)
    # Candidate colours are vivid; avoid deep brown and low contrast against
    # the retained dark skin. Include any requested anchors.
    chosen=[]
    for col in anchors:
        if col not in pool:raise ValueError("anchor not in candidate pool: "+col)
        idx=pool.index(col)
        if idx not in chosen:chosen.append(idx)
    rng=np.random.default_rng(seed)
    while len(chosen)<quantity:
        d=np.linalg.norm(labs[:,None,:]-labs[np.asarray(chosen)][None,:,:],axis=2)
        nearest=d.min(axis=1)
        nearest[np.asarray(chosen)]=-1
        # Greedy farthest selection makes 41 colours unique globally,
        # while the graph later prioritizes pairs that actually touch.
        bests=np.flatnonzero(nearest>=nearest.max()-1e-6)
        chosen.append(int(rng.choice(bests)))
    selected=[pool[i] for i in chosen]
    chosenlab=lab_hex(selected)
    dd=np.linalg.norm(chosenlab[:,None,:]-chosenlab[None,:,:],axis=2)
    np.fill_diagonal(dd,np.inf)
    return selected,chosenlab,float(dd.min())

def optimize(regions,view_conflicts,rig,*,seed=20261102):
    names=list(regions["selected_bone_names"])
    if len(names)!=41:raise ValueError("unexpected list of nonzero dominant Skin Bones")
    if set(regions["categories"].values())!={"hind_legs","forearms_hands","tail"}:
        raise ValueError("unexpected categories")
    anchor={
        "CC_Dinosaur_L_Thigh":"#2979FF",
        "CC_Dinosaur_R_Thigh":"#E53935",
        "CC_Dinosaur_L_Upperarm":"#00F0FF",
        "CC_Dinosaur_R_Upperarm":"#FFC850",
        "CC_Dinosaur_Tail07":"#FF28F0",
    }
    anchor_names=list(anchor)
    colors,labs,global_min=pick_palette(len(names),list(anchor.values()),seed=seed)
    anchors={names.index(n):colors.index(c) for n,c in anchor.items()}
    aa,bb,weights,tag=graph(regions,view_conflicts,rig)
    labdiff=np.linalg.norm(labs[:,None,:]-labs[None,:,:],axis=2)
    node_weights=np.bincount(np.concatenate((aa,bb)),weights=np.concatenate((weights,weights)),minlength=len(names))
    options=np.asarray([k for k in range(len(names)) if k not in anchors],int)
    open_colors=np.asarray([k for k in range(len(colors)) if k not in anchors.values()],int)
    rng=np.random.default_rng(seed)
    w=np.log1p(weights)*3.
    importance=w>np.quantile(w,.66)
    # Cameras where the model already works are the key: every pair that
    # visually approaches within 10 source pixels in Front/Side gets a
    # strong perceptual-near constraint, not just true Mesh-edge adjacency.
    focus=[]
    fweight=[]
    for pair in view_conflicts['near_pairs']:
        count=int(pair['view_count']['front'])+int(pair['view_count']['side'])
        if count<=0:continue
        focus.append((names.index(pair['a']),names.index(pair['b'])))
        fweight.append(min(8.,1.+np.sqrt(count)))
    focus=np.asarray(focus,dtype=int)
    focusw=np.asarray(fweight,dtype=float)
    def score(mapping):
        delta=labdiff[mapping[aa],mapping[bb]]
        close=np.maximum(0,58.-delta)
        # Primary criterion: adjacent/near pixels should be >~58 DeltaE76.
        loss=np.dot(w,close**2)
        frontside=labdiff[mapping[focus[:,0]],mapping[focus[:,1]]]
        # Eliminate low-contrast same-frame matches before optimizing the
        # higher-level overall palette. Heavy weight below DeltaE76=35.
        loss+=380.*np.dot(focusw,np.maximum(0.,36.-frontside)**2)
        # Improve differences even above threshold on most connected nodes.
        reward=np.dot(w,np.minimum(delta,90))
        worst=np.min(delta[importance])
        return (float(loss-.14*reward-3.*worst),
                float((w*(delta<40)).sum()),float(np.min(delta[importance])),
                float(np.average(delta,weights=w)))
    best=None
    for restart in range(12):
        order=np.empty(len(names),int)
        for i,colidx in anchors.items():order[i]=colidx
        order[options]=rng.permutation(open_colors)
        cur=score(order)
        baseline=cur if restart==0 else None
        for _ in range(55):
            guess=cur
            temp=None
            # Most severe visual collisions get priority in greedy search.
            for ii,pos in enumerate(options):
                for other in options[ii+1:]:
                    trial=order.copy()
                    trial[pos],trial[other]=trial[other],trial[pos]
                    candidate=score(trial)
                    if candidate[0]<guess[0]-1e-5:
                        guess=candidate;temp=trial
            if temp is None:break
            cur=guess
            order=temp
        if best is None or cur[0]<best[0][0]:best=(cur,order.copy())
        print("ASSIGNMENT_RESTART",restart+1,"edges",len(aa),"loss",round(cur[0],2),
              "colliding-edge-weight",round(cur[1],2),
              "critical_min_delta",round(cur[2],2),flush=True)
    cur,assign=best
    bonecolors={name:colors[assign[i]] for i,name in enumerate(names)}
    if len(set(bonecolors.values()))!=len(names):
        raise AssertionError("41 colors must be globally distinct")
    for n,c in anchor.items():
        if bonecolors[n]!=c:raise AssertionError("semantic anchor moved")
    edge_details=[]
    for a,b,weight in zip(aa,bb,weights):
        dist=float(labdiff[assign[a],assign[b]])
        edge_details.append({"a":names[a],"b":names[b],
                             "deltaE76":round(dist,2),
                             "weight":round(float(weight),2),
                             "tags":sorted(tag[(int(a),int(b))])})
    near_view={v:{"problem_pairs_deltaE_lt_35":[],"close_pairs":0}
               for v in ("front","side","back","opposite_side")}
    for pair in view_conflicts["near_pairs"]:
        d=float(np.linalg.norm(labs[assign[names.index(pair["a"])]]-
                               labs[assign[names.index(pair["b"])]]))
        for view,num in pair["view_count"].items():
            if num>0:
                near_view[view]["close_pairs"]+=1
                if d<35:near_view[view]["problem_pairs_deltaE_lt_35"].append({
                  "a":pair["a"],"b":pair["b"],"deltaE76":round(d,1),
                  "poses_near":int(num)})
    return {
        "schema":"anymove.dino_skinmax_camera_palette/1",
        "seed":seed,
        "class_count":len(names),
        "bone_colors":bonecolors,
        "selected_vertex_count":regions["colorable_vertex_count"],
        "total_mesh_vertices":regions["total_mesh_vertices"],
        "categories":regions["categories"],
        "counts_by_bone":regions["dominant_vertex_count"],
        "excluded_zero_weight_bones":regions["unused_or_zero_dominant_weight_bones"],
        "active_unpainted_bones":regions["active_unpainted_bones"],
        "source_for_projection":"training 32 P4 poses, Front/Side weighted 2x over back/opposite, 10px camera proximity",
        "candidate_color_count":len(candidates()),
        "all_selected_colors_globally_unique":True,
        "globally_minimum_deltaE76":round(global_min,2),
        "projection_and_rig_graph_edges":len(aa),
        "weighted_adjacency_min_critical_deltaE76":round(cur[2],2),
        "weighted_adjacency_mean_deltaE76":round(cur[3],2),
        "view_conflict_diagnostics":near_view,
        "worst_edges":sorted(edge_details,key=lambda q:q["deltaE76"])[:24],
        "source_mesh_invariant":True,
        "unpainted_skin_original_albedo":True,
        "caveats":["41 unique colors does not prevent geometric occlusion",
                   "Lab DeltaE76 in Blender Cycles is approximate and may change under lighting, AA and video generation",
                   "No demonstrated Reader accuracy gains until matched training A/B"],
    }

if __name__=="__main__":
    base=ROOT/"runs/dino/skinmax_camera_aware_v1"
    region=json.loads((base/"skinmax_regions.json").read_text(encoding="utf8"))
    graphdata=json.loads((base/"camera_conflicts.json").read_text(encoding="utf8"))
    rig=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
    output=base/"optimized_palette_v2.json"
    if output.exists():raise FileExistsError("palette exists, do not replace")
    data=optimize(region,graphdata,rig)
    output.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf8")
    print("DINO_CAMERA_OPTIMIZED",data["class_count"],
          "unique colors, min global DeltaE76",data["globally_minimum_deltaE76"],
          "view-neighbor scores",data["view_conflict_diagnostics"],flush=True)
