"""Constrained Dino camera palette repair: do not worsen Front/Side near pairs.

v2 primary palette has exactly one borderline Front near pair (<35 Lab)
and zero Side. Optimize actual rendered weak pairs without increasing
the number of projection-near conflicts on 32 *TRAINING* P4 poses.
"""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(r"F:\anymove")
sys.path.insert(0,str(ROOT))
from tools.optimize_dino_camera_palette import candidates,lab_hex,graph

base=ROOT/"runs/dino/skinmax_camera_aware_v1"
orig=json.loads((base/"optimized_palette_final.json").read_text(encoding="utf8"))
reg=json.loads((base/"skinmax_regions.json").read_text(encoding="utf8"))
proj=json.loads((base/"camera_conflicts.json").read_text(encoding="utf8"))
render=json.loads((base/"rendered_camera_colour_audit.json").read_text(encoding="utf8"))
rig=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
names=reg["selected_bone_names"]
palette=[orig["bone_colors"][n] for n in names]
fixed={"CC_Dinosaur_L_Thigh","CC_Dinosaur_R_Thigh","CC_Dinosaur_L_Upperarm",
       "CC_Dinosaur_R_Upperarm","CC_Dinosaur_Tail07"}
observed=set()
for v in ("front","side"):
    for ex in render["view_metrics"][v]["near_color_examples"]:
        observed.add(tuple(sorted((names.index(ex["a"]),names.index(ex["b"])))))
h=np.asarray(sorted(observed),int)
near=[]
for pair in proj["near_pairs"]:
    count=pair["view_count"]["front"]+pair["view_count"]["side"]
    if count>0:
        near.append((names.index(pair["a"]),names.index(pair["b"]),count))
near=np.asarray(near,int)
p,q,w,tags=graph(reg,proj,rig)
wg=np.log1p(w)*3
def score(colors):
    labs=lab_hex(colors)
    n1=np.linalg.norm(labs[near[:,0]]-labs[near[:,1]],axis=1)
    d=np.linalg.norm(labs[h[:,0]]-labs[h[:,1]],axis=1)
    gen=np.linalg.norm(labs[p]-labs[q],axis=1)
    return (
      int(np.count_nonzero(n1<35)),
      float(900*np.sum(np.maximum(0.,69-d)**2)+
            150*np.dot((np.minimum(8,1+np.sqrt(near[:,2]))),np.maximum(0.,36-n1)**2)+
            np.dot(wg,np.maximum(0.,58-gen)**2)-.11*np.dot(wg,np.minimum(gen,90))),
      int(np.count_nonzero(d<69)),
      float(d.min())
    )
before=score(palette)
pool=candidates()
for itr in range(18):
    current=score(palette)
    bad=np.linalg.norm(lab_hex(palette)[h[:,0]]-lab_hex(palette)[h[:,1]],axis=1)
    active={int(x) for pair in h for x in pair if names[int(x)] not in fixed}
    # Also allow fixing a single borderline front/sided projected pair.
    neardist=np.linalg.norm(lab_hex(palette)[near[:,0]]-lab_hex(palette)[near[:,1]],axis=1)
    for i in np.where(neardist<37)[0]:
        active.update(int(x) for x in near[i,:2] if names[int(x)] not in fixed)
    chosen=None
    best=current
    labs=lab_hex(palette)
    for ix in sorted(active):
        for col in pool:
            if col in palette:continue
            other=np.linalg.norm(lab_hex([col])[0]-labs,axis=1)
            other[ix]=np.inf
            if other.min()<18:continue
            cand=palette.copy()
            cand[ix]=col
            val=score(cand)
            if (val[0]<best[0] or val[0]==best[0] and val[1]<best[1]-1e-5):
                best=val;chosen=cand
    if chosen is None:break
    palette=chosen
    print("CONSTRAINED_COLOUR_CHANGE",itr+1,best,flush=True)
    if best[0]==0 and best[2]==0:break
after=score(palette)
if after[0]>before[0]:
    raise AssertionError("camera projection Front/Side conflict worsened")
if len(set(palette))!=len(palette):raise AssertionError("selected color duplicate")
for i,n in enumerate(names):
    if n in fixed and palette[i]!=orig["bone_colors"][n]:
        raise AssertionError("important color identity changed")
lab=lab_hex(palette)
summary={}
for v in ("front","side","back","opposite_side"):
    exceptions=[]
    for pair in proj["near_pairs"]:
        if not pair["view_count"][v]:continue
        dist=np.linalg.norm(lab[names.index(pair["a"])]-lab[names.index(pair["b"])])
        if dist<35:
            exceptions.append({"a":pair["a"],"b":pair["b"],"deltaE76":round(float(dist),1),
                               "near_pose_count":pair["view_count"][v]})
    summary[v]={"pairs_below_deltaE35":len(exceptions),"problems":exceptions}
out={**orig,"schema":"anymove.dino_skinmax_camera_palette/2",
     "bone_colors":dict(zip(names,palette)),
     "postrefinement_camera_near_conflicts":summary,
     "source_for_projection":"32 training poses, Front and Side priority; supplementary 3 diagnostic image renders used for colour QA",
     "render_diagnostic_colours_used_for_refinement":True,
     "constrained_optim_initial_score":str(before),
     "constrained_optim_final_score":str(after),
     "validation_pose_not_used_for_reader_score":True,
     "min_overall_selected_deltaE76":round(float(np.min(np.linalg.norm(lab[:,None]-lab[None,:],axis=2)+999*np.eye(len(names)))),2)}
file=base/"optimized_palette_constrained_v4.json"
if file.exists():raise FileExistsError(file)
file.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf8")
print("CONSTRAINED_COLOR_RESULT",before,"=>",after,"cam",[(k,v["pairs_below_deltaE35"]) for k,v in summary.items()])
