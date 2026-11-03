"""Additional rendered-colour hard-pair repair (Dino camera palette v3).

Prior heuristic used training Front/Side projected Part-ID proximity. This
inspection refinement additionally repairs concrete low rendered-Lab pairs
from three DIAGNOSTIC renders; those samples are not independent validation
after this change. New unseen tests are required to assert generalization.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(r"F:\anymove")
sys.path.insert(0,str(ROOT))
from tools.optimize_dino_camera_palette import candidates,lab_hex,graph

base=ROOT/"runs/dino/skinmax_camera_aware_v1"
palette=json.loads((base/"optimized_palette_final.json").read_text(encoding="utf8"))
regions=json.loads((base/"skinmax_regions.json").read_text(encoding="utf8"))
projection=json.loads((base/"camera_conflicts.json").read_text(encoding="utf8"))
render=json.loads((base/"rendered_camera_colour_audit.json").read_text(encoding="utf8"))
rig=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
names=list(regions["selected_bone_names"])
colors=[palette["bone_colors"][n] for n in names]
pool=candidates()
anchors={"CC_Dinosaur_L_Thigh","CC_Dinosaur_R_Thigh",
         "CC_Dinosaur_L_Upperarm","CC_Dinosaur_R_Upperarm","CC_Dinosaur_Tail07"}
a,b,w,tags=graph(regions,projection,rig)
weight=np.log1p(w)*3
fs=[]
fweights=[]
for p in projection["near_pairs"]:
    near=p["view_count"]["front"]+p["view_count"]["side"]
    if near>0:
        fs.append((names.index(p["a"]),names.index(p["b"])))
        fweights.append(min(8.,1+np.sqrt(near)))
fs=np.asarray(fs,int)
fw=np.asarray(fweights,float)
observed=set()
for v in ("front","side"):
    for item in render["view_metrics"][v]["near_color_examples"]:
        observed.add(tuple(sorted((names.index(item["a"]),names.index(item["b"])))))
hard=np.asarray(sorted(observed),int)
print("RENDER_HARD_PAIRS",[(names[a],names[b]) for a,b in hard],flush=True)

def score(colours):
    lab=lab_hex(colours)
    dd=np.linalg.norm(lab[a]-lab[b],axis=-1)
    local=np.linalg.norm(lab[fs[:,0]]-lab[fs[:,1]],axis=-1)
    harddist=np.linalg.norm(lab[hard[:,0]]-lab[hard[:,1]],axis=-1)
    penalty=np.dot(weight,np.maximum(0.,58.-dd)**2)
    penalty+=350*np.dot(fw,np.maximum(0,36-local)**2)
    penalty+=2000*np.sum(np.maximum(0,73-harddist)**2)
    penalty-=.15*np.dot(weight,np.minimum(dd,90))
    return float(penalty),int((harddist<70).sum()),float(harddist.min()),int((local<35).sum())

before=score(colors)
for step in range(20):
    lab=lab_hex(colors)
    focus={int(x) for pair in hard for x in pair}
    harddist=np.linalg.norm(lab[hard[:,0]]-lab[hard[:,1]],axis=-1)
    for j in np.where(harddist<75)[0]:
        focus.update(map(int,hard[j]))
    focus={k for k in focus if names[k] not in anchors}
    if not focus:break
    baseline=score(colors)
    candidate=None
    selected=set(colors)
    for ix in sorted(focus):
        for candidatehex in pool:
            if candidatehex in selected:continue
            dd=np.linalg.norm(lab_hex([candidatehex])[0]-lab,axis=-1)
            dd[ix]=np.inf
            if dd.min()<18:continue
            trial=colors.copy()
            trial[ix]=candidatehex
            test=score(trial)
            if test[0]<baseline[0]-1e-4:
                baseline=test
                candidate=trial
    if candidate is None:break
    colors=candidate
    print("HARD_PAIR_REFINE",step+1,score(colors),flush=True)
    if score(colors)[1]==0 and score(colors)[3]==0:
        break

if len(set(colors))!=len(colors):raise AssertionError("duplicate colours introduced")
for i,n in enumerate(names):
    if n in anchors and colors[i]!=palette["bone_colors"][n]:
        raise AssertionError("semantic anchor changed")
labs=lab_hex(colors)
summaries={}
for view in ("front","side","back","opposite_side"):
    failures=[]
    for pair in projection["near_pairs"]:
        num=pair["view_count"][view]
        if num<=0:continue
        distance=float(np.linalg.norm(labs[names.index(pair["a"]) ]-
                                      labs[names.index(pair["b"]) ]))
        if distance<35:
            failures.append({"a":pair["a"],"b":pair["b"],
                             "deltaE76":round(distance,1),"near_pose_count":int(num)})
    summaries[view]={"pairs_below_deltaE35":len(failures),"problems":failures}
out={
 **palette,
 "schema":"anymove.dino_skinmax_camera_palette/2",
 "bone_colors":dict(zip(names,colors)),
 "min_overall_selected_deltaE76":round(float(np.min(np.linalg.norm(labs[:,None]-labs[None,:],axis=-1)+999*np.eye(len(names)))),2),
 "postrefinement_camera_near_conflicts":summaries,
 "rendered_diagnostic_hard_pairs":[{"a":names[a],"b":names[b],
      "theoretical_deltaE76":round(float(np.linalg.norm(labs[a]-labs[b])),2)} for a,b in hard],
 "postrefinement_rendered_hardpair_objective_before":round(before[0],1),
 "postrefinement_rendered_hardpair_objective_after":round(score(colors)[0],1),
 "diagnostic_test_poses_used_for_design":[9001,9002,9007],
 "future_unseen_pose_required_for_unbiased_test":True,
}
dest=base/"optimized_palette_render_v3.json"
if dest.exists():raise FileExistsError(dest)
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
print("FINAL_CAMERA_RGB_COLOUR_REPAIR",before,"=>",score(colors),
      {v:x["pairs_below_deltaE35"] for v,x in summaries.items()},flush=True)
