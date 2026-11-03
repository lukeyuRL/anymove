"""Local candidate-color repair after camera-aware graph optimization.

Repair any Front/Side near-projection pair with DeltaE76<35, without
changing fixed thigh/arm/tail identity anchors or introducing duplicate
colors. Does not mutate previous optimizer outputs.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(r"F:\anymove")
sys.path.insert(0,str(ROOT))
from tools.optimize_dino_camera_palette import candidates,lab_hex,graph

base=ROOT/"runs/dino/skinmax_camera_aware_v1"
original=json.loads((base/"optimized_palette_v2.json").read_text(encoding="utf8"))
region=json.loads((base/"skinmax_regions.json").read_text(encoding="utf8"))
views=json.loads((base/"camera_conflicts.json").read_text(encoding="utf8"))
rig=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
names=list(region["selected_bone_names"])
pairs=views["near_pairs"]
anchor={name for name in names if name in (
    "CC_Dinosaur_L_Thigh","CC_Dinosaur_R_Thigh",
    "CC_Dinosaur_L_Upperarm","CC_Dinosaur_R_Upperarm","CC_Dinosaur_Tail07")}
pool=candidates()
palette=[original["bone_colors"][name] for name in names]
a,b,w,tags=graph(region,views,rig)
weight=np.log1p(w)*3
fs=[(names.index(x["a"]),names.index(x["b"]),int(x["view_count"]["front"])+int(x["view_count"]["side"]))
    for x in pairs if x["view_count"]["front"]+x["view_count"]["side"]>0]
f1=np.asarray([r[0] for r in fs],int)
f2=np.asarray([r[1] for r in fs],int)
fw=np.asarray([min(8.,1+np.sqrt(r[2])) for r in fs],float)
def score(colors):
    lab=lab_hex(colors)
    dd=np.linalg.norm(lab[a]-lab[b],axis=-1)
    critical=np.linalg.norm(lab[f1]-lab[f2],axis=-1)
    loss=np.dot(weight,np.maximum(0,58-dd)**2)
    loss+=550*np.dot(fw,np.maximum(0,37-critical)**2)
    loss-=.14*np.dot(weight,np.minimum(dd,90))
    return float(loss),int(np.sum(critical<35)),float(critical.min())

initial=score(palette)
for it in range(30):
    lab=lab_hex(palette)
    critical=np.linalg.norm(lab[f1]-lab[f2],axis=-1)
    bad=np.nonzero(critical<35)[0]
    if len(bad)==0:break
    priority=set()
    for i in bad:
        priority.add(int(f1[i]));priority.add(int(f2[i]))
    priority={i for i in priority if names[i] not in anchor}
    current=score(palette)
    chosen=None
    selected=set(palette)
    for node in sorted(priority):
        for code in pool:
            if code in selected:continue
            vals=np.linalg.norm(lab_hex([code]+palette)[0]-lab,axis=-1)
            vals[node]=np.inf
            if vals.min()<18:continue
            trial=palette.copy()
            trial[node]=code
            ss=score(trial)
            if ss[0]<current[0]-1e-5:
                current=ss;chosen=trial
    if chosen is None:break
    palette=chosen
    print("REPLACEMENT_STEP",it+1,"remaining",current[1],"critical_min",round(current[2],1),flush=True)

if len(set(palette))!=len(palette):raise AssertionError("duplicate selected Color")
for i,name in enumerate(names):
    if name in anchor and palette[i]!=original["bone_colors"][name]:
        raise AssertionError("semantic anchor moved")
labs=lab_hex(palette)
near={}
for view in ("front","side","back","opposite_side"):
    problem=[]
    for p in pairs:
        nearcnt=int(p["view_count"][view])
        if nearcnt<1:continue
        d=float(np.linalg.norm(labs[names.index(p["a"])]-labs[names.index(p["b"])]))
        if d<35:
            problem.append({"a":p["a"],"b":p["b"],
                            "deltaE76":round(d,1),"near_pose_count":nearcnt})
    near[view]={"pairs_below_deltaE35":len(problem),"problems":problem}
final={
    **original,
    "schema":"anymove.dino_skinmax_camera_palette/2",
    "bone_colors":dict(zip(names,palette)),
    "candidate_colour_count":len(pool),
    "postrefinement_objective_before":round(initial[0],1),
    "postrefinement_objective_after":round(score(palette)[0],1),
    "min_overall_selected_deltaE76":round(float(np.min(np.linalg.norm(
        labs[:,None]-labs[None,:],axis=-1)+np.eye(len(names))*999)),2),
    "postrefinement_camera_near_conflicts":near,
    "render_test_required":True
}
out=base/"optimized_palette_final.json"
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf8")
print("CAMERA_COLOR_REPAIR",initial,"->",score(palette),"FS problems",near["front"]["pairs_below_deltaE35"],near["side"]["pairs_below_deltaE35"])
