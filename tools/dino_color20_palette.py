"""Dino P4 Color20: deterministic anatomy-aware palette assignment.

20 *surface regions*: ten on left hind leg/foot and ten on right.
No source skins or existing original diffuse images are overwritten.

All 20 distinct colors are assigned to a side/region by minimizing a
perceptual Lab contrast penalty on adjacent toe branches, distal tip/shaft,
and camera-overlap left/right region collisions. This is an engineering
heuristic, NOT a mathematically/global perceptual optimum.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import cv2

REGIONS=("calf","foot","back_shaft","inner_shaft","middle_shaft",
         "outer_shaft","back_tip","inner_tip","middle_tip","outer_tip")
SIDES=("L","R")
PALETTE=(
 "#2979FF", "#E53935", "#A0F000", "#00F0FF", "#C800B4",
 "#FFC850", "#14783C", "#FFB4FF", "#00C88C", "#B478FF",
 "#00B4FF", "#783CFF", "#64A000", "#FF28F0", "#8CA0F0",
 "#B43C64", "#C8C864", "#FF8C8C", "#FF64A0", "#A0DC8C",
)
NODES=tuple(f"{side}_{region}" for side in SIDES for region in REGIONS)

def _edge_list():
    out={}
    def add(a,b,weight):
        a=NODES.index(a);b=NODES.index(b)
        if a==b:return
        e=(min(a,b),max(a,b))
        out[e]=out.get(e,0)+weight
    for s in SIDES:
        add(s+"_calf",s+"_foot",6)
        for k in ("back","inner","middle","outer"):
            add(s+"_foot",s+f"_{k}_shaft",5)
            add(s+f"_{k}_shaft",s+f"_{k}_tip",9)
        for k in ("back","inner","middle","outer"):
            for n in ("back","inner","middle","outer"):
                if k>=n:continue
                add(s+f"_{k}_shaft",s+f"_{n}_shaft",3)
                add(s+f"_{k}_tip",s+f"_{n}_tip",5)
        for k in range(10):
            for n in range(k+1,10):
                add(s+"_"+REGIONS[k], s+"_"+REGIONS[n],.25)
    for k in REGIONS:
        add("L_"+k,"R_"+k,8)
    for a in REGIONS[2:]:
        for b in REGIONS[2:]:
            add("L_"+a,"R_"+b,.16)
    return [(a,b,w) for (a,b),w in out.items()]

def srgb_to_lab(s):
    pix=np.asarray([[int(v[i:i+2],16)/255. for i in (1,3,5)] for v in s],np.float32)
    return cv2.cvtColor(pix[:,None,:],cv2.COLOR_RGB2Lab)[:,0,:]

def assign(seed=20261102):
    lab=srgb_to_lab(PALETTE)
    dif=np.linalg.norm(lab[:,None,:]-lab[None,:,:],axis=2)
    # Perceptual minimum for high-priority face adjacency, plus weighted score.
    e=_edge_list()
    leftanchor=PALETTE.index("#2979FF")
    rightanchor=PALETTE.index("#E53935")
    rng=np.random.default_rng(seed)
    ww=np.asarray([q[2] for q in e],float)
    indA=np.asarray([q[0] for q in e],int)
    indB=np.asarray([q[1] for q in e],int)
    def evaluate(perm):
        distance=dif[perm[indA],perm[indB]]
        # Heavy quadratic penalty for near-indistinguishable adjacent colors.
        contrast_penalty=float(np.sum(ww*(np.maximum(0,61-distance)**2)))
        # Encourage separation when two toes cross the same view.
        priority=ww>4.5
        minscore=float(np.min(distance[priority]))
        total=float(np.sum(ww*np.sqrt(distance)))
        return (-contrast_penalty+1.5*minscore+.015*total,
                minscore,float(np.mean(distance[priority])))
    best=None
    for restart in range(18):
        perm=np.arange(len(PALETTE))
        others=[i for i in range(20) if i not in (0,10)]
        available=[i for i in range(20) if i not in (leftanchor,rightanchor)]
        perm[others]=rng.permutation(available)
        perm[0]=leftanchor;perm[10]=rightanchor
        if len(set(perm))!=20:raise AssertionError('palette colors not unique')
        score=evaluate(perm)
        for iteration in range(200):
            candidate=None;cs=score
            for x in range(1,20):
                if x==10:continue
                for y in range(x+1,20):
                    if y==10:continue
                    temp=perm.copy()
                    temp[x],temp[y]=temp[y],temp[x]
                    q=evaluate(temp)
                    if q[0]>cs[0]+1e-7:
                        candidate=temp;cs=q
            if candidate is None:break
            perm,score=candidate,cs
        if best is None or score[0]>best[0][0]:
            best=(score,perm.copy())
    score,assignment=best
    selected=[PALETTE[i] for i in assignment]
    return {
        "schema":"anymove.dino_color20_palette/1",
        "not_a_globally_optimal_color_science_claim":True,
        "seed":seed,"source_palette":list(PALETTE),
        "side_semantics":"L calf blue (#2979FF), R calf red (#E53935); each foot has 4 toe shafts and 4 distal surface markers",
        "regions":{k:{"hex":selected[i],"RGB_srgb":np.round(np.asarray([int(selected[i][j:j+2],16) for j in (1,3,5)]) /255,5).tolist(),
                       "Lab_D65":np.round(lab[assignment[i]],2).tolist()}
                   for i,k in enumerate(NODES)},
        "weighted_adjacency_edges":len(e),
        "priority_adjacent_min_deltaE76":round(score[1],2),
        "priority_adjacent_mean_deltaE76":round(score[2],2),
        "warning":"Color separation scored in LAB, but final result must be checked on rendered textured/occluded views. Some terminal toe bones have no Mesh Skinning weights.",
    }

if __name__=="__main__":
    out=pathlib.Path(r"F:\anymove\runs\dino\color20_candidate_v1\palette.json")
    if out.exists():raise FileExistsError("do not replace existing test palette")
    out.parent.mkdir(parents=True,exist_ok=True)
    result=assign()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
