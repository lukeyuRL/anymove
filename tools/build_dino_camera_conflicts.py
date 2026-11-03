"""Build a camera-aware conflict graph from actual Dino skinned Mesh masks.

Graph nodes: active argmax Skin Weight bones selected for color.
Source masks: original geometry/rasterized frontmost surface part IDs from
P4 training poses, NOT 2D Reader predictions or held-out test images.

Front and Side score more highly than Back/Opposite. Never claims colors
can reveal two occluded objects in the same image pixel.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

ROOT=Path(r"F:\anymove")
DIR=ROOT/"runs/dino/skinmax_camera_aware_v1"
src=ROOT/"runs/dino/p4_source_material_pilot_v1"
selection=json.loads((DIR/"skinmax_regions.json").read_text(encoding="utf8"))
info=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text(encoding="utf8"))
split=json.loads((ROOT/"runs/dino/p4_source_material_split_v1.json").read_text(encoding="utf8"))
bone_order={b["name"]:i for i,b in enumerate(info["bones"])}
selected=selection["selected_bone_names"]
identifiers={bone_order[n]:n for n in selected}
rng=np.random.default_rng(41111)
train=sorted(rng.choice(split["pilot_train_ids"],size=32,replace=False).tolist())
views={"front":2., "side":2., "back":1., "opposite_side":1.}
edges=defaultdict(float)
per_view_close={v:defaultdict(int) for v in views}
per_view_visible={v:defaultdict(int) for v in views}
per_view_areas={v:defaultdict(list) for v in views}
r=5 # half-res 5 px radius ~=10 original camera pixels
ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
for id in train:
    for view,scale in views.items():
        srcfile=src/f"{id:05d}_{view}_labels.npz"
        with np.load(srcfile,allow_pickle=False) as z:
            part=z["part_id"].copy()
        hh,ww=part.shape
        mask=cv2.resize(part.astype(np.float32),(ww//2,hh//2),
                        interpolation=cv2.INTER_NEAREST).astype(np.int16)
        valid=np.unique(mask[mask>=0])
        valid=[int(x) for x in valid if int(x) in identifiers and int((mask==x).sum())>=2]
        valid.sort()
        for va in valid:
            per_view_areas[view][identifiers[va]].append(int((mask==va).sum()*4))
        # Unique co-visible pair: distinct not occluded, may be spatially far.
        for ix,va in enumerate(valid):
            for vb in valid[ix+1:]:
                pair=tuple(sorted((identifiers[va],identifiers[vb])))
                per_view_visible[view][pair]+=1
        # Near-surface projected edges from frontmost visible mask only.
        for va in valid:
            m=(mask==va).astype(np.uint8)
            near=cv2.dilate(m,ker)
            neigh=np.unique(mask[near!=0])
            for vb in neigh:
                vb=int(vb)
                if vb<=va or vb not in identifiers:continue
                pair=tuple(sorted((identifiers[va],identifiers[vb])))
                count=int((mask[near!=0]==vb).sum())
                if count<1:continue
                per_view_close[view][pair]+=1
                # Edge strength per pose, with saturation to avoid bias from
                # broad body surfaces; actual small fingertip still counts.
                edges[pair]+=float(scale*(1.5+0.6*np.sqrt(min(count,150))))
print("PROJECTION_MASK_SCAN",len(train),"poses x",len(views),"views",flush=True)
report={
    "schema":"anymove.dino_skinmax_camera_conflicts/1",
    "input_source":"P4 source-material Blender *training* pose Part-ID masks (no test data used)",
    "poses":train,
    "sample_count":len(train),
    "view_weights":views,
    "near_radius_source_pixels":r*2,
    "colour_cannot_solve_complete_occlusion":True,
    "view_summary":{},
    "near_pairs":[],
    "view_pixel_coverage":{}
}
pairs=sorted(edges)
for view in views:
    report["view_summary"][view]={
      "visible_pair_count":len(per_view_visible[view]),
      "close_pair_count":len(per_view_close[view]),
      "close_pair_observation_total":sum(per_view_close[view].values())
    }
    report["view_pixel_coverage"][view]={
      name:{"observed_poses":len(pixel_areas),
            "median_visible_pixels":int(np.median(pixel_areas)),
            "min_visible_pixels":min(pixel_areas)}
      for name,pixel_areas in per_view_areas[view].items() if len(pixel_areas)}
for pair in pairs:
    report["near_pairs"].append({
      "a":pair[0],"b":pair[1],
      "view_count":{v:int(per_view_close[v][pair]) for v in views},
      "co_visible_view_count":{v:int(per_view_visible[v][pair]) for v in views},
      "weight":round(float(edges[pair]),4)
    })
out=DIR/"camera_conflicts.json"
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
print("SOURCE_GRAPH",report["view_summary"],"near_pairs",len(pairs),flush=True)
print("SMALL REGIONS",[ (n,m["median_visible_pixels"]) for n,m in report["view_pixel_coverage"]["front"].items()
                           if m["median_visible_pixels"]<70 ][:30],flush=True)
