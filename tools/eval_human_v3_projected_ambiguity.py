"""Paired left/right-joint 2D separation from V3 human GT poses.

Does not measure Reader prediction accuracy. Uses only 3D pose GT projected to
calibrated two-view orthographic camera angles.
"""
import json
from pathlib import Path

import numpy as np

from mocap.camera_geometry import project_orthographic

ROOT = Path("runs/human/lora_synth_A/v6_reach_source")
OUT = Path("runs/human/camera_angle_p0_v3/human_v3_gt_left_right_separation.json")
PAIRS = {"baseline_0_90": (0,90),
         "symmetric_315_45": (315,45),
         "oblique_45_135": (45,135)}
WIDTH, HEIGHT = 384, 448
MPP = 2.432802200317383 / 448
ZC = 0.8688581585884094

data = {}
for p in sorted(ROOT.iterdir()):
    if p.is_dir() and (p / "joints.json").is_file():
        rec=json.loads((p / "joints.json").read_text(encoding="utf-8"))
        data[p.name]=(rec["names"],np.asarray(rec["joint_position_m"],dtype=np.float64))
names=next(iter(data.values()))[0]
assert all(n==names for n,points in data.values())
pairs=[]
for n in names:
    if "_L_" in n:
        right=n.replace("_L_","_R_",1)
        if right in names:
            pairs.append((n,right,names.index(n),names.index(right)))
if not pairs:
    raise RuntimeError("No symmetry pairs in human GT")
out={"schema":"anymove.human_v3_lr_projection_separation/1",
     "source":"Existing V6 3D GT; no Reader predictions",
     "left_right_corresponding_joint_pairs":len(pairs),
     "examples":[[a,b] for a,b,_,_ in pairs[:8]],
     "clip_count":len(data),"per_view_separation_threshold_px":8.0,
     "pairs":{}}
idx_l=[a for _,_,a,_ in pairs]
idx_r=[b for _,_,_,b in pairs]
for key,angles in PAIRS.items():
    all_d=[]
    for clip,(name,pts) in data.items():
        per=[]
        for a in angles:
            uv=project_orthographic(pts,a,WIDTH,HEIGHT,MPP,ZC)
            per.append(np.linalg.norm(uv[:,idx_l]-uv[:,idx_r],axis=-1))
        all_d.append(np.stack(per,axis=-1))
    d=np.concatenate(all_d,axis=0)
    out["pairs"][key]={
        "angles_deg":list(angles),
        "fraction_lr_pairs_close_both_views_lt8px":round(float(np.mean(np.all(d<8,axis=-1))),6),
        "fraction_lr_pairs_close_any_view_lt8px":round(float(np.mean(np.any(d<8,axis=-1))),6),
        "fraction_lr_pairs_separated_in_both_views_ge8px":round(float(np.mean(np.all(d>=8,axis=-1))),6),
        "mean_lr_pair_separation_px_view1":round(float(d[...,0].mean()),2),
        "mean_lr_pair_separation_px_view2":round(float(d[...,1].mean()),2),
    }
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
