"""Dino view-aware skin-color rendered pixel/geometry camera audit.

For three independent test poses, measures actual rendered mean Lab colors
inside *frontmost* per-part masks, excluding borders. Pixels hidden behind
another part cannot be distinguished by any color assignment.

P1 Part-ID proxy is dominant Skinning of triangles. Per-VERTEX color mixing
may be present at boundaries; this audit reports rather than hides it.
"""
import cv2,json,os
import numpy as np
from pathlib import Path
from collections import defaultdict
ROOT=Path(r"F:\anymove")
BASE=ROOT/"runs/dino/skinmax_camera_aware_v1"
RENDER_SET=os.environ.get('DINO_COLOR_RENDER_SET','test_render_v1')
if RENDER_SET not in ('test_render_v1','test_render_v4'):
    raise ValueError('unsupported camera-colour render audit variant')
DEST=BASE/RENDER_SET
ORIG=ROOT/"runs/dino/p4_blind_fourview_v1"
RIG=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text())
PAL=json.loads((BASE/"optimized_palette_final.json").read_text())
BONE={b["name"]:i for i,b in enumerate(RIG["bones"])}
SEL={BONE[n]:n for n in PAL["bone_colors"]}
VIEWS=("front","side","back","opposite_side")
SAMPLES=(9001,9002,9007)
out={"schema":"anymove.dino_camera_palette_render_colour_audit/1",
     "pose_ids":list(SAMPLES),
     "color_count":len(SEL),
     "view_metrics":{},
     "masked_skin_area_measured_by_triangle_part_proxy_not_exact_point_interpolation":True,
     "no_colour_can_restore_fully_occluded_mesh":True,
     "training_poses_used_for_optimizer_not_this_test":True}
mask_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
adj_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
for view in VIEWS:
    all_adjacent=[]
    soft_colors=[]
    visible_num=[]
    for pose in SAMPLES:
        imgp=DEST/f"{pose:05d}_{view}.png"
        img=cv2.imread(str(imgp),cv2.IMREAD_UNCHANGED)
        with np.load(DEST/f"{pose:05d}_{view}_labels.npz",allow_pickle=False) as f:
            part=f["part_id"].copy()
        with np.load(ORIG/f"{pose:05d}_{view}_labels.npz",allow_pickle=False) as f:
            if not np.array_equal(part,f["part_id"]):
                raise AssertionError("color assignment changed mesh Part-ID geometry")
            if not np.allclose(f["depth_m"],np.load(DEST/f"{pose:05d}_{view}_labels.npz")["depth_m"],equal_nan=True):
                raise AssertionError("color assignment changed depth")
        labs={}
        for ix,bone in SEL.items():
            blob=(part==ix).astype(np.uint8)
            raw_pixel_num=int(blob.sum())
            if raw_pixel_num<10:continue
            mask=cv2.erode(blob,mask_kernel)
            pixels=img[:,:,:3][mask>0]
            if len(pixels)<6:continue
            # Sample median observed sRGB and transform to Lab.
            median_bgr=np.median(pixels,axis=0).astype(np.float32)/255
            lab=cv2.cvtColor(median_bgr.reshape(1,1,3),cv2.COLOR_BGR2LAB)[0,0]
            labs[ix]=lab
        visible_num.append(len(labs))
        for ix in labs:
            binary=(part==ix).astype(np.uint8)
            halo=cv2.dilate(binary,adj_kernel)
            others=np.unique(part[halo>0])
            for jx in others:
                jx=int(jx)
                if jx<=ix or jx not in labs:continue
                lab_distance=float(np.linalg.norm(labs[ix]-labs[jx]))
                all_adjacent.append((lab_distance,SEL[ix],SEL[jx],pose))
        print("RENDER_CAMERA",view,"pose",pose,"readable regions",len(labs),flush=True)
    near_below25=[{"deltaE76":round(v,1),"a":a,"b":b,"sample_id":i}
                  for v,a,b,i in all_adjacent if v<25]
    closest=sorted(all_adjacent)[:10]
    out["view_metrics"][view]={
        "readable_colored_region_count_by_pose":visible_num,
        "near_screen_neighbor_pairs":len(all_adjacent),
        "rendered_color_neighbor_pairs_deltaE76_below25":len(near_below25),
        "observed_median_lab_adjacent_min_deltaE76":round(min([x[0] for x in all_adjacent],default=0),2),
        "near_color_examples":near_below25[:12],
        "closest_10_pairs":[{"deltaE76":round(v,1),"a":a,"b":b,"pose":i}
                            for v,a,b,i in closest],
    }
outpath=BASE/('rendered_camera_colour_audit.json' if RENDER_SET=='test_render_v1'
              else 'rendered_camera_colour_audit_v4.json')
if outpath.exists():raise FileExistsError(outpath)
outpath.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf8")
print("RENDERED_COLOR_AUDIT",json.dumps(out["view_metrics"],ensure_ascii=False)[:6000],flush=True)
