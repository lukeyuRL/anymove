"""Independent pixel-level color mask IoU QA for V6 FOURCAM generated A/B.

IoU is computed between original 2x2 Blender RGB and fal generated output
at normalized motion phase, independently for each panel and each
anatomical RED-left / BLUE-right upper arm. Supports occlusion with
minimum-mask-area reporting. Does NOT claim 3D joint accuracy.
No paid inference.
"""
import json,pathlib,cv2,numpy as np
H3=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va")
ROOT=H3/"remote"/"h3_ref2va_v6_28_fourview_400step"/"heldout_ab_v1"
SOURCE=H3/"fourview_v6_reach_augmented"/"clips"/"heldout"
CASES={
 "forward_left":"full_heldout_reach_forward_left_a55_s8201",
 "backward_right":"full_heldout_reach_backward_right_a52_s8204"
}
VIEWS=[("left_45",0,0),("right_45",1,0),("back",0,1),("top",1,1)]
def frames(p,expect):
 cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 assert n==expect,(p,n)
 images=[]
 while len(images)<n:
  good,frame=cap.read()
  if not good:raise RuntimeError("Truncated "+str(p))
  if expect==107:frame=cv2.resize(frame,(768,768),interpolation=cv2.INTER_AREA)
  images.append(frame)
 cap.release()
 return images
def mask(im,color):
 h=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
 x,s,v=h[:,:,0],h[:,:,1],h[:,:,2]
 if color=="red":return ((x<=11)|(x>=173))&(s>=70)&(v>=60)
 if color=="blue":return (x>=90)&(x<=124)&(s>=60)&(v>=50)
 raise ValueError(color)
def iou(a,b):
 inter=(a&b).sum();union=(a|b).sum()
 if union==0:return None
 return float(inter/union)
def assess(case):
 base=ROOT/case
 orig=frames(SOURCE/CASES[case]/"rgb_fourview.mp4",107)
 result={"case":case,"status":"PASS","frame_count":124,
  "status_reason":"Every 2x2 grid view compared independently by anatomical color pixel mask, source time warped to generated normalized phase.",
  "metrics":{}}
 for scale in ("scale0","scale1"):
  target=frames(base/scale/f"h3_ref2va_{scale}.mp4",124)
  matches=[]
  for j,img in enumerate(target):
   k=round(j*(106/123))
   src=orig[k]
   row={}
   for view,col,rr in VIEWS:
    src_tile=src[384*rr:384*(rr+1),384*col:384*(col+1)]
    out_tile=img[384*rr:384*(rr+1),384*col:384*(col+1)]
    for color in ("red","blue"):
     s=mask(src_tile,color);g=mask(out_tile,color)
     row[f"{view}_{color}"]=iou(s,g)
     row[f"{view}_{color}_source_area"]=int(s.sum())
     row[f"{view}_{color}_gen_area"]=int(g.sum())
   matches.append(row)
  summary={}
  for v,_,_ in VIEWS:
   for color in ("red","blue"):
    key=f"{v}_{color}"
    entries=[x for x in matches if x[f"{key}_source_area"]>=20]
    # report n_valid, mean, mid-action and min source area, not just average
    vals=np.asarray([x[key] for x in entries if x[key] is not None],dtype=float)
    active=[matches[j][key] for j in range(24,97) if matches[j][key] is not None and matches[j][key+"_source_area"]>=20]
    mid=[matches[j][key] for j in (24,48,72,96)]
    summary[key]={
      "mean_mask_IoU_active_frames24_96":round(float(np.mean(active)),4) if active else None,
      "valid_active_frame_count":len(active),
      "mean_mask_IoU_all_nonoccluded":round(float(np.mean(vals)),4) if len(vals) else None,
      "p10_mask_IoU":round(float(np.percentile(vals,10)),4) if len(vals) else None,
      "frames_source_has_min20_color_px":len(entries),
      "mid_frames_IoU":[round(float(x),4) if x is not None else None for x in mid],
      "source_mid_area_px":[matches[j][key+"_source_area"] for j in (24,48,72,96)],
      "gen_mid_area_px":[matches[j][key+"_gen_area"] for j in (24,48,72,96)],
    }
  result["metrics"][scale]=summary
 out=base/"analysis"/f"{case}_color_pixel_iou.json"
 out.write_text(json.dumps(result,indent=2),encoding="utf8")
 print(json.dumps({"case":case,"scale_results":{
  lab:{p:{v:result["metrics"][lab][f"{v}_{p}"]["mean_mask_IoU_all_nonoccluded"] for v,_,_ in VIEWS} for p in ("red","blue")} for lab in ("scale0","scale1")
 },"report":str(out)},indent=2),flush=True)
 return result
summary={}
for c in CASES:summary[c]=assess(c)
(ROOT/"combined_color_pixel_iou.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
