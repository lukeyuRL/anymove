"""Analyze V6 FOURCAM forward-left and backward-right A/B fal videos.

Creates 3-column (Blender source, scale0, scale1) timepoint contact sheets
and 2-column dynamic comparison MP4 for each heldout action.
Per-view color-region tracking in TL front-left45, TR front-right45,
BL back, BR true overhead. Measures RGB segment motion versus same
source rendered colored upper-arm regions, normalized by clip phase.
Read-only fal: no uploads/inference/training, only local video decode.
"""
import csv,hashlib,json,math,pathlib,subprocess
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va")
PILOT=ROOT/"remote"/"h3_ref2va_v6_28_fourview_400step"
EXPS=PILOT/"heldout_ab_v1"
SOURCE=ROOT/"fourview_v6_reach_augmented"/"clips"/"heldout"
FFMPEG=r"C:\ffmpeg\bin\ffmpeg.exe"
CASES={
 "forward_left":"full_heldout_reach_forward_left_a55_s8201",
 "backward_right":"full_heldout_reach_backward_right_a52_s8204"
}
VIEWS=[("left_45",0,0),("right_45",1,0),("back",0,1),("top",1,1)]
LABELS={"red":"actor_LEFT_upperarm_RED","blue":"actor_RIGHT_upperarm_BLUE"}
def rms(x):
 arr=np.asarray(x,dtype=float);return float(np.sqrt(np.mean(arr*arr)))
def load(path):return json.loads(path.read_text(encoding="utf8"))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for block in iter(lambda:f.read(1048576),b""):h.update(block)
 return h.hexdigest()
def extract_color(frame,color):
 # frame is one 384 x 384 panel. Unique red/blue upperarm skin colors.
 h,w=frame.shape[:2]
 hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
 H,S,V=cv2.split(hsv)
 if color=="red":
  mask=(((H<=10)|(H>=173))&(S>=80)&(V>=70)).astype(np.uint8)
 elif color=="blue":
  mask=((H>=90)&(H<=122)&(S>=64)&(V>=57)).astype(np.uint8)
 else:raise ValueError(color)
 mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
 n,_,stats,centroids=cv2.connectedComponentsWithStats(mask,connectivity=8)
 if n<2:return [None,None,0]
 j=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
 area=int(stats[j,cv2.CC_STAT_AREA])
 if area<12:return [None,None,0]
 x,y=centroids[j]
 return [round(float(x),4),round(float(y),4),area]
def decode(path,case,label):
 cap=cv2.VideoCapture(str(path))
 if not cap.isOpened():raise FileNotFoundError(path)
 w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 fps=float(cap.get(cv2.CAP_PROP_FPS))
 if label=="source" and (w,h,n,round(fps))!=(896,896,107,24):raise RuntimeError("Source video has changed")
 if label!="source" and (w,h,n,round(fps))!=(768,768,124,24):raise RuntimeError("fal generated video changed")
 cell=384; frames={}
 rows=[]
 for i in range(n):
  ok,frame=cap.read()
  if not ok:raise IOError("Video ended early: "+str(path))
  if label=="source":
   frame=cv2.resize(frame,(768,768),interpolation=cv2.INTER_AREA)
  if i in (0,24,48,72,96,120) or (label=="source" and i in (24,48,72,96,106)):
   frames[i]=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  entry={"frame":i,"phase":i/(n-1)}
  for view,c,r in VIEWS:
   roi=frame[r*cell:(r+1)*cell,c*cell:(c+1)*cell]
   for color in ("red","blue"):
    entry[f"{view}_{color}"]=extract_color(roi,color)
  rows.append(entry)
 cap.release()
 return rows,frames
def tracking(records,source_records):
 summary={}; tracks={}
 source_phase=np.linspace(0,1,len(source_records))
 gen_phase=np.linspace(0,1,len(records))
 for view,_,_ in VIEWS:
  for color in ("red","blue"):
   key=f"{view}_{color}"
   vv=np.asarray([[np.nan,np.nan] if q[key][0] is None else q[key][:2] for q in records],dtype=float)
   ss=np.asarray([[np.nan,np.nan] if q[key][0] is None else q[key][:2] for q in source_records],dtype=float)
   def interp(x):
    for axis in range(2):
     arr=x[:,axis]
     good=np.isfinite(arr)
     if good.sum()<max(65,int(.60*len(arr))) or not good[0]:
      return None
     if not np.all(good):arr[~good]=np.interp(np.flatnonzero(~good),np.flatnonzero(good),arr[good])
    return x
   detected_gen=int(np.isfinite(vv[:,0]).sum())
   detected_src=int(np.isfinite(ss[:,0]).sum())
   frame_diff={"generated_detected":detected_gen,"source_detected":detected_src,
               "generated_count":len(records),"source_count":len(source_records)}
   vv=interp(vv);ss=interp(ss)
   if vv is not None and ss is not None:
    src_delta=ss-ss[0]
    gen_delta=vv-vv[0]
    src_interp=np.column_stack([np.interp(gen_phase,source_phase,src_delta[:,a]) for a in range(2)])
    e=gen_delta-src_interp
    frame_diff.update({
      "usable":True,
      "source_displacement_RMS_px":round(rms(np.linalg.norm(src_interp,axis=1)),3),
      "generated_displacement_RMS_px":round(rms(np.linalg.norm(gen_delta,axis=1)),3),
      "displacement_error_MAE_px":round(float(np.mean(np.linalg.norm(e,axis=1))),3),
      "displacement_error_P95_px":round(float(np.percentile(np.linalg.norm(e,axis=1),95)),3),
      "generated_rest_offset_px":round(float(np.linalg.norm(gen_delta[-1])),3),
      "source_peak_up_px":round(float(np.max(-src_interp[:,1])),3),
      "generated_peak_up_px":round(float(np.max(-gen_delta[:,1])),3),
      "generated_x_delta_rms_px":round(rms(gen_delta[:,0]),3),
      "source_x_delta_rms_px":round(rms(src_interp[:,0]),3)
    })
    tracks[key]={"gen":gen_delta,"src":src_interp}
   else:
    frame_diff["usable"]=False
   summary[key]=frame_diff
 return summary,tracks

def contact_sheet(case,sources,output):
 # Fixed frames by physical time: generated is 124 frames at 24FPS.
 # Original 107 frames ~4.458s, use normalized phase instead.
 target_frames=(0,24,48,72,96,120)
 tile=768; gap=32
 final=Image.new("RGB",(tile*3,(tile+gap)*len(target_frames)),(241,241,241))
 pen=ImageDraw.Draw(final)
 for ri,gf in enumerate(target_frames):
  sf=min(106,int(round((gf/123)*106)))
  for ci,label,fr in ((0,"Blender source",sources["source"][1].get(sf)),
                      (1,"LoRA scale=0",sources["scale0"][1][gf]),
                      (2,"LoRA scale=1",sources["scale1"][1][gf])):
   if fr is None:
    # Source 107-frame capture store all samples for exact match by
    # reading one arbitrary frame directly.
    if ci!=0:raise ValueError("Missing generated frame")
    cap=cv2.VideoCapture(str(SOURCE/CASES[case]/"rgb_fourview.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    ok,bgr=cap.read();cap.release()
    if not ok:raise ValueError("Can't load original Blender frame")
    fr=cv2.cvtColor(cv2.resize(bgr,(768,768)),cv2.COLOR_BGR2RGB)
   x=ci*tile;y=ri*(tile+gap)
   final.paste(Image.fromarray(fr),(x,y))
   pen.text((x+10,y+tile+4),f"{case} / {label} / source f{sf:03d} / generated f{gf:03d}",fill=(20,20,20))
 path=output/f"{case}_source_vs_scale0_vs_scale1.jpg"
 final.save(path,quality=87,optimize=True)
 return str(path)

def compare(case,out):
 ab=EXPS/case
 vids={
  "source":SOURCE/CASES[case]/"rgb_fourview.mp4",
  "scale0":ab/"scale0"/"h3_ref2va_scale0.mp4",
  "scale1":ab/"scale1"/"h3_ref2va_scale1.mp4"
 }
 rows={}
 frames={}
 for label,path in vids.items():
  rows[label],frames[label]=decode(path,case,label)
 for label in ("scale0","scale1"):
  req=load(ab/label/"request.json")["request"]
  if req["aspect_ratio"]!="1:1" or req["seed"]!=42 or req["resolution"]!="768P":
   raise ValueError("Uncontrolled comparison")
  if req["loras"][0]["scale"]!=(0.0 if label=="scale0" else 1.0):
   raise ValueError("Wrong LoRA scale")
 a=load(ab/"scale0"/"request.json")["request"]
 b=load(ab/"scale1"/"request.json")["request"]
 ca=json.loads(json.dumps(a));cb=json.loads(json.dumps(b))
 w0=ca.pop("loras");w1=cb.pop("loras")
 if ca!=cb or w0[0]["path"]!=w1[0]["path"]:raise ValueError("More than LoRA scale differs")
 summary={}
 for label in ("scale0","scale1"):
  summary[label],_ =tracking(rows[label],rows["source"])
 out.mkdir(parents=True,exist_ok=True)
 contact=contact_sheet(case,{k:(rows[k],frames[k]) for k in rows},out)
 aligned=out/f"{case}_scale0_vs_scale1.mp4"
 subprocess.run([FFMPEG,"-hide_banner","-loglevel","error","-y",
                  "-i",str(vids["scale0"]),"-i",str(vids["scale1"]),
                  "-filter_complex","[0:v]scale=768:768,setsar=1[a];[1:v]scale=768:768,setsar=1[b];[a][b]hstack=inputs=2:shortest=1[v]",
                  "-map","[v]","-an","-c:v","libx264","-preset","fast",
                  "-crf","21","-pix_fmt","yuv420p",str(aligned)],check=True)
 report={
  "schema":"anymove.v6_fourcam_400_ref2va_two_motion_ab/1",
  "case":case,"source_motion":CASES[case],
  "status":"PASS_GENERATED_AND_MEASURED",
  "view_positions":{v:[c,r] for v,c,r in VIEWS},
  "source_resolution":[896,896],"generated_resolution":[768,768],
  "generated_frames":124,"source_frames":107,
  "controlled_input_differs_only_lora_scale":True,
  "reference_is_soft_not_hard_control":True,
  "generated_request_ids":{s:load(ab/s/"submitted.json")["request_id"] for s in ("scale0","scale1")},
  "color_region_tracking":summary,
  "contact_sheet":contact,"paired_video":str(aligned),
  "limitations":["Same source timeline normalized by phase, not exact frame time",
                 "2D color region centroids are not anatomical 3D joints",
                 "Occlusion, depth changes and region deformation make centroids unreliable",
                 "One random seed cannot establish generalization"]
 }
 (out/f"{case}_metrics.json").write_text(json.dumps(report,indent=2),encoding="utf8")
 print(json.dumps({
  "case":case,"status":"PASS","paired":str(aligned),"contact_sheet":contact,
  "metric_tracked":{lab:sum(x["usable"] for x in summary[lab].values()) for lab in summary},
  "table":{lab:{color:{"mae_px":[summary[lab][f"{v}_{color}"].get("displacement_error_MAE_px") for v,_,_ in VIEWS],
                        "motion_rms_px":[summary[lab][f"{v}_{color}"].get("generated_displacement_RMS_px") for v,_,_ in VIEWS]} for color in ("red","blue")} for lab in ("scale0","scale1")}
 },indent=2),flush=True)
 return report

def main():
 final={}
 for case in CASES:
  out=EXPS/case/"analysis"
  final[case]=compare(case,out)
 (EXPS/"combined_overview.json").write_text(json.dumps({
   "status":"PASS_ALL_4_VIDEOS","model":"V6 FOURCAM H3 native Ref2VA 400 Steps",
   "cases":{k:{"report":str(EXPS/k/"analysis"/f"{k}_metrics.json"),
               "requests":v["generated_request_ids"],
               "comparison_video":v["paired_video"],"contact":v["contact_sheet"]} for k,v in final.items()}
 },indent=2),encoding="utf8")
if __name__=="__main__":main()
