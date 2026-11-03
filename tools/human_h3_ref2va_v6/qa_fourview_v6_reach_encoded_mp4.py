"""V5 four-camera actual H264 encoded RGB+Pose MP4 acceptance.

Loads every source frame for all 16 clips, decodes corresponding
107+107 MP4 frames, confirms frame sequence/profile and evaluates
color-coded 3D Bone segments after H264 compression in all four panels.
No fal paid actions.
"""
import collections,json,pathlib
import cv2
import numpy as np
from PIL import Image
H3=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va")
DATA=H3/"fourview_v6_reach_augmented"
V2=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\v6_reach_source")
palette=json.loads((H3/"anatomical_pose_v4_depth_verified"/"skeleton_schema.json").read_text(encoding="utf8"))["segments"]
prototypes=np.array([item["RGB"] for item in palette]+[[255,255,255]],dtype=np.float32)
color_tuple=lambda q:(int(q[0])<<16)|(int(q[1])<<8)|int(q[2])
VALID={"left_45","right_45","back","top"}
Q={"left_45":(0,0),"right_45":(1,0),"back":(0,1),"top":(1,1)}
SAMPLE_FRAMES=(0,13,26,39,53,66,79,92,106)
plan=json.loads((V2/"v2_dataset_plan.json").read_text(encoding="utf8"))
report={
 "schema":"anymove.h3_ref2va_fourview_v6_h264_decode_qa/1",
 "status":"FAIL_UNTIL_COMPLETE",
 "rgb_mp4_frames_decoded":0,"pose_mp4_frames_decoded":0,
 "frames_per_clip":107,"clip_count":20,"native_resolution":[896,896],
 "encoded_pose_stable_color_accuracy":None,
 "mean_rgb_frame_l1_mae_255":None,
 "mean_pose_frame_l1_mae_255":None,
 "fully_decoded_videos":0,
 "records":{},"new_paid_jobs":0}
totalrgb=0.0
totalpose=0.0
rgb_cnt=pose_cnt=0
correct=total=0
for i,rec in enumerate(plan["results"]):
 clip=rec["id"];split=rec["split"]
 dir=DATA/"clips"/split/clip
 caps={
   "RGB":cv2.VideoCapture(str(dir/"rgb_fourview.mp4")),
   "POSE":cv2.VideoCapture(str(dir/"pose_fourview.mp4"))
 }
 for label,c in caps.items():
  if not c.isOpened():raise AssertionError("Cannot open "+label+" "+clip)
  profile=(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           int(c.get(cv2.CAP_PROP_FRAME_COUNT)),round(float(c.get(cv2.CAP_PROP_FPS))))
  if profile!=(896,896,107,24):
   raise AssertionError("Wrong MP4 profile "+label+clip+str(profile))
 row={"split":split,"family":rec["family"],"rgb_frames":0,"pose_frames":0,"pose_line_color_accuracy":None}
 local_correct=local_total=0
 for frame in range(107):
  for key,col in (("RGB","fourview_rgb_png"),("POSE","fourview_pose_png")):
   ok,bgr=caps[key].read()
   if not ok:raise RuntimeError("Encoded frame truncated "+clip+key+str(frame))
   rendered=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
   path=dir/col/f"f_{frame:04d}.png"
   with Image.open(path) as im:
    source=np.asarray(im.convert("RGB"))
   if rendered.shape!=source.shape:
    raise RuntimeError("Mismatch source/encoded dims")
   diff=np.abs(rendered.astype(np.int16)-source.astype(np.int16))
   score=float(np.mean(diff))
   if key=="RGB":
    totalrgb+=score;rgb_cnt+=1
    report["rgb_mp4_frames_decoded"]+=1
    row["rgb_frames"]+=1
   else:
    totalpose+=score;pose_cnt+=1
    report["pose_mp4_frames_decoded"]+=1
    row["pose_frames"]+=1
   if score>24:
    raise AssertionError("Excessive H264 color distortion "+str((clip,key,frame,score)))
   if key=="POSE" and frame in SAMPLE_FRAMES:
    # Check centerline of bones where all immediate neighbors share
    # EXACT RGB material value (avoid edge compression confounders).
    foreground=np.any(source!=0,axis=2)
    center=cv2.erode(foreground.astype(np.uint8),np.ones((3,3),np.uint8))>0
    for axis,sign in ((0,1),(0,-1),(1,1),(1,-1)):
     same=np.all(source==np.roll(source,sign,axis=axis),axis=2)
     center &= same
    center &= np.all(source!=255,axis=2)
    # Exclude blank, white markers; only distinct anatomical material
    # wireframe pixel cores are evaluated.
    mask=center & foreground
    actual=rendered[mask].astype(np.float32)
    expected=source[mask].astype(np.float32)
    if actual.shape[0]<150:raise AssertionError("Too few bone color center pixels "+clip+str(frame))
    ids_expected=np.argmin(np.sum((expected[:,None,:]-prototypes[None,:,:])**2,axis=2),axis=1)
    ids_actual=np.argmin(np.sum((actual[:,None,:]-prototypes[None,:,:])**2,axis=2),axis=1)
    good=int(np.sum(ids_expected==ids_actual))
    count=len(ids_actual)
    if count==0 or good/count<.90:
     raise AssertionError("Reference compressed color semantics below 90% "+str((clip,frame,good,count)))
    local_correct+=good;local_total+=count
 for cap in caps.values():
  ok,_=cap.read()
  cap.release()
  if ok:raise AssertionError("Unexpected extra MP4 frame "+clip)
 row["pose_line_color_accuracy"]=round(local_correct/local_total,6)
 report["records"][clip]=row
 correct+=local_correct;total+=local_total
 report["fully_decoded_videos"]+=2
 print("V5_ENCODED_MP4_QA",i+1,"/16",clip,"line_center_accuracy",row["pose_line_color_accuracy"],flush=True)
if rgb_cnt!=20*107 or pose_cnt!=20*107 or total==0:
 raise AssertionError("Missing decoded frames")
if correct/total<.95:raise AssertionError("Overall bone color classification below 95%")
report["encoded_pose_stable_color_accuracy"]=round(correct/total,6)
report["mean_rgb_frame_l1_mae_255"]=round(totalrgb/rgb_cnt,5)
report["mean_pose_frame_l1_mae_255"]=round(totalpose/pose_cnt,5)
report["status"]="PASS_ACTUAL_H264_RGB_AND_POSE"
path=DATA/"ENCODED_MP4_INDEPENDENT_QA.json"
path.write_text(json.dumps(report,indent=2),encoding="utf8")
print("V5_ENCODED_ALL_PASS",json.dumps({k:v for k,v in report.items() if k!="records"},indent=2),flush=True)
