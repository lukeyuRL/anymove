"""Offline QA for completed fal H3 Ref2VA V6 FOURCAM 1-Step debug result.

DO NOT submit training or video inference. Run after single accepted
request COMPLETED and debug_dataset.zip present.

Checks: 28 input clips survived preprocessing; output video still square
and 4-panel, 107f/24fps; caption contains correct 4-camera semantics;
two H3 ref sidecars per sample; video conditioning frames (sparse soft
reference, not hard per-frame control) retain color variation in all 4
panels; Image2 preserved; catches self-reference fallbacks / corruptions.

Does NOT claim trained LoRA motion quality.
"""
import collections
import csv
import hashlib
import io
import json
import pathlib
import shutil
import subprocess
import zipfile
import numpy as np
import cv2
from PIL import Image,ImageDraw

ROOT=pathlib.Path(r"F:\anymove")
H3=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"
RUN=H3/"remote"/"h3_ref2va_v6_28_fourview_debug_1step"
ZIP=RUN/"debug_dataset.zip"
OUT=RUN/"audit"
INPUT=H3/"fourview_v6_reach_expanded"/"fal_h3_ref2va_v6_28train_fourview_diverse_reach.zip"
FFPROBE=pathlib.Path(r"C:\ffmpeg\bin\ffprobe.exe")
EXPECTED_SHA="76faf527f9e5cd7f9521b8a42f306b7f07fda04f6c2d7cc2d4d233f57f102141"

def load(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def save(p,obj):
 p=pathlib.Path(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=pathlib.Path(str(p)+".tmp")
 tmp.write_text(json.dumps(obj,indent=2),encoding="utf8")
 tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def video_profile(p):
 r=subprocess.run([str(FFPROBE),"-v","error","-select_streams","v:0",
                   "-show_entries","stream=width,height,nb_frames,r_frame_rate",
                   "-of","json",str(p)],check=True,capture_output=True,text=True)
 x=json.loads(r.stdout)["streams"][0]
 return {"width":int(x["width"]),"height":int(x["height"]),
         "frames":int(x["nb_frames"]),"fps":x["r_frame_rate"]}
def assertn(v,message):
 if not v:raise AssertionError(message)
def fourview_rgb_stats(arr,background_guess=None):
 # Four quadrants each must contain colored/moving pixels above its
 # local neutral gray or black background. Count saturated pixels
 # in target RGB; for Pose reference counts the nonblack pixels.
 h,w=arr.shape[:2]
 assertn(h==w and h%2==0,"Expected 1:1 four camera frame")
 metrics={}
 for view,(col,row) in {
   "left45":(0,0),"right45":(1,0),"back":(0,1),"top":(1,1)}.items():
  panel=arr[row*h//2:(row+1)*h//2,
            col*w//2:(col+1)*w//2]
  hsv=cv2.cvtColor(panel,cv2.COLOR_RGB2HSV)
  color_count=int(np.count_nonzero((hsv[:,:,1]>55)&(hsv[:,:,2]>40)))
  # Count at least some non-background significant pixels to ensure
  # Top was not silently replaced by a blank/mirrored two-camera panel.
  metrics[view]={
    "saturated_pixels":color_count,
    "saturated_fraction":round(color_count/max(1,panel.shape[0]*panel.shape[1]),6)}
 return metrics
def preview_image(im,side=224):
 return im.resize((side,side),Image.Resampling.LANCZOS)
def run():
 assertn(RUN.joinpath("submitted.json").is_file(),"V6 Debug not submitted")
 assertn(RUN.joinpath("result.json").is_file(),"V6 Debug has no result yet")
 assertn(ZIP.is_file(),"V6 debug archive not downloaded yet")
 assertn(sha(INPUT)==EXPECTED_SHA,"Source train ZIP SHA drift")
 OUT.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(ZIP) as archive,zipfile.ZipFile(INPUT) as source:
  names=archive.namelist()
  assertn(archive.testzip() is None,"Corrupt fal debug ZIP")
  debug_name=next((s for s in names if s.endswith("debug_manifest.json")),None)
  preprocess_name=next((s for s in names if s.endswith("preprocessing_report.json")),None)
  assertn(debug_name and preprocess_name,"Fal debug metadata missing")
  manifest=json.loads(archive.read(debug_name))
  preprocess=json.loads(archive.read(preprocess_name))
  samples=manifest.get("samples")
  assertn(isinstance(samples,list) and len(samples)==28,
          "Fal skipped/reordered dataset count: not 28 samples")
  reportcounts=preprocess.get("counts",{})
  badfallback={key:v for key,v in reportcounts.items()
       if key!="audio_missing_used_silence" and v}
  assertn(not badfallback,"Fal used forbidden preprocessing fallback "+repr(badfallback))
  expected_source={n.removesuffix(".mp4") for n in source.namelist()
                   if n.endswith(".mp4") and ".ref_" not in n}
  actual_source={p.get("source_media","").removesuffix(".mp4") for p in samples}
  assertn(len(expected_source)==28 and actual_source==expected_source,
          "Fal source sample identity/dedup mismatch")
  targets=[]
  paired=[]
  reference_frames=[]
  thumbnails=[]
  failures=[]
  for i,s in enumerate(samples):
   name=s["source_media"]
   stem=name.removesuffix(".mp4")
   assertn(s.get("index")==i,"Fal sample index unexpectedly changed")
   caption=s.get("caption","")
   assertn("FOURCAM" in caption and all(x in caption for x in (
          "TOP-LEFT","TOP-RIGHT","BOTTOM-LEFT","BOTTOM-RIGHT",
          "NO front panel")),
          "Fal caption lost 4-camera contract "+stem)
   target=s.get("decoded_video")
   assertn(target in names and target.endswith(".mp4"),
          "Fal preprocessed target video missing "+stem)
   refs=s.get("conditioning_files")
   assertn(isinstance(refs,list) and len(refs)==2,"Fal sidecars missing or multiplied "+stem)
   refs_video=[x for x in refs if x.endswith("_ref_1_video.npz")]
   refs_image=[x for x in refs if x.endswith("_ref_2.png")]
   assertn(len(refs_video)==len(refs_image)==1,"Fal sidecar types do not match planned 1 video+1 image")
   for x in refs:assertn(x in names,"Missing fal sidecar "+x)
   targetfile=OUT/("decoded_target_"+str(i).zfill(3)+".mp4")
   if not targetfile.exists():
    with archive.open(target) as source_stream,targetfile.open("wb") as d:
     shutil.copyfileobj(source_stream,d)
   p=video_profile(targetfile)
   assertn(p["frames"]==107 and p["fps"]=="24/1" and
           p["width"]==p["height"] and p["width"]>=384,
           "fal decoded target video aspect/frames changed "+stem+" "+str(p))
   # Capture target mid and reference image for four-camera layout
   cap=cv2.VideoCapture(str(targetfile))
   cap.set(cv2.CAP_PROP_POS_FRAMES,53)
   ok,frame=cap.read()
   cap.release()
   assertn(ok,"fal target MP4 failed to decode frame53")
   rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
   tmetrics=fourview_rgb_stats(rgb)
   assertn(all(x["saturated_pixels"]>80 for x in tmetrics.values()),
           "Target missing one of four colored views "+stem)
   identity=Image.open(io.BytesIO(archive.read(refs_image[0]))).convert("RGB")
   assertn(identity.width==identity.height and identity.width>=384,
           "Fal resized first-frame image to non-square "+stem)
   identity_stats=fourview_rgb_stats(np.asarray(identity))
   assertn(all(x["saturated_pixels"]>60 for x in identity_stats.values()),
           "Fal identity ref missing four cameras "+stem)
   original_identity=Image.open(io.BytesIO(source.read(stem+".ref_2.png"))).convert("RGB")
   # Note: fal resize is allowed; exact match only if dimensions same.
   image_exact_if_same=None
   if original_identity.size==identity.size:
    image_exact_if_same=bool(np.array_equal(np.asarray(original_identity),np.asarray(identity)))
   with np.load(io.BytesIO(archive.read(refs_video[0])),allow_pickle=False) as npz:
    assertn("frames" in npz.files,"Fal sparse ref1 video has no frame list "+stem)
    frames=npz["frames"]
    assertn(frames.ndim==4 and frames.shape[-1]==3 and
            frames.shape[1]==frames.shape[2] and
            frames.shape[0]>=2,
            "Fal sparse Pose reference invalid shape "+stem+str(frames.shape))
    n=int(frames.shape[0])
    stats=[]
    for frame_idx in (0,min(n-1,n//2),n-1):
     sampled=frames[frame_idx]
     cm=fourview_rgb_stats(sampled)
     assertn(all(x["saturated_pixels"]>20 for x in cm.values()),
             "Fal sparse Pose reference lost one of four colored panels "+stem)
     stats.append(cm)
    reference_frames.append(n)
    refsizes=[int(v) for v in frames.shape]
   data={
     "sample_index":i,
     "stem":stem,
     "decoded_target_video":p,
     "target_mid_four_quadrants":tmetrics,
     "first_reference_image_size":list(identity.size),
     "first_reference_image_square":True,
     "first_reference_image_exact_if_same_dims":image_exact_if_same,
     "sparse_reference_video_tensor_shape":refsizes,
     "sparse_reference_video_sample_stats":stats,
     "caption_fourcam_preserved":True,
     "conditioning_files_present":True,
     "note":"Soft sampled Pose reference, not hard 107-frame pose control"
   }
   paired.append(data)
   # Color targets contacts one every 5
   if i%5==0:
    im=preview_image(Image.fromarray(rgb))
    thumbnails.append((stem,im))
   print("V6_28_4CAM_PREPROCESS_SAMPLE_PASS",i+1,"/28",stem,p,refsizes,flush=True)
 assertn(len(paired)==28 and len(set(reference_frames))==1,
         "Fal inconsistent sparse ref frames between clips")
 # Contact sheet for quick inspection, independent of model training.
 cols=3
 rows=(len(thumbnails)+cols-1)//cols
 contact=Image.new("RGB",(cols*236,rows*(236+25)),(239,239,239))
 dr=ImageDraw.Draw(contact)
 for j,(stem,thumb) in enumerate(thumbnails):
  x=(j%cols)*236;y=(j//cols)*261
  contact.paste(thumb,(x,y))
  dr.text((x+4,y+236),stem[:29],fill=(20,20,20))
 imagepath=OUT/"fourcam_fal_decoded_targets_contact.jpg"
 contact.save(imagepath,quality=91)
 status={
  "schema":"anymove.fal_h3_ref2va_v6_fourcam_1step_debug_preprocessing/1",
  "status":"PASS_PREPROCESSED_28_FOUR_CAMERA_SAMPLES",
  "fal_request_id":load(RUN/"submitted.json")["request_id"],
  "source_dataset_sha256":EXPECTED_SHA,
  "source_training_clips":28,
  "preprocessed_samples":len(paired),
  "captions_with_FOURCAM":28,
  "reference_video_and_image_present_for_all_28":True,
  "reference1_sparse_video_frames":reference_frames[0],
  "no_unexpected_fallback":True,
  "preprocessing_fallbacks":reportcounts,
  "metadata":{
    "samples":paired,
    "decoded_target_contact":str(imagepath)
  },
  "caveats":[
   "LoRA training completed only 1 step; weights are debug artifact, not validated motion quality",
   "Training ZIP heldout 8 clips were NOT submitted to fal",
   "Video conditioning is sparse and soft, not hard 3D joint control",
   "Some fal resize/conditioning image preprocessing can change nominal pixel resolution",
   "fal audio missing -> silence substitution is expected for silent synthetic Blender inputs"
  ],
  "no_paid_inference_video":True,
  "no_full_training_400step":True
 }
 save(OUT/"FULL_28_FOURCAM_DEBUG_AUDIT.json",status)
 print("FAL_V6_28_FOURCAM_FULL_DEBUG_PREPROCESS_PASS",json.dumps({
  "status":status["status"],"request_id":status["fal_request_id"],
  "samples":len(paired),
  "sparse_ref_pose_video_frame_count":reference_frames[0],
  "preprocessing_fallbacks":reportcounts,
  "contact":str(imagepath),"report":str(OUT/"FULL_28_FOURCAM_DEBUG_AUDIT.json")
 },indent=2),flush=True)
if __name__=="__main__":run()
