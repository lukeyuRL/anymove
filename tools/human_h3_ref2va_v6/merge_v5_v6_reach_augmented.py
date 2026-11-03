"""Merge accepted Human V5 4-camera 12-train baseline with V6 16 varied Reach.

Produces 28-train final dataset + manifest of 8 heldout, without
modifying either source archive or previous LoRA. Neither fal upload
nor paid inference/training happens here. Fails closed on QA.
"""
import collections,hashlib,json,pathlib,zipfile
H3=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va")
OLD=H3/"fourview_v5_l45_r45_back_top"
NEW=H3/"fourview_v6_reach_augmented"
OUT=H3/"fourview_v6_reach_expanded"
V2=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\v2")
V6=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\v6_reach_source")
def read(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for buf in iter(lambda:f.read(1048576),b""):h.update(buf)
 return h.hexdigest()
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True)
 tmp=pathlib.Path(str(p)+".partial")
 tmp.write_text(json.dumps(obj,indent=2),encoding="utf8")
 tmp.replace(p)
def check_preflight():
 oldqa=read(OLD/"FINAL_ACCEPTANCE.json")
 if oldqa["status"]!="PASS_OFFLINE_DATASET":
  raise ValueError("Original V5 data not formally accepted")
 if sha(OLD/"fal_h3_ref2va_v5_l45_r45_back_top_12_train.zip")!=oldqa["training_zip"]["sha256"]:
  raise ValueError("Original V5 ZIP hash mismatch")
 newqa=read(NEW/"STATIC_INDEPENDENT_QA.json")
 enc=read(NEW/"ENCODED_MP4_INDEPENDENT_QA.json")
 if newqa["status"]!="PASS_STATIC_FOURVIEW_GEOMETRY_AND_MEDIA" or enc["status"]!="PASS_ACTUAL_H264_RGB_AND_POSE":
  raise ValueError("V6 augmented Reach not independently accepted")
 if newqa["train"]!=16 or newqa["heldout"]!=4 or newqa["verified_rgb_and_pose_png_frames"]!=20*107:
  raise ValueError("Insufficient V6 QA coverage")
 meta=read(NEW/"media_manifest.json")
 if sha(meta["training_zip"]["path"])!=meta["training_zip"]["sha256"]:
  raise ValueError("V6 augmented Reach ZIP hash mismatch")
 if meta["canvas"]!=[896,896] or meta["view_order"]!=["left_45","right_45","back","top"]:
  raise ValueError("V6 view layout differs")
 before=read(OLD/"media_manifest.json")
 if before["canvas"]!=meta["canvas"] or before["view_order"]!=meta["view_order"]:
  raise ValueError("V5/V6 render format differs; can not safely mix")
 diversity=read(V6/"INDEPENDENT_MOTION_DIVERSITY_QA.json")
 if diversity["status"]!="PASS_20_NEW_REACH_WORLD_GEOMETRY" or len(diversity["metrics"])!=20:
  raise ValueError("3D Reach diversity audit not accepted")
 oldplan=read(V2/"v2_dataset_plan.json")
 newplan=read(V6/"v2_dataset_plan.json")
 if oldplan["status"]!="PASS" or newplan["status"]!="PASS":
  raise ValueError("Source motion datasets not accepted")
 old_train=[x for x in oldplan["results"] if x["split"]=="train"]
 new_train=[x for x in newplan["results"] if x["split"]=="train"]
 old_holdout=[x for x in oldplan["results"] if x["split"]=="heldout"]
 new_holdout=[x for x in newplan["results"] if x["split"]=="heldout"]
 if not(len(old_train)==12 and len(new_train)==16 and len(old_holdout)==len(new_holdout)==4):
  raise ValueError("Wrong train/holdout composition")
 train=old_train+new_train
 holdout=old_holdout+new_holdout
 ids=[x["id"] for x in train+holdout]
 if len(ids)!=36 or len(set(ids))!=36:
  raise ValueError("Train/heldout leakage by clip identifier")
 if {x["id"] for x in train}&{x["id"] for x in holdout}:
  raise ValueError("Train/heldout overlap")
 if collections.Counter(x["family"] for x in new_train)!={
     f"reach_{kind}_{side}":2 for kind in ("forward","diagonal","high","backward") for side in ("left","right")}:
  raise ValueError("Augmented Reach family count differs")
 oldsha={}
 newsha={}
 for c in old_train:
  oldsha[c["id"]]=sha(pathlib.Path(c["path"])/"motion.json")
 for c in new_train+new_holdout:
  newsha[c["id"]]=sha(pathlib.Path(c["path"])/"motion.json")
 if len(set(oldsha.values()))!=len(oldsha) or len(set(newsha.values()))!=len(newsha):
  raise ValueError("Duplicate source motion file content")
 return oldqa,newqa,enc,meta,oldplan,newplan,train,holdout
def main():
 oldqa,newqa,enc,meta,oldplan,newplan,train,holdout=check_preflight()
 OUT.mkdir(parents=True,exist_ok=True)
 archive=OUT/"fal_h3_ref2va_v6_28train_fourview_diverse_reach.zip"
 orig=OLD/"fal_h3_ref2va_v5_l45_r45_back_top_12_train.zip"
 aug=pathlib.Path(meta["training_zip"]["path"])
 if not archive.exists():
  tmp=pathlib.Path(str(archive)+".partial")
  try:
   with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as combined:
    for filename in (orig,aug):
     with zipfile.ZipFile(filename) as z:
      if z.testzip() is not None:raise ValueError("Broken source ZIP "+str(filename))
      for name in z.namelist():
       combined.writestr(name,z.read(name))
   tmp.replace(archive)
  finally:tmp.unlink(missing_ok=True)
 sources={}
 for file in (orig,aug):
  with zipfile.ZipFile(file) as z:
   for name in z.namelist():
    if name in sources:raise ValueError("Training stem duplicate")
    sources[name]=z.read(name)
 with zipfile.ZipFile(archive) as z:
  if len(z.namelist())!=28*4 or len(set(z.namelist()))!=28*4 or z.testzip() is not None:
   raise ValueError("Final archive not exactly 112 valid members")
  for name,data in sources.items():
   if z.read(name)!=data:
    raise ValueError("Final archive member content drift "+name)
  for clip in train:
   stem=clip["id"]
   for suffix in (".mp4",".txt",".ref_1.mp4",".ref_2.png"):
    if stem+suffix not in z.namelist():
     raise ValueError("Missing paired training Sidecar "+stem+suffix)
   text=z.read(stem+".txt").decode("utf8")
   if not all(s in text for s in ("TOP-LEFT","TOP-RIGHT","BOTTOM-LEFT","BOTTOM-RIGHT")):
    raise ValueError("Training caption four cameras not described "+stem)
   if "NO front panel" not in text and "NO front panel." not in text and "There is NO front panel." not in text:
    raise ValueError("No-front rule absent in caption")
 output={
  "schema":"anymove.h3_fourview_diverse_reach_28train_8heldout/1",
  "status":"PASS_MERGED_OFFLINE_FILES_AND_INTEGRITY",
  "new_training_status":"NOT_STARTED",
  "new_paid_fal_jobs":0,
  "fal_trainer":"minimax/h3/ref2va/trainer",
  "fal_inference":"minimax/h3/reference-to-video/lora",
  "recommended_trigger_phrase":"FOURCAM",
  "same_view_layout":["left_45","right_45","back","top"],
  "canvas":[896,896],"aspect_ratio":"1:1",
  "frames":107,"fps":24,
  "training_clips":28,"heldout_clips":8,
  "training_reach_clips":20,"training_lean_clips":8,
  "original_v5_training_clips":12,
  "added_reach_training_clips":16,
  "added_reach_heldout_clips":4,
  "source_V5_archive_sha256":sha(orig),
  "source_V6_aug_archive_sha256":sha(aug),
  "merged_archive":{"path":str(archive),"bytes":archive.stat().st_size,
                    "sha256":sha(archive),"entries":112},
  "train_records":train,"heldout_records":holdout,
  "new_reach_family_counts":dict(collections.Counter(x["family"] for x in newplan["results"] if x["split"]=="train")),
  "known_motion_similarity_warning":{
     "high_reach_overlaps_prior_lateral_reach":True,
     "new_heldout_high_left_nearest_original_reach_peak_hand_distance_m":0.0052,
     "do_not_treat_heldout_high_left_as_independent_novel_motion_test":True,
     "main_new_direction_families":["forward","diagonal","backward"],
     "high_reach_family_is_amplitude_and_timing_variation_not_a_new_axis":True
  },
  "static_v6_QA":newqa["status"],
  "encoded_v6_QA":enc["status"],
  "H3_sparsely_sampled_soft_reference_poses":True,
  "3D_generation_quality_unverified":True,
  "does_not_authorize_new_training":True
 }
 save(OUT/"MERGED_DATASET_QA.json",output)
 print("V6_FOURVIEW_DIVERSE_REACH_MERGE_PASS",json.dumps({
  k:v for k,v in output.items() if k not in ("train_records","heldout_records","new_reach_family_counts")
 },indent=2),flush=True)
if __name__=="__main__":main()
