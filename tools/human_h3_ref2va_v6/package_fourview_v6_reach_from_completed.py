"""Build exact 16-paired native fal H3 Ref2VA V6 ZIP from completed media.

Does NOT re-render or retrain. Verifies every clip's existing manifest,
actual H264 profile, canonical 896x896 layout, and source 3D identity.
Then uses already-reviewed package() function, writes V6 media manifest.
"""
import collections,importlib.util,json,pathlib
ROOT=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A")
H3=ROOT/"conditioned_backend"/"fal_h3_ref2va"
BASE=H3/"fourview_v6_reach_augmented"
V6=ROOT/"v6_reach_source"
mod=H3/"build_fourview_v6_reach_media.py"
spec=importlib.util.spec_from_file_location("h3_v6_reach_media",mod)
lib=importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)
plan=json.loads((V6/"v2_dataset_plan.json").read_text(encoding="utf8"))
if plan["status"]!="PASS" or len(plan["results"])!=20:
 raise ValueError("Source 20 clips incomplete")
if collections.Counter(x["split"] for x in plan["results"])!={"train":16,"heldout":4}:
 raise ValueError("Source splits incorrect")
rows=[]
for i,c in enumerate(plan["results"]):
 root=BASE/"clips"/c["split"]/c["id"]
 m=root/"fourview_clip_manifest.json"
 if not m.is_file():raise FileNotFoundError(m)
 item=json.loads(m.read_text(encoding="utf8"))
 if item["stem"]!=c["id"] or item["split"]!=c["split"] or item["family"]!=c["family"]:
  raise ValueError("Mixed V6 clip manifest "+c["id"])
 if item["view_names"]!=["left_45","right_45","back","top"] or item["canvas"]!=[896,896]:
  raise ValueError("V6 fourview layout changed")
 if not all((root/p).exists() for p in
          ("rgb_fourview.mp4","pose_fourview.mp4","identity_first_frame.png","caption.txt")):
  raise ValueError("Incomplete V6 output "+c["id"])
 for stem_key,source_path in (
     ("rgb_fourview_video",root/"rgb_fourview.mp4"),
     ("pose_fourview_video",root/"pose_fourview.mp4")):
  if item[stem_key]["sha256"]!=lib.sha(source_path):
   raise ValueError("V6 media output video checksum drift")
  if lib.profile(source_path)!={k:item[stem_key][k] for k in ("sha256","bytes","width","height","fps","frames")}:
   raise ValueError("V6 MP4 profile changed "+c["id"])
 if item["first_frame_identity_sha256"]!=lib.sha(root/"identity_first_frame.png"):
  raise ValueError("V6 identity changed")
 if (root/"caption.txt").read_bytes()!=(item["train_caption"]+"\n").encode("utf8"):
  raise ValueError("Caption LF mismatch "+c["id"])
 rows.append(item)
 print("V6_REACH_MEDIA_BUNDLE_ACCEPTED",i+1,"/20",c["id"],flush=True)
pkg=lib.package(rows)
media={
 "schema":"anymove.human_h3_ref2va_diverse_reach_v6_media/1",
 "status":"MEDIA_PACKAGED_INDEPENDENT_QA_REQUIRED",
 "view_order":["left_45","right_45","back","top"],
 "quadrants":{k:list(v) for k,v in lib.POS.items()},
 "canvas":[896,896],
 "fal_native_trainer":"minimax/h3/ref2va/trainer",
 "fal_native_inference":"minimax/h3/reference-to-video/lora",
 "suggested_training_aspect_ratio":"1:1",
 "suggested_resolution":"medium",
 "number_of_frames":107,"frame_rate":24,
 "trigger_phrase":"FOURCAM",
 "clip_records":rows,
 "training_zip":pkg,
 "source_model_unchanged":True,
 "existing_400step_lora_unchanged":True,
 "new_paid_training_or_video_jobs":0
}
lib.save(BASE/"media_manifest.json",media)
print("V6_REACH_NATIVE_FAL_ARCHIVE_PACKAGED_AWAIT_INDEPENDENT_QA",
      json.dumps(pkg,indent=2),flush=True)
