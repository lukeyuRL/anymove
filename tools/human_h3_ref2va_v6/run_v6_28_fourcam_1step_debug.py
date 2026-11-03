"""One authorized paid fal H3 Ref2VA Human V6 28-clip FOUR-CAMERA 1-step debug.

User has authorized exactly ONE new 1-Step Debug training job, NOT full
400 Step training or generation. 100 billed-step floor ~US$1.50.
This script refuses new training if an accepted/uncertain request exists.

Fourview V6: 28 train, 8 heldout, 896x896 1:1, 107 frames@24fps,
TL left-45, TR right-45, BL back, BR top, no front panel.
Uses new FOURCAM trigger and official native fal ref2va trainer only.
"""
import collections
import hashlib
import io
import json
import pathlib
import sys
import zipfile
import requests

ROOT=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(ROOT))
from mocap import fal

HERE=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"
DATA=HERE/"fourview_v6_reach_expanded"
ZIP=DATA/"fal_h3_ref2va_v6_28train_fourview_diverse_reach.zip"
RUN=HERE/"remote"/"h3_ref2va_v6_28_fourview_debug_1step"
ENDPOINT="minimax/h3/ref2va/trainer"
EXPECTED_SHA256="76faf527f9e5cd7f9521b8a42f306b7f07fda04f6c2d7cc2d4d233f57f102141"
EXPECTED_NUMBER=28
PRICE_USD_PER_STEP=.015
BILLING_FLOOR=100
BILL_ESTIMATE_USD=1.50

INPUT={
 "rank":16,
 "number_of_steps":1,
 "learning_rate":.0001,
 "number_of_frames":107,
 "frame_rate":24,
 "resolution":"medium",
 "aspect_ratio":"1:1",
 "trigger_phrase":"FOURCAM",
 "reference_conditioning_p":.95,
 "auto_scale_input":False,
 "split_input_into_scenes":False,
 "strict_dataset":False,
 "debug_dataset":True
}

def load(p):
 return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
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
def verify_zip():
 if not ZIP.is_file() or ZIP.stat().st_size<12000000:
  raise RuntimeError("Final 28-train ZIP missing or suspiciously small")
 digest=sha(ZIP)
 if digest!=EXPECTED_SHA256:raise RuntimeError("Validated 28-train ZIP SHA drift")
 mapping=collections.defaultdict(set)
 bytecount=collections.Counter()
 observed={}
 with zipfile.ZipFile(ZIP) as archive:
  files=archive.namelist()
  if len(files)!=112 or len(set(files))!=112 or archive.testzip() is not None:
   raise RuntimeError("Wrong ZIP member count or CRC")
  if any("/" in n or "\\" in n for n in files):
   raise RuntimeError("Expected a flat fal ZIP with sidecars")
  for name in files:
   if name.endswith(".ref_1.mp4"):
    stem=name.removesuffix(".ref_1.mp4");kind="pose"
   elif name.endswith(".ref_2.png"):
    stem=name.removesuffix(".ref_2.png");kind="identity"
   elif name.endswith(".mp4"):
    stem=name.removesuffix(".mp4");kind="rgb"
   elif name.endswith(".txt"):
    stem=name.removesuffix(".txt");kind="caption"
   else:raise RuntimeError("Unexpected training ZIP member "+name)
   if kind in mapping[stem]:raise RuntimeError("Duplicate sidecar type "+stem)
   mapping[stem].add(kind)
   if kind=="caption":
    caption=archive.read(name).decode("utf8")
    if not all(s in caption for s in ("TOP-LEFT","TOP-RIGHT","BOTTOM-LEFT","BOTTOM-RIGHT",
                                       "NO front panel","anatomical","3D")):
     raise RuntimeError("Four-panel instructions missing from caption "+stem)
    if "TWOCAM" in caption or "Front view on the LEFT" in caption:
     raise RuntimeError("Two-camera caption mixed in "+stem)
    if len(caption)>3500 or len(caption)<280:
     raise RuntimeError("Suspicious caption size "+stem)
   if kind=="identity":
    # PNG header IHDR, 896x896 (we do not re-encode it).
    payload=archive.read(name)
    if payload[:8]!=b"\x89PNG\r\n\x1a\n":
     raise RuntimeError("Invalid RGB identity PNG magic "+stem)
    import struct
    w,h=struct.unpack(">II",payload[16:24])
    if (w,h)!=(896,896):raise RuntimeError("Identity dimensions not 896x896 "+stem)
   if kind in ("rgb","pose"):
    # media already verified by independent full 1712/2140 frame H264 QA;
    # check ISO BMFF bytes for each archive entry too.
    payload=archive.read(name)
    if b"ftyp" not in payload[:128] or len(payload)<10000:
     raise RuntimeError("Corrupt MP4 bytes "+name)
   bytecount[kind]+=1
 if len(mapping)!=EXPECTED_NUMBER or any(
     kinds!={"rgb","pose","identity","caption"} for kinds in mapping.values()):
  raise RuntimeError("Broken 28 x 4 fal Ref2VA sample/sidecar pairing")
 if bytecount!={"rgb":28,"pose":28,"identity":28,"caption":28}:
  raise RuntimeError("Unbalanced 28 sample types")
 return {"sha256":digest,"samples":len(mapping),"files":len(files),
         "bytes":ZIP.stat().st_size,"kinds":dict(bytecount)}
def preflight():
 final=load(DATA/"FINAL_ACCEPTANCE.json")
 merged=load(DATA/"MERGED_DATASET_QA.json")
 v5=load(HERE/"fourview_v5_l45_r45_back_top"/"FINAL_ACCEPTANCE.json")
 v6s=load(HERE/"fourview_v6_reach_augmented"/"STATIC_INDEPENDENT_QA.json")
 v6e=load(HERE/"fourview_v6_reach_augmented"/"ENCODED_MP4_INDEPENDENT_QA.json")
 diversity=load(ROOT/"runs"/"human"/"lora_synth_A"/"v6_reach_source"/"INDEPENDENT_MOTION_DIVERSITY_QA.json")
 if final["status"]!="PASS_OFFLINE_DATASET" or merged["status"]!="PASS_MERGED_OFFLINE_FILES_AND_INTEGRITY":
  raise RuntimeError("28-clip dataset not finally accepted")
 if v5["status"]!="PASS_OFFLINE_DATASET" or v6s["status"]!="PASS_STATIC_FOURVIEW_GEOMETRY_AND_MEDIA":
  raise RuntimeError("Source V5/V6 media not independently validated")
 if v6e["status"]!="PASS_ACTUAL_H264_RGB_AND_POSE" or diversity["status"]!="PASS_20_NEW_REACH_WORLD_GEOMETRY":
  raise RuntimeError("Decoded H264 or original 3D source motion QA failed")
 if v6s["verified_rgb_and_pose_png_frames"]!=20*107 or v6s["independent_depth_owner_witnesses"]!=320:
  raise RuntimeError("Not all source frames have been verified")
 if final["final_zip"]["sha256"]!=EXPECTED_SHA256 or final["final_zip"]["training_clips"]!=28 or final["final_zip"]["heldout_clips"]!=8:
  raise RuntimeError("Dataset final contract differs")
 if final["layout"]["camera_extrinsics_consumed_by_fal_as_numbers"] is not False:
  raise RuntimeError("Misrepresented fal H3 camera support")
 if merged["aspect_ratio"]!="1:1" or merged["same_view_layout"]!=["left_45","right_45","back","top"]:
  raise RuntimeError("User-approved 4-camera 1:1 contract differs")
 if merged["new_paid_fal_jobs"]!=0:
  raise RuntimeError("New dataset already has paid job record; reconcile first")
 verify=verify_zip()
 if not fal.key():
  raise RuntimeError("fal API credentials unavailable")
 return {
  "state":"PREFLIGHT_PASS",
  "endpoint":ENDPOINT,
  "training_zip":str(ZIP),
  "dataset":verify,
  "parameters":INPUT,
  "billing_floor_steps":BILLING_FLOOR,
  "expected_success_billing_usd":BILL_ESTIMATE_USD,
  "already_accepted":(RUN/"submitted.json").is_file(),
  "uncertain_prior_paid_POST":(RUN/"submission_in_progress.json").is_file(),
  "existing_uploaded_URL":(RUN/"uploaded_dataset.json").is_file(),
  "user_authorization":"Exactly one 1-Step paid debug; no 400-Step or inference"
 }
def status():
 info=preflight()
 # Redact raw upload URLs / keys in preflight output.
 print("V6_FOURVIEW_REF2VA_1STEP_PREFLIGHT",json.dumps(info,indent=2),flush=True)
 if (RUN/"submitted.json").exists():
  print("SAVED_REQUEST_ID",load(RUN/"submitted.json")["request_id"],flush=True)
 if (RUN/"last_status.json").exists():
  print("LAST_POLL",load(RUN/"last_status.json").get("status"),flush=True)
def submit():
 info=preflight()
 RUN.mkdir(parents=True,exist_ok=True)
 marker=RUN/"submission_in_progress.json"
 accepted=RUN/"submitted.json"
 if accepted.exists():
  print("V6_ALREADY_SUBMITTED_NO_DUPLICATE",load(accepted).get("request_id"),flush=True)
  return
 if marker.exists():
  raise RuntimeError("Paid submission may already be accepted: no retry")
 uploaded=RUN/"uploaded_dataset.json"
 if uploaded.is_file():
  upload=load(uploaded)
  if upload.get("sha256")!=EXPECTED_SHA256 or not upload.get("url","").startswith("https://"):
   raise RuntimeError("Untrusted previously uploaded URL")
  url=upload["url"]
  print("REUSING_PREVIOUS_VALIDATED_ZIP_UPLOAD",flush=True)
 else:
  print("UPLOADING_H3_V6_28_SAMPLE_4CAM_ZIP",ZIP.stat().st_size,flush=True)
  url=fal.upload(ZIP)
  if not isinstance(url,str) or not url.startswith("https://"):
   raise RuntimeError("Fal upload did not return URL")
  save(uploaded,{"url":url,"sha256":EXPECTED_SHA256,"bytes":ZIP.stat().st_size,
                 "local_zip_path":str(ZIP)})
  print("H3_4CAM_ZIP_UPLOADED",flush=True)
 request=dict(INPUT)
 request["training_data_url"]=url
 if "resume_from_lora_url" in request:
  raise RuntimeError("Debug must train fresh, no warm start")
 staged=RUN/"request.json"
 if staged.exists() and load(staged)["request"]!=request:
  raise RuntimeError("Existing request parameters changed; refuse")
 save(staged,{"endpoint":ENDPOINT,"request":request,
   "dataset_sha256":EXPECTED_SHA256,"expected_billed_USD":BILL_ESTIMATE_USD,
   "steps":1,"user_approved_paid_training_scope":"ONE 1-step debug only"})
 save(marker,{"endpoint":ENDPOINT,"request":request,
    "danger":"Paid HTTP POST may have been accepted; NEVER retry before reconciliation"})
 print("SUBMITTING_ONLY_ONE_PAID_FAL_REF2VA_V6_1STEP_DEBUG",flush=True)
 # Deliberately one HTTP POST without retry after uncertain HTTP failures.
 resp=requests.post("https://queue.fal.run/"+ENDPOINT,
                    headers=fal.headers(),json=request,timeout=90)
 if resp.status_code>=400:
  save(RUN/"submit_error.json",{"status_code":resp.status_code,
                               "body":resp.text[:1400]})
  if resp.status_code in (400,401,403,404,422):
   marker.unlink(missing_ok=True)
  raise RuntimeError("Fal rejected 1-Step Debug, HTTP "+str(resp.status_code)+" "+resp.text[:350])
 doc=resp.json()
 if not all(doc.get(k) for k in ("request_id","status_url","response_url")):
  save(RUN/"unrecognized_submit_response.json",doc)
  raise RuntimeError("Ambiguous successful submission; keep safety marker")
 save(accepted,{**doc,"endpoint":ENDPOINT})
 marker.unlink(missing_ok=True)
 print("V6_28_FOURCAM_1STEP_DEBUG_SUBMITTED",json.dumps({
   "request_id":doc["request_id"],"endpoint":ENDPOINT,
   "steps":1,"samples":28,"aspect_ratio":"1:1",
   "billed_step_floor":100,
   "estimated_cost_usd":BILL_ESTIMATE_USD,
   "no_full_training_or_video_gen":True,
   "job_dir":str(RUN)},indent=2),flush=True)
def poll():
 subfile=RUN/"submitted.json"
 if not subfile.is_file():
  raise FileNotFoundError("No accepted V6 1-Step job; poll NEVER submits")
 sub=load(subfile)
 response=requests.get(sub["status_url"],headers=fal.headers(),timeout=60)
 response.raise_for_status()
 st=response.json()
 save(RUN/"last_status.json",st)
 print("V6_28_4CAM_REMOTE_STATUS",st.get("status"),sub["request_id"],flush=True)
 if st.get("status")!="COMPLETED":
  if st.get("status") not in ("IN_QUEUE","IN_PROGRESS"):
   print("V6_4CAM_UNEXPECTED_STATUS",json.dumps(st)[:1200],flush=True)
  return False
 result=RUN/"result.json"
 if result.exists():
  res=load(result)
 else:
  response=requests.get(sub["response_url"],headers=fal.headers(),timeout=120)
  response.raise_for_status()
  res=response.json()
  save(result,res)
 for name in ("lora_file","config_file","debug_dataset"):
  if not isinstance(res.get(name),dict) or not res[name].get("url"):
   raise RuntimeError("Completed fal job lacks "+name)
 target=RUN/"debug_dataset.zip"
 expected=int(res["debug_dataset"].get("file_size") or 0)
 if target.exists() and (not expected or target.stat().st_size==expected):
  print("V6_REUSE_DEBUG_ZIP",target.stat().st_size,flush=True)
 else:
  tmp=RUN/"debug_dataset.partial.zip"
  print("V6_DOWNLOADING_DEBUG_ZIP",flush=True)
  fal.download(res["debug_dataset"]["url"],tmp)
  if expected and tmp.stat().st_size!=expected:
   raise RuntimeError("Downloaded debug.zip size incorrect")
  with zipfile.ZipFile(tmp) as z:
   if z.testzip() is not None:
    raise RuntimeError("Downloaded fal debug ZIP CRC corrupted")
  tmp.replace(target)
 print("V6_1STEP_DEBUG_ARCHIVE_SAVED",json.dumps({
  "request_id":sub["request_id"],"bytes":target.stat().st_size,
  "path":str(target)},indent=2),flush=True)
 return True
if __name__=="__main__":
 method=sys.argv[1] if len(sys.argv)>1 else "preflight"
 {"preflight":status,"status":status,"submit":submit,"poll":poll}[method]()
