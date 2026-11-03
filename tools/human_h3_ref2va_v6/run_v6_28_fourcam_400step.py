"""User-approved ONE paid fal native H3 Ref2VA V6 FOURCAM 400-step training.

Fresh from H3 base (NO debug 1step checkpoint warm start).
Preserves same 28 x RGB / caption / Pose ref_1 / RGB identity ref_2
dataset validated by fal 1-Step debug (28/28, refs 9-frame sparse).
Uses same already uploaded ZIP; 400 billed x $0.015 = $6.00.

Modes preflight/submit/poll. Exactly one HTTP paid POST. When uncertain,
retain marker to avoid duplicate billing. Poll NEVER creates a job.
Does not generate any paid inference video.
"""
import hashlib,json,pathlib,requests,struct,sys

ROOT=pathlib.Path(r"F:\anymove")
sys.path.insert(0,str(ROOT))
from mocap import fal

H3=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"
DATA=H3/"fourview_v6_reach_expanded"
DEBUG=H3/"remote"/"h3_ref2va_v6_28_fourview_debug_1step"
RUN=H3/"remote"/"h3_ref2va_v6_28_fourview_400step"
ENDPOINT="minimax/h3/ref2va/trainer"
EXPECTED_SHA="76faf527f9e5cd7f9521b8a42f306b7f07fda04f6c2d7cc2d4d233f57f102141"
EXPECTED_REQUEST_ID="01a1a02c-599c-7281-9144-7c2ef7b46c1e"
ZIP=DATA/"fal_h3_ref2va_v6_28train_fourview_diverse_reach.zip"
STEPS=400
COST_USD=6.00
def load(p):
 return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def save(path,doc):
 path=pathlib.Path(path)
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=pathlib.Path(str(path)+".tmp")
 tmp.write_text(json.dumps(doc,indent=2),encoding="utf8")
 tmp.replace(path)
def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()

def preflight():
 if digest(ZIP)!=EXPECTED_SHA:
  raise RuntimeError("The exact user-approved 28-train ZIP changed")
 final=load(DATA/"FINAL_ACCEPTANCE.json")
 merged=load(DATA/"MERGED_DATASET_QA.json")
 if final["status"]!="PASS_OFFLINE_DATASET" or merged["status"]!="PASS_MERGED_OFFLINE_FILES_AND_INTEGRITY":
  raise RuntimeError("Final V6 dataset not accepted")
 if final["final_zip"]["sha256"]!=EXPECTED_SHA or final["final_zip"]["training_clips"]!=28:
  raise RuntimeError("Invalid final acceptance SHA/size")
 if final["final_zip"]["heldout_clips"]!=8 or merged["aspect_ratio"]!="1:1":
  raise RuntimeError("Heldout or aspect ratio has changed")
 if merged["same_view_layout"]!=["left_45","right_45","back","top"]:
  raise RuntimeError("No-front four-camera contract mismatch")
 fal_audit=load(DEBUG/"audit"/"FULL_28_FOURCAM_DEBUG_AUDIT.json")
 if (fal_audit.get("status")!="PASS_PREPROCESSED_28_FOUR_CAMERA_SAMPLES" or
     fal_audit.get("fal_request_id")!=EXPECTED_REQUEST_ID or
     fal_audit.get("source_dataset_sha256")!=EXPECTED_SHA or
     fal_audit.get("preprocessed_samples")!=28 or
     fal_audit.get("captions_with_FOURCAM")!=28 or
     fal_audit.get("reference_video_and_image_present_for_all_28") is not True or
     fal_audit.get("reference1_sparse_video_frames")!=9 or
     fal_audit.get("no_unexpected_fallback") is not True):
  raise RuntimeError("H3 1-Step preprocessing audit not entirely PASS")
 if fal_audit["preprocessing_fallbacks"]!={"audio_missing_used_silence":28}:
  raise RuntimeError("Unexpected fal 1-step preprocessing fallbacks")
 old_status=load(DEBUG/"last_status.json")
 if old_status["status"]!="COMPLETED" or not (DEBUG/"result.json").is_file():
  raise RuntimeError("1-Step Debug must be completed before 400-Step")
 old_result=load(DEBUG/"result.json")
 if not all(isinstance(old_result.get(k),dict) and old_result[k].get("url")
            for k in ("lora_file","config_file","debug_dataset")):
  raise RuntimeError("1-Step Debug result is incomplete")
 uploaded=load(DEBUG/"uploaded_dataset.json")
 if uploaded.get("sha256")!=EXPECTED_SHA or uploaded.get("bytes")!=ZIP.stat().st_size:
  raise RuntimeError("Existing fal upload digest or size differs")
 url=uploaded.get("url")
 if not isinstance(url,str) or not url.startswith("https://"):
  raise RuntimeError("No verified original uploaded ZIP URL")
 previous=load(DEBUG/"request.json")
 original=previous["request"]
 expected={
  "rank":16,"number_of_steps":1,"learning_rate":.0001,
  "number_of_frames":107,"frame_rate":24,"resolution":"medium",
  "aspect_ratio":"1:1","trigger_phrase":"FOURCAM",
  "reference_conditioning_p":.95,"auto_scale_input":False,
  "split_input_into_scenes":False,"strict_dataset":False,
  "debug_dataset":True,"training_data_url":url}
 if set(original)!=set(expected) or any(original[k]!=v for k,v in expected.items()):
  raise RuntimeError("Actual completed 1-step request differs from verified template")
 if previous.get("dataset_sha256")!=EXPECTED_SHA or previous["endpoint"]!=ENDPOINT:
  raise RuntimeError("Original 1-Step DEBUG dataset/endpoint mismatch")
 request=dict(original)
 request["number_of_steps"]=400
 request["debug_dataset"]=False
 if "resume_from_lora_url" in request:
  raise RuntimeError("400-Step must start from official H3 base, no warm start")
 if not fal.key():raise RuntimeError("FAL_KEY missing")
 return request,{
  "status":"PRE_FLIGHT_PASS",
  "endpoint":ENDPOINT,
  "source_1step_request_id":EXPECTED_REQUEST_ID,
  "source_fal_preprocess_accepted_samples":28,
  "training_zip_sha256":EXPECTED_SHA,
  "reuses_existing_uploaded_zip":True,
  "steps":400,"rank":16,"learning_rate":.0001,
  "trigger_phrase":"FOURCAM","aspect_ratio":"1:1",
  "number_of_frames":107,"frame_rate":24,
  "reference_conditioning_p":.95,
  "debug_dataset":False,"warm_start":False,
  "expected_success_cost_usd":COST_USD,
  "prior_400_submitted":(RUN/"submitted.json").is_file(),
  "prior_uncertain_paid_submit":(RUN/"submission_in_progress.json").is_file()
 }

def status():
 _,info=preflight()
 print("V6_FOURCAM_400STEP_PREFLIGHT",json.dumps(info,indent=2),flush=True)
 if (RUN/"submitted.json").is_file():
  print("EXISTING_400_STEP_REQUEST_ID",load(RUN/"submitted.json")["request_id"],flush=True)
 if (RUN/"last_status.json").is_file():
  print("LAST_STATUS",load(RUN/"last_status.json").get("status"),flush=True)

def submit():
 request,info=preflight()
 RUN.mkdir(parents=True,exist_ok=True)
 accepted=RUN/"submitted.json"
 uncertain=RUN/"submission_in_progress.json"
 staged=RUN/"request.json"
 if accepted.is_file():
  print("ALREADY_SUBMITTED_NO_DUPLICATE",load(accepted)["request_id"],flush=True)
  return
 if uncertain.exists():
  raise RuntimeError("Uncertain previous POST: do not duplicate 400-step paid training")
 if staged.exists() and load(staged).get("request")!=request:
  raise RuntimeError("Existing staged 400-step request changed; no submit")
 save(staged,{"endpoint":ENDPOINT,"request":request,
              "training_dataset_sha256":EXPECTED_SHA,
              "prevalidated_1step_request_id":EXPECTED_REQUEST_ID,
              "expected_billed_steps":400,
              "estimated_billed_usd":COST_USD,
              "authorization":"One 400-step user-approved fal training only"})
 save(uncertain,{"endpoint":ENDPOINT,"request":request,
     "warning":"One paid POST may have been accepted; never retry without reconciliation"})
 print("SENDING_EXACTLY_ONE_PAID_REF2VA_400STEP_TRAINING_REQUEST",flush=True)
 response=requests.post("https://queue.fal.run/"+ENDPOINT,
                        headers=fal.headers(),json=request,timeout=90)
 if response.status_code>=400:
  save(RUN/"submit_error.json",{"status_code":response.status_code,
                                "body":response.text[:1400]})
  if response.status_code in (400,401,403,404,422):
   uncertain.unlink(missing_ok=True)
  raise RuntimeError("fal 400-step rejected/errored HTTP "+str(response.status_code)+": "+response.text[:350])
 doc=response.json()
 if not all(doc.get(k) for k in ("request_id","status_url","response_url")):
  save(RUN/"unexpected_submission_result.json",doc)
  raise RuntimeError("Ambiguous accepted POST response; marker retained")
 save(accepted,{**doc,"endpoint":ENDPOINT})
 uncertain.unlink(missing_ok=True)
 print("V6_28_FOURCAM_400STEP_SUBMITTED",json.dumps({
  "request_id":doc["request_id"],
  "endpoint":ENDPOINT,"steps":400,"estimated_cost_usd":6.00,
  "dataset_samples":28,"trained_from_base":True,
  "new_paid_inference_requests":0,"run_dir":str(RUN)},indent=2),flush=True)

def safetensor_details(path):
 with open(path,"rb") as f:
  header_len=struct.unpack("<Q",f.read(8))[0]
  if header_len<16 or header_len>32000000:
   raise RuntimeError("Downloaded safetensors invalid header")
  head=json.loads(f.read(header_len))
 keys=[k for k in head if k!="__metadata__"]
 if len(keys)<20:
  raise RuntimeError("Downloaded safetensors tensor count implausible")
 return {"sha256":digest(path),"bytes":path.stat().st_size,
         "tensors":len(keys)}

def poll():
 if not (RUN/"submitted.json").is_file():
  raise RuntimeError("Not submitted; poll NEVER starts paid jobs")
 sub=load(RUN/"submitted.json")
 response=requests.get(sub["status_url"],headers=fal.headers(),timeout=60)
 response.raise_for_status()
 status_doc=response.json()
 save(RUN/"last_status.json",status_doc)
 state=status_doc.get("status","UNKNOWN")
 print("V6_FOURCAM_400STEP_STATUS",state,sub["request_id"],flush=True)
 if state!="COMPLETED":
  if state not in ("IN_QUEUE","IN_PROGRESS"):
   print("V6_400STEP_UNEXPECTED_REMOTE_STATUS",json.dumps(status_doc)[:1200],flush=True)
  return False
 resultfile=RUN/"result.json"
 if resultfile.is_file():res=load(resultfile)
 else:
  response=requests.get(sub["response_url"],headers=fal.headers(),timeout=120)
  response.raise_for_status()
  res=response.json()
  save(resultfile,res)
 for key in ("lora_file","config_file"):
  if not isinstance(res.get(key),dict) or not res[key].get("url"):
   raise RuntimeError("Completed fal 400-step is missing "+key)
 modeldir=RUN/"model";modeldir.mkdir(parents=True,exist_ok=True)
 files={}
 for key,name in (("lora_file","human_v6_fourcam_h3_400.safetensors"),
                  ("config_file","config.json")):
  item=res[key]
  dest=modeldir/name
  size=int(item.get("file_size") or 0)
  if not dest.exists() or (size and dest.stat().st_size!=size):
   temp=pathlib.Path(str(dest)+".partial")
   print("DOWNLOADING",key,flush=True)
   fal.download(item["url"],temp)
   if size and temp.stat().st_size!=size:
    raise RuntimeError("Downloaded "+key+" size differs from fal")
   temp.replace(dest)
  files[key]={"local_path":str(dest),"bytes":dest.stat().st_size,
              "sha256":digest(dest)}
 cfg=load(modeldir/"config.json")
 if cfg.get("training_type")!="ref2va" or cfg.get("instance_prompt")!="FOURCAM":
  raise RuntimeError("Unexpected LoRA config: "+str(cfg)[:200])
 tensors=safetensor_details(modeldir/"human_v6_fourcam_h3_400.safetensors")
 report={
  "status":"PASS_400STEP_MODEL_ARTIFACTS",
  "request_id":sub["request_id"],
  "steps":400,"training_source":"H3_BASE","dataset_sha256":EXPECTED_SHA,
  "files":files,"weights":tensors,"config":cfg,
  "no_paid_video_inference":True}
 save(RUN/"MODEL_VALIDATION.json",report)
 print("V6_FOURCAM_400STEP_MODEL_SAVED_AND_VALIDATED",json.dumps(report,indent=2),flush=True)
 return True

if __name__=="__main__":
 mode=sys.argv[1] if len(sys.argv)>1 else "preflight"
 {"preflight":status,"status":status,"submit":submit,"poll":poll}[mode]()
