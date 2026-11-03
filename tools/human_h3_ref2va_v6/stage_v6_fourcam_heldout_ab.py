"""Offline stage four fal H3 Ref2VA V6 FOURCAM heldout A/B requests.

Two distinct heldout action families:
  1) forward reach with anatomical left arm
  2) backward reach with anatomical right arm
Each same endpoint and same input references between LoRA scales 0.0/1.0.
Read-only preparation; any paid submit is done later by generated scripts,
each of which preserves single-POST and anti-duplicate markers.
"""
import hashlib,json,pathlib
ROOT=pathlib.Path(r"F:\anymove")
BASE=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"
PILOT=BASE/"remote"/"h3_ref2va_v6_28_fourview_400step"
SOURCE=BASE/"fourview_v6_reach_augmented"/"clips"/"heldout"
MODEL=PILOT/"model"/"human_v6_fourcam_h3_400.safetensors"
RUNROOT=PILOT/"heldout_ab_v1"
TEMPLATE=BASE/"run_400step_heldout_ab.py"
EXPECTED={
 "forward_left":"full_heldout_reach_forward_left_a55_s8201",
 "backward_right":"full_heldout_reach_backward_right_a52_s8204",
}
def load(p):return json.loads(p.read_text(encoding="utf8"))
def save(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if load(p)!=data:raise RuntimeError("Existing plan differs: "+str(p))
  return
 tmp=p.with_suffix(p.suffix+".tmp")
 tmp.write_text(json.dumps(data,indent=2),encoding="utf8")
 tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def main():
 model=load(PILOT/"MODEL_VALIDATION.json")
 if model["status"]!="PASS_400STEP_MODEL_ARTIFACTS" or model["steps"]!=400 or model["config"]!={"instance_prompt":"FOURCAM","training_type":"ref2va"}:
  raise RuntimeError("400-Step FOURCAM LoRA invalid")
 if sha(MODEL)!=model["weights"]["sha256"] or MODEL.stat().st_size!=model["weights"]["bytes"]:
  raise RuntimeError("Weights changed")
 url=load(PILOT/"result.json")["lora_file"]["url"]
 if not (url.startswith("https://") and model["weights"]["bytes"]==65628056):
  raise RuntimeError("Invalid model artifact")
 prior=TEMPLATE.read_text(encoding="utf8")
 report={"status":"READY_TO_CHECK_PAID_API","train_id":model["request_id"],
         "model_sha256":model["weights"]["sha256"],"experiments":{}}
 for case,stem in EXPECTED.items():
  source=SOURCE/stem
  manifest=load(source/"fourview_clip_manifest.json")
  if manifest["split"]!="heldout" or manifest["stem"]!=stem or manifest["view_names"]!=["left_45","right_45","back","top"] or manifest["canvas"]!=[896,896]:
   raise RuntimeError("Unexpected heldout cam layout "+stem)
  identity=source/"identity_first_frame.png"
  pose=source/"pose_fourview.mp4"
  rgb=source/"rgb_fourview.mp4"
  media=[
    (identity,manifest["first_frame_identity_sha256"]),
    (pose,manifest["pose_fourview_video"]["sha256"]),
    (rgb,manifest["rgb_fourview_video"]["sha256"])
  ]
  if any(not p.is_file() or sha(p)!=hash for p,hash in media):
   raise RuntimeError("Heldout source media mismatch "+stem)
  prompt="FOURCAM "+(source/"caption.txt").read_text(encoding="utf8").strip()
  if prompt.count("FOURCAM")!=1 or not all(s in prompt for s in ("TOP-LEFT","TOP-RIGHT","BOTTOM-LEFT","BOTTOM-RIGHT")):
   raise RuntimeError("Fourcam prompt missing layout")
  if case=="forward_left" and not all(x in prompt for x in ("LEFT","FORWARD","RIGHT blue/green arm stays lowered")):
   raise RuntimeError("Forward action prompt mismatch")
  if case=="backward_right" and not all(x in prompt for x in ("RIGHT","BACKWARD","LEFT red/orange arm stays lowered")):
   raise RuntimeError("Backward action prompt mismatch")
  models={}
  for label,scale in (("scale0",0.),("scale1",1.)):
   models[label]={
     "prompt":prompt,
     "reference_image_urls":["<UPLOAD_SAME_IDENTITY_IMAGE_ONCE>"],
     "reference_video_urls":["<UPLOAD_SAME_POSE_VIDEO_ONCE>"],
     "duration":5,"resolution":"768P","aspect_ratio":"1:1",
     "seed":42,"prompt_expansion_mode":"disabled",
     "loras":[{"path":url,"scale":scale}]
   }
  plan={
    "status":"STAGED_NOT_SUBMITTED",
    "fal_endpoint":"minimax/h3/reference-to-video/lora",
    "pilot_request_id":model["request_id"],
    "training_status":"COMPLETED",
    "candidate_heldout_case":stem,
    "reference_video_local":str(pose),"reference_video_sha256":sha(pose),
    "reference_image_local":str(identity),"reference_image_sha256":sha(identity),
    "source_rgb":str(rgb),"source_rgb_sha256":sha(rgb),
    "view_order":["left_45","right_45","back","top"],
    "layout":[2,2],"canvas":[896,896],"target_ratio":"1:1",
    "matched_inference_requests_if_approved":2,
    "reference_assets_to_upload_once_if_approved":2,
    "requests":models,"paid_video_jobs_submitted":0,
    "important":"Visual references are soft, not hard per-frame pose or camera conditioning."
  }
  dst=RUNROOT/case
  save(dst/"plan.json",plan)
  script=prior
  changes=[
    ('PILOT=BASE/"remote"/"h3_ref2va_pilot_400"','PILOT=BASE/"remote"/"h3_ref2va_v6_28_fourview_400step"'),
    ('PLAN=PILOT/"heldout_side_lean_ab_plan.json"',f'PLAN=PILOT/"heldout_ab_v1"/"{case}"/"plan.json"'),
    ('WEIGHTS=PILOT/"model"/"human_h3_ref2va_400.safetensors"','WEIGHTS=PILOT/"model"/"human_v6_fourcam_h3_400.safetensors"'),
    ('VALIDATION=PILOT/"model"/"checkpoint_validation.json"','VALIDATION=PILOT/"MODEL_VALIDATION.json"'),
    ('OUT=PILOT/"heldout_ab"',f'OUT=PILOT/"heldout_ab_v1"/"{case}"'),
    ('CANDIDATE="full_heldout_lean_left_a21_s9901"',f'CANDIDATE="{stem}"'),
    ('val["status"]!="PASS" or val["steps"]!=400 or val["model"]["training_type"]!="ref2va"',
     'val["status"]!="PASS_400STEP_MODEL_ARTIFACTS" or val["steps"]!=400 or val["config"]["training_type"]!="ref2va"'),
    ('val["weights"]["bytes"]','val["weights"]["bytes"]'),
    ('(768,448)','(896,896)'),
    ('source_profile["width"]!=768 or source_profile["height"]!=448','source_profile["width"]!=896 or source_profile["height"]!=896'),
    ('"TWOCAM"','"FOURCAM"'),
    ('request.get("aspect_ratio")!="16:9"','request.get("aspect_ratio")!="1:1"'),
    ('if v["width"]<960 or v["height"]<500 or v["width"]%2 or v["height"]%2:',
     'if v["width"]<720 or v["height"]<720 or v["width"]%2 or v["height"]%2 or v["width"]!=v["height"]:'),
    ('"heldout_left_lean_400_ref2va"',f'"heldout_v6_fourcam_{case}_400_ref2va"')
  ]
  for old,new in changes:
   if old==new:continue
   n=script.count(old)
   if n!=1:raise RuntimeError(f"expected one replace match for {old!r}, got {n}")
   script=script.replace(old,new)
  compile(script,str(BASE/f"run_v6_fourcam_{case}_ab.py"),"exec")
  scriptpath=BASE/f"run_v6_fourcam_{case}_ab.py"
  if scriptpath.exists() and scriptpath.read_text(encoding="utf8")!=script:
   raise RuntimeError("Existing runner differs: "+str(scriptpath))
  scriptpath.write_text(script,encoding="utf8")
  report["experiments"][case]={"stem":stem,"source_rgb_sha256":sha(rgb),
     "pose_ref_sha256":sha(pose),"identity_sha256":sha(identity),
     "runner":str(scriptpath),"plan":str(dst/"plan.json"),
     "existing_paid_jobs":[(dst/x/"submitted.json").exists() for x in ("scale0","scale1")]}
 # same first frame identity across cases: reuse uploaded URL if exists for
 # either experiment; no need to re-upload identical image reference.
 a=report["experiments"]["forward_left"]["identity_sha256"]
 b=report["experiments"]["backward_right"]["identity_sha256"]
 if a!=b:raise RuntimeError("First-frame identity is not shared between heldout clips")
 report["shared_first_frame_sha256"]=a
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
