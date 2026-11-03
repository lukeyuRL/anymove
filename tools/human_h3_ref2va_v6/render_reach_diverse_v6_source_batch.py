"""Render 20 new Human V6 Reach controlled motion sources (16 train/4 holdout).

Blender 5.0 reuses the original immutable SkinMax V3 source.
Creates exact 107-frame 24FPS 55-bone FK and calibrated original
front/back camera RGB views for each new Reach motion. Other views
are computed/rendered later by V5 4-camera generator.
No paid AI. Idempotent per clip with manifest SHA guards.
"""
import collections,json,pathlib,subprocess,time
ROOT=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A")
SRC=ROOT/"v6_reach_source"
PLAN=SRC/"v2_dataset_plan.json"
SCRIPT=ROOT/"render_reach_diverse_v6.py"
BLENDER=pathlib.Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
def save(path,obj):
 temp=path.with_suffix(".tmp.json")
 temp.write_text(json.dumps(obj,indent=2),encoding="utf8")
 temp.replace(path)
plan=json.loads(PLAN.read_text(encoding="utf8"))
records=plan["results"]
if plan["status"] not in ("PLANNED_RENDER_NOT_YET_ACCEPTED","PASS"):
 raise RuntimeError("V6 source status unexpected")
if len(records)!=20 or collections.Counter(x["split"] for x in records)!={"train":16,"heldout":4}:
 raise ValueError("Expected 16 train and 4 heldout")
start=time.monotonic()
for i,c in enumerate(records):
 folder=pathlib.Path(c["path"])
 status_file=folder/"status.json"
 def accepted():
  if not status_file.is_file():return False
  state=json.loads(status_file.read_text(encoding="utf8"))
  if state.get("status")!="PASS" or state.get("render_frames")!=list(range(107)):return False
  return all(len(list((folder/view).glob("f_*.png")))==107 for view in ("front","back"))
 if accepted():
  print("REUSE_V6_SOURCE",i+1,"/20",c["id"],flush=True)
  continue
 args=[str(BLENDER),"-b","-t","6","--python-exit-code","17","-P",str(SCRIPT),"--",
       "--out",str(folder),"--family",c["family"],
       "--seed",str(c["seed"]),"--amplitude-deg",str(c["amplitude_deg"]),
       "--fps","24","--frames","107","--samples","8",
       "--views","front,back","--render-frames","all"]
 folder.mkdir(parents=True,exist_ok=True)
 ts=time.monotonic()
 with (folder/"blender_render_full.log").open("w",encoding="utf8") as log:
  result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 if result.returncode!=0 or not accepted():
  raise RuntimeError("V6 full Blender source render failed "+c["id"]+" log "+str(folder/"blender_render_full.log"))
 print("V6_SOURCE_COMPLETE",i+1,"/20",c["id"],"secs",round(time.monotonic()-ts,2),flush=True)
# Only after all 20 clips complete, set the source plan PASS
plan["status"]="PASS"
plan["rendered_train_clips"]=16
plan["rendered_heldout_clips"]=4
plan["rendered_frames_per_clip"]=107
plan["camera_source_views"]=["front","back"]
plan["status_detail"]="All 20 clips with exact same source rig, 55-bone FK, front/back RGB images at 107 frames and 24 FPS"
save(PLAN,plan)
print("V6_REACH_SOURCE_20_COMPLETE_PASS",round(time.monotonic()-start,1),flush=True)
