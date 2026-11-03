"""Blender V6 four directional Reach motion smoke, both anatomical sides.

8 x 4 keyframes only. Inspect generated 55-Bone FK, no H3 training.
"""
import json,pathlib,subprocess
ROOT=pathlib.Path(r"F:\anymove")
HERE=ROOT/"runs"/"human"/"lora_synth_A"
BLENDER=pathlib.Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
SCRIPT=HERE/"render_reach_diverse_v6.py"
V6=HERE/"v6_reach_source"
plan=json.loads((V6/"v2_dataset_plan.json").read_text(encoding="utf8"))["results"]
candidates={}
for entry in plan:
 if entry["split"]=="train" and entry["seed"]==7301:
  candidates[entry["family"]]=entry
assert len(candidates)==8
for n,item in enumerate(candidates.values()):
 folder=HERE/"v6_smoke"/item["id"]
 args=[str(BLENDER),"-b","-t","4","--python-exit-code","17","-P",str(SCRIPT),"--",
       "--out",str(folder),"--family",item["family"],
       "--seed",str(item["seed"]),"--amplitude-deg",str(item["amplitude_deg"]),
       "--fps","24","--frames","107","--samples","8",
       "--views","front,side,back,opposite_side",
       "--render-frames","0,24,53,106"]
 folder.mkdir(parents=True,exist_ok=True)
 with (folder/"blender_smoke_log.txt").open("w",encoding="utf8") as out:
  r=subprocess.run(args,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
 if r.returncode!=0:
  raise RuntimeError("V6 smoke Blender failed "+item["id"]+" log "+str(folder/"blender_smoke_log.txt"))
 stats=json.loads((folder/"status.json").read_text(encoding="utf8"))
 assert stats["status"]=="PASS" and stats["render_frames"]==[0,24,53,106]
 print("V6_REACH_DIRECTION_SMOKE_PASS",n+1,"/8",item["family"],flush=True)
print("V6_ALL_8_REACH_MOTION_SMOKES_PASS",flush=True)
