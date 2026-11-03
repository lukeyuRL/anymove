"""Batch background Blender renderer for existing V2 -> H3 V5 four obliques.

Locally render only the two missing 45-degree images at 384x448 for
each of 16 original sources. Original front/back are reused unmodified.
No paid fal jobs and no source .blend modifications.
Fail fast if Blender returns error, idempotent clip completion records.
"""
import json,pathlib,subprocess,time
ROOT=pathlib.Path(r"F:\anymove")
HERE=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"
OUT=HERE/"fourview_v6_reach_augmented"
V2=ROOT/"runs"/"human"/"lora_synth_A"/"v6_reach_source"
BLENDER=pathlib.Path(r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
SCRIPT=HERE/"render_fourview_v6_reach_blender.py"
plan=json.loads((V2/"v2_dataset_plan.json").read_text(encoding="utf8"))
cases=plan["results"]
assert len(cases)==20 and BLENDER.is_file()
start=time.monotonic()
for i,item in enumerate(cases):
 stem=item["id"]
 sub=OUT/"clips"/item["split"]/stem
 result=sub/"rgb_render_status.json"
 if result.exists():
  a=json.loads(result.read_text(encoding="utf8"))
  if a.get("status")=="COMPLETE" and all(len(list((sub/"rgb"/v).glob("f_*.png")))==107
      for v in ("left_45","right_45","top")):
   print("REUSE_ALREADY_RENDERED",i+1,"/16",stem,flush=True)
   continue
 print("V5_START_BLENDER_RGB",i+1,"/16",stem,flush=True)
 args=[str(BLENDER),"-b","-t","6","--python-exit-code","17","-P",str(SCRIPT),
       "--","--clip",stem,"--render-frames","all"]
 sub.mkdir(parents=True,exist_ok=True)
 with (sub/"blender_render.log").open("w",encoding="utf8") as logfile:
  t=time.monotonic()
  process=subprocess.run(args,stdout=logfile,stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
 if process.returncode!=0:
  print("V5_BLENDER_RENDER_FAILED",stem,"exit",process.returncode,
        "see",sub/"blender_render.log",flush=True)
  raise RuntimeError("Failed Blender render for "+stem)
 if not result.exists():raise RuntimeError("Missing Blender RGB completion evidence "+stem)
 b=json.loads(result.read_text(encoding="utf8"))
 if b.get("status")!="COMPLETE" or b["source_V2_55_bone_fk_max_error_m"]>5e-5:
  raise RuntimeError("Source 3D FK replay invalid "+stem)
 print("V5_COMPLETED_BLENDER_RGB",i+1,"/16",stem,
       "runtime_sec",round(time.monotonic()-t,1),flush=True)
print("V5_ALL_16_RGB_OBLIQUES_RENDERED","elapsed_s",round(time.monotonic()-start,1),flush=True)
