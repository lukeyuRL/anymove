"""Human H3 native Ref2VA — V5 four-view dataset media builder.

FOUR views only: TOP-LEFT front-left 45, TOP-RIGHT front-right 45,
BOTTOM-LEFT back 180, BOTTOM-RIGHT overhead true top.
No front panel. 896x896 square 2x2 canvas; 448x448 tile.
Each original 384x448 camera image is pasted at x+32, y+0
without geometric resizing. Four cameras share same exact V2 55-bone FK.

RGB: existing V2 back PNG + Blender 45/45/top RGB PNGs.
Pose: 19 anatomically color-mapped wireframe edges with independent
per-pixel 64bit depth buffer reused from V4, same original 3D motion.
New reference identity image is EXACT first RGB composite frame.
ZIP flat: 12 x RGB MP4, 12 x pose ref_1 MP4, 12 x image ref_2,
12 x captions, no paid fal actions.

Use --only <clip-id> --smoke to generate only frames 0/53, without ZIP.
Full run packages 12 train and 4 heldout output video pairs.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import zipfile
import numpy as np
from PIL import Image

ROOT=pathlib.Path(r"F:\anymove")
H3=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"
BASE=H3/"fourview_v6_reach_augmented"
V2=ROOT/"runs"/"human"/"lora_synth_A"/"v6_reach_source"
BLD=H3/"build_h3_depth_zbuffer_v4.py"
FFMPEG=pathlib.Path(r"C:\ffmpeg\bin\ffmpeg.exe")
FFPROBE=pathlib.Path(r"C:\ffmpeg\bin\ffprobe.exe")
VIEW_NAMES=("left_45","right_45","back","top")
POS={"left_45":(0,0),"right_45":(1,0),"back":(0,1),"top":(1,1)}
N=107
FPS=24
PANEL_W=PANEL_H=448
IMW,IMH=384,448
CANVAS=(896,896)
BG_DEFAULT=(158,158,158)
def load(path):return json.loads(pathlib.Path(path).read_text(encoding="utf8"))
def save(path,data):
 path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=pathlib.Path(str(path)+".temp")
 tmp.write_text(json.dumps(data,indent=2),encoding="utf8")
 tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):
   h.update(b)
 return h.hexdigest()
def profile(path):
 res=subprocess.run([str(FFPROBE),"-v","error","-select_streams","v:0",
                     "-show_entries","stream=width,height,nb_frames,r_frame_rate",
                     "-of","json",str(path)],capture_output=True,text=True,check=True)
 s=json.loads(res.stdout)["streams"][0]
 got=(int(s["width"]),int(s["height"]),int(s["nb_frames"]),s["r_frame_rate"])
 if got!=(896,896,107,"24/1"):
  raise ValueError("Fourview H264 contract failed "+str(path)+str(got))
 return {"sha256":sha(path),"bytes":path.stat().st_size,"width":896,"height":896,"fps":FPS,"frames":107}
def encode(directory,path):
 if path.exists():return profile(path)
 temp=pathlib.Path(str(path)+".part.mp4")
 try:
  cmd=[str(FFMPEG),"-hide_banner","-loglevel","error","-y",
       "-framerate","24","-start_number","0","-i",str(directory/"f_%04d.png"),
       "-frames:v","107","-c:v","libx264","-crf","16","-pix_fmt","yuv420p",
       "-r","24",str(temp)]
  subprocess.run(cmd,check=True)
  profile(temp)
  temp.replace(path)
  return profile(path)
 finally:temp.unlink(missing_ok=True)
spec=importlib.util.spec_from_file_location("h3v4_zbuffer",BLD)
v4=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)
assert len(v4.SEGMENTS)==19 and v4.W==384 and v4.H==448
JOINT_NAMES=tuple(v4.ALL_BONE_NODES)
PALETTE=v4.COLORS
LEFT_RED=tuple(PALETTE[9])
LEFT_ORANGE=tuple(PALETTE[10])
RIGHT_BLUE=tuple(PALETTE[6])
RIGHT_GREEN=tuple(PALETTE[7])
MOTIONS={
 "lean_left":"The human smoothly bends the upper torso toward the actor's anatomical LEFT, then returns upright. No forward bow.",
 "lean_right":"The human smoothly bends the upper torso toward the actor's anatomical RIGHT, then returns upright. No forward bow.",
 "lean_alternating":"The human bends to the actor's anatomical LEFT, returns to neutral, then to the actor's anatomical RIGHT and back to neutral. No forward bow.",
 "reach_left":"Only the actor's anatomical LEFT arm (RED upperarm, ORANGE forearm) raises laterally and extends, while the anatomical RIGHT arm (BLUE upperarm, GREEN forearm) remains down; then the left arm lowers.",
 "reach_right":"Only the actor's anatomical RIGHT arm (BLUE upperarm, GREEN forearm) raises laterally and extends, while the anatomical LEFT arm (RED upperarm, ORANGE forearm) remains down; then the right arm lowers.",
 "reach_forward_left":"Only the anatomical LEFT red/orange arm reaches FORWARD toward the person's front, then returns to rest; RIGHT blue/green arm stays lowered.",
 "reach_forward_right":"Only the anatomical RIGHT blue/green arm reaches FORWARD toward the person's front, then returns to rest; LEFT red/orange arm stays lowered.",
 "reach_diagonal_left":"Only the anatomical LEFT red/orange arm extends FORWARD and OUTWARD diagonally, then lowers; RIGHT blue/green arm stays lowered.",
 "reach_diagonal_right":"Only the anatomical RIGHT blue/green arm extends FORWARD and OUTWARD diagonally, then lowers; LEFT red/orange arm stays lowered.",
 "reach_high_left":"Only the anatomical LEFT red/orange arm rises HIGH and outward above the shoulder then lowers; RIGHT blue/green arm stays lowered.",
 "reach_high_right":"Only the anatomical RIGHT blue/green arm rises HIGH and outward above the shoulder then lowers; LEFT red/orange arm stays lowered.",
 "reach_backward_left":"Only the anatomical LEFT red/orange arm reaches BACKWARD behind the torso then returns to neutral; RIGHT blue/green arm stays lowered.",
 "reach_backward_right":"Only the anatomical RIGHT blue/green arm reaches BACKWARD behind the torso then returns to neutral; LEFT red/orange arm stays lowered."}
CAPTION_COMMON=(
 "Four synchronized fixed ORTHOGRAPHIC cameras show the SAME multicolored human "
 "performing the SAME 3D motion at exactly the SAME time in a 2x2 square grid. "
 "TOP-LEFT: front-left 45 degree oblique view. "
 "TOP-RIGHT: front-right 45 degree oblique view. "
 "BOTTOM-LEFT: back view. BOTTOM-RIGHT: true overhead TOP view looking down on the head and arms. "
 "There is NO front panel. Camera views do not move or swap. "
 "Human anatomical LEFT upperarm RED, LEFT forearm ORANGE, "
 "RIGHT upperarm BLUE, RIGHT forearm GREEN; limb colors never switch. "
 "Image 1 is the exact appearance and four-view layout reference. "
 "Video 1 is a colored anatomical skeleton reference for poses; "
 "render the colored 3D HUMAN, not the skeleton drawing. ")
def caption(family):
 if family not in MOTIONS:raise ValueError("Action family missing "+family)
 return CAPTION_COMMON+MOTIONS[family]
def canvas_with_four(rgb_data,background=(0,0,0)):
 if set(rgb_data)!=set(VIEW_NAMES):
  raise ValueError("Missing/extra camera view in fourview layout "+str(rgb_data.keys()))
 page=Image.new("RGB",CANVAS,background)
 for view in VIEW_NAMES:
  image=rgb_data[view]
  if not isinstance(image,Image.Image) or image.size!=(IMW,IMH):
   raise ValueError("Camera image not 384x448 "+view)
  tx,ty=POS[view]
  page.paste(image,(tx*448+32,ty*448))
 return page
def load_source_rgb(src,frame,view,target):
 if view=="back":
  path=src/"back"/f"f_{frame:04d}.png"
 else:
  path=target/"rgb"/view/f"f_{frame:04d}.png"
 if not path.is_file():
  raise FileNotFoundError("Missing V5 RGB camera render "+str(path))
 with Image.open(path) as image:
  if image.size!=(IMW,IMH):
   raise ValueError("Bad Blender PNG size "+str(path))
  return image.convert("RGB")
def get_camera_points(target,src):
 pose=load(target/"fourview_pose_projections.json")
 original=load(src/"joints.json")
 if pose["frames"]!=N or pose["fps"]!=FPS or pose["joint_names"]!=original["names"]:
  raise ValueError("V5 projected Joint ID or fps mismatch")
 all_points={}
 ids=np.array([pose["joint_names"].index(name) for name in JOINT_NAMES],dtype=int)
 for view in VIEW_NAMES:
  pts=np.asarray(pose["views"][view],dtype=np.float64)
  if pts.shape!=(N,55,4) or not np.isfinite(pts).all():
   raise ValueError("Invalid camera 4-vector "+view)
  if not np.all(pts[:,:,3]>0):
   raise ValueError("H3 pose outside viewport "+view)
  all_points[view]=pts[:,ids,:3]
 return all_points
def prepare_case(entry,smoke):
 stem=entry["id"]
 split=entry["split"]
 src=pathlib.Path(entry["path"])
 target=BASE/"clips"/split/stem
 conf=load(target/"fourview_camera_contract.json")
 if conf["views"]!=list(VIEW_NAMES) or conf["layout_canvas_size"]!=list(CANVAS):
  raise ValueError("V5 view order/canvas contract differs "+stem)
 if conf["panel_positions"]!={x:list(POS[x]) for x in VIEW_NAMES}:
  raise ValueError("V5 panel order not canonical")
 joints=get_camera_points(target,src)
 frames=[0,53] if smoke else list(range(N))
 colored_counts=collections.Counter()
 for frame in frames:
  out_rgb=target/"fourview_rgb_png"/f"f_{frame:04d}.png"
  out_pose=target/"fourview_pose_png"/f"f_{frame:04d}.png"
  images={view:load_source_rgb(src,frame,view,target) for view in VIEW_NAMES}
  background=images["back"].getpixel((0,0))
  rgb=canvas_with_four(images,background=background)
  pose_views={}
  for view in VIEW_NAMES:
   rendered,depth,owner=v4.depth_panel(joints[view][frame])
   if rendered.shape!=(IMH,IMW,3):
    raise ValueError("Wrong per-camera wireframe dimensions")
   pose_views[view]=Image.fromarray(rendered)
   if frame in (0,24,53,106):
    witness=target/"qa_depth"/view/f"frame_{frame:04d}.npz"
    witness.parent.mkdir(parents=True,exist_ok=True)
    if not witness.exists():
     np.savez_compressed(witness,depth_m=depth,owner_id=owner)
   for key,code in (("red_left_upperarm",LEFT_RED),("orange_left_forearm",LEFT_ORANGE),
                     ("blue_right_upperarm",RIGHT_BLUE),("green_right_forearm",RIGHT_GREEN)):
    visible=int(np.count_nonzero(np.all(rendered==code,axis=2)))
    colored_counts[f"{view}_{key}_pixel_sum"]+=visible
    if frame==53:colored_counts[f"{view}_{key}_peak_pixels"]=visible
  pose=canvas_with_four(pose_views)
  for path,image in ((out_rgb,rgb),(out_pose,pose)):
   path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():
    with Image.open(path) as prev:
     if prev.size!=CANVAS or np.any(np.asarray(prev.convert("RGB"))!=np.asarray(image)):
      raise ValueError("Output V5 frame already exists with different pixels "+str(path))
   else:image.save(path)
  if frame==0:
   identity=target/"identity_first_frame.png"
   if identity.exists():
    with Image.open(identity) as prev:
     if prev.size!=CANVAS or np.any(np.asarray(prev.convert("RGB"))!=np.asarray(rgb)):
      raise ValueError("Existing V5 exact identity image changed")
   else:rgb.save(identity,optimize=True)
 if smoke:
  return {"clip":stem,"split":split,"stage":"SMOKE_TWO_FRAMES",
          "visibility_samples":dict(colored_counts),
          "first_frame":str(target/"fourview_rgb_png"/"f_0000.png"),
          "peak_frame":str(target/"fourview_rgb_png"/"f_0053.png")}
 rgb_info=encode(target/"fourview_rgb_png",target/"rgb_fourview.mp4")
 pose_info=encode(target/"fourview_pose_png",target/"pose_fourview.mp4")
 text=caption(entry["family"])
 (target/"caption.txt").write_bytes((text+"\n").encode("utf8"))
 details={
  "schema":"anymove.h3_fourview_v6_reach_clip/1",
  "stage":"ENCODED_AWAIT_INDEPENDENT_QA",
  "stem":stem,"split":split,"family":entry["family"],
  "view_names":list(VIEW_NAMES),
  "canvas":[896,896],"panel_canvas":[448,448],
  "source_crop":[384,448],"source_paste_offset":[32,0],
  "original_motion_sha256":sha(src/"motion.json"),
  "original_joints_sha256":sha(src/"joints.json"),
  "camera_contract_sha256":sha(target/"fourview_camera_contract.json"),
  "pose_projection_sha256":sha(target/"fourview_pose_projections.json"),
  "rgb_fourview_video":rgb_info,
  "pose_fourview_video":pose_info,
  "first_frame_identity_sha256":sha(target/"identity_first_frame.png"),
  "train_caption":text,
  "visibility_counts":dict(colored_counts),
  "new_fal_paid_jobs_submitted":0}
 save(target/"fourview_clip_manifest.json",details)
 return details
def package(rows):
 train={r["stem"]:r for r in rows if r["split"]=="train"}
 if len(train)!=16:
  raise ValueError("Expected exactly 16 train Reach clips")
 archive=BASE/"fal_h3_ref2va_v6_reach_aug_16_train.zip"
 if archive.exists():
  raise FileExistsError("New fourview ZIP already exists, refusing to overwrite: "+str(archive))
 tmp=pathlib.Path(str(archive)+".temp")
 try:
  with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as out:
   for stem in sorted(train):
    t=BASE/"clips"/"train"/stem
    if len(train[stem]["train_caption"])<300:
     raise ValueError("New caption lacks fourview anatomy specifics")
    out.write(t/"rgb_fourview.mp4",arcname=stem+".mp4")
    out.writestr(stem+".txt",train[stem]["train_caption"]+"\n")
    out.write(t/"pose_fourview.mp4",arcname=stem+".ref_1.mp4")
    out.write(t/"identity_first_frame.png",arcname=stem+".ref_2.png")
  with zipfile.ZipFile(tmp) as check:
   if len(check.namelist())!=64 or check.testzip() is not None:
    raise ValueError("Invalid fal fourview train archive")
   if len(set(check.namelist()))!=64:raise ValueError("Duplicate members in ZIP")
  tmp.replace(archive)
 finally:
  tmp.unlink(missing_ok=True)
 return {"path":str(archive),"sha256":sha(archive),"bytes":archive.stat().st_size,
         "samples":16,"entries":64,
         "structure":"16 RGB target MP4 + 16 TXT captions + 16 Pose reference MP4 + 16 firstframe identity PNG"}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--only",default="")
 ap.add_argument("--smoke",action="store_true")
 args=ap.parse_args()
 if args.smoke and not args.only:
  raise ValueError("Smoke mode requires exact clip name")
 plan=load(V2/"v2_dataset_plan.json")
 entries=plan["results"]
 if len(entries)!=20 or collections.Counter(r["split"] for r in entries)!={"train":16,"heldout":4}:
  raise ValueError("V2 base split invalid")
 cases=[c for c in entries if not args.only or c["id"]==args.only]
 if not cases:raise ValueError("No matching clip")
 rows=[]
 for i,entry in enumerate(cases):
  result=prepare_case(entry,args.smoke)
  rows.append(result)
  print("V6_REACH_FOURVIEW_MEDIA",i+1,"/",len(cases),entry["id"],
        "SMOKE" if args.smoke else "ENCODED",flush=True)
 if args.smoke:
  save(BASE/"smoke_media_result.json",{"status":"SMOKE_PASS",
    "candidate":rows[0],"paid_jobs":0})
  print("FOURVIEW_SMOKE_MEDIA_PASS",flush=True)
  return
 if args.only:
  print("V5_SINGLE_CLIP_READY_NO_ZIP",flush=True);return
 package_info=package(rows)
 out={
  "schema":"anymove.human_h3_ref2va_diverse_reach_v6_media/1",
  "status":"MEDIA_PACKAGED_INDEPENDENT_QA_REQUIRED",
  "view_order":list(VIEW_NAMES),
  "quadrants":{k:list(v) for k,v in POS.items()},
  "canvas":[896,896],
  "fal_native_trainer":"minimax/h3/ref2va/trainer",
  "fal_native_inference":"minimax/h3/reference-to-video/lora",
  "suggested_training_aspect_ratio":"1:1",
  "suggested_resolution":"medium",
  "number_of_frames":107,
  "frame_rate":24,
  "trigger_phrase":"FOURCAM",
  "clip_records":rows,
  "training_zip":package_info,
  "source_model_unchanged":True,
  "existing_400step_lora_unchanged":True,
  "new_paid_training_or_video_jobs":0}
 save(BASE/"media_manifest.json",out)
 print("V6_REACH_FOURVIEW_FAL_TRAINING_ZIP_CREATED_AWAIT_QA",
  json.dumps(package_info,indent=2),flush=True)
if __name__=="__main__":main()
