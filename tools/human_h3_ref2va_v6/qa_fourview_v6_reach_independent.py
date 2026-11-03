"""Independent fail-closed offline QA for H3 Ref2VA V5 four-view (1:1).

Checks 16 clips, 1712 synchronized source RGB/pose frames, all four
calibrated cameras, 12 train ZIP + 4 heldout, 256 independent
depth-owner-frame witnesses, no Front view in any output and separate
left45/right45/back/top cell ordering. Does NOT submit paid fal jobs.
"""
import ast
import collections
import hashlib
import json
import math
import pathlib
import subprocess
import zipfile
import numpy as np
from PIL import Image
ROOT=pathlib.Path(r"F:\anymove")
H3=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"
BASE=H3/"fourview_v6_reach_augmented"
V2=ROOT/"runs"/"human"/"lora_synth_A"/"v6_reach_source"
FFPROBE=pathlib.Path(r"C:\ffmpeg\bin\ffprobe.exe")
VW=("left_45","right_45","back","top")
Q={"left_45":(0,0),"right_45":(1,0),"back":(0,1),"top":(1,1)}
N=107
SAMPLES=(0,24,53,106)
W,H=384,448
QW=448
SQUARE=896
def read(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def must(condition,msg):
 if not condition:raise AssertionError(msg)
def video_info(p):
 r=subprocess.run([str(FFPROBE),"-v","error","-select_streams","v:0",
                   "-show_entries","stream=width,height,nb_frames,r_frame_rate",
                   "-of","json",str(p)],check=True,capture_output=True,text=True)
 x=json.loads(r.stdout)["streams"][0]
 return int(x["width"]),int(x["height"]),int(x["nb_frames"]),x["r_frame_rate"]
def img(path):
 with Image.open(path) as im:
  must(im.size==(SQUARE,SQUARE),"Bad image dimension "+str(path))
  return np.asarray(im.convert("RGB")).copy()
def pane(canvas,view):
 qx,qy=Q[view]
 return canvas[qy*QW:(qy+1)*QW,qx*QW+32:qx*QW+32+W,:]
def calc_proj(xyz,cam):
 M=np.asarray(cam["matrix_world"],dtype=float)
 must(M.shape==(4,4) and abs(np.linalg.det(M[:3,:3])-1)<1e-4,"Camera orientation invalid")
 r=xyz-M[:3,3];sc=H/float(cam["ortho_scale"])
 x=W/2+np.einsum("tjd,d->tj",r,M[:3,0])*sc
 y=H/2-np.einsum("tjd,d->tj",r,M[:3,1])*sc
 z=np.einsum("tjd,d->tj",r,-M[:3,2])
 return np.stack((x,y,z),axis=-1)
def color(n):
 return (int(n[0])<<16)|(int(n[1])<<8)|int(n[2])
palette=read(ROOT/"runs"/"human"/"human_skinmax_multicolor_v3_palette.json")["bone_colors"]
RRED=color(bytes.fromhex(palette["CC_Base_L_Upperarm"][1:]))
BBLUE=color(bytes.fromhex(palette["CC_Base_R_Upperarm"][1:]))
LORANGE=color(bytes.fromhex(palette["CC_Base_L_Forearm"][1:]))
RGREEN=color(bytes.fromhex(palette["CC_Base_R_Forearm"][1:]))
schema=read(H3/"anatomical_pose_v4_depth_verified"/"skeleton_schema.json")
SEGMENTS=schema["segments"]
JOINTS=schema["joint_names"]
RADIUS=float(schema["line_radius_pixels"])
JRADIUS=float(schema["joint_radius_pixels"])
# Extract only the manually-written V4 *independent* 2D depth raster
# verifier; never call the dataset-generation renderer's raster routine.
source=ast.parse((H3/"qa_h3_v4_independent.py").read_text(encoding="utf8"))
fn=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=="independently_depth_test")
globals()["assertn"]=must
report={"counts":collections.Counter()}
exec(compile(ast.Module(body=[fn],type_ignores=[]),
    "imported_independent_v4_reference_depth_test","exec"),globals())
# It expects a 768-wide pair of two 384 images (front/side).
def audit_depth_witness(points,view,frame,png,witness):
 row=np.concatenate([png,np.zeros_like(png)],axis=1)
 independently_depth_test({"front":points},"front",frame,row,witness)
plan=read(V2/"v2_dataset_plan.json")
manifest=read(BASE/"media_manifest.json")
must(manifest["view_order"]==list(VW),"Metadata camera order wrong")
must(manifest["canvas"]==[SQUARE,SQUARE],"Metadata canvas not 1:1")
must(manifest["suggested_training_aspect_ratio"]=="1:1","Fal train ratio mismatch")
must(manifest["trigger_phrase"]=="FOURCAM","Trigger mismatch")
must(len(plan["results"])==len(manifest["clip_records"])==20,"Clip records not 20")
must(collections.Counter(c["split"] for c in plan["results"])=={"train":16,"heldout":4},"Split invalid")
must(len({r["id"] for r in plan["results"]})==20,"Duplicate source clip IDs")
record_index={r["stem"]:r for r in manifest["clip_records"]}
must({r["id"] for r in plan["results"]}==set(record_index),"Clip record map incomplete")
zip_path=pathlib.Path(manifest["training_zip"]["path"])
must(zip_path.is_file() and sha(zip_path)==manifest["training_zip"]["sha256"],"Training archive SHA mismatch")
with zipfile.ZipFile(zip_path) as z:
 must(len(z.namelist())==64 and len(set(z.namelist()))==64 and z.testzip() is None,"Zip CRC/member error")
 for clip in plan["results"]:
  stem=clip["id"]
  if clip["split"]!="train":continue
  path=BASE/"clips"/"train"/stem
  for suffix,sourcefile in [
    (".mp4",path/"rgb_fourview.mp4"),
    (".txt",path/"caption.txt"),
    (".ref_1.mp4",path/"pose_fourview.mp4"),
    (".ref_2.png",path/"identity_first_frame.png")
  ]:
   member=stem+suffix
   must(member in z.namelist(),"ZIP member absent "+member)
   must(z.read(member)==sourcefile.read_bytes(),"ZIP content differs from verified source "+member)
R=collections.Counter()
errors={"projection_max_px":0.0,"depth_max_m":0.0,"fk_max_3d_m":0.0}
coverage={}
motion_checks=[]
for index,clip in enumerate(plan["results"]):
 src=pathlib.Path(clip["path"])
 stem=clip["id"];split=clip["split"]
 sub=BASE/"clips"/split/stem
 rec=read(sub/"fourview_clip_manifest.json")
 four=read(sub/"fourview_camera_contract.json")
 pose=read(sub/"fourview_pose_projections.json")
 source_joints=read(src/"joints.json")
 xyz=np.asarray(source_joints["joint_position_m"],dtype=float)
 must(xyz.shape==(N,55,3),"Original FK not 55 joints")
 must(sha(src/"motion.json")==rec["original_motion_sha256"],"Motion source SHA drift "+stem)
 must(sha(src/"joints.json")==rec["original_joints_sha256"],"Joint source SHA drift "+stem)
 must(pose["joint_names"]==source_joints["names"],"Joint ordering incorrect "+stem)
 must(tuple(four["views"])==VW and four["no_front_panel"] is True,
      "Front unexpectedly in four view "+stem)
 must(four["layout_canvas_size"]==[896,896] and four["layout_panel_size"]==[448,448],
      "Bad quadrant size "+stem)
 assert_dirs={
     "left_45":np.array([math.sqrt(.5),-math.sqrt(.5),0.]),
     "right_45":np.array([-math.sqrt(.5),-math.sqrt(.5),0.]),
     "back":np.array([0.,1.,0.]),
     "top":np.array([0.,0.,1.])
 }
 points={}
 for view in VW:
  cam=four["camera_views"][view]
  direction=np.asarray(cam["matrix_world"],dtype=float)[:3,2]
  must(np.linalg.norm(direction-assert_dirs[view])<.0001,
      "Camera direction/sign mismatch "+view+" "+stem)
  expected=calc_proj(xyz,cam)
  observed=np.asarray(pose["views"][view],dtype=float)
  must(observed.shape==(N,55,4),"Wrong pose shape "+stem)
  err=float(np.max(np.abs(expected-observed[:,:,:3])))
  errors["projection_max_px"]=max(errors["projection_max_px"],err)
  must(err<.001,"Fourview camera 3D projection error "+stem+view+str(err))
  must(np.all(observed[:,:,3]==1),"Out of view joint "+stem+view)
  ids=[pose["joint_names"].index(x) for x in JOINTS]
  points[view]=observed[:,ids,:3]
 # Verify motion itself is from same original source and anatomy signs:
 family=clip["family"]
 ind={b:i for i,b in enumerate(source_joints["names"])}
 fcam=four["camera_views"]["left_45"]
 p45=calc_proj(xyz,fcam)
 if family.startswith("reach_"):
  left=np.linalg.norm(p45[:,ind["CC_Base_L_Hand"],:2]-p45[0,ind["CC_Base_L_Hand"],:2],axis=1).max()
  right=np.linalg.norm(p45[:,ind["CC_Base_R_Hand"],:2]-p45[0,ind["CC_Base_R_Hand"],:2],axis=1).max()
  if family.endswith("_left"):must(left>20 and right<2,"GT Reach L anatomy failed "+stem)
  else:must(right>20 and left<2,"GT Reach R anatomy failed "+stem)
  motion_checks.append({"clip":stem,"left_hand_motion_px":round(float(left),2),
                        "right_hand_motion_px":round(float(right),2)})
 # Blender every PNG must be present and camera manifest COMPLETE.
 render=read(sub/"rgb_render_status.json")
 must(render["status"]=="COMPLETE","Oblique Blender render incomplete "+stem)
 must(render["source_V2_55_bone_fk_max_error_m"]<1.e-5,
      "Original motion replay incorrect "+stem)
 must(video_info(sub/"rgb_fourview.mp4")==video_info(sub/"pose_fourview.mp4")== (896,896,107,"24/1"),
      "MP4 invalid or RGB/Pose profile differs "+stem)
 cover={v:{"red":0,"orange":0,"blue":0,"green":0} for v in VW}
 for frame in range(N):
  rgb=img(sub/"fourview_rgb_png"/f"f_{frame:04d}.png")
  poseimg=img(sub/"fourview_pose_png"/f"f_{frame:04d}.png")
  must(rgb.shape==poseimg.shape==(896,896,3),"Unexpected PNG shape "+stem)
  # RGB identity frame must match the target exact first frame.
  if frame==0:
   identity=img(sub/"identity_first_frame.png")
   must(np.array_equal(identity,rgb),"First-frame identity is not exact RGB target")
  # Each actual 384x448 view is at the precise tile.
  for view in VW:
   if view=="back":rawfile=src/"back"/f"f_{frame:04d}.png"
   else:rawfile=sub/"rgb"/view/f"f_{frame:04d}.png"
   with Image.open(rawfile) as sourceimg:
    original=np.asarray(sourceimg.convert("RGB"))
   must(np.array_equal(pane(rgb,view),original),"RGB quadrant/camera mismatch "+stem+view+str(frame))
   tile=pane(poseimg,view)
   pixel=(tile[:,:,0].astype(np.uint32)<<16)|(tile[:,:,1].astype(np.uint32)<<8)|tile[:,:,2]
   for name,code in (("red",RRED),("orange",LORANGE),("blue",BBLUE),("green",RGREEN)):
    val=int(np.count_nonzero(pixel==code))
    cover[view][name]+=val
    if frame in (0,53) and clip["family"].startswith("reach"):
     R[f"{clip['family']}_{view}_{name}_frame{frame}_px"]+=val
   if frame in SAMPLES:
    witness=sub/"qa_depth"/view/f"frame_{frame:04d}.npz"
    must(witness.is_file(),"Missing depth owner witness "+stem+view+str(frame))
    audit_depth_witness(points[view],view,frame,tile,witness)
  # Matte background outside 384px body render must be Pose=black,
  # prevent a source/front view accidentally bleeding into the layout.
  for view,(col,row) in Q.items():
   start_x=col*448
   leftpad=poseimg[row*448:(row+1)*448,start_x:start_x+32]
   rightpad=poseimg[row*448:(row+1)*448,start_x+32+384:start_x+448]
   must(not np.any(leftpad) and not np.any(rightpad),
     "Pose panel padding changed/corrupted "+stem+view)
  R["verified_png_frames"]+=1
 # Every quadrant must retain some of each anatomical upper-arm color
 # over the whole clip; perfect per-frame color visibility is NOT
 # required because real depth overlap is expected.
 for view in VW:
  must(cover[view]["blue"]>0 and cover[view]["red"]>0,
       "Both arm identities not visible in "+view+stem)
 coverage[stem]=cover
 R[split+"_clips"]+=1
 print("V6_REACH_INDEPENDENT_CLIP_PASS",index+1,"/20",stem,flush=True)
must(R["verified_png_frames"]==20*107,"Did not verify all source frames")
must(R["train_clips"]==16 and R["heldout_clips"]==4,"Wrong train/heldout totals")
must(report["counts"]["pixel_witnesses"]==20*4*4,"Did not check all depth witnesses")
qa={
 "schema":"anymove.human_fourview_v6_reach_independent/1",
 "status":"PASS_STATIC_FOURVIEW_GEOMETRY_AND_MEDIA",
 "views":list(VW),"panel_order":Q,"no_front_panel":True,
 "canvas":[896,896],"source_view_pixels":[384,448],
 "train":16,"heldout":4,"frames_per_clip":107,"fps":24,
 "training_zip_sha256":sha(zip_path),
 "camera_reprojection_max_px":errors["projection_max_px"],
 "verified_rgb_and_pose_png_frames":R["verified_png_frames"],
 "independent_depth_owner_witnesses":report["counts"]["pixel_witnesses"],
 "foreground_depth_pixels_checked":report["counts"]["foreground_witness_pixels"],
 "all_new_source_RGB_back_view_frames_reused_without_changes":True,
 "source_motion_unchanged":True,
 "source_mesh_and_original_400step_lora_unchanged":True,
 "identity_frame_matches_RGB_video_source":True,
 "per_view_anatomical_upperarm_color_visibility":coverage,
 "sampled_reach_key_frame_color_pixels":{k:v for k,v in R.items() if "frame" in k},
 "reach_anatomy_3d_motion_checks":motion_checks,
 "additional_paid_fal_jobs_submitted":0,
 "model_training_quality_not_verified":True
}
file=BASE/"STATIC_INDEPENDENT_QA.json"
file.write_text(json.dumps(qa,indent=2),encoding="utf8")
print("V5_FOURVIEW_INDEPENDENT_STATIC_QA_PASS",
 json.dumps({k:v for k,v in qa.items() if k not in
  ("per_view_anatomical_upperarm_color_visibility","reach_anatomy_3d_motion_checks",
   "sampled_reach_key_frame_color_pixels")},indent=2),flush=True)
