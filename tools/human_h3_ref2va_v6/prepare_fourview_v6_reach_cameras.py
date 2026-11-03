"""AnyMove Human V5 four-view calibrated camera projection prep.

Defines reproducible fixed 2x2 four-view layout:
TL Front (0deg), TR Front-Right 45deg (camera world -X,-Y),
BL Back (180deg), BR Front-Left 45deg (+X,-Y).

Source V2 front/back extrinsics preserved; oblique fixed cameras derived
from source front camera + rest bounds; no mirroring, no coordinate cheats.
Each panel has 384x448 orthographic pixel source (same per-view detail
as V2) centered into 800x450 tile of a 1600x900 16:9 canvas.

This file is a SOURCE DATA preparer only, runs via Windows Python 3.14.
No Blender invocation, no model download, no fal API call, no paid work.
"""
import argparse,collections,hashlib,json,math,pathlib,sys
import numpy as np
ROOT=pathlib.Path(r"F:\anymove")
V2=ROOT/"runs"/"human"/"lora_synth_A"/"v6_reach_source"
HERE=ROOT/"runs"/"human"/"lora_synth_A"/"conditioned_backend"/"fal_h3_ref2va"/"fourview_v6_reach_augmented"
VIEWS=("left_45","right_45","back","top")
ANGLES=(-45,45,180,"OVERHEAD") # -45 = camera world +X (actor-left), +45 = world -X (actor-right)
TILE_POS={
    "left_45":[0,0],
    "right_45":[1,0],
    "back":[0,1],
    "top":[1,1]}
SOURCE_SIZE=(384,448)
CANVAS_SIZE=(896,896)
TILE_SIZE=(448,448)
SPLIT_TRAIN=12
SPLIT_HELDOUT=4
MAX_PROJECTION_ERROR=0.002
def read(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def write(path,obj):
 path=pathlib.Path(path)
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=pathlib.Path(str(path)+".partial")
 tmp.write_text(json.dumps(obj,indent=2),encoding="utf8")
 tmp.replace(path)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
 return h.hexdigest()
def camera_for_front_oblique(original,angle_deg):
 front=original["views"]["front"]
 cx=np.asarray(original["rest_bounds"]["center_xyz"],dtype=np.float64)
 fM=np.asarray(front["matrix_world"],dtype=np.float64)
 radius=float(np.linalg.norm(fM[:3,3]-cx))
 rad=math.radians(angle_deg)
 direction=np.array([-math.sin(rad),-math.cos(rad),0.0],dtype=float)
 right=np.array([-direction[1],direction[0],0.0],dtype=float)
 up=np.array([0.0,0.0,1.0],dtype=float)
 M=np.eye(4,dtype=float)
 M[:3,:3]=np.stack([right,up,direction],axis=1)
 M[:3,3]=cx+radius*direction
 if abs(float(np.linalg.det(M[:3,:3]))-1)>1.e-10:
  raise ValueError("New camera orientation not rigid right-handed")
 ret={**front,"matrix_world":M.tolist(),"camera_axis_order":"pixels x right, y down; no rescale",
      "angle_from_front_deg":angle_deg,"camera_direction_world":direction.tolist()}
 return ret
def camera_overhead(original):
 # TOP: camera looking straight down on world XY plane.
 # screen-right=+world X (actor-left), screen-up=+world Y.
 # This camera is right-handed: right X, up Y, camera-back Z.
 front=original["views"]["front"]
 cx=np.asarray(original["rest_bounds"]["center_xyz"],dtype=np.float64)
 fM=np.asarray(front["matrix_world"],dtype=np.float64)
 radius=float(np.linalg.norm(fM[:3,3]-cx))
 M=np.eye(4,dtype=float)
 M[:3,3]=cx+radius*np.array([0.,0.,1.],dtype=float)
 assert abs(np.linalg.det(M[:3,:3])-1.0)<1e-10
 return {**front,"matrix_world":M.tolist(),
         "camera_axis_order":"pixels x right, y down; no rescale",
         "angle_from_front_deg":"TOP_ORTHOGRAPHIC",
         "camera_direction_world":[0.,0.,1.],
         "top_screen_right_world":[1.,0.,0.],
         "top_screen_up_world":[0.,1.,0.]}
def project(xyz,cam):
 M=np.asarray(cam["matrix_world"],dtype=np.float64)
 h=int(cam["height"]);w=int(cam["width"])
 sc=h/float(cam["ortho_scale"])
 p=xyz-M[:3,3]
 x=w/2+np.einsum("tjd,d->tj",p,M[:3,0])*sc
 y=h/2-np.einsum("tjd,d->tj",p,M[:3,1])*sc
 z=np.einsum("tjd,d->tj",p,-M[:3,2])
 in_frame=(x>=0)&(x<w)&(y>=0)&(y<h)&(z>0)
 return np.stack((x,y,z,in_frame.astype(float)),axis=-1)
def prep(clip):
 src=pathlib.Path(clip["path"])
 if not src.is_dir():raise FileNotFoundError(src)
 meta=read(src/"manifest.json")
 if meta["frames"]!=107 or meta["fps"]!=24:raise ValueError("Unexpected source fps/frames")
 skel=read(src/"skeleton_contract.json")
 joints=read(src/"joints.json")
 cameras=read(src/"cameras.json")
 pose=read(src/"pose_camera_conditions.json")
 if len(skel["bones"])!=55 or [b["name"] for b in skel["bones"]]!=joints["names"]:
  raise ValueError("55-bone contract mismatch")
 xyz=np.asarray(joints["joint_position_m"],dtype=np.float64)
 if xyz.shape!=(107,55,3) or not np.isfinite(xyz).all():
  raise ValueError("Source 3D joint trajectory invalid")
 m={
     "left_45":camera_for_front_oblique(cameras,-45),
     "right_45":camera_for_front_oblique(cameras,45),
     "back":cameras["views"]["back"],
     "top":camera_overhead(cameras)}
 v={}
 source_errors={}
 for name in VIEWS:
  arr=project(xyz,m[name])
  if not np.all(arr[:,:,3]>0):raise ValueError("Joint outside "+name+" "+clip["id"])
  if name in pose["views"]:
   srcproject=np.asarray(pose["views"][name],dtype=np.float64)
   error=float(np.max(np.abs(arr-srcproject)))
   if error>MAX_PROJECTION_ERROR:
    raise ValueError("Regenerated source camera projection differs "+name+f" {error}")
   source_errors[name]=error
  v[name]=arr.tolist()
 path=HERE/"clips"/clip["split"]/clip["id"]
 output={
  "schema":"anymove.human_v5_fourview_clip/1",
  "source_clip":str(src),
  "source_clip_id":clip["id"],"source_split":clip["split"],
  "family":clip["family"],"frames":107,"fps":24,
  "source_3d_joints_sha256":sha(src/"joints.json"),
  "source_motion_sha256":sha(src/"motion.json"),
  "source_back_rgb_frame0_sha256":sha(src/"back"/"f_0000.png"),
  "rest_bounds":cameras["rest_bounds"],
  "camera_views":m,
  "camera_coordinate_contract":"Blender right-handed world, Z up, orthographic, pixel X right Y down, positive camera depth toward inside camera",
  "views":list(VIEWS),
  "angles_from_front_degrees":dict(zip(VIEWS,ANGLES)),
  "source_projection_max_abs_error":source_errors,
  "render_input_size":[384,448],
  "layout_canvas_size":list(CANVAS_SIZE),
  "layout_panel_size":list(TILE_SIZE),
  "layout_source_paste_offset":[32,0],
  "panel_positions":TILE_POS,
  "source_body_mesh_unchanged":True,
  "no_front_panel":True,
  "top_camera_is_true_overhead":True,
  "source_3d_motion_unchanged":True,
  "inference_uses_numeric_camera_extrinsics":False
 }
 write(path/"fourview_camera_contract.json",output)
 write(path/"fourview_pose_projections.json",{
  "schema":"anymove.human_fourview_camera_projection/1",
  "views":v,"fps":24,"frames":107,
  "joint_names":joints["names"],
  "point_format":["pixel_u","pixel_v","camera_depth_m","in_frustum"],
  "source_3d_joints_sha256":output["source_3d_joints_sha256"]})
 return {"split":clip["split"],"family":clip["family"],"id":clip["id"],
         "contract":str(path/"fourview_camera_contract.json"),
         "pose":str(path/"fourview_pose_projections.json"),
         "max_source_camera_projection_error_px":max(source_errors.values()),
         "source_sha256":sha(src/"joints.json")}
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--only",default="",help="Optional exact one clip ID")
 a=parser.parse_args()
 plan=read(V2/"v2_dataset_plan.json")
 if plan["status"]!="PASS" or len(plan["results"])!=20:
  raise ValueError("Source V2 plan not accepted")
 clips=plan["results"]
 if collections.Counter(x["split"] for x in clips)!={"train":16,"heldout":4}:
  raise ValueError("Source V2 split incorrect")
 selected=[c for c in clips if not a.only or c["id"]==a.only]
 if not selected:raise ValueError("Unknown case "+a.only)
 log=[]
 for idx,clip in enumerate(selected):
  out=prep(clip);log.append(out)
  print("V5_FOURVIEW_PROJECTED",idx+1,"/",len(selected),clip["id"],out["max_source_camera_projection_error_px"],flush=True)
 root={"schema":"anymove.fal_h3_v6_diverse_reach_l45_r45_back_top/1",
       "stage":"CAMERA_AND_3D_PROJECTED","views":list(VIEWS),
       "panel_positions":TILE_POS,
       "output_canvas":[1600,900],
       "source_panel_resolution":[384,448],
       "fps":24,"frames":107,
       "clips":log,"new_paid_ai_jobs":0}
 write(HERE/("camera_smoke.json" if a.only else "camera_contract_report.json"),root)
 print("V5_CAMERA_CONTRACT_READY",len(log),flush=True)
if __name__=="__main__":main()
