"""Summarize completed V6 FOURCAM four fal clips without further paid work."""
import json, pathlib, statistics,numpy as np
ROOT=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A")
HERE=ROOT/"conditioned_backend"/"fal_h3_ref2va"
PILOT=HERE/"remote"/"h3_ref2va_v6_28_fourview_400step"
BASE=PILOT/"heldout_ab_v1"
CASES={"forward_left":"full_heldout_reach_forward_left_a55_s8201",
       "backward_right":"full_heldout_reach_backward_right_a52_s8204"}
VIEWS=("left_45","right_45","back","top")
def read(p):return json.loads(p.read_text(encoding="utf8"))
validation=read(PILOT/"MODEL_VALIDATION.json")
assert validation["status"]=="PASS_400STEP_MODEL_ARTIFACTS"
overview={"status":"PASS_FOUR_PAID_VIDEOS_COMPLETED",
 "model":{"request_id":validation["request_id"],"training_steps":400,
          "dataset":"V6 28 train, 8 heldout","weights_sha256":validation["weights"]["sha256"],
          "trigger":"FOURCAM","fal_inference_endpoint":"minimax/h3/reference-to-video/lora"},
 "full_videos":{},"statistics":{},"warnings":[]}
wins=0
allmean={}
for case,stem in CASES.items():
 d=BASE/case
 data=read(d/"analysis"/f"{case}_color_pixel_iou.json")
 batch={}
 for label in ("scale0","scale1"):
  req=read(d/label/"request.json")
  submitted=read(d/label/"submitted.json")
  status=read(d/label/"last_status.json")
  res=read(d/label/"response.json")
  val=read(d/label/"validation.json")
  assert status["status"]=="COMPLETED" and submitted["request_id"]==status["request_id"]
  assert val["width"]==768 and val["height"]==768 and val["nb_frames"]==124
  assert req["endpoint"]=="minimax/h3/reference-to-video/lora"
  assert req["request"]["loras"][0]["scale"]== (0 if label=="scale0" else 1)
  assert req["request"]["prompt"].startswith("FOURCAM")
  assert req["request"]["seed"]==42
  assert res["video"]["url"].startswith("https://")
  batch[label]={"request_id":submitted["request_id"],"source_url":res["video"]["url"],
     "local_video":str(d/label/f"h3_ref2va_{label}.mp4"),
     "bytes":val["file_size"],"sha256":val["sha256"]}
 a=read(d/"scale0"/"request.json")["request"];b=read(d/"scale1"/"request.json")["request"]
 aa={k:v for k,v in a.items() if k!="loras"}
 bb={k:v for k,v in b.items() if k!="loras"}
 assert aa==bb and a["loras"][0]["path"]==b["loras"][0]["path"]
 overview["full_videos"][case]=batch
 group={}
 for label in ("scale0","scale1"):
  g=data["metrics"][label]
  g8=[g[f"{v}_{c}"]["mean_mask_IoU_all_nonoccluded"] for v in VIEWS for c in ("red","blue")]
  g8a=[g[f"{v}_{c}"]["mean_mask_IoU_active_frames24_96"] for v in VIEWS for c in ("red","blue")]
  target="red" if case=="forward_left" else "blue"
  moving=[g[f"{v}_{target}"]["mean_mask_IoU_active_frames24_96"] for v in VIEWS]
  group[label]={
   "eight_regions_mean_IoU_whole":round(float(np.mean(g8)),5),
   "eight_regions_mean_IoU_active":round(float(np.mean(g8a)),5),
   "four_views_target_arm_active_IoU":round(float(np.mean(moving)),5),
   "per_view_target_arm_active_IoU":{v:g[f"{v}_{target}"]["mean_mask_IoU_active_frames24_96"] for v in VIEWS}
  }
 for v in VIEWS:
  for color in ("red","blue"):
   if data["metrics"]["scale1"][f"{v}_{color}"]["mean_mask_IoU_all_nonoccluded"]>data["metrics"]["scale0"][f"{v}_{color}"]["mean_mask_IoU_all_nonoccluded"]:wins+=1
 group["scale1_minus_scale0_active_IoU"]=round(group["scale1"]["eight_regions_mean_IoU_active"]-group["scale0"]["eight_regions_mean_IoU_active"],5)
 group["scale1_minus_scale0_full_IoU"]=round(group["scale1"]["eight_regions_mean_IoU_whole"]-group["scale0"]["eight_regions_mean_IoU_whole"],5)
 overview["statistics"][case]=group
 for label in ("scale0","scale1"):
  allmean[(case,label)]=group[label]["eight_regions_mean_IoU_active"]
overview["statistics"]["combined"]={
 "same_seed_controlled_paired_cases":2,
 "fal_paid_inference_videos_submitted":4,
 "moving_upperarm_color_association_preserved_in_both_video_families":"Visual evidence only; not certified 3D joint rig",
 "sixteen_view_color_pairs_where_scale1_IoU_higher":wins,
 "out_of_total_pairs":16,
 "all_color_regions_mean_active_IoU_scale0":round(float(np.mean([allmean[(c,"scale0")] for c in CASES])),5),
 "all_color_regions_mean_active_IoU_scale1":round(float(np.mean([allmean[(c,"scale1")] for c in CASES])),5),
}
source={"families":{}}
for case,stem in CASES.items():
 j=read(ROOT/"v6_reach_source"/stem/"joints.json")
 names=j["names"]
 pts=np.asarray(j["joint_position_m"],dtype=float)
 source["families"][case]={}
 for side in ("L","R"):
  hand=pts[:,names.index(f"CC_Base_{side}_Hand"),:]
  delta=hand-hand[0]
  source["families"][case][side]={
   "hand_joint_total_3d_travel_from_rest_RMS_m":round(float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),4),
   "hand_joint_max_3d_displacement_from_rest_m":round(float(np.max(np.linalg.norm(delta,axis=1))),4)
  }
overview["blender_3d_gt"]=source
overview["warnings"]=[
 "Four-view output preserves 2x2 layout but this does not prove same 3D motion in all views.",
 "IoU is 2D pixel color-mask overlap with Blender reference, not 3D skeletal joint error.",
 "Reference Pose Video uses fal native sparse / soft video conditioning, not hard per-frame joint-camera constraints.",
 "Model generated 124 frames versus Blender source 107; metrics use normalized clip phase, not exact time.",
 "Only one seed (42), two heldout cases; results cannot establish robust generalization.",
 "Forward LEFT red upperarm/backward RIGHT blue upperarm are anatomically designated per V6 colors; 45-degree camera silhouettes and self-occlusion can shift apparent 2D centroids.",
]
output=BASE/"FINAL_4CLIP_SUMMARY.json"
output.write_text(json.dumps(overview,indent=2),encoding="utf8")
print(json.dumps({"output":str(output),"data":overview},indent=2),flush=True)
