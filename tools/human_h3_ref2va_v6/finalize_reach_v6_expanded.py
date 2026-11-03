"""Release final 28-train/8-heldout Human V6 Reach augmented fourview dataset.

Only if both old V5 and new V6 independent offline QA PASS.
Never start new fal paid training or inference automatically.
"""
import hashlib,json,pathlib,zipfile,collections
H3=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\conditioned_backend\fal_h3_ref2va")
OLD=H3/"fourview_v5_l45_r45_back_top"
NEW=H3/"fourview_v6_reach_augmented"
OUT=H3/"fourview_v6_reach_expanded"
V6=pathlib.Path(r"F:\anymove\runs\human\lora_synth_A\v6_reach_source")
def read(p):return json.loads(pathlib.Path(p).read_text(encoding="utf8"))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for buf in iter(lambda:f.read(1048576),b""):h.update(buf)
 return h.hexdigest()
def save(p,obj):
 tmp=p.with_suffix(p.suffix+".tmp")
 tmp.write_text(json.dumps(obj,indent=2),encoding="utf8")
 tmp.replace(p)
original=read(OLD/"FINAL_ACCEPTANCE.json")
static=read(NEW/"STATIC_INDEPENDENT_QA.json")
encoded=read(NEW/"ENCODED_MP4_INDEPENDENT_QA.json")
diverse=read(V6/"INDEPENDENT_MOTION_DIVERSITY_QA.json")
merged=read(OUT/"MERGED_DATASET_QA.json")
assert original["status"]=="PASS_OFFLINE_DATASET"
assert static["status"]=="PASS_STATIC_FOURVIEW_GEOMETRY_AND_MEDIA"
assert encoded["status"]=="PASS_ACTUAL_H264_RGB_AND_POSE"
assert diverse["status"]=="PASS_20_NEW_REACH_WORLD_GEOMETRY"
assert merged["status"]=="PASS_MERGED_OFFLINE_FILES_AND_INTEGRITY"
assert static["train"]==16 and static["heldout"]==4
assert static["verified_rgb_and_pose_png_frames"]==20*107
assert static["independent_depth_owner_witnesses"]==20*4*4
assert encoded["rgb_mp4_frames_decoded"]==encoded["pose_mp4_frames_decoded"]==20*107
assert encoded["encoded_pose_stable_color_accuracy"]>.95
assert merged["training_clips"]==28 and merged["heldout_clips"]==8
assert merged["training_reach_clips"]==20 and merged["training_lean_clips"]==8
assert merged["same_view_layout"]==["left_45","right_45","back","top"]
assert merged["canvas"]==[896,896] and merged["aspect_ratio"]=="1:1"
assert merged["new_paid_fal_jobs"]==0
file=pathlib.Path(merged["merged_archive"]["path"])
assert file.is_file() and sha(file)==merged["merged_archive"]["sha256"]
with zipfile.ZipFile(file) as z:
 assert len(z.namelist())==112 and z.testzip() is None
 counted=collections.Counter()
 for sample in merged["train_records"]:
  name=sample["id"]
  assert all(name+s in z.namelist() for s in
             (".mp4",".txt",".ref_1.mp4",".ref_2.png"))
  counted[sample["family"]]+=1
assert counted["reach_left"]==2 and counted["reach_right"]==2
for typ in ("forward","diagonal","high","backward"):
 assert counted[f"reach_{typ}_left"]==2
 assert counted[f"reach_{typ}_right"]==2
data={
 "schema":"anymove.human_fal_h3_ref2va_v6_diverse_reach_final_acceptance/1",
 "status":"PASS_OFFLINE_DATASET",
 "go_for_native_fal_ref2va_1step_debug":"REQUIRES_USER_APPROVAL",
 "go_for_full_400step_paid_training":"HOLD_UNTIL_DEBUG_AND_FURTHER_QUALITY_REVIEW",
 "no_paid_training_or_inference_executed":True,
 "final_zip":{
   "path":str(file),"sha256":sha(file),"bytes":file.stat().st_size,
   "members":112,"training_clips":28,"heldout_clips":8,
   "source_V5_train":12,"new_V6_reach_train":16,
   "total_reach_train":20,"total_lean_train":8},
 "layout":{
   "top_left":"front-left 45",
   "top_right":"front-right 45",
   "bottom_left":"back",
   "bottom_right":"top-overhead",
   "canvas":[896,896],"aspect_ratio":"1:1",
   "camera_extrinsics_consumed_by_fal_as_numbers":False,
   "fourview_RGB_Pose_and_Identity_match":True
 },
 "source_validation":{
   "new_source_motions":20,
   "new_augmented_source_world_3D_joint_signs_and_nonmoving_arm":"PASS",
   "new_augmented_source_deduplication":"PASS",
   "new_augmented_20x107_source_4camera_projection":"PASS",
   "new_RGB_Pose_20x107_png_pair_QA":"PASS",
   "new_20x4x4_independent_depth_pixel_witness_QA":"PASS",
   "new_20x107_RGB_and_Pose_H264_mp4_decode_QA":"PASS",
   "encoded_bone_color_accuracy":encoded["encoded_pose_stable_color_accuracy"]
 },
 "remaining_risks":[
   "Only 20 Reach training examples, generated procedurally; not natural movement diversity",
   "Original four Reach training samples only side-lateral; high V6 family partially overlaps old reaching trajectories",
   "New heldout high-left near an old lateral left Reach, so do not treat it as a fully novel action generalization test",
   "fal H3 Ref2VA conditions on sparse/soft visual references, not a hard multi-view 3D joint/camera model",
   "No model trained/evaluated yet with the expanded fourview distribution"
 ],
 "recommended_next":"Review final RGB/Pose contact and optionally authorize at most one fal 1-Step Debug on THIS exact ZIP. Full training only after separate approval and matched heldout experiment.",
 "reports":[
  str(OLD/"FINAL_ACCEPTANCE.json"),
  str(NEW/"STATIC_INDEPENDENT_QA.json"),
  str(NEW/"ENCODED_MP4_INDEPENDENT_QA.json"),
  str(V6/"INDEPENDENT_MOTION_DIVERSITY_QA.json"),
  str(OUT/"MERGED_DATASET_QA.json")],
 "new_paid_jobs":0
}
OUT.mkdir(parents=True,exist_ok=True)
save(OUT/"FINAL_ACCEPTANCE.json",data)
print("V6_FOURVIEW_REACH_EXPANDED_FINAL_ACCEPTANCE_PASS",json.dumps(data,indent=2),flush=True)
