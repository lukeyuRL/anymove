"""Validate actual Blender human rig replay against existing V6 3D GT.
No original .blend save, no render or generated video.
"""
import json, math, pathlib, sys
import bpy
import numpy as np
from mathutils import Matrix

args=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
base=pathlib.Path(args[0]) if args else pathlib.Path("runs/human/lora_synth_A/v6_reach_source/full_train_reach_forward_left_a44_s7301")
m=json.loads((base/"motion.json").read_text(encoding="utf-8"))
j=json.loads((base/"joints.json").read_text(encoding="utf-8"))
arm=bpy.data.objects.get("Armature")
assert arm and len(arm.pose.bones)==55
assert j["names"]==m["bones"], "motion and GT names differ"
errs=[]
frame_summaries=[]
for frame in (0,20,53,80,106):
    if frame>=len(j["joint_position_m"]):continue
    for name,vals in zip(m["bones"],m["pose_euler_xyz_rad"][frame]):
        pb=arm.pose.bones.get(name)
        if pb is None:raise RuntimeError(f"Unknown bone: {name}")
        pb.rotation_mode="XYZ"
        pb.rotation_euler=vals
    arm.matrix_world=Matrix(m["rig_world_matrices"][frame])
    bpy.context.view_layer.update()
    world=np.asarray([arm.matrix_world @ arm.pose.bones[n].head for n in j["names"]],dtype=float)
    gt=np.asarray(j["joint_position_m"][frame],dtype=float)
    err=np.linalg.norm(world-gt,axis=-1)
    errs.extend(err)
    frame_summaries.append({"frame":frame,"mean_cm":float(err.mean()*100),"max_cm":float(err.max()*100)})
result={"status":"PASS" if max(errs)<1e-3 else "FAIL_GT_REPLAY", "frames":frame_summaries,
        "max_cm":round(max(errs)*100,5),"mean_cm":round(np.mean(errs)*100,5)}
print("HUMAN_MOTION_REPLAY_AUDIT",json.dumps(result))
if result["status"]!="PASS":raise SystemExit(1)
