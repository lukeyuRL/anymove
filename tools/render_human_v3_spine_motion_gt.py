"""Generate full-rotation Human V3 Nose *Motion Prior* GT from Blender 5.

No Reader images or retraining. 144 independent clips (12 action families,
8 Train + 2 Validation + 2 Test each), 107 fully observed 24FPS frames,
55 local XYZ bone rotations, 1 root quaternion, root translation, 67 3D
marker positions and proxy foot contact. Source rig and weights untouched.

Programmatic canonical output is local NPZ and manifest; not a trained
neural prior or physically accurate Mocap/contact data.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time
from collections import Counter

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools import human_v3_spine_motion_prior_v1 as motion
from tools import render_human_v3_hr_v2_315_45 as renderer

SOURCE=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
SOURCE_SHA=renderer.SHA256
OUT=ROOT/"runs/human/fullbody_finger_v2_hr_315_45/spine_motion_prior_v1"
DATA=OUT/"clips"


def filehash(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data,separators=(",",":"),ensure_ascii=False),encoding="utf8")
    tmp.replace(path)


def synthesize(arm,rest,names,bones,desc):
    frame_count=motion.FRAMES
    euler=np.zeros((frame_count,55,3),dtype=np.float64)
    root_q=np.zeros((frame_count,4),dtype=np.float64)
    root_xyz=np.zeros((frame_count,3),dtype=np.float64)
    points=np.zeros((frame_count,67,3),dtype=np.float64)
    toe_min=np.zeros((frame_count,2),dtype=np.float64)
    rest_offset=rest.translation.copy()
    rig_index={name:i for i,name in enumerate(bones)}
    for frame in range(frame_count):
        changes,world_delta,root_euler=motion.pose_changes(
            desc["family"],desc["seed"],frame)
        for pb in arm.pose.bones:
            val=np.asarray(motion.BASE.get(pb.name,(0.,0.,0.)))+np.asarray(changes.get(pb.name,(0.,0.,0.)))
            if not np.isfinite(val).all():
                raise RuntimeError("Invalid local GT")
            pb.rotation_mode="XYZ"
            pb.rotation_euler=tuple(float(v) for v in val)
            euler[frame,rig_index[pb.name]]=val
        q=Euler(tuple(root_euler),"XYZ").to_quaternion()
        world_rot=q.to_matrix().to_4x4()
        arm.matrix_world=Matrix.Translation(Vector(world_delta))@world_rot@rest
        bpy.context.view_layer.update()
        joints=renderer.prior.joints_world(arm,names)
        if joints.shape!=(67,3) or not np.isfinite(joints).all():
            raise RuntimeError("Invalid GT world joints")
        toe_ids=[names.index(n) for n in motion.toe_tip_names()]
        feet_ids=[names.index(f"CC_Base_{side}_Foot") for side in ("L","R")]
        lowest=min(float(joints[k,2]) for k in toe_ids+feet_ids)
        if lowest<.002:
            matrix=arm.matrix_world.copy()
            matrix.translation.z+=.002-lowest
            arm.matrix_world=matrix
            bpy.context.view_layer.update()
            joints=renderer.prior.joints_world(arm,names)
        points[frame]=joints
        root_q[frame]=[q.w,q.x,q.y,q.z]
        root_xyz[frame]=np.asarray(arm.matrix_world.translation-rest_offset)
        toe_min[frame]=joints[toe_ids,2]
    if not np.isfinite(euler).all() or not np.isfinite(root_xyz).all():
        raise RuntimeError("Nonfinite bone animation GT")
    if not np.array_equal(euler[:,rig_index["RL_BoneRoot"],:],
                          np.zeros((frame_count,3),dtype=np.float64)):
        raise RuntimeError("RL_BoneRoot must remain static: Root rotation belongs to Armature Object")
    # Torso action is present, including ROOT variation. All 12 families are
    # intentionally non-stationary; no family should silently collapse.
    waist=rig_index["CC_Base_Waist"]
    spine1=rig_index["CC_Base_Spine01"]
    spine2=rig_index["CC_Base_Spine02"]
    yaw_deg=np.rad2deg(np.ptp(np.asarray([motion.pose_changes(desc["family"],desc["seed"],f)[2][2] for f in range(frame_count)])))
    torso_deg=np.rad2deg(max(np.ptp(euler[:,waist,ax]) for ax in range(3)))
    legs=[i for i,n in enumerate(names) if any(q in n for q in ("Thigh","Calf","Foot","ToeBase","ToeTip"))]
    fingers=[names.index(name) for name in motion.finger_tip_names()]
    max_leg=float(np.max(np.linalg.norm(points[:,legs]-points[0:1,legs],axis=-1)))
    max_finger=float(np.max(np.linalg.norm(points[:,fingers]-points[0:1,fingers],axis=-1)))
    if min(yaw_deg,torso_deg)<4.0 or min(max_leg,max_finger)<.025:
        raise RuntimeError("Insufficient Root/Spine/limb animation diversity "+str(desc)+
                           f" yaw={yaw_deg} waist={torso_deg} leg={max_leg} finger={max_finger}")
    footids=[names.index(f"CC_Base_{s}_Foot") for s in ("L","R")]
    toeids=[names.index(n) for n in motion.toe_tip_names()]
    # Contact is a GEOMETRIC/VELOCITY proxy. Root turning around a planted
    # foot may still slide; never interpret this proxy as physical truth.
    foot_anchor=(points[:,footids]+points[:,toeids])/2.
    foot_vel=np.zeros((frame_count,2),dtype=np.float64)
    foot_vel[1:]=np.linalg.norm(np.diff(foot_anchor[:,:,:2],axis=0),axis=-1)
    foot_proxy=(toe_min<.03)&(foot_vel<.024)
    quality={
        "root_yaw_range_deg":float(yaw_deg),
        "waist_axis_range_max_deg":float(torso_deg),
        "spine01_axis_range_max_deg":float(np.rad2deg(max(np.ptp(euler[:,spine1,ax]) for ax in range(3)))),
        "spine02_axis_range_max_deg":float(np.rad2deg(max(np.ptp(euler[:,spine2,ax]) for ax in range(3)))),
        "max_leg_world_travel_m":max_leg,
        "max_fingertip_world_travel_m":max_finger,
        "lowest_toe_world_z_m":float(toe_min.min()),
        "contact_proxy_fraction_L":float(foot_proxy[:,0].mean()),
        "contact_proxy_fraction_R":float(foot_proxy[:,1].mean()),
        "contact_proxy_method":"toe z<3cm AND foot/Toe anchor horizontal step <2.4cm/frame. NOT physical foot-plant GT",
    }
    return {"frames":np.arange(frame_count,dtype=np.int32),
            "bone_euler_xyz_rad":euler.astype(np.float32),
            "root_rotation_wxyz":root_q.astype(np.float32),
            "root_translation_m":root_xyz.astype(np.float32),
            "world_joints_m":points.astype(np.float32),
            "toe_height_m":toe_min.astype(np.float32),
            "contact_proxy":foot_proxy.astype(np.uint8)}, quality


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--family-index",type=int,default=-1)
    args=ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if pathlib.Path(bpy.data.filepath).resolve()!=SOURCE.resolve() or filehash(SOURCE)!=SOURCE_SHA:
        raise RuntimeError("Wrong V3 Nose input Blend or hash")
    all_clips=motion.all_unique_clips()
    clips=all_clips
    if args.smoke:
        clips=[all_clips[0],all_clips[8],all_clips[10]]
    elif args.family_index!=-1:
        if args.family_index not in range(len(motion.FAMILIES)):
            raise ValueError("Unknown motion family index")
        clips=[r for r in all_clips if r["family"]==motion.FAMILIES[args.family_index]]
    arm=bpy.data.objects["Armature"]
    mesh=bpy.data.objects["Dummy"]
    if len(arm.pose.bones)!=55 or len(mesh.data.vertices)!=30188:
        raise RuntimeError("V3 Nose source model is not 55 bones")
    if mesh.data.color_attributes.get("AnyMoveHumanSkinMax53") is None:
        raise RuntimeError("V3 colour attribute missing")
    rest=arm.matrix_world.copy()
    names=renderer.marker_names(arm)
    bones=[pb.name for pb in arm.pose.bones]
    if len(names)!=67:
        raise RuntimeError("Wrong tracking marker count")
    start=time.monotonic()
    dest=OUT/("smoke" if args.smoke else "clips")
    summary=[]
    for i,desc in enumerate(clips):
        path=dest/desc["split"]/(desc["clip"]+".npz")
        metadata=dest/desc["split"]/(desc["clip"]+".json")
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() or metadata.exists():
            # Always fail closed rather than writing mismatched motion to a
            # reused source path. The separate manifest audit can resume.
            if not path.exists() or not metadata.exists():
                raise RuntimeError("Incomplete existing GT clip "+desc["clip"])
            row=json.loads(metadata.read_text(encoding="utf8"))
            if row["clip"]!=desc or row["source_model_sha256"]!=SOURCE_SHA:
                raise RuntimeError("Existing motion GT clip mismatch")
            quality=row["quality"]
        else:
            arrays,quality=synthesize(arm,rest,names,bones,desc)
            np.savez_compressed(path,**arrays)
            row={"schema":"anymove.human_v3_spine_motion_gt_clip/1",
                 "clip":desc,
                 "source_model_sha256":SOURCE_SHA,
                 "fps":motion.FPS,
                 "source_frames":motion.FRAMES,
                 "bone_names":bones,
                 "marker_names":names,
                 "world_root_motion_convention":"Root object translation in metres + Root world quaternion wxyz; per-bone Blender local Euler XYZ",
                 "quality":quality,
                 "grounding_warning":"Foot contact is NOT physically enforced",
                 }
            atomic_json(metadata,row)
        summary.append({"clip":desc["clip"],"family":desc["family"],
                        "split":desc["split"],**quality})
        print("SPINE_MOTION_CLIP",i+1,"of",len(clips),
              desc["family"],desc["split"],"rootYawDeg",round(quality["root_yaw_range_deg"],1),
              "elapsed_s",round(time.monotonic()-start,1),flush=True)
    batch={
        "schema":"anymove.human_v3_spine_motion_gt_batch/1",
        "state":"COMPLETED",
        "source_model_sha256":SOURCE_SHA,
        "human_bones":55,"tracking_markers":67,
        "fps":motion.FPS,"frames_per_clip":motion.FRAMES,
        "families":list(motion.FAMILIES),
        "clip_count":len(summary),
        "rows":summary,
        "notes":"Rotation/time-series GT; not Reader training images or genuine human Mocap",
    }
    name=("smoke_manifest.json" if args.smoke else f"family_{args.family_index:02d}_manifest.json")
    atomic_json(OUT/name,batch)
    print("SPINE_MOTION_BATCH_COMPLETE",json.dumps({
        "clips":len(summary),"frames":len(summary)*107,"manifest":str(OUT/name),
        "elapsed_s":round(time.monotonic()-start,1)}),flush=True)


if __name__=="__main__":
    main()
