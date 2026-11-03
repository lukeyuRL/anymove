"""Render Dino Top training images for EXACT SAME POSES in existing HR dataset.

blender -b assets/dino/rigged/rigged.blend -P blender/render_dino_top_dataset.py -- \
  runs/dino/dataset_hr_v1 assets/dino/rigged/rig_info.json NEW_TOP_DATASET_DIR \
  --start 0 --count 30 --samples 8

Reads original pose/root/yaw/joints_world metadata without editing them.
Each pose is re-applied to the original skinned Dino Blender rig and verified
against original 3D world joints. Reproduced top view is static (does NOT
pretend to reproduce original Front/Side 20% motion blur). The Top training
samples are a NEW independent dataset profile, never mixed with original HR.
"""
import hashlib
import json
import math
import pathlib
import sys

import bpy
import numpy as np
from mathutils import Euler,Vector

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import common

PROFILE="anymove.dino_top_same_pose_static_v1"


def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as fd:
        for block in iter(lambda:fd.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def main():
    args=sys.argv[sys.argv.index("--")+1:]
    if len(args)<3:
        raise ValueError("source_hr_dir rig_info.json NEW_TOP_OUTPUT [--start 0 --count 30 --samples 8]")
    source=pathlib.Path(args[0]).resolve()
    rigpath=pathlib.Path(args[1]).resolve()
    out=pathlib.Path(args[2]).resolve()
    opts=args[3:]
    def get(flag,default):
        return opts[opts.index(flag)+1] if flag in opts else default
    first=int(get("--start","0"))
    count=int(get("--count","30"))
    samples=int(get("--samples","8"))
    if first<0 or count<1 or count>2200 or not 1<=samples<=128:
        raise ValueError("invalid Top dataset range/render samples")
    if out.exists():
        raise FileExistsError("Top dataset output already exists; no mixing old/new samples")
    for i in range(first,first+count):
        if not (source/f"{i:05d}_meta.json").is_file():
            raise FileNotFoundError("missing source HR pose metadata "+str(i))
    info=json.loads(rigpath.read_text(encoding="utf-8"))
    names=[b["name"] for b in info["bones"]]
    kpnames=[k["name"] for k in info["keypoints"]]
    if len(names)!=67 or len(kpnames)!=92:
        raise ValueError("original Dino HR Rig must have 67 bones and 92 keypoints")
    scene=bpy.context.scene
    arm,body=common.skinned_character_objects()
    common.studio(scene,samples=samples)
    scene.cycles.use_adaptive_sampling=True
    scene.cycles.denoising_use_gpu=True
    scene.render.image_settings.color_mode="RGBA"
    scene.render.image_settings.file_format="PNG"
    scene.render.use_motion_blur=False
    scene.render.resolution_x=880
    scene.render.resolution_y=720
    scene.render.resolution_percentage=100
    top=bpy.data.objects.new("cam_top_dataset",bpy.data.cameras.new("cam_top_dataset"))
    scene.collection.objects.link(top)
    top.location=(0,0,30)
    top.rotation_euler=(0,0,math.pi/2)
    top.data.type="ORTHO"
    top.data.ortho_scale=common.M_PER_PX*880
    top.data.clip_end=100
    scene.camera=top
    bpy.context.view_layer.update()
    cam=common.camera_json(top,880,720)
    R=np.asarray(cam["R"],np.float64)
    t=np.asarray(cam["t"],np.float64)
    f=float(cam["f"])
    out.mkdir(parents=True,exist_ok=False)
    (out/"cam_top.json").write_text(json.dumps(cam,indent=2),encoding="utf-8")
    key=bpy.data.objects["key"]
    fill=bpy.data.objects["fill"]
    rim=bpy.data.objects["rim"]
    for i in range(first,first+count):
        src=source/f"{i:05d}_meta.json"
        meta=json.loads(src.read_text(encoding="utf-8"))
        if (set(meta["pose"])!=set(names) or
                np.asarray(meta["joints_world"]).shape!=(92,3) or
                float(meta["scale"])!=1.0):
            raise ValueError("incompatible original Dino pose/scale metadata "+str(src))
        arm.animation_data_clear()
        for pb in arm.pose.bones:
            pb.rotation_mode="XYZ"
            pb.location=(0,0,0)
            pb.rotation_euler=(0,0,0)
        for bone,angles in meta["pose"].items():
            arm.pose.bones[bone].rotation_euler=Euler(angles,"XYZ")
        arm.location=Vector(meta["root"])
        arm.rotation_euler=(0,0,float(meta["yaw"]))
        scene.frame_set(0)
        bpy.context.view_layer.update()
        actual=[]
        for kp in info["keypoints"]:
            pb=arm.pose.bones[names[kp["bone"]]]
            p=arm.matrix_world@(pb.head if kp["end"]=="head" else pb.tail)
            actual.append(list(p))
        original=np.asarray(meta["joints_world"],float)
        error=np.linalg.norm(np.asarray(actual)-original,axis=-1)
        if error.max()>.001:
            raise RuntimeError(
                f"Source Dino pose {i} could not be reconstructed: max {error.max():.6f}m")
        local=original@R.T+t
        top_uv=np.stack([880/2+f*local[:,0],720/2-f*local[:,1]],-1)
        if np.any((top_uv[:,0]<0)|(top_uv[:,0]>=880)|
                  (top_uv[:,1]<0)|(top_uv[:,1]>=720)):
            raise ValueError("Top pose goes out of 880x720 camera frame "+str(i))
        # Deterministic independent top-domain lighting across images.
        rng=np.random.default_rng(500000+int(meta["sample_seed"]))
        key.rotation_euler=(math.radians(rng.uniform(25,70)),0,rng.uniform(-math.pi,math.pi))
        key.data.energy=rng.uniform(1.5,4.0)
        fill.rotation_euler=(math.radians(rng.uniform(30,80)),0,rng.uniform(-math.pi,math.pi))
        fill.data.energy=rng.uniform(.2,1.5)
        rim.data.energy=rng.uniform(0,2.0)
        scene.render.filepath=str(out/f"{i:05d}_top.png")
        bpy.ops.render.render(write_still=True)
        top_meta={
            "profile":PROFILE,
            "index":int(i),
            "names":kpnames,
            "top_uv":top_uv.round(4).tolist(),
            "render_size":[880,720],
            "scale":1.0,
            "joints_world":original.tolist(),
            "source_meta_sha256":digest(src),
            "original_blur":bool(meta["blur"]),
            "top_blur":False,
            "restored_pose_max_3d_cm":round(float(error.max()*100),6),
        }
        (out/f"{i:05d}_top_meta.json").write_text(
            json.dumps(top_meta,indent=1),encoding="utf-8")
        print("TOP STATIC DATASET",i,"/",first+count-1,
              "FK label max diff cm",round(float(error.max()*100),5),
              flush=True)
    (out/"dataset_top_audit.json").write_text(json.dumps({
        "profile":PROFILE,"source_hr_dir":str(source),
        "rig_info_sha256":digest(rigpath),
        "start":first,"count":count,
        "top_image_size":[880,720],
        "top_pose_policy":"reconstruct original HR training poses, no new independent 3D targets",
        "blur_policy":"Top static RGBA, original HR blur not reproduced",
        "no_source_files_modified":True,
    },indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
