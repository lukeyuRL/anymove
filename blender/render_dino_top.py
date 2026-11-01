"""Render synchronized Dino Top View from an EXISTING keyed Blender GT scene.

blender -b gt_render/animation.blend -P blender/render_dino_top.py -- \
    assets/dino/rigged/rig_info.json gt_render/joints.json NEW_TOP_DIR \
    --engine cycles --samples 8

This script NEVER synthesizes a new animation. It validates frame 0 and every
joint against the independent existing Blender `gt_render/joints.json`.
Output is a separate 880 x 720 Top sequence + Blender camera projection.
No model inference and NO oracle positional data is written to images.
"""
import hashlib
import json
import pathlib
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import common


def get_opts():
    args=sys.argv[sys.argv.index("--")+1:]
    if len(args)<3:
        raise ValueError("usage: rig_info.json joints.json NEW_TOP_DIR [--engine cycles|workbench] [--samples 8]")
    info_path,gt_path,out=map(pathlib.Path,args[:3])
    trailing=args[3:]
    def get(key,default):
        return trailing[trailing.index(key)+1] if key in trailing else default
    engine=get("--engine","cycles")
    samples=int(get("--samples","8"))
    if engine not in ("cycles","workbench") or not 1<=samples<=128:
        raise ValueError("invalid Top Blender engine or samples")
    return info_path,gt_path,out,engine,samples


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as fd:
        for block in iter(lambda:fd.read(1048576),b""):
            h.update(block)
    return h.hexdigest()


def main():
    info_path,gt_path,out,engine,samples=get_opts()
    if out.exists():
        raise FileExistsError("Top output directory already exists; do not overwrite prior experiments")
    info=json.loads(info_path.read_text(encoding="utf-8"))
    gt=json.loads(gt_path.read_text(encoding="utf-8"))
    scene=bpy.context.scene
    arm,body=common.skinned_character_objects()
    names=[x["name"] for x in info["keypoints"]]
    truth=np.asarray(gt["joints"],np.float64)
    T,J=truth.shape[:2]
    if (J!=92 or names!=gt["names"] or truth.shape!=(T,J,3) or
            not np.isfinite(truth).all() or T<3 or
            scene.frame_start!=0 or scene.frame_end!=T-1):
        raise ValueError("Top renderer needs original keyed 92-joint Blender GT and full identical frame range")
    if scene.render.fps/scene.render.fps_base != float(gt["fps"]):
        raise ValueError("Top keyed scene FPS does not match Blender GT")
    if bpy.data.objects.get("cam_top") is not None:
        raise ValueError("GT scene already includes cam_top; refusing camera overwrite")
    # Camera orientation identity: image +U is world +X, image -V is
    # world +Y. Orthographic ray travels from +Z toward world -Z.
    w,h=880,720
    top=bpy.data.objects.new("cam_top",bpy.data.cameras.new("cam_top"))
    scene.collection.objects.link(top)
    top.location=(0.0,0.0,30.0)
    top.rotation_euler=(0.0,0.0,0.0)
    top.data.type="ORTHO"
    top.data.ortho_scale=common.M_PER_PX*max(w,h)
    top.data.clip_end=100
    if engine=="cycles":
        common.studio(scene,samples=samples)
    else:
        scene.render.engine="BLENDER_WORKBENCH"
        scene.display.shading.light="STUDIO"
        scene.display.shading.color_type="TEXTURE"
        scene.render.film_transparent=True
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGBA"
    scene.render.resolution_x=w
    scene.render.resolution_y=h
    scene.render.resolution_percentage=100
    scene.camera=top
    out.mkdir(parents=True,exist_ok=False)
    (out/"top").mkdir()
    # Keyed GT and per-frame camera transform must be identical, with
    # Blender evaluating the real rig, NOT solving an FK approximation.
    for t in range(T):
        scene.frame_set(t)
        actual=[]
        for kp in info["keypoints"]:
            pb=arm.pose.bones[info["bones"][kp["bone"]]["name"]]
            p=arm.matrix_world@(pb.head if kp["end"]=="head" else pb.tail)
            actual.append(list(p))
        err=float(np.max(np.abs(np.asarray(actual)-truth[t])))
        if err>0.0003:
            raise RuntimeError(f"Top camera animation differs from GT at frame {t}: {err:.6f}m")
        if t==0:
            (out/"cam_top.json").write_text(
                json.dumps(common.camera_json(top,w,h),indent=2),encoding="utf-8")
        scene.render.filepath=str(out/"top"/f"f_{t:04d}.png")
        bpy.ops.render.render(write_still=True)
        if t%20==0 or t==T-1:
            print("TOP VIEW RENDER",t+1,"/",T,flush=True)
    report={
        "profile":"anymove.dino_synced_top_blender_render_v1",
        "frames":T,"fps":float(gt["fps"]),"keypoints":J,
        "resolution":[w,h],"width_world_m":w*common.M_PER_PX,
        "blender_truth_sha256":sha256(gt_path),
        "rig_info_sha256":sha256(info_path),
        "camera":"cam_top.json",
        "top_view":"synchronized exact keyed Blender GT animation, NOT independent AI generated camera",
        "no_animation_modified":True,
        "no_joint_truth_written_to_images":True,
    }
    (out/"top_render_audit.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
