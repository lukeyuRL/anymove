"""Render saved 107-frame Human V3 Root-static Blender animation, no re-authoring.

Applies to the procedural turn GT and TRAIN-neutral-init SO3 FK solution.
Every image is generated from Blender's reopened Action, not from a
source JSON or synthetic interpolation. Checks RL_BoneRoot pose and 24FPS.
"""
import argparse,bpy,json,pathlib,sys,numpy as np

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
BASE=ROOT/"runs/human/fullbody_finger_v2_hr_315_45"
DIR=BASE/"solver/spine_so3_train_neutral_init_research/blender_qa"
CLIP="spine_validation_pivot_turn_left_s405752"


def main(mode):
    expected=DIR/(CLIP+"_"+mode+"_rl_boneroot_static.blend")
    if pathlib.Path(bpy.data.filepath).resolve()!=expected.resolve():
        raise RuntimeError("Unexpected saved Root-static turn Blender project")
    scene=bpy.context.scene
    arm=bpy.data.objects.get("Armature")
    if not arm or (scene.frame_start,scene.frame_end,scene.render.fps)!=(0,106,24):
        raise RuntimeError("Missing 107-frame 24FPS Human action")
    if (scene.get("AnyMove_RL_BoneRoot_Rest_Pose") is not True or
        scene.get("AnyMove_Root_Animation_Target")!="ArmatureObject"):
        raise RuntimeError("Root-static metadata is missing")
    cameras={"front":bpy.data.objects.get("HRV2_front"),
             "side":bpy.data.objects.get("HRV2_side")}
    if any(c is None for c in cameras.values()):
        raise RuntimeError("Source calibrated 315/45 cameras are missing")
    from tools.verify_human_v3_root_static_blender import action_paths
    if any("RL_BoneRoot" in p for p in action_paths(arm)):
        raise RuntimeError("Forbidden RL_BoneRoot animation FCurve")
    root=arm.pose.bones["RL_BoneRoot"]
    scene.render.resolution_x=672
    scene.render.resolution_y=672
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGBA"
    new=0
    for f in range(107):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        if not np.allclose(np.asarray(root.matrix_basis),np.eye(4),atol=1e-6):
            raise RuntimeError("RL_BoneRoot no longer at rest")
        for slot,cam in cameras.items():
            scene.camera=cam
            png=DIR/"video_frames"/mode/slot/f"frame_{f:04d}.png"
            png.parent.mkdir(parents=True,exist_ok=True)
            if not png.exists() or png.stat().st_size<1000:
                scene.render.filepath=str(png)
                bpy.ops.render.render(write_still=True)
                new+=1
            if not png.is_file() or png.stat().st_size<1000:
                raise RuntimeError("Blender render not saved")
        if f in (0,26,53,80,106):
            print("SPINE_GT_SOLVER_RENDER",mode,f,"newimages",new,flush=True)
    complete={"status":"PASS","clip":CLIP,"mode":mode,
        "frames":107,"fps":24,"total_images":214,
        "new_images":new,"RL_BoneRoot_static":True,
        "armature_object_motion":True,
        "original_source_unmodified":True}
    (DIR/(CLIP+"_"+mode+"_107frame_render_audit.json")).write_text(
        json.dumps(complete,indent=2),encoding="utf8")
    print("SPINE_GT_SOLVER_RENDER_COMPLETE",json.dumps(complete),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("gt","solver","root_so3"),required=True)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    main(args.mode)
