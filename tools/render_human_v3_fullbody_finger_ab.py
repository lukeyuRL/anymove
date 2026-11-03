"""Render V3-nose fullbody+finger A/B dataset using identical procedural GT.

Purely offline Blender Eevee. Same generated poses/GT for 0/90 and 315/45,
including 55 pose-bone heads, 10 animated fingertips and 2 toe endpoints.
Never changes/saves the V3 Nose .blend. Each split is clip-disjoint.
"""
import argparse,hashlib,json,math,pathlib,sys

import bpy
import numpy as np
from mathutils import Matrix,Vector

ROOT=pathlib.Path("F:/anymove")
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.human_v3_fullbody_motion import (
    BASE,FAMILIES,pose_changes,finger_tip_names,toe_tip_names
)

SCENE=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
OUT=ROOT/"runs/human/fullbody_finger_v3_nose"
SHA="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
CAMERAS={"baseline_0_90":(0,90),"symmetric_315_45":(315,45)}
W=H=448
ORTHO_SCALE=2.432802200317383
Z_CENTER=.8688581585884094
M_PER_PX=ORTHO_SCALE/H
FRAME_COUNT=107
TRAIN_FRAMES=(0,8,16,24,32,40,48,56,64,72,80,88,96,106)
OTHER_FRAMES=(0,16,32,48,64,80,96,106)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda:f.read(1<<20),b""):
            h.update(x)
    return h.hexdigest()

def clip_defs(smoke=False):
    train=[{"split":"train","family":family,"seed":4000+i*31+j*791,
            "clip":"fb_train_"+family+"_"+str(4000+i*31+j*791)}
           for i,family in enumerate(FAMILIES) for j in range(2)]
    val=[{"split":"validation","family":FAMILIES[i],"seed":6500+i*41,
          "clip":"fb_val_"+FAMILIES[i]+"_"+str(6500+i*41)} for i in range(4)]
    test=[{"split":"test","family":FAMILIES[i],"seed":8500+i*43,
           "clip":"fb_test_"+FAMILIES[i]+"_"+str(8500+i*43)} for i in range(4,8)]
    if smoke:return [train[0],val[0],test[0]]
    return train+val+test

def projection(positions,angle):
    angle=math.radians(angle)
    x=W/2+(math.cos(angle)*positions[:,0]-math.sin(angle)*positions[:,1])/M_PER_PX
    y=H/2-(positions[:,2]-Z_CENTER)/M_PER_PX
    return np.stack((x,y),axis=-1)

def apply_pose(arm,rest_world,names,family,seed,frame):
    adjustments,offset=pose_changes(family,seed,frame,FRAME_COUNT)
    for bone in arm.pose.bones:
        a=BASE.get(bone.name,(0,0,0))
        d=adjustments.get(bone.name,(0,0,0))
        bone.rotation_mode="XYZ"
        bone.rotation_euler=tuple(float(a[i]+d[i]) for i in range(3))
    arm.matrix_world=Matrix.Translation(Vector(offset)) @ rest_world
    bpy.context.view_layer.update()
    toes=[arm.matrix_world @ arm.pose.bones["CC_Base_"+s+"_ToeBase"].tail for s in ("L","R")]
    ankles=[arm.matrix_world @ arm.pose.bones["CC_Base_"+s+"_Foot"].head for s in ("L","R")]
    foot_ground=min(float(p.z) for p in toes+ankles)
    # Keep at least one plantar endpoint above the floor, not a measured gait/IK.
    if foot_ground<.002:
        m=arm.matrix_world.copy()
        m.translation.z+=.002-foot_ground
        arm.matrix_world=m
        bpy.context.view_layer.update()

def joints_world(arm,bone_names):
    head=[arm.matrix_world @ arm.pose.bones[n].head for n in bone_names[:55]]
    tips=[arm.matrix_world @ arm.pose.bones[n.replace("Tip","3")].tail for n in finger_tip_names()]
    toes=[arm.matrix_world @ arm.pose.bones[n.replace("ToeTip","ToeBase")].tail for n in toe_tip_names()]
    return np.asarray(head+tips+toes,dtype=np.float64)

def make_scene():
    if pathlib.Path(bpy.data.filepath).resolve()!=SCENE.resolve() or sha(SCENE)!=SHA:
        raise RuntimeError("Must open verified Human V3-nose scene with matching SHA")
    rig=bpy.data.objects["Armature"]
    mesh=bpy.data.objects["Dummy"]
    if len(rig.pose.bones)!=55 or len(mesh.data.vertices)!=30188:
        raise RuntimeError("Invalid source mesh or skeleton")
    if "AnyMoveHumanSkinMax53" not in mesh.data.color_attributes:
        raise RuntimeError("V3 color attribute missing")
    rest=rig.matrix_world.copy()
    scene=bpy.context.scene
    scene.render.engine="BLENDER_EEVEE"
    scene.eevee.taa_render_samples=12
    scene.render.film_transparent=True
    scene.render.resolution_percentage=100
    scene.render.resolution_x=W;scene.render.resolution_y=H
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGBA"
    scene.view_settings.view_transform="Standard"
    try:scene.view_settings.look="Medium High Contrast"
    except (TypeError,ValueError):pass
    world=scene.world or bpy.data.worlds.new("FBFullBodyReaderWorld")
    scene.world=world
    world.use_nodes=True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value=.75
    for name,rot,power in (
      ("FBKey",(math.radians(45),0,math.radians(-35)),3.),
      ("FBFill",(math.radians(60),0,math.radians(110)),1.3),
      ("FBRim",(math.radians(35),0,math.radians(200)),1.7)):
        ob=bpy.data.objects.new(name,bpy.data.lights.new(name,"SUN"))
        scene.collection.objects.link(ob)
        ob.rotation_euler=rot;ob.data.energy=power;ob.data.angle=math.radians(8)
    camera=bpy.data.objects.new("FB_Cam",bpy.data.cameras.new("FB_Cam"))
    scene.collection.objects.link(camera)
    camera.data.type="ORTHO";camera.data.ortho_scale=ORTHO_SCALE
    camera.data.clip_end=100;scene.camera=camera
    return rig,rest,scene,camera

def main():
    args=argparse.ArgumentParser()
    args.add_argument("--smoke",action="store_true")
    a=args.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if W!=448 or H!=448:raise ValueError("Reader contract requires 448")
    base=OUT/("smoke" if a.smoke else "dataset")
    rig,rest,scene,camera=make_scene()
    names=[b.name for b in rig.pose.bones]+finger_tip_names()+toe_tip_names()
    if len(names)!=67 or len(set(names))!=67:raise AssertionError("Expected 67 independent named markers")
    jointgroups={
        "finger_30_bone_heads":[n for n in names if any(f in n for f in ("Pinky","Ring","Mid","Index","Thumb")) and not n.endswith("Tip")],
        "finger_10_tips":finger_tip_names(),
        "toe_2_tips":toe_tip_names(),
        "leg_8":[n for n in names if any(t in n for t in ("Thigh","Calf","Foot","ToeBase"))],
        "arm_6":[n for n in names if any(t in n for t in ("Upperarm","Forearm","_Hand"))],
        "torso_4":[n for n in names if any(t in n for t in ("Hip","Waist","Spine01","Spine02"))],
    }
    for name,group in jointgroups.items():
        if not group:raise AssertionError("Empty category "+name)
    rows={key:[] for key in CAMERAS}
    stats=[]
    for cd in clip_defs(a.smoke):
        frames_to_render=(0,53,106) if a.smoke else (TRAIN_FRAMES if cd["split"]=="train" else OTHER_FRAMES)
        xyz_seq=[]
        finger_flex=[]
        for frame in range(FRAME_COUNT):
            apply_pose(rig,rest,names,cd["family"],cd["seed"],frame)
            xyz_seq.append(joints_world(rig,names))
            finger_flex.append({
                "L":float(rig.pose.bones["CC_Base_L_Index1"].rotation_euler[0]),
                "R":float(rig.pose.bones["CC_Base_R_Index1"].rotation_euler[0])})
        xyz_seq=np.asarray(xyz_seq,dtype=np.float64)
        # Joint trajectories must actually move, rather than always returning
        # static feet/finger markers that make Reader tasks deceptively easy.
        def travel(group):
            indices=[names.index(n) for n in group]
            d=np.linalg.norm(xyz_seq[:,indices,:]-xyz_seq[0:1,indices,:],axis=-1)
            return float(d.max())
        finger_rel=[]
        for side in ("L","R"):
            hand=names.index("CC_Base_"+side+"_Hand")
            for tip in ("Pinky","Ring","Mid","Index","Thumb"):
                ti=names.index("CC_Base_"+side+"_"+tip+"Tip")
                radial=np.linalg.norm(xyz_seq[:,ti,:]-xyz_seq[:,hand,:],axis=-1)
                finger_rel.append(float(np.ptp(radial)))
        motion_stats={"clip":cd["clip"],"split":cd["split"],"family":cd["family"],
            "frames":FRAME_COUNT,
            "max_foot_toe_travel_m":travel(jointgroups["toe_2_tips"]),
            "max_leg_travel_m":travel(jointgroups["leg_8"]),
            "max_hand_travel_m":travel(["CC_Base_L_Hand","CC_Base_R_Hand"]),
            "max_finger_tip_travel_m":travel(finger_tip_names()),
            "min_finger_tip_to_wrist_radius_delta_m":min(finger_rel),
            "max_finger_tip_to_wrist_radius_delta_m":max(finger_rel),
            "index_flex_radians":[round(max(f["L"] for f in finger_flex)-min(f["L"] for f in finger_flex),5),
                                  round(max(f["R"] for f in finger_flex)-min(f["R"] for f in finger_flex),5)],
            "min_foot_tip_z_m":float(xyz_seq[:,[names.index(n) for n in toe_tip_names()],2].min()),
        }
        if (motion_stats["max_leg_travel_m"]<.025 or
            motion_stats["max_hand_travel_m"]<.03 or
            motion_stats["min_finger_tip_to_wrist_radius_delta_m"]<.001 or
            min(motion_stats["index_flex_radians"])<.04 or
            motion_stats["min_foot_tip_z_m"]<-.01):
            raise RuntimeError("Not really fullbody+finger motion: "+str(motion_stats))
        stats.append(motion_stats)
        clipmeta=base/"motion_gt"/(cd["clip"]+".json")
        clipmeta.parent.mkdir(parents=True,exist_ok=True)
        gt={"schema":"anymove.human_v3_nose_fullbody_finger_gt/1",
            "source_blend_sha256":SHA,"clip":cd,"fps":24,"bone_head_count":55,
            "additional_tip_markers":finger_tip_names()+toe_tip_names(),
            "joint_names":names,"joint_position_m":np.round(xyz_seq,7).tolist(),
            "qa":motion_stats}
        if clipmeta.exists():
            previous=json.loads(clipmeta.read_text(encoding="utf-8"))
            if previous["joint_names"]!=names or previous["clip"]!=cd:
                raise RuntimeError("GT output conflicts with this run")
        clipmeta.write_text(json.dumps(gt),encoding="utf-8")
        for frame in frames_to_render:
            apply_pose(rig,rest,names,cd["family"],cd["seed"],frame)
            same=np.max(np.linalg.norm(joints_world(rig,names)-xyz_seq[frame],axis=-1))
            if same>1e-7:raise RuntimeError("Frame GT replay failed")
            for pair,angles in CAMERAS.items():
                for slot,azimuth in zip(("front","side"),angles):
                    pix=projection(xyz_seq[frame],azimuth)
                    inside=(pix[...,0]>=1)&(pix[...,0]<(W-1))&(pix[...,1]>=1)&(pix[...,1]<(H-1))
                    if not np.all(inside):
                        bad=[names[i] for i in np.where(~inside)[0]]
                        raise RuntimeError("Camera clipping "+cd["clip"]+" "+str(frame)+" "+str(azimuth)+" "+str(bad[:10]))
                    theta=math.radians(azimuth)
                    camera.location=(-20*math.sin(theta),-20*math.cos(theta),Z_CENTER)
                    camera.rotation_euler=(math.pi/2,0,-theta)
                    image=base/pair/cd["split"]/(cd["clip"]+"_f"+str(frame).zfill(3)+"_"+slot+".png")
                    image.parent.mkdir(parents=True,exist_ok=True)
                    if not image.exists() or image.stat().st_size<1000:
                        scene.render.filepath=str(image)
                        bpy.ops.render.render(write_still=True)
                    if not image.is_file() or image.stat().st_size<1000:
                        raise RuntimeError("Image missing "+str(image))
                    rows[pair].append({"split":cd["split"],"clip":cd["clip"],
                        "frame":frame,"slot":slot,"camera_azimuth_deg":azimuth,
                        "path":str(image.relative_to(base)).replace("\\","/"),
                        "uv":np.round(pix,5).tolist()})
        print("FULLBODY_CLIP_PASS",cd["split"],cd["clip"],"foot_travel",round(motion_stats["max_foot_toe_travel_m"],3),
              "finger_flex",motion_stats["index_flex_radians"],flush=True)
    for pair,angles in CAMERAS.items():
        manifest={"schema":"anymove.human_v3_nose_fullbody_finger_reader_dataset/1",
           "status":"complete","source_blend_sha256":SHA,
           "source_blend":str(SCENE),"joint_names":names,
           "joint_groups":jointgroups,
           "angles_deg":list(angles),"width":W,"height":H,
           "metres_per_pixel":M_PER_PX,"z_center":Z_CENTER,
           "frame_count":FRAME_COUNT,
           "train_clips":[c["clip"] for c in clip_defs(a.smoke) if c["split"]=="train"],
           "validation_clips":[c["clip"] for c in clip_defs(a.smoke) if c["split"]=="validation"],
           "test_clips":[c["clip"] for c in clip_defs(a.smoke) if c["split"]=="test"],
           "train_frames":list((0,53,106) if a.smoke else TRAIN_FRAMES),
           "validation_frames":list((0,53,106) if a.smoke else OTHER_FRAMES),
           "test_frames":list((0,53,106) if a.smoke else OTHER_FRAMES),
           "samples":rows[pair]}
        path=base/pair/"manifest.json"
        path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (base/"fullbody_motion_qa.json").write_text(json.dumps({"status":"PASS","clips":stats},indent=2),encoding="utf-8")
    print("FULLBODY_DATASET_PASS",json.dumps({"source":str(SCENE),"pairs":list(CAMERAS),
        "joint_count":len(names),"images":sum(map(len,rows.values())),
        "clips":len(stats),"train":len([x for x in stats if x["split"]=="train"]),
        "validation":len([x for x in stats if x["split"]=="validation"]),
        "test":len([x for x in stats if x["split"]=="test"]),
        "split_dir":str(base)}),flush=True)

if __name__=="__main__":
    main()
