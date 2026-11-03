"""Human V3 Nose HR Dataset V2, calibrated 315/45, fullbody/fingers.

192 motion clips (12 families x 16 seeds), 14 non-rest sampled frames/clip:
2,688 synchronized stereo poses, 5,376 native 672x672 RGBA images.
TRAIN 144 clips / 2,016 pose pairs; VAL 24 / 336; TEST 24 / 336.
Train/val/test share only rig appearance, NEVER motion seed or clip frames.

Per-camera rows store UV (67 markers), visibility proxy based on sampled
bone-dominant skinned surface vertices + BVH first-hit raycast, uncertainty
as a qualitative proxy only, and Blender calibrated camera matrices.
The source .blend is never saved or modified. No H3 or paid training.
"""
import argparse, hashlib, json, math, pathlib, sys, time
from collections import defaultdict

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view

ROOT=pathlib.Path("F:/anymove")
sys.path.insert(0,str(ROOT))
from tools import human_v3_fullbody_motion_v2 as movement
from tools import render_human_v3_fullbody_finger_ab as prior

SOURCE=ROOT/"runs/human/human_skinmax_multicolor_v3_nose.blend"
SHA256="677f0d7421fd1671c279f104a471610ec68b8050fe75a994de97ed78d9245ccc"
OUT=ROOT/"runs/human/fullbody_finger_v2_hr_315_45"
ANGLES={"front":315.0, "side":45.0}  # names are Reader slots, not physical front/side
WIDTH=HEIGHT=672
SCALE=2.432802200317383
Z_CENTER=.8688581585884094
METRES_PER_PIXEL=SCALE/HEIGHT
FPS=24
TOTAL_FRAMES=107
# Non-rest frames; no exact identical neutral pose leaking between splits.
SELECTED_FRAMES=tuple(int(v) for v in np.linspace(8,98,14).round().astype(int))
SURFACE_SAMPLES_PER_MARKER=10
SURFACE_HIT_TOLERANCE_M=.003
SURFACE_VISIBLE_THRESHOLD=.20
SURFACE_OCCLUDED_THRESHOLD=.0

def shafile(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def atomic_json(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(payload,separators=(",",":"),ensure_ascii=False),encoding="utf-8")
    temp.replace(path)

def png_size(path):
    import struct
    try:
        with path.open("rb") as f:head=f.read(24)
        if head[:8]!=b"\x89PNG\r\n\x1a\n" or head[12:16]!=b"IHDR":return None
        return struct.unpack(">II",head[16:24])
    except (FileNotFoundError,OSError):return None

def clip_defs(smoke=False):
    rows=[]
    for family_idx,family in enumerate(movement.FAMILIES):
        for take in range(16):
            split="train" if take<12 else ("validation" if take<14 else "test")
            # Entirely distinct deterministic seed per family/take, never
            # duplicate even if 12 new motion families include common motion.
            seed=72000+family_idx*1901+take*791
            rows.append({"split":split,"family":family,"seed":seed,
                         "clip":f"hrv2_{split}_{family}_s{seed}"})
    if len(rows)!=192 or len({x["seed"] for x in rows})!=192:
        raise RuntimeError("Dataset V2 source seeds are not unique")
    if smoke:
        return [next(x for x in rows if x["split"]==part) for part in ("train","validation","test")]
    return rows

def marker_names(arm):
    names=[pb.name for pb in arm.pose.bones]+movement.finger_tip_names()+movement.toe_tip_names()
    if len(names)!=67 or len(set(names))!=67:
        raise RuntimeError("67-marker human schema mismatch")
    return names

def projection(xyz, angle):
    theta=math.radians(angle)
    u=WIDTH/2+(math.cos(theta)*xyz[:,0]-math.sin(theta)*xyz[:,1])/METRES_PER_PIXEL
    v=HEIGHT/2-(xyz[:,2]-Z_CENTER)/METRES_PER_PIXEL
    return np.stack((u,v),axis=-1)

def studio():
    if pathlib.Path(bpy.data.filepath).resolve()!=SOURCE.resolve() or shafile(SOURCE)!=SHA256:
        raise RuntimeError("Wrong V3 Nose model or source SHA-256")
    arm=bpy.data.objects.get("Armature")
    mesh=bpy.data.objects.get("Dummy")
    if not arm or not mesh or len(arm.pose.bones)!=55 or len(mesh.data.vertices)!=30188:
        raise RuntimeError("Human V3 Nose rig or geometry changed")
    if mesh.data.color_attributes.get("AnyMoveHumanSkinMax53") is None:
        raise RuntimeError("SkinMax53 color attribute missing")
    rig_rest=arm.matrix_world.copy()
    scene=bpy.context.scene
    scene.render.engine="BLENDER_EEVEE"
    scene.eevee.taa_render_samples=12
    scene.render.film_transparent=True
    scene.render.resolution_x=WIDTH
    scene.render.resolution_y=HEIGHT
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGBA"
    scene.view_settings.view_transform="Standard"
    try:scene.view_settings.look="Medium High Contrast"
    except (TypeError,ValueError):pass
    world=scene.world or bpy.data.worlds.new("HRV2Studio")
    scene.world=world
    world.use_nodes=True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value=.75
    for name,rot,power in (
        ("HRV2Key",(math.radians(45),0,math.radians(-35)),3.0),
        ("HRV2Fill",(math.radians(60),0,math.radians(110)),1.3),
        ("HRV2Rim",(math.radians(35),0,math.radians(200)),1.7)):
        lamp=bpy.data.objects.new(name,bpy.data.lights.new(name,"SUN"))
        scene.collection.objects.link(lamp)
        lamp.rotation_euler=rot
        lamp.data.energy=power
        lamp.data.angle=math.radians(8)
    cameras={}
    for slot,angle in ANGLES.items():
        rad=math.radians(angle)
        cam=bpy.data.objects.new("HRV2_"+slot,bpy.data.cameras.new("HRV2_"+slot))
        scene.collection.objects.link(cam)
        cam.location=(-20*math.sin(rad),-20*math.cos(rad),Z_CENTER)
        cam.rotation_euler=(math.pi/2,0,-rad)
        cam.data.type="ORTHO"
        cam.data.ortho_scale=SCALE
        cam.data.clip_end=100
        cameras[slot]=cam
    bpy.context.view_layer.update()
    return arm,mesh,rig_rest,scene,cameras

def candidate_skin_vertices(mesh, arm, names):
    """Each marker has a proxy patch from strongest skinned bone region.

    Some rig/root controls have NO corresponding surface patch and must be
    marked unknown rather than fabricated as visible.
    """
    groups={g.index:g.name for g in mesh.vertex_groups}
    bybone=defaultdict(list)
    for v in mesh.data.vertices:
        if not v.groups:continue
        top=max(v.groups,key=lambda g:(g.weight,-g.group))
        bybone[groups[top.group]].append(v.index)
    rng=np.random.default_rng(20261104)
    patches={}
    missing=[]
    for name in names:
        if name.endswith("ToeTip"):
            region=name.replace("ToeTip","ToeBase")
        elif name.endswith("Tip"):
            region=name.replace("Tip","3")
        else:
            region=name
        choices=bybone.get(region,[])
        if len(choices)<3:
            patches[name]=[]
            missing.append(name)
            continue
        sample=rng.choice(choices,size=min(len(choices),SURFACE_SAMPLES_PER_MARKER),replace=False)
        patches[name]=[int(i) for i in sample]
    print("V2_VISIBILITY_REGIONS",json.dumps({
        "available":len(names)-len(missing),"missing":missing,
        "max_sample_points":SURFACE_SAMPLES_PER_MARKER}),flush=True)
    return patches

def visibility_by_view(mesh, scene, cameras, patches, projected):
    """Raycast against the actual posed mesh. Proxy for surface coverage only."""
    depsgraph=bpy.context.evaluated_depsgraph_get()
    obj=mesh.evaluated_get(depsgraph)
    geom=obj.to_mesh()
    try:
        world_matrix=obj.matrix_world.copy()
        verts=[world_matrix @ v.co for v in geom.vertices]
        poly=[tuple(p.vertices) for p in geom.polygons]
        bvh=BVHTree.FromPolygons(verts,poly,all_triangles=False,epsilon=1e-6)
        visible={}
        for slot,cam in cameras.items():
            origin=cam.matrix_world.translation.copy()
            pose=[]
            uv=projected[slot]
            for idx,(name,v_indices) in enumerate(patches.items()):
                if not v_indices:
                    pose.append({"fraction":None,"label":"unknown","sample_count":0})
                    continue
                if not (1<=uv[idx,0]<=WIDTH-2 and 1<=uv[idx,1]<=HEIGHT-2):
                    pose.append({"fraction":0.0,"label":"out_of_frame","sample_count":len(v_indices)})
                    continue
                found=0
                for v_idx in v_indices:
                    target=verts[v_idx]
                    ray=target-origin
                    length=ray.length
                    if length<1e-5:continue
                    ray.normalize()
                    hit,normal,face,distance=bvh.ray_cast(origin,ray,length+SURFACE_HIT_TOLERANCE_M)
                    if hit is None or distance is None or abs(distance-length)<SURFACE_HIT_TOLERANCE_M:
                        found+=1
                frac=found/len(v_indices)
                if frac>=SURFACE_VISIBLE_THRESHOLD:
                    label="surface_visible"
                elif frac<=SURFACE_OCCLUDED_THRESHOLD:
                    label="surface_occluded"
                else:
                    label="uncertain"
                pose.append({"fraction":round(frac,4),"label":label,"sample_count":len(v_indices)})
            visible[slot]=pose
        return visible
    finally:
        obj.to_mesh_clear()

def make_trajectory(arm,rest,names,meta):
    xyz=[]
    flex=[]
    for frame in range(TOTAL_FRAMES):
        original=prior.pose_changes
        try:
            prior.pose_changes=movement.pose_changes
            prior.apply_pose(arm,rest,names,meta["family"],meta["seed"],frame)
        finally:
            prior.pose_changes=original
        xyz.append(prior.joints_world(arm,names))
        flex.append(tuple(float(arm.pose.bones[f"CC_Base_{side}_Index1"].rotation_euler[0])
                          for side in ("L","R")))
    xyz=np.asarray(xyz,dtype=np.float64)
    ix={n:i for i,n in enumerate(names)}
    finger=[ix[n] for n in movement.finger_tip_names()]
    legs=[i for i,name in enumerate(names) if any(x in name for x in ("Thigh","Calf","Foot","ToeBase","ToeTip"))]
    hand=[ix[f"CC_Base_{side}_Hand"] for side in ("L","R")]
    moves=lambda inds:float(np.max(np.linalg.norm(xyz[:,inds]-xyz[0:1,inds],axis=-1)))
    finger_ranges=[]
    for side in ("L","R"):
        wrist=ix[f"CC_Base_{side}_Hand"]
        for digit in movement.FINGER_NAMES:
            tip=ix[f"CC_Base_{side}_{digit}Tip"]
            length=np.linalg.norm(xyz[:,tip]-xyz[:,wrist],axis=-1)
            finger_ranges.append(float(np.ptp(length)))
    audit={
        "leg_max_travel_m":moves(legs),
        "hand_max_travel_m":moves(hand),
        "finger_tip_max_travel_m":moves(finger),
        "smallest_finger_radius_change_m":min(finger_ranges),
        "index_flex_range_rad":[float(np.ptp(np.asarray(flex)[:,i])) for i in (0,1)],
        "min_toe_world_z_m":float(np.min(xyz[:,[ix[n] for n in movement.toe_tip_names()],2])),
    }
    if (audit["leg_max_travel_m"]<.025 or audit["hand_max_travel_m"]<.03
        or audit["smallest_finger_radius_change_m"]<.001
        or min(audit["index_flex_range_rad"])<.04
        or audit["min_toe_world_z_m"]<-.01):
        raise RuntimeError("V2 clip failed whole-body/finger audit: "+str(meta)+str(audit))
    return xyz,audit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--family-index",type=int,default=-1,
                    help="0..11 to generate one family; -1 generates all")
    a=ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if len(SELECTED_FRAMES)!=14 or len(set(SELECTED_FRAMES))!=14:
        raise AssertionError("Invalid unique keyframe selection")
    cliplist=clip_defs(a.smoke)
    if a.family_index!=-1:
        if not 0<=a.family_index<len(movement.FAMILIES):raise ValueError("Invalid family index")
        cliplist=[x for x in cliplist if x["family"]==movement.FAMILIES[a.family_index]]
    if not cliplist:raise ValueError("No clips selected")
    folder=OUT/("smoke" if a.smoke else "dataset")
    arm,mesh,rest,scene,cameras=studio()
    names=marker_names(arm)
    patches=candidate_skin_vertices(mesh,arm,names)
    start=time.monotonic()
    samples=0
    rows=defaultdict(list)
    local_summary=[]
    for clip_i,cd in enumerate(cliplist):
        gt_file=folder/"motion_gt"/(cd["clip"]+".json")
        if gt_file.is_file():
            old=json.loads(gt_file.read_text(encoding="utf-8"))
            if old["clip"]!=cd or old["joint_names"]!=names or old["source_blend_sha256"]!=SHA256:
                raise RuntimeError("Existing GT conflicts with V2 contract")
            xyz=np.asarray(old["joint_position_m"],dtype=np.float64)
            audit=old["motion_quality"]
        else:
            xyz,audit=make_trajectory(arm,rest,names,cd)
            atomic_json(gt_file,{
                "schema":"anymove.human_v3_nose_hrv2_motion_gt/1",
                "source_blend_sha256":SHA256,"clip":cd,"fps":FPS,
                "frames":TOTAL_FRAMES,"joint_names":names,
                "joint_position_m":np.round(xyz,7).tolist(),
                "motion_quality":audit,
                "note":"Procedural pose GT, not verified natural human gait",
            })
        if xyz.shape!=(TOTAL_FRAMES,67,3):
            raise RuntimeError("Bad saved 3D motion GT shape")
        local_summary.append({"clip":cd["clip"],"family":cd["family"],"split":cd["split"],**audit})
        for frame in ((0,53,106) if a.smoke else SELECTED_FRAMES):
            original=prior.pose_changes
            try:
                prior.pose_changes=movement.pose_changes
                prior.apply_pose(arm,rest,names,cd["family"],cd["seed"],frame)
            finally:
                prior.pose_changes=original
            actual=prior.joints_world(arm,names)
            if np.max(np.linalg.norm(actual-xyz[frame],axis=-1))>5e-6:
                raise RuntimeError("GT motion replay mismatch: "+cd["clip"]+" "+str(frame))
            projections={slot:projection(xyz[frame],angle) for slot,angle in ANGLES.items()}
            for slot,cam in cameras.items():
                uv=projections[slot]
                good=(uv[:,0]>=1)&(uv[:,0]<WIDTH-1)&(uv[:,1]>=1)&(uv[:,1]<HEIGHT-1)
                if not bool(np.all(good)):
                    bad=[names[i] for i in np.flatnonzero(~good)]
                    raise RuntimeError("Clipped V2 keypoints "+cd["clip"]+" "+str(frame)+":"+str(bad))
                # Independently compare actual Blender camera's view matrix.
                for j in (0,4,11,16,30,42,56,63,66):
                    b=world_to_camera_view(scene,cam,Vector(xyz[frame,j]))
                    project=np.array((b.x*WIDTH,(1-b.y)*HEIGHT))
                    if np.max(np.abs(project-uv[j]))>.003:
                        raise RuntimeError("Actual camera extrinsics differ from UV: "+str(slot))
            vis=visibility_by_view(mesh,scene,cameras,patches,projections)
            for slot,cam in cameras.items():
                angle=ANGLES[slot]
                dest=folder/cd["split"]/slot/(cd["clip"]+"_f"+str(frame).zfill(3)+".png")
                dest.parent.mkdir(parents=True,exist_ok=True)
                if png_size(dest)!=(WIDTH,HEIGHT):
                    scene.camera=cam
                    scene.render.filepath=str(dest)
                    bpy.ops.render.render(write_still=True)
                if png_size(dest)!=(WIDTH,HEIGHT):
                    raise RuntimeError("V2 render corrupted: "+str(dest))
                rows[cd["split"]].append({
                    "clip":cd["clip"],"split":cd["split"],
                    "family":cd["family"],"frame":frame,
                    "slot":slot,"camera_azimuth_deg":angle,
                    "file":str(dest.relative_to(folder)).replace("\\","/"),
                    "uv":np.round(projections[slot],4).tolist(),
                    "visibility_proxy":vis[slot],
                })
                samples+=1
        # Incremental clip audit: can be resumed without re-rendering.
        clip_report=folder/"clip_reports"/(cd["clip"]+".json")
        atomic_json(clip_report,{"schema":"anymove.hr_v2_clip_report/1",
            "source_blend_sha256":SHA256,
            "clip":cd,"motion_quality":audit,
            "images":len(((0,53,106) if a.smoke else SELECTED_FRAMES))*2,
            "render_height":HEIGHT,"render_width":WIDTH})
        print("HRV2_CLIP_COMPLETE",clip_i+1,len(cliplist),cd["split"],cd["family"],
              cd["seed"],"images_so_far",samples,
              "elapsed_s",round(time.monotonic()-start,1),flush=True)
    # Atomic manifest per family batch; after all 12 batches are produced,
    # a separate independent audit assembles one global manifest.
    run_manifest={
        "schema":"anymove.human_v3_nose_hr_v2_family_batch/1",
        "state":"COMPLETE",
        "source_blend_sha256":SHA256,
        "camera_angles_deg":[315,45],
        "slot_to_angle":ANGLES,
        "render_size":[WIDTH,HEIGHT],
        "metres_per_pixel":METRES_PER_PIXEL,
        "z_center":Z_CENTER,
        "frames_per_clip":TOTAL_FRAMES,
        "sampled_frames":list((0,53,106) if a.smoke else SELECTED_FRAMES),
        "total_clips":len(cliplist),
        "joint_names":names,
        "visibility_method":"skinned dominant-vertex regions, sampled rays against deformed full mesh BVH; surface proxy only",
        "visibility_thresholds":{"visible":SURFACE_VISIBLE_THRESHOLD,"occluded":SURFACE_OCCLUDED_THRESHOLD},
        "visibility_missing_labels":[name for name,ids in patches.items() if not ids],
        "rows":rows,
        "clip_audits":local_summary,
    }
    report_path=folder/("smoke_manifest.json" if a.smoke else f"family_{a.family_index:02d}_manifest.json")
    atomic_json(report_path,run_manifest)
    print("HRV2_BATCH_PASS",json.dumps({
        "clips":len(cliplist),"images":samples,"total_pose_pairs":samples//2,
        "output":str(report_path),"elapsed_s":round(time.monotonic()-start,1),
        "source_model":str(SOURCE)},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
