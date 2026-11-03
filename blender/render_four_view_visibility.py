"""Replay verified existing Blender pose metadata and render four-view mesh visibility GT.

Usage:
blender -b -P blender/render_four_view_visibility.py -- \
  --blend assets/dino/rigged/rigged.blend \
  --rig assets/dino/rigged/rig_info.json \
  --source-dataset runs/dino/dataset_hr_v1 \
  --indices 0,1 --out runs/dino/four_view_visibility_smoke \
  --scale 0.5 --render-stills

Only STILL images (optional) and rasterized NPZ/JSON labels. No H3 generation,
no animation videos, no dataset retraining. Exact pose replay validated against
source per-frame 3D joints.

Default appearance: source-material (Cycles), preserving original creature
textures and color-coded left/right limbs and feet for Reader training.
Optional --appearance neutral-geometry retains the original gray Workbench
render ONLY for pure geometric Part-ID/Depth debugging. No edits to original Reader dataset.
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys
import time
import shutil

import bpy
import numpy as np
from mathutils import Euler,Vector

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"blender"))
from mocap.surface_visibility import (
    dominant_triangle_parts,rasterize_camera_mesh,part_visibility_targets)
import common


PROFILE="anymove.four_view_surface_part_gt/1"


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def replay_pose(rig,meta,info):
    """Validate Blender FK reproduces the 3D ground truth of the original pose."""
    if "pose" not in meta or not isinstance(meta.get("pose"),dict):
        raise ValueError("dataset metadata lacks replayable pose")
    bpy.context.scene.frame_set(1)
    rig.animation_data_clear()
    all_bones={b["name"] for b in info["bones"]}
    if set(meta["pose"])!=all_bones:
        raise ValueError("source pose bones mismatch target rig")
    for pb in rig.pose.bones:
        pb.rotation_mode="XYZ"
        pb.location=(0,0,0)
        if pb.name in all_bones:
            pb.rotation_euler=Euler(meta["pose"][pb.name],"XYZ")
    rig.rotation_mode="XYZ"
    rig.rotation_euler=(0,0,float(meta["yaw"]))
    rig.location=Vector(meta["root"])
    bpy.context.view_layer.update()
    mw=rig.matrix_world
    result=np.array([list(mw@(rig.pose.bones[info["bones"][j["bone"]]["name"]].head
                         if j["end"]=="head" else
                         rig.pose.bones[info["bones"][j["bone"]]["name"]].tail))
                    for j in info["keypoints"]],dtype=float)
    original=np.asarray(meta["joints_world"],dtype=float)
    if result.shape!=original.shape:
        raise ValueError("source GT joint count changed")
    deviation=np.linalg.norm(result-original,axis=-1)
    error=float(np.max(deviation))
    if error>.001:
        raise ValueError(f"source dataset poses cannot be replayed: {error:.6g} m")
    return result,error


def create_cameras(scene):
    """Keep current Front and Side; add the exact 180�X orthographic opposites."""
    current=common.cameras(scene)
    views={"front":current["cam_front"],"side":current["cam_side"]}
    zc=common.Z_CENTER
    for name,loc,width in [
            ("back",(0.,20.,zc),common.W_FRONT),
            ("opposite_side",(20.,0.,zc),common.W_SIDE)]:
        obj=bpy.data.objects.get("cam_"+name)
        if obj is None:
            obj=bpy.data.objects.new("cam_"+name,bpy.data.cameras.new("cam_"+name))
            scene.collection.objects.link(obj)
        obj.location=loc
        target=Vector((0.,0.,zc))
        obj.rotation_euler=(target-obj.location).to_track_quat("-Z","Y").to_euler()
        obj.data.type="ORTHO"
        obj.data.ortho_scale=common.M_PER_PX*max(width,common.H)
        obj.data.clip_end=100
        views[name]=(obj,width)
    bpy.context.view_layer.update()
    return views


def pixel_projection(world_xyz,cam,width,height,meters_per_px):
    inv=np.asarray(cam.matrix_world.inverted(),dtype=np.float64)
    pts=np.asarray(world_xyz,dtype=np.float64)
    local=pts@inv[:3,:3].T+inv[:3,3]
    uv=np.stack([width/2+local[:,0]/meters_per_px,
                 height/2-local[:,1]/meters_per_px],axis=-1)
    return uv,local,inv


def extract_mesh(body,rig_names):
    """Evaluated deformed mesh and per-vertex skinning weights in its current pose."""
    deps=bpy.context.evaluated_depsgraph_get()
    obj=body.evaluated_get(deps)
    mesh=obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        src_groups={g.index:g.name for g in body.vertex_groups}
        name_to_id={n:i for i,n in enumerate(rig_names)}
        weights=np.zeros((len(mesh.vertices),len(rig_names)),dtype=np.float32)
        for vert in mesh.vertices:
            for group in vert.groups:
                name=src_groups.get(group.group)
                idx=name_to_id.get(name)
                if idx is not None:
                    weights[vert.index,idx]+=group.weight
        raw=np.empty(len(mesh.vertices)*3,dtype=np.float64)
        mesh.vertices.foreach_get("co",raw)
        raw=raw.reshape(-1,3)
        world=np.asarray(body.matrix_world,dtype=np.float64)
        world_verts=raw@world[:3,:3].T+world[:3,3]
        triangles=np.array([list(x.vertices) for x in mesh.loop_triangles],dtype=np.int32)
        face_parts,stats=dominant_triangle_parts(weights,triangles)
        return world_verts,triangles,face_parts,stats
    finally:
        obj.to_mesh_clear()


def configure_render_appearance(scene,appearance,samples):
    """Reader training must preserve textured/painted body colours.

    Workbench SINGLE was intentionally neutral for geometry P1, but must
    NOT silently be used for four-view Reader training: it removes the
    source texture's L/R limb and toe colour-code information.
    """
    if appearance == "source-material":
        common.studio(scene,samples=samples)
        scene.render.engine="CYCLES"
        scene.render.use_motion_blur=False
    elif appearance == "neutral-geometry":
        scene.render.engine="BLENDER_WORKBENCH"
        scene.display.shading.light="STUDIO"
        scene.display.shading.color_type="SINGLE"
        scene.display.shading.single_color=(.58,.66,.73)
    else:
        raise ValueError("unsupported appearance mode")
    scene.render.film_transparent=True
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGBA"
    scene.render.resolution_percentage=100


def static_still_render(scene,cam,w,h,path):
    """Render ONE original-material or neutral still. Never render video."""
    scene.render.resolution_x=w
    scene.render.resolution_y=h
    scene.camera=cam
    scene.render.filepath=str(path)
    path.unlink(missing_ok=True)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size<512:
        raise RuntimeError("empty still: "+str(path))


def verify_camera_opposites(projections,width_front,width_side):
    f=projections["front"];b=projections["back"]
    s=projections["side"];o=projections["opposite_side"]
    # the two opposite camera pairs have matching image V and mirrored U
    checks={
        "front_back_horizontal_max_abs_px":float(np.max(np.abs(f[:,0]+b[:,0]-width_front))),
        "front_back_vertical_max_abs_px":float(np.max(np.abs(f[:,1]-b[:,1]))),
        "side_opposite_horizontal_max_abs_px":float(np.max(np.abs(s[:,0]+o[:,0]-width_side))),
        "side_opposite_vertical_max_abs_px":float(np.max(np.abs(s[:,1]-o[:,1]))),
    }
    if max(checks.values())>.02:
        raise ValueError("opposite-view camera orientation not geometrically consistent: "+str(checks))
    return checks


def raycast_validate(scene,camera,raster,*,m_per_pixel,seed,num_checks=80):
    """Independently compare raster depth with Blender's evaluated scene Raycast."""
    from mathutils import Vector
    dep=bpy.context.evaluated_depsgraph_get()
    ids=np.argwhere(np.isfinite(raster["depth_m"]))
    if not len(ids):raise ValueError("mesh has zero visible surface pixels")
    rng=np.random.default_rng(seed)
    # Avoid silhouette edges: stable local 3x3 same part/depth neighbourhood.
    part=raster["part_id"]
    if len(ids)>0:
        stable=np.ones(part.shape,bool)
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            valid=np.roll(part,shift=(dy,dx),axis=(0,1))==part
            stable&=valid
        ids=np.argwhere(np.isfinite(raster["depth_m"])&stable)
    if not len(ids):raise ValueError("no stable mesh depth pixels for raycast QA")
    chosen=ids[rng.choice(len(ids),size=min(num_checks,len(ids)),replace=False)]
    cammat=camera.matrix_world
    direct=cammat.to_3x3()@Vector((0,0,-1))
    origin_z=0
    errors=[]
    misses=0
    hit_names={}
    for y,x in chosen:
        ox=(float(x)+.5-raster["depth_m"].shape[1]/2)*m_per_pixel
        oy=(raster["depth_m"].shape[0]/2-float(y)-.5)*m_per_pixel
        origin=cammat@Vector((ox,oy,origin_z))
        hit,location,normal,poly,ob,matrix=scene.ray_cast(dep,origin,direct,distance=100.)
        if not hit:
            misses+=1
            continue
        hit_names[ob.name]=hit_names.get(ob.name,0)+1
        ray_depth=float((location-origin).dot(direct))
        errors.append(abs(ray_depth-float(raster["depth_m"][y,x])))
    if not errors:
        raise AssertionError("no depth rays hit geometry")
    errors=np.asarray(errors)
    report={"sample_count":int(len(chosen)),"ray_hit_count":len(errors),
            "misses":misses,"depth_error_m_median":round(float(np.median(errors)),6),
            "depth_error_m_p95":round(float(np.percentile(errors,95)),6),
            "depth_error_m_max":round(float(np.max(errors)),6),
            "hit_objects":hit_names}
    # One or two AA/silhouette samples can deviate; interior should agree
    # well with direct geometric ray casting (not a trained model).
    if misses>max(2,int(len(chosen)*.06)) or report["depth_error_m_p95"]>.03:
        raise AssertionError("software depth does not agree with Blender raycast: "+str(report))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--blend",required=True)
    ap.add_argument("--rig",required=True)
    ap.add_argument("--source-dataset",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--indices",default="0")
    ap.add_argument("--scale",type=float,default=.5)
    ap.add_argument("--render-stills",action="store_true")
    ap.add_argument('--resume',action='store_true',help='Skip only fully verified existing samples in this dedicated output')
    ap.add_argument('--reuse-source-front-side',action='store_true',help='Use already rendered source Front/Side PNG for unblurred source pose; back/opposite freshly rendered')
    ap.add_argument("--appearance",choices=("source-material","neutral-geometry"),
                    default="source-material",
                    help="Use source textures/leg paint for Reader training; neutral is diagnostic only")
    ap.add_argument("--samples",type=int,default=16,
                    help="Cycles samples when appearance=source-material")
    ap.add_argument('--color20-palette',default='',
                    help='Apply 20 real Bone argmax-SkinWeight colours to Dino, retaining uncoloured original texture')
    ap.add_argument('--viewaware-palette',default='',
                    help='Apply arbitrary number of effective Dino SkinMax colors optimized for real Front/Side camera visual proximity')
    args=ap.parse_args(sys.argv[sys.argv.index("--")+1:])
    if args.samples < 1 or args.samples > 128:
        raise ValueError("Cycles render samples must be 1..128")
    if not np.isfinite(args.scale) or not 0<args.scale<=2:
        raise ValueError("invalid render scale")
    indices=[int(x) for x in args.indices.split(",")]
    if len(indices)!=len(set(indices)) or min(indices)<0:
        raise ValueError("duplicate/negative sample indices")
    blend=pathlib.Path(args.blend).resolve()
    rig_info=pathlib.Path(args.rig).resolve()
    source=pathlib.Path(args.source_dataset).resolve()
    out=pathlib.Path(args.out).resolve()
    if out==source or str(out).startswith(str(source)+str(pathlib.os.sep)):
        raise ValueError("visibility outputs cannot overwrite existing Reader dataset")
    if not out.exists():out.mkdir(parents=True)
    if list(out.iterdir()) and not args.resume:
        raise FileExistsError('output directory is not empty; use --resume only for the same documented run')
    if args.reuse_source_front_side and (not args.render_stills or args.scale!=1.0 or args.appearance!='source-material'):
        raise ValueError('source Front/Side reuse requires source-material 1.0 scale and render-stills')
    if args.color20_palette and (not args.render_stills or args.appearance!='source-material' or args.reuse_source_front_side):
        raise ValueError('Dino color20 requires new source-material still rendering for every camera; cannot reuse unpainted source images')
    if args.viewaware_palette and (args.color20_palette or not args.render_stills or args.appearance!='source-material' or args.reuse_source_front_side):
        raise ValueError('view-aware color requires exclusive source-material RGB stills with all cameras freshly rendered')
    info=json.loads(rig_info.read_text(encoding="utf8"))
    bones=[b["name"] for b in info["bones"]]
    if len(bones)!=len(set(bones)) or not isinstance(info.get("layout"),dict):
        raise ValueError("invalid rig_info")
    layout=info["layout"]
    for k in ("W_FRONT","W_SIDE","M_PER_PX","H","Z_CENTER"):
        if abs(float(layout[k])-float(getattr(common,k)))>1e-6:
            raise ValueError("MOCAP_CREATURE or camera layout does not match Rig: "+k)

    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene=bpy.context.scene
    rig,body=common.skinned_character_objects()
    if set(bones)!=set(pb.name for pb in rig.pose.bones):
        raise ValueError("rig_info bone names mismatch Blender armature")
    color_audit=None
    viewaware_audit=None
    if args.color20_palette:
        from dino_skinmax_color20 import apply as apply_skinmax_color20
        color_audit=apply_skinmax_color20(body,info,args.color20_palette,original_blend_path=blend)
        (out/'color_assignment_audit.json').write_text(json.dumps(color_audit,indent=2,ensure_ascii=False),encoding='utf8')
        print('SKINMAX_COLOR20_APPLIED',color_audit['selected_vertex_count'],'of',color_audit['total_mesh_vertices'],flush=True)
    if args.viewaware_palette:
        from dino_skinmax_camera import apply as apply_viewaware
        viewaware_audit=apply_viewaware(body,info,args.viewaware_palette,source_blend=blend)
        (out/'viewaware_color_audit.json').write_text(json.dumps(viewaware_audit,indent=2,ensure_ascii=False),encoding='utf8')
        print('SKINMAX_CAMERA_AWARE_APPLIED',viewaware_audit['colour_count'],'classes',viewaware_audit['coloured_vertex_count'],'verts',flush=True)
    cameras=create_cameras(scene)
    if args.render_stills:
        configure_render_appearance(scene,args.appearance,args.samples)
    blend_sha=sha256(blend);rig_sha=sha256(rig_info)
    start=time.perf_counter()
    for index in indices:
        meta_path=source/f'{index:05d}_meta.json'
        meta=json.loads(meta_path.read_text(encoding='utf8'))
        completed_manifest=out/f'{index:05d}_visibility_meta.json'
        if args.resume and completed_manifest.is_file():
            old=json.loads(completed_manifest.read_text(encoding='utf8'))
            if (old.get('source_meta_sha256')!=sha256(meta_path) or
                    old.get('rig_info_sha256')!=rig_sha or
                    old.get('blend_sha256')!=blend_sha or
                    old.get('appearance_mode')!=args.appearance or
                    old.get('source_front_side_reused',False)!=bool(args.reuse_source_front_side) or
                    old.get('render_scale')!=args.scale or
                    old.get('color20_palette_sha256')!=(sha256(args.color20_palette) if args.color20_palette else None) or
                    old.get('viewaware_palette_sha256')!=(sha256(args.viewaware_palette) if args.viewaware_palette else None) or
                    any(not (out/f'{index:05d}_{v}_labels.npz').is_file() or
                            args.render_stills and not (out/f'{index:05d}_{v}.png').is_file()
                            for v in cameras)):
                raise RuntimeError('resume manifest changed/incomplete: '+str(completed_manifest))
            print('RESUME_SKIP',index,flush=True)
            continue
        if args.reuse_source_front_side and bool(meta.get('blur')):
            raise ValueError('cannot reuse motion-blurred Front/Side as exact static-pose images')
        jw,pose_error=replay_pose(rig,meta,info)
        world,triangles,partids,mesh_stats=extract_mesh(body,bones)
        all_uv={}
        view_results={}
        for v,(cam,source_width) in cameras.items():
            w=int(round(source_width*args.scale))
            h=int(round(common.H*args.scale))
            m=common.M_PER_PX/args.scale
            uv,cam_joint,cam_inverse=pixel_projection(jw,cam,w,h,m)
            all_uv[v]=uv
            calib_error=None
            if v in ("front","side"):
                source_uv=np.asarray(meta[v+"_uv"],float)*args.scale
                calib_error=float(np.linalg.norm(uv-source_uv,axis=1).max())
                if calib_error>.1:
                    raise ValueError(v+f" camera disagrees with source 2D GT by {calib_error:.5f}px")
            cam_verts=world@cam_inverse[:3,:3].T+cam_inverse[:3,3]
            t0=time.perf_counter()
            data=rasterize_camera_mesh(cam_verts,triangles,partids,
                width=w,height=h,meters_per_px=m,n_parts=len(bones),
                near=float(cam.data.clip_start),far=float(cam.data.clip_end))
            targets=part_visibility_targets(data,np.array([j["bone"] for j in info["keypoints"]]))
            if data["total_surface_pixels"]<50:
                raise AssertionError(v+" camera has no visible mesh")
            ray=raycast_validate(scene,cam,data,m_per_pixel=m,seed=10000+index)
            suffix=out/f"{index:05d}_{v}"
            np.savez_compressed(str(suffix)+"_labels.npz",
                part_id=data["part_id"],
                depth_m=data["depth_m"],
                amodal_pixel_count=data["amodal_pixel_count"],
                visible_pixel_count=data["visible_pixel_count"],
                surface_visible_fraction=data["surface_visible_fraction"],
                joint_surface_part_visible=targets,
                joints_uv=uv.astype(np.float32))
            if args.render_stills:
                image_path=pathlib.Path(str(suffix)+'.png')
                if args.reuse_source_front_side and v in ('front','side'):
                    original=source/f'{index:05d}_{v}.png'
                    if not original.is_file():
                        raise FileNotFoundError('missing original source image '+str(original))
                    # Original source PNG is the actual Cycles texture render;
                    # never link or rewrite the original data.
                    import struct
                    with original.open('rb') as handle:
                        header=handle.read(26)
                    if (len(header)<26 or header[:8]!=bytes((137,80,78,71,13,10,26,10)) or
                            struct.unpack('>II',header[16:24])!=(w,h) or header[25]!=6):
                        raise ValueError('source view PNG mismatch or not RGBA: '+str(original))
                    shutil.copy2(original,image_path)
                else:
                    static_still_render(scene,cam,w,h,image_path)
            view_results[v]={
                "image_size_px":[w,h],"meters_per_pixel":m,
                "camera_world_to_local":cam_inverse.tolist(),
                "joints_projection_source_max_error_px":calib_error,
                "surface_pixels":data["total_surface_pixels"],
                "labeled_surface_pixels":data["labeled_surface_pixels"],
                "ambiguous_surface_pixels":data["ambiguous_surface_pixels"],
                "rasterized_triangles":data["rasterized_triangles"],
                "clipped_triangles":data["near_far_clipped_triangles"],
                "known_joint_surface_labels":int(np.count_nonzero(targets>=0)),
                "visible_joint_surface_labels":int(np.count_nonzero(targets==1)),
                "occluded_joint_surface_labels":int(np.count_nonzero(targets==0)),
                "unknown_joint_surface_labels":int(np.count_nonzero(targets==-1)),
                "raycast_validation":ray,
                "raster_seconds":round(time.perf_counter()-t0,3),
                "label_file":str(suffix)+"_labels.npz",
                "render_file":str(suffix)+".png" if args.render_stills else None,
                "joint_surface_part_visibility":targets.tolist(),
            }
            print(f"sample={index} view={v} pixels={data['total_surface_pixels']} "
                  f"labels={len(targets)-int(np.sum(targets<0))}/{len(targets)} "
                  f"occluded={int(np.count_nonzero(targets==0))} "
                  f"ray_p95={ray['depth_error_m_p95']}m",flush=True)
        geometry=verify_camera_opposites(all_uv,
            round(common.W_FRONT*args.scale),round(common.W_SIDE*args.scale))
        manifest={
            "schema_version":PROFILE,"sample_index":index,
            "input_blend":str(blend),"source_pose_metadata":str(meta_path),
            "source_meta_sha256":sha256(meta_path),"rig_info_sha256":rig_sha,
            "blend_sha256":blend_sha,"source_blur":bool(meta.get("blur",False)),
            "source_framing_signature":meta.get("framing_signature"),
            "replay_max_joint_world_error_m":pose_error,
            "camera_opposite_geometry_checks":geometry,
            "surface_part_semantics":"Triangle ID from dominant Skinning Weight, visible mask is nearest rasterized mesh surface; per-Joint target is owning surface-part visibility, NOT raycast visibility of skeleton center.",
            "unknown_label":-1,"visible_label":1,"occluded_label":0,
            "minimum_part_projected_area_px":12,
            "visibility_threshold_fraction":0.025,
            "no_new_motion_or_h3_video":True,
            "appearance_mode":args.appearance,
            "source_material_colors_preserved":bool(args.render_stills and args.appearance=="source-material"),
            "source_front_side_reused":bool(args.reuse_source_front_side),
            "source_front_side_lighting_note":('Reused original randomized source lighting for Front/Side; Back/Opp use the fixed studio lighting; not identical lighting across cameras' if args.reuse_source_front_side else None),
            "render_scale":args.scale,
            "render_samples":args.samples if args.render_stills and args.appearance=="source-material" else None,
            "color20_applied":bool(color_audit),
            "color20_palette_sha256":color_audit['palette_sha256'] if color_audit else None,
            "viewaware_color_applied":bool(viewaware_audit),
            "viewaware_color_count":viewaware_audit['colour_count'] if viewaware_audit else None,
            "viewaware_palette_sha256":viewaware_audit['palette_sha256'] if viewaware_audit else None,
            "no_motion_blur_in_gt":"Replayed one static pose, independent of original possible blurred render.",
            "mesh":mesh_stats,"views":view_results,
        }
        (out/f"{index:05d}_visibility_meta.json").write_text(
            json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print("FOUR_VIEW_VISIBILITY_COMPLETE",len(indices),
          "elapsed_seconds",round(time.perf_counter()-start,3),"output",out,flush=True)


if __name__=="__main__":
    main()
