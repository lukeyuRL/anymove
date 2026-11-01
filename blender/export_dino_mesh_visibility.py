"""Evaluate *actual deformed Blender mesh* between camera and Dino keypoint.

Usage (already keyed animation.blend, no re-render):
blender -b PATH/gt_render/animation.blend -P blender/export_dino_mesh_visibility.py -- \
    EXACT_RIG_INFO.json EXACT_GT_RENDER/joints.json OUTPUT_GEOMETRY.npz

Bone endpoints are generally INSIDE the body. A single first skin hit is
NOT sufficient to call a joint occluded. Count intersected mesh surfaces
from an orthographic source through the keypoint, and mark 2+ crossings
as intervening geometry. This is a conservative line-of-sight PROXY,
not expert-annotated keypoint visibility / true occlusion probability.

Crucially, Tail08 is ~1m outside the model mesh for this Dino. If the
nearest skinned surface is farther than 0.5m, do not infer visibility
from skin geometry: mark geometry support UNKNOWN; never silently
interpret it as a reliable occlusion observation.
"""
import hashlib
import json
import pathlib
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common

PROFILE="anymove.blender_dino_mesh_surface_crossing_proxy_v1"


def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,"rb") as fd:
        for block in iter(lambda:fd.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def surfaces_before_joint(bvh, origin, target, *, max_surfaces=8, epsilon=0.005):
    """Cast in OBJECT local space from orthographic source to the 3D point."""
    ray=target-origin
    length=ray.length
    if length <= 0.001:
        raise ValueError("invalid near-zero occlusion test ray")
    direction=ray/length
    current=origin.copy()
    remaining=length
    crossings=0
    for _ in range(max_surfaces):
        # Exclude a surface exactly at the joint point.
        if remaining <= 0.003:
            break
        hit,normal,polygon,dist=bvh.ray_cast(current,direction,remaining-0.003)
        if hit is None:
            break
        crossings+=1
        advance=max(float(dist),0.0)+epsilon
        current+=direction*advance
        remaining-=advance
    return crossings


def compute(arm, mesh, info, gt, *, max_proxy_distance_m=0.5):
    scene=bpy.context.scene
    cams=common.cameras(scene)
    if len(info["keypoints"])!=92 or len(gt["names"])!=92:
        raise ValueError("Dino experiment expects exactly 92 rig/GT joints")
    names=[k["name"] for k in info["keypoints"]]
    if names!=gt["names"]:
        raise ValueError("rig order does not match independent Blender 3D GT")
    truth=np.asarray(gt["joints"],np.float64)
    T,J,_=truth.shape
    if T<3 or truth.shape[1:]!=(92,3) or not np.isfinite(truth).all():
        raise ValueError("invalid full-frame Blender GT joint tensor")
    if max_proxy_distance_m<=0 or not np.isfinite(max_proxy_distance_m):
        raise ValueError("invalid geometric support distance")
    counts={v:np.zeros((T,J),np.uint8) for v in ("front","side")}
    supported={v:np.zeros((T,J),bool) for v in ("front","side")}
    nearest_distance=np.empty((T,J),np.float32)
    # Rest pose and animated GT must use the same evaluated Armature.
    for t in range(T):
        scene.frame_set(t)
        depsgraph=bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        bvh=BVHTree.FromObject(mesh,depsgraph,deform=True)
        if bvh is None:
            raise RuntimeError("failed to build evaluated skin BVH")
        local_from_world=mesh.matrix_world.inverted()
        points=[]
        for kp in info["keypoints"]:
            bone=info["bones"][kp["bone"]]["name"]
            pb=arm.pose.bones[bone]
            world=arm.matrix_world @ (pb.head if kp["end"]=="head" else pb.tail)
            points.append(world)
        actual=np.array([tuple(p) for p in points])
        if not np.allclose(actual,truth[t],rtol=0,atol=0.0003):
            raise ValueError(
                "Blender animated FK joint mismatch at frame %d, max %.6f m"
                %(t,np.abs(actual-truth[t]).max()))
        locals_=[local_from_world@p for p in points]
        for j,local in enumerate(locals_):
            location,normal,face,d=bvh.find_nearest(local)
            if location is None:
                raise RuntimeError("evaluated mesh has no nearest surface")
            nearest_world=mesh.matrix_world@location
            nearest_distance[t,j]=(points[j]-nearest_world).length
        # The exact orthographic rays are reconstructed independently of
        # Reader confidence, 2D detections and predicted errors.
        scene.view_layers[0].update()
        for view in ("front","side"):
            cam=cams["cam_"+view][0]
            dir_world=(cam.matrix_world.to_3x3()@Vector((0,0,-1))).normalized()
            for j,point in enumerate(points):
                start=local_from_world @ (point-dir_world*60.0)
                count=surfaces_before_joint(bvh,start,locals_[j])
                counts[view][t,j]=min(count,255)
                supported[view][t,j]=nearest_distance[t,j] <= max_proxy_distance_m
        if t%20==0 or t==T-1:
            print("mesh geometry frame",t+1,"/",T,flush=True)
    result={"names":np.asarray(names),
            "fps":np.asarray(float(gt["fps"])),
            "frame_count":np.asarray(T),
            "nearest_mesh_surface_distance_m":nearest_distance}
    for view in ("front","side"):
        result[view+"_surface_crossings"]=counts[view]
        result[view+"_mesh_support"]=supported[view]
        result[view+"_mesh_blocked_proxy"]=(counts[view]>=2) & supported[view]
    audit={
        "profile":PROFILE,"frames":T,"keypoints":J,
        "mesh_surface_distance_gate_m":float(max_proxy_distance_m),
        "blocked_definition":"at least TWO mesh surface crossings before keypoint, with skin proximity <= gate",
        "no_mesh_support":"Unknown, NOT visible or occluded; keep raw observation until a separate reliability test",
        "not_ground_truth":"Geometric line-of-sight proxy is not anatomical keypoint visibility. Internal joints often have one skin crossing; virtual external tips may have zero. No Ray label is based on known 2D error.",
        "views":{},
        "special_joints":{},
    }
    for view in ("front","side"):
        audit["views"][view]={
            "blocked_joint_frames":int(result[view+"_mesh_blocked_proxy"].sum()),
            "supported_joint_frames":int(supported[view].sum()),
            "unknown_joint_frames":int((~supported[view]).sum()),
            "crossing_histogram":{
                str(k):int(np.count_nonzero(counts[view]==k))
                for k in np.unique(counts[view])},
        }
    for fragment in ("Tail08","Tail07","LeftToe03","Tongue03","Hip"):
        ix=next((i for i,n in enumerate(names)
                 if fragment in n and n.endswith(":tail")),None)
        if ix is not None:
            audit["special_joints"][fragment]={
                "name":names[ix],
                "mean_distance_to_mesh_m":round(float(nearest_distance[:,ix].mean()),5),
                "front_blocked_frames":int(result["front_mesh_blocked_proxy"][:,ix].sum()),
                "side_blocked_frames":int(result["side_mesh_blocked_proxy"][:,ix].sum()),
                "unknown_frames":int((nearest_distance[:,ix]>max_proxy_distance_m).sum()),
            }
    return result,audit


def main():
    args=sys.argv[sys.argv.index("--")+1:]
    if len(args)!=3:
        raise SystemExit("usage: blender -b animation.blend -P blender/export_dino_mesh_visibility.py -- rig_info.json gt_joints.json NEW_geometry.npz")
    info_path,gt_path,out=map(pathlib.Path,args)
    audit_path=out.with_suffix(".mesh_geometry.json")
    if out.exists() or audit_path.exists():
        raise FileExistsError("refuse to overwrite Dino mesh visibility results")
    info=json.loads(info_path.read_text(encoding="utf-8"))
    gt=json.loads(gt_path.read_text(encoding="utf-8"))
    arm,mesh=common.skinned_character_objects()
    result,audit=compute(arm,mesh,info,gt)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(out),**result)
    audit["rig_info_sha256"]=sha256_file(info_path)
    audit["blender_truth_sha256"]=sha256_file(gt_path)
    audit["geometry_npz_sha256"]=sha256_file(out)
    audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2),flush=True)


if __name__=="__main__":
    main()
