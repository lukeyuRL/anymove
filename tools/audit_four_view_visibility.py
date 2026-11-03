"""Audit 4-view surface visibility labels, source provenance, and coverage.

This is an independent NumPy/OpenCV verifier. Optional contact sheet is a
STATIC PNG from already-rendered Blender images and structured Part-ID/depth;
never synthesizes a video and does not train the Reader.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from mocap.surface_visibility import part_visibility_targets

VIEWS=("front","side","back","opposite_side")
PROFILE="anymove.four_view_surface_part_gt/1"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()


def _colorize(label):
    label=np.asarray(label,dtype=np.int32)
    h,w=label.shape
    color=np.zeros((h,w,3),np.uint8)
    color[:]=(31,33,36)
    mask=label== -2
    color[mask]=(108,108,108)
    ids=np.unique(label[label>=0])
    for idx in ids:
        hue=int((int(idx)*137.507764)%180)
        hsv=np.uint8([[[hue,185,237]]])
        bgr=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)[0,0]
        color[label==idx]=bgr
    return color


def _alpha_rgb(img):
    if img.ndim!=3 or img.shape[2]!=4:
        raise ValueError("expected four-channel Blender RGBA still")
    a=img[:,:,3:4].astype(np.float32)/255
    return np.clip(img[:,:,:3]*a+(1-a)*np.array([32,32,36])[None,None,:],0,255).astype(np.uint8)


def _tile(view,raster_png,data,label_status,max_width=450):
    p=_alpha_rgb(raster_png)
    ids=data["part_id"]
    w0=p.shape[1];h0=p.shape[0]
    # Support source-resolution material renders (Side may be 880px wide)
    # as well as prior 0.5-scale geometry smoke images.
    w=max(max_width,w0);h=h0
    # Keep camera pixel geometry; pad rather than resize.
    left=(w-w0)//2
    body=np.full((h,w,3),(32,32,36),np.uint8)
    body[:,left:left+w0]=p
    uv=data["joints_uv"]
    for (u,v),label in zip(uv,label_status):
        x=int(round(float(u)))+left
        y=int(round(float(v)))
        if not (0<=x<w and 0<=y<h):
            continue
        if label==1:
            col=(70,213,99)
        elif label==0:
            col=(226,62,221)
        else:
            col=(155,155,155)
        cv2.circle(body,(x,y),2,col,-1,lineType=cv2.LINE_AA)
    part=np.full_like(body, (32,32,36))
    part[:,left:left+w0]=_colorize(ids)
    depth=data["depth_m"]
    valid=np.isfinite(depth)
    image=np.full((h,w0),0,np.uint8)
    if np.any(valid):
        lo,hi=np.percentile(depth[valid],[2,98])
        value=(depth[valid]-lo)/max(hi-lo,1e-6)
        image[valid]=(255-205*np.clip(value,0,1)).astype(np.uint8)
    heat=cv2.applyColorMap(image,cv2.COLORMAP_TURBO)
    heat[~valid]=(32,32,36)
    depth_rgb=np.full_like(body,(32,32,36))
    depth_rgb[:,left:left+w0]=heat
    header=np.full((36,w,3),(43,44,49),np.uint8)
    cv2.putText(header,view.replace("_"," ").upper(),(12,24),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1,cv2.LINE_AA)
    return np.concatenate((header,body,part,depth_rgb),axis=0)


def audit_one(out,idx,rig,source_meta,create_contact):
    p=out/f"{idx:05d}_visibility_meta.json"
    m=json.loads(p.read_text(encoding="utf8"))
    if m.get("schema_version")!=PROFILE or m.get("sample_index")!=idx:
        raise ValueError("wrong P1 four-view schema or sample ID")
    if set(m["views"])!=set(VIEWS):
        raise ValueError("missing/extra camera views")
    if m["source_meta_sha256"]!=sha(source_meta) or m["rig_info_sha256"]!=sha(rig):
        raise ValueError("source provenance hash mismatch")
    blend=Path(m["input_blend"])
    if not blend.is_file() or m["blend_sha256"]!=sha(blend):
        raise ValueError("Blender asset provenance hash mismatch")
    source_gt=np.asarray(json.loads(source_meta.read_text(encoding="utf8"))["joints_world"],dtype=float)
    if m["replay_max_joint_world_error_m"]>.001:
        raise ValueError("source 3D pose replay error >1mm")
    if any(v>.02 for v in m["camera_opposite_geometry_checks"].values()):
        raise ValueError("camera opposite geometry check not passed")
    rig_data=json.loads(rig.read_text(encoding="utf8"))
    bone=np.asarray([kp["bone"] for kp in rig_data["keypoints"]],np.int64)
    num_bones=len(rig_data["bones"])
    all_labels={}
    per_view={}
    preview=[]
    for v in VIEWS:
        f=out/f"{idx:05d}_{v}_labels.npz"
        png=out/f"{idx:05d}_{v}.png"
        if not f.is_file() or not png.is_file():
            raise FileNotFoundError("missing P1 four view artifact "+str(f))
        with np.load(f) as z:
            zdata={k:z[k] for k in z.files}
        detail=m["views"][v]
        width,height=detail["image_size_px"]
        if (zdata["part_id"].shape!=(height,width) or
                zdata["depth_m"].shape!=(height,width)):
            raise ValueError("raster file size mismatch "+v)
        if (len(zdata["amodal_pixel_count"])!=num_bones or
                len(zdata["visible_pixel_count"])!=num_bones or
                len(zdata["surface_visible_fraction"])!=num_bones or
                zdata["joints_uv"].shape!=(len(bone),2)):
            raise ValueError("raster bone/joint shape mismatch "+v)
        # Recompute 2D projections independently from original 3D source GT
        # and the stored per-view camera matrix; do not trust Blender's emitted UV.
        world_to_camera=np.asarray(detail["camera_world_to_local"],dtype=float)
        if world_to_camera.shape!=(4,4) or not np.isfinite(world_to_camera).all():
            raise ValueError("invalid four-view extrinsic matrix "+v)
        cam_xyz=source_gt@world_to_camera[:3,:3].T+world_to_camera[:3,3]
        mpp=float(detail["meters_per_pixel"])
        uv_again=np.stack((width/2+cam_xyz[:,0]/mpp,
                           height/2-cam_xyz[:,1]/mpp),axis=-1)
        max_uv_error=float(np.linalg.norm(uv_again-zdata["joints_uv"],axis=-1).max())
        if max_uv_error>.1:
            raise ValueError(f"camera reprojected 3D GT mismatch {v}: {max_uv_error}px")
        pix=zdata["part_id"]
        valid=np.isfinite(zdata["depth_m"])
        if (np.any(pix[valid]==-1) or np.any(pix[~valid]!=-1) or
                np.any(zdata["depth_m"][valid]<=0) or
                np.any((pix>=num_bones)) or np.any(pix < -2)):
            raise ValueError("invalid depth-to-part correspondence "+v)
        calc=np.bincount(pix[pix>=0].astype(np.int64),minlength=num_bones)
        if not np.array_equal(calc,zdata["visible_pixel_count"]):
            raise ValueError("visible part count disagrees with pixels "+v)
        if np.any(zdata["visible_pixel_count"]>zdata["amodal_pixel_count"]):
            raise ValueError("surface part visible > amodal "+v)
        recomputed=part_visibility_targets(zdata,bone)
        labels=zdata["joint_surface_part_visible"]
        if not np.array_equal(recomputed,labels):
            raise ValueError("incorrect joint surface-part targets "+v)
        if labels.tolist()!=detail["joint_surface_part_visibility"]:
            raise ValueError("stored JSON labels differ from NPZ "+v)
        if (int(np.sum(labels==1))!=detail["visible_joint_surface_labels"] or
                int(np.sum(labels==0))!=detail["occluded_joint_surface_labels"] or
                int(np.sum(labels==-1))!=detail["unknown_joint_surface_labels"]):
            raise ValueError("visibility label counts mismatch "+v)
        ray=detail["raycast_validation"]
        if (ray["sample_count"]<60 or ray["misses"]>4 or
                ray["depth_error_m_p95"]>.03):
            raise ValueError("Blender independent raycast failed "+v)
        img=cv2.imread(str(png),cv2.IMREAD_UNCHANGED)
        if img is None or img.shape!=(height,width,4):
            raise ValueError("render PNG shape/alpha mismatch "+v)
        silhouette=img[:,:,3]>8
        miou=float(np.logical_and(silhouette,valid).sum()/
                  max(np.logical_or(silhouette,valid).sum(),1))
        if miou<.88:
            raise ValueError(f"silhouette/raster mismatch {v}: {miou}")
        all_labels[v]=labels
        per_view[v]={
            "size":[width,height],"surface_pixels":int(valid.sum()),
            "part_labeled_surface_pixels":int((pix>=0).sum()),
            "visible_joint_surface_count":int((labels==1).sum()),
            "occluded_joint_surface_count":int((labels==0).sum()),
            "unknown_joint_surface_count":int((labels==-1).sum()),
            "render_mask_iou":round(miou,5),
            "source_3d_gt_projection_max_error_px":round(max_uv_error,6),
            "ray_depth_p95_error_m":ray["depth_error_m_p95"],
        }
        if create_contact:
            preview.append(_tile(v,img,zdata,labels))
    two=(all_labels["front"]==1)|(all_labels["side"]==1)
    four=np.logical_or.reduce([all_labels[v]==1 for v in VIEWS])
    any_known=np.logical_or.reduce([all_labels[v]>=0 for v in VIEWS])
    coverage={
        "keypoints":len(bone),"visible_in_front_or_side":int(two.sum()),
        "visible_in_any_four":int(four.sum()),
        "newly_visible_in_back_or_opposite_only":int(np.logical_and(four,~two).sum()),
        "known_in_at_least_one_view":int(any_known.sum()),
        "not_observed_in_any_view":int((~four).sum()),
    }
    if create_contact:
        image=np.hstack(preview)
        file=out/f"{idx:05d}_four_view_contact.png"
        if not cv2.imwrite(str(file),image):
            raise RuntimeError("unable to write static QA contact sheet")
    return {"sample_index":idx,"replay_max_joint_world_error_m":m["replay_max_joint_world_error_m"],
            "camera_opposite_max_error_px":max(m["camera_opposite_geometry_checks"].values()),
            "views":per_view,"surface_part_coverage":coverage}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--rig",required=True)
    ap.add_argument("--source-dataset",required=True)
    ap.add_argument("--indices",default="0,1")
    ap.add_argument("--create-contact-sheets",action="store_true")
    ap.add_argument("--audit-name",default="four_view_audit.json")
    args=ap.parse_args()
    if (Path(args.audit_name).name != args.audit_name or
            not args.audit_name.endswith(".json")):
        raise ValueError("--audit-name must be a JSON basename")
    out=Path(args.out).resolve()
    rig=Path(args.rig).resolve()
    dataset=Path(args.source_dataset).resolve()
    ids=[int(n) for n in args.indices.split(",")]
    reports=[audit_one(out,i,rig,dataset/f"{i:05d}_meta.json",args.create_contact_sheets)
             for i in ids]
    summary={
        "schema":"anymove.four_view_surface_part_gt_audit/1",
        "status":"PASS",
        "source_dataset":str(dataset),"output":str(out),
        "joint_visibility_semantics":"Skin-weight assigned SURFACE PART, not buried joint center",
        "no_new_h3_videos":True,"no_new_motion":True,
        "num_samples":len(ids),"num_views_per_sample":4,
        "source_samples":[r["sample_index"] for r in reports],
        "totals":{
            "newly_visible_in_extra_two_views":sum(
                r["surface_part_coverage"]["newly_visible_in_back_or_opposite_only"] for r in reports),
            "four_view_visibility_labels_known":sum(
                r["surface_part_coverage"]["known_in_at_least_one_view"] for r in reports),
            "raycast_validated_views":sum(
                r["views"][v]["surface_pixels"]>0 for r in reports for v in VIEWS),
        },
        "samples":reports,
    }
    file=out/args.audit_name
    if file.exists():raise FileExistsError("audit already exists: "+str(file))
    file.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
