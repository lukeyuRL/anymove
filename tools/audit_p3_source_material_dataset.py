"""P3 source-material four-view dataset integrity and colour QA.

Reads EXISTING Blender dataset; never writes/changes images or model weights.
Reports view-specific saturated-foreground coverage, source pose provenance,
known visibility labels and camera/raycast integrity. Unlike label training,
this appearance audit does not infer occlusion from colour.
"""
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from mocap.fourview_data import split_poses
from mocap.fourview_data import sha256

VIEWS=("front","side","back","opposite_side")


def audit(root,rig_path,expected_poses=28):
    root,rig_path=Path(root),Path(rig_path)
    train,val=split_poses(root)
    if len(train)+len(val)!=expected_poses or set(train)&set(val):
        raise ValueError("bad sample count or overlapping pose IDs")
    if not train or not val:raise ValueError("no independent validation samples")
    rig=json.loads(rig_path.read_text(encoding="utf8"))
    bone_count=len(rig["bones"]);J=len(rig["keypoints"])
    report={"schema":"anymove.p3_source_material_dataset_audit/1","path":str(root),
        "pose_ids":{"train":list(train),"validation":list(val)},
        "rendered_views":0,"source_poses":0,
        "all_appearance_source_material":True,
        "view_color_saturation":{},
        "unknown_surface_part_labels_by_view":{},
        "min_mask_iou":None,"max_depth_raycast_p95_m":0.}
    stats={v:[] for v in VIEWS};unknown={v:[] for v in VIEWS}
    max_error=0.; iou_values=[]
    rigsha=sha256(rig_path)
    for idx in sorted(train+val):
        manifest=root/f"{idx:05d}_visibility_meta.json"
        m=json.loads(manifest.read_text(encoding="utf8"))
        if m.get("schema_version")!="anymove.four_view_surface_part_gt/1" or m["sample_index"]!=idx:
            raise ValueError("source manifest sample mismatch")
        if (m.get("appearance_mode")!="source-material" or
                not m.get("source_material_colors_preserved") or
                m["rig_info_sha256"]!=rigsha or
                sha256(m["source_pose_metadata"])!=m["source_meta_sha256"]):
            raise ValueError("source material renderer/rig or pose metadata corrupted")
        if m["replay_max_joint_world_error_m"]>.001:
            raise ValueError("replay 3D position differs from source GT")
        report["source_poses"]+=1
        max_error=max(max_error,m["replay_max_joint_world_error_m"])
        for v in VIEWS:
            meta=m["views"][v]
            fname=root/f"{idx:05d}_{v}.png"
            label=root/f"{idx:05d}_{v}_labels.npz"
            pix=cv2.imread(str(fname),cv2.IMREAD_UNCHANGED)
            w,h=meta["image_size_px"]
            if pix is None or pix.shape!=(h,w,4):
                raise ValueError("image missing/shape mismatch "+str(fname))
            if h!=720 or w not in (rig["layout"]["W_FRONT"],rig["layout"]["W_SIDE"]):
                raise ValueError("not original layout resolution")
            hsv=cv2.cvtColor(pix[:,:,:3],cv2.COLOR_BGR2HSV)
            opaque=pix[:,:,3]>64
            saturated=opaque&(hsv[:,:,1]>110)
            frac=float(saturated.sum()/max(opaque.sum(),1))
            if np.count_nonzero(opaque)<100:
                raise ValueError("render lacks creature")
            stats[v].append(frac)
            with np.load(label,allow_pickle=False) as z:
                kp=z["joints_uv"]
                surf=z["joint_surface_part_visible"]
                part=z["part_id"]
                depth=z["depth_m"]
                if (kp.shape!=(J,2) or surf.shape!=(J,) or
                        part.shape!=(h,w) or depth.shape!=(h,w) or
                        len(z["surface_visible_fraction"])!=bone_count or
                        not np.isin(surf,[-1,0,1]).all()):
                    raise ValueError("source Surface Part label corruption")
                unknown[v].append(int(np.sum(surf<0)))
            ray=meta["raycast_validation"]
            if (ray["depth_error_m_p95"]>.03 or ray["misses"]>4 or
                    m["camera_opposite_geometry_checks"]["front_back_horizontal_max_abs_px"]>.02):
                raise ValueError("raster depth, projection or raycast mismatch")
            report["max_depth_raycast_p95_m"]=max(report["max_depth_raycast_p95_m"],ray["depth_error_m_p95"])
            report["rendered_views"]+=1
    report["max_replay_error_m"]=max_error
    for v in VIEWS:
        report["view_color_saturation"][v]={"min":round(min(stats[v]),6),
                                              "mean":round(float(np.mean(stats[v])),6),
                                              "max":round(max(stats[v]),6)}
        report["unknown_surface_part_labels_by_view"][v]=sorted(set(unknown[v]))
    if report["rendered_views"]!=4*expected_poses:
        raise ValueError("missing or extra views")
    report["status"]="PASS"
    return report


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--dataset",required=True)
    a.add_argument("--rig",required=True)
    a.add_argument("--out",required=True)
    a.add_argument("--poses",type=int,default=28)
    args=a.parse_args()
    target=Path(args.out)
    if target.exists():raise FileExistsError("preserve original QA reports; select new output")
    rep=audit(args.dataset,args.rig,args.poses)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(rep,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(rep,indent=2))


if __name__=="__main__":
    main()
