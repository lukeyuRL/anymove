"""Package 2/3/4-view orthographic observations from existing Blender P1 poses.

Modes:
  synthetic_clean: exact 2D projection of STORED independent 3D poses.
  synthetic_noisy: deterministic per-view pixel errors (occluded parts receive
                   stronger corruption), to stress geometric reconstruction.
  reader: actual previously trained 2-view Dino/Dragon DINOv2 Reader on the
          new four-view Workbench stills; back/opposite are OOD and untrained.

3D ground-truth is written ONLY into a separate evaluation file, never to
the solver packet. Static independent poses are not a motion sequence.
No H3/AI video generation, no retraining.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.multiview import save_packet,project_np

VIEWS=("front","side","back","opposite_side")


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(2**20),b""):
            h.update(block)
    return h.hexdigest()


def load_source(src, indices, rig_path):
    info=json.loads(Path(rig_path).read_text(encoding="utf-8"))
    names=[x["name"] for x in info["keypoints"]]
    frames=[];labels=[];cameras=None
    sources=[]
    for i in indices:
        manifest=src/f"{i:05d}_visibility_meta.json"
        data=json.loads(manifest.read_text(encoding="utf8"))
        if data.get("schema_version")!="anymove.four_view_surface_part_gt/1":
            raise ValueError("P1 manifest version mismatch")
        if data["rig_info_sha256"]!=sha(rig_path):
            raise ValueError("rig provenance mismatch")
        source_meta=Path(data["source_pose_metadata"])
        if sha(source_meta)!=data["source_meta_sha256"]:
            raise ValueError("source pose metadata sha mismatch")
        gt=np.asarray(json.loads(source_meta.read_text(encoding="utf8"))["joints_world"],float)
        if gt.shape!=(len(names),3):
            raise ValueError("source 3D GT wrong joint order/shape")
        frames.append(gt)
        one={}
        for v in VIEWS:
            path=src/f"{i:05d}_{v}_labels.npz"
            with np.load(path) as z:
                name_data={key:z[key].copy() for key in z.files}
            size=np.asarray(data["views"][v]["image_size_px"],int)
            extrinsic=np.asarray(data["views"][v]["camera_world_to_local"],float)
            mpp=float(data["views"][v]["meters_per_pixel"])
            if name_data["joints_uv"].shape!=(len(names),2):
                raise ValueError("P1 UV shape wrong")
            reprojection=project_np(gt[None],extrinsic[None],size[None],np.array([mpp]))[0,0]
            if np.max(np.linalg.norm(reprojection-name_data["joints_uv"],axis=-1))>.08:
                raise ValueError("P1 camera/GT projection mismatch")
            one[v]={"uv":name_data["joints_uv"],"visible":name_data["joint_surface_part_visible"],
                    "size":size,"extrinsic":extrinsic,"mpp":mpp,
                    "png":str(src/f"{i:05d}_{v}.png")}
        if cameras is None:
            cameras={v:(one[v]["size"],one[v]["extrinsic"],one[v]["mpp"]) for v in VIEWS}
        for v in VIEWS:
            size,ext,mpp=cameras[v]
            if (not np.array_equal(size,one[v]["size"]) or
                    not np.allclose(ext,one[v]["extrinsic"],atol=1e-5) or
                    abs(mpp-one[v]["mpp"])>1e-12):
                raise ValueError("P1 camera differs across poses, cannot fit synchronized N-view camera contract")
        labels.append(one)
        sources.append({"sample_id":i,"metadata":str(source_meta),"metadata_sha256":sha(source_meta),
                        "visibility_manifest":str(manifest),"visibility_manifest_sha256":sha(manifest)})
    return info,names,np.stack(frames),labels,cameras,sources


def run_actual_reader(checkpoint,labels,names):
    from mocap.detect import load,run
    from mocap.model import JointReader
    from mocap.reader_config import spec_from_checkpoint
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("existing Reader requires CUDA for current load() API")
    meta=torch.load(checkpoint,map_location='cpu',weights_only=True)
    quality_spec=meta.get('quality_spec') or {}
    q_version=quality_spec.get('version')
    if q_version==2:
        if (quality_spec.get('target')!='mesh_surface_part_visibility' or
                quality_spec.get('visibility_semantics')!='mesh_surface_part_visibility_v1_uncalibrated' or
                not bool((meta.get('reader_spec') or {}).get('quality_head'))):
            raise ValueError('unsupported P3 Mesh Visibility checkpoint contract')
        spec=spec_from_checkpoint(meta)
        trained_names=list(meta['names'])
        model=JointReader(len(trained_names),input_size=spec.input_size,
                          heatmap_size=spec.heatmap_size,quality_head=True).cuda().eval()
        model.load_state_dict(meta['state_dict'],strict=True)
        model.backbone.set_attn_implementation('sdpa')
        semantics='mesh_surface_part_visibility_v1_uncalibrated'
        print('Loaded experimental P3 Mesh Surface Visibility Reader',flush=True)
    else:
        del meta
        model,trained_names=load(str(checkpoint))
        semantics='synthetic_cutout_only_uncalibrated'
    if list(map(str,trained_names))!=names:
        raise ValueError("Reader trained names/order differ from Rig")
    T=len(labels);J=len(names)
    predictions=[];confidence=[]
    predicted_visibility=[];predicted_error=[]
    for view in VIEWS:
        rgba=[]
        for pose in labels:
            png=cv2.imread(pose[view]["png"],cv2.IMREAD_UNCHANGED)
            if png is None or png.ndim!=3 or png.shape[2]!=4:
                raise ValueError("P1 Reader input must be RGBA PNG")
            a=png[:,:,3:4].astype(np.float32)/255
            # Original Reader expects a studio background, NOT transparency
            # displayed as black. Neutral background is an OOD compromise.
            bgr=(png[:,:,:3]*a+np.array([128,130,134],np.float32)*(1-a)).clip(0,255).astype(np.uint8)
            rgba.append(bgr[:,:,::-1].copy())
        result=run(model,rgba,bs=1,with_quality=model.quality_head)
        xy,c=result[:2]
        if model.quality_head:
            predicted_visibility.append(result[2])
            predicted_error.append(result[3])
        if xy.shape!=(T,J,2) or c.shape!=(T,J):
            raise ValueError("Reader returned wrong dimensions")
        predictions.append(xy);confidence.append(c)
        print("Reader view",view,"peak_mean",round(float(c.mean()),4),
              "uv_range",xy.min(axis=(0,1)),xy.max(axis=(0,1)),flush=True)
    quality=(None if not model.quality_head else {
        "visibility_score":np.stack(predicted_visibility).astype(np.float32),
        "expected_error_px":np.stack(predicted_error).astype(np.float32),
        "quality_semantics":np.asarray(semantics)})
    del model
    torch.cuda.empty_cache()
    return np.stack(predictions),np.stack(confidence),quality


def make_observations(args):
    if args.mode not in ("synthetic_clean","synthetic_noisy","reader"):
        raise ValueError("invalid mode")
    indices=[int(n) for n in args.indices.split(",")]
    if not indices or len(set(indices))!=len(indices):
        raise ValueError("empty/duplicate indices")
    src=Path(args.source).resolve()
    info,names,gt,labels,cameras,sources=load_source(src,indices,Path(args.rig).resolve())
    uv_gt=np.stack([[pose[v]["uv"] for pose in labels] for v in VIEWS],axis=0).astype(float)
    visibility=np.stack([[pose[v]["visible"] for pose in labels] for v in VIEWS],axis=0)
    world_to_camera=np.stack([cameras[v][1] for v in VIEWS])
    size=np.stack([cameras[v][0] for v in VIEWS]).astype(np.int32)
    mpp=np.asarray([cameras[v][2] for v in VIEWS],dtype=np.float64)
    quality=None
    if args.mode=="reader":
        if not args.checkpoint:
            raise ValueError("--checkpoint required for Reader mode")
        uv,conf,quality=run_actual_reader(Path(args.checkpoint).resolve(),labels,names)
    elif args.mode=="synthetic_clean":
        uv=uv_gt.copy()
        conf=np.ones(uv_gt.shape[:-1],dtype=np.float32)
    else:
        uv=uv_gt.copy()
        conf=np.ones(uv_gt.shape[:-1],dtype=np.float32)
        for vi,v in enumerate(VIEWS):
            rng=np.random.default_rng(args.seed+vi*1000003)
            jitter=rng.normal(size=uv[vi].shape)
            sigma=np.where(visibility[vi]==0,args.occluded_noise_px,
                    np.where(visibility[vi]==-1,args.noise_px*1.5,args.noise_px))
            uv[vi]+=jitter*sigma[...,None]
    valid=np.zeros(conf.shape,bool)
    for vi,(w,h) in enumerate(size):
        u=uv[vi,:,:,0];v=uv[vi,:,:,1]
        valid[vi]=(u>=0)&(u<w)&(v>=0)&(v<h)&np.isfinite(u)&np.isfinite(v)
    conf=np.maximum(conf,0)
    conf[~valid]=0
    if not np.isfinite(uv).all() or not np.isfinite(conf).all():
        raise ValueError("Reader observations contain nonfinite values")
    # Known GT masks are diagnostic only, saved separately in the GT file.
    p={
        "view_ids":np.asarray(VIEWS),"names":np.asarray(names),
        "uv":uv.astype(np.float32),"confidence":conf.astype(np.float32),
        "valid":valid,"world_to_camera":world_to_camera,
        "image_size_px":size,"meters_per_px":mpp,
        "frame_ids":np.asarray(indices,dtype=np.int32),
        "fps":np.asarray(24.,dtype=np.float64),
        "independent_stills":np.asarray(True),
        "source_kind":np.asarray({
            "synthetic_clean":"blender_gt_projection",
            "synthetic_noisy":"synthetic_noisy",
            "reader":"reader_inference"}[args.mode])
    }
    if quality is not None:
        p.update(quality)
    output=Path(args.out).resolve()
    gt_path=output.with_suffix(".ground_truth.npz")
    report=output.with_suffix(".manifest.json")
    if gt_path.exists() or report.exists():
        raise FileExistsError("P2 output exists, use new output prefix")
    packet_meta=save_packet(output,p,rig_names=names)
    np.savez_compressed(gt_path,
        joints_world=gt.astype(np.float64),names=np.asarray(names),
        frame_ids=np.asarray(indices,np.int32),
        part_visibility=visibility.astype(np.int8),
        exact_uv=uv_gt.astype(np.float32))
    gt3d=project_np(gt,world_to_camera,size,mpp)
    verify=float(np.max(np.abs(gt3d-uv_gt)))
    error=np.linalg.norm(uv-uv_gt,axis=-1)
    details={"schema_version":"anymove.multiview_source_manifest/1",
        "mode":args.mode,"packet":str(output),"ground_truth_eval_only":str(gt_path),
        "frame_ids":indices,"view_ids":list(VIEWS),"rig_info":str(Path(args.rig).resolve()),
        "rig_sha256":sha(Path(args.rig).resolve()),
        "quality_semantics":str(quality["quality_semantics"].item()) if quality is not None else None,
        "source_pose_metadata":sources,
        "camera_gt_reprojection_max_error_px":verify,
        "per_view_input_2d_error_mean_px":{v:round(float(error[vi].mean()),4) for vi,v in enumerate(VIEWS)},
        "reader_checkpoint":str(args.checkpoint) if args.mode=="reader" else None,
        "seed":args.seed if args.mode=="synthetic_noisy" else None,
        "synthesis_noise_sigma_px":args.noise_px if args.mode=="synthetic_noisy" else None,
        "synthesis_occluded_sigma_px":args.occluded_noise_px if args.mode=="synthetic_noisy" else None,
        "warning":( "Experimental four-view Reader trained on a small Blender dataset; "
                    "visibility/uncertainty uncalibrated and H3 generalization unverified"
                    if args.mode=="reader" and quality is not None and
                       quality["quality_semantics"].item()=="mesh_surface_part_visibility_v1_uncalibrated"
                    else "Original Reader trained on Front/Side; Back/Opposite Side "
                         "are out-of-distribution"
                    if args.mode=="reader" else
                    "Synthetic 2D observations use 3D GT projections; "
                    "this validates geometry, NOT learned inference"),
        "no_h3_video_generated":True,"no_motion_training":True}
    report.write_text(json.dumps(details,indent=2),encoding="utf8")
    print("PACKET",packet_meta)
    print(json.dumps(details,indent=2))
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    ap.add_argument("--rig",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--indices",default="0,1")
    ap.add_argument("--mode",choices=("synthetic_clean","synthetic_noisy","reader"),default="synthetic_noisy")
    ap.add_argument("--checkpoint",default=None)
    ap.add_argument("--seed",type=int,default=1729)
    ap.add_argument("--noise-px",type=float,default=2.)
    ap.add_argument("--occluded-noise-px",type=float,default=12.)
    args=ap.parse_args()
    if (not np.isfinite(args.noise_px) or args.noise_px<0 or
            not np.isfinite(args.occluded_noise_px) or args.occluded_noise_px<0):
        raise ValueError("noise parameters must be finite and nonnegative")
    make_observations(args)


if __name__=="__main__":
    main()
