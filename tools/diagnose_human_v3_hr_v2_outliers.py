"""Posthoc forensic audit of HR V2 2D/3D outlier joints.

For both Validation and previously-inspected Test, run the *frozen* Reader,
collect per-joint pixel errors and triangulated 3D errors. No retraining, no
threshold selection, no test-dependent inference adjustment.
"""
import json, pathlib
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from mocap.camera_geometry import triangulate_orthographic
from mocap.reader_hr_v2 import HumanReaderHRV2,decode_hr
from tools.train_human_v3_hr_v2_reader import DATA,OUT,V2Images,load_manifest,SHA

@torch.inference_mode()
def diagnostics():
    mf=load_manifest()
    ck=torch.load(OUT/"full"/"best.pt",map_location="cpu",weights_only=True)
    if ck["source_blend_sha256"]!=SHA or ck["camera_azimuth_deg"]!=[315,45] or ck["names"]!=mf["joint_names"]:
        raise RuntimeError("Reader provenance mismatch")
    names=mf["joint_names"]
    net=HumanReaderHRV2(len(names)).cuda().eval()
    net.load_state_dict(ck["state_dict"],strict=True)
    result={"status":"POSTHOC_ANALYSIS_ONLY","source_blend_sha256":SHA,
            "checkpoint_selected_using":"validation 2d mean",
            "camera_angles_deg":[315,45],"splits":{}}
    for split in ("validation","test"):
        ds=V2Images(mf,split,augment=False)
        loader=DataLoader(ds,batch_size=12,shuffle=False,num_workers=0,pin_memory=True)
        preds=defaultdict(dict)
        err_2d={}
        for images,truth,inds in loader:
            with torch.autocast("cuda",dtype=torch.float16):
                logits=net(images.cuda(non_blocking=True))
            uv,conf=decode_hr(logits.float())
            uv=uv.cpu().numpy()
            conf=conf.cpu().numpy()
            truth=truth.numpy()
            for i,j in enumerate(inds.tolist()):
                row=ds.samples[j]
                key=(row["clip"],int(row["frame"]))
                slot=row["slot"]
                pixerr=np.linalg.norm(uv[i]-truth[i],axis=-1)
                preds[key][slot]=uv[i].astype(np.float64)
                err_2d[(key,slot)]={"error":pixerr,"peak":conf[i]}
        gt_cache={}
        joint_3d=Counter()
        view_2d=Counter()
        outliers=[]
        total=0
        over_counts=Counter()
        all3d=[]
        for (clip,frame),views in sorted(preds.items()):
            if set(views)!={"front","side"}:raise RuntimeError("Missing view")
            if clip not in gt_cache:
                gt=json.loads((DATA/"motion_gt"/(clip+".json")).read_text(encoding="utf-8"))
                gt_cache[clip]=np.asarray(gt["joint_position_m"],dtype=np.float64)
            world=gt_cache[clip][frame]
            recon=triangulate_orthographic(views["front"],views["side"],np.ones(67),np.ones(67),
                       front_width=672,side_width=672,height=672,
                       metres_per_pixel=mf["metres_per_pixel"],z_center=mf["z_center"],
                       front_angle_deg=315,side_angle_deg=45,front_height_weight=1.)
            errors=np.linalg.norm(recon-world,axis=-1)*100
            all3d.extend(errors.tolist())
            total+=len(names)
            for idx,err in enumerate(errors):
                for threshold in (2,5,10,20,50):
                    if err>threshold:over_counts[str(threshold)+"cm"]+=1
                if err<=5:continue
                name=names[idx]
                joint_3d[name]+=1
                ea=err_2d[((clip,frame),"front")]
                eb=err_2d[((clip,frame),"side")]
                va=float(ea["error"][idx])
                vb=float(eb["error"][idx])
                if va>vb:view_2d["315_degrees"]+=1
                else:view_2d["45_degrees"]+=1
                outliers.append({
                    "clip":clip,"frame":frame,"joint":name,
                    "error_cm":round(float(err),4),
                    "315_2d_error_px":round(va,3),"45_2d_error_px":round(vb,3),
                    "315_peak":round(float(ea["peak"][idx]),5),
                    "45_peak":round(float(eb["peak"][idx]),5),
                    "true_world_m":np.round(world[idx],5).tolist(),
                    "reconstructed_world_m":np.round(recon[idx],5).tolist(),
                    "315_pred_uv":np.round(views["front"][idx],3).tolist(),
                    "45_pred_uv":np.round(views["side"][idx],3).tolist(),
                })
        errors=np.asarray(all3d)
        split_out={
            "frames":len(preds),
            "joint_samples":total,
            "3d_error_mean_cm":round(float(np.mean(errors)),5),
            "3d_error_p95_cm":round(float(np.percentile(errors,95)),5),
            "3d_error_p99_cm":round(float(np.percentile(errors,99)),5),
            "3d_error_max_cm":round(float(np.max(errors)),5),
            "outlier_counts":dict(over_counts),
            "outlier_joint_frequency_gt5cm":dict(joint_3d.most_common()),
            "outlier_worse_view":dict(view_2d),
            "worst_outliers":sorted(outliers,key=lambda r:-r["error_cm"])[:30],
        }
        result["splits"][split]=split_out
        print("HRV2_OUTLIER_DIAG",split,json.dumps({
            "frames":split_out["frames"],
            "mean_3d_cm":split_out["3d_error_mean_cm"],
            "max_cm":split_out["3d_error_max_cm"],
            "threshold_counts":split_out["outlier_counts"],
            "outlier_worse_view":split_out["outlier_worse_view"],
            "worst":split_out["worst_outliers"][:6],
        }),flush=True)
    path=OUT/"full"/"outlier_diagnostic_posthoc.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("HRV2_OUTLIER_DIAGNOSTIC_SAVED",str(path),flush=True)

if __name__=="__main__":
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    diagnostics()
