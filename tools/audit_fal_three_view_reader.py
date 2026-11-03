"""Audit fal-generated Dino three-panel video using frozen REAL Joint Reader.

Only measures cross-view *consistency*, NOT 3D GT accuracy (no known
3D motion exists for AI-generated video). Extract each 16:9 panel via
SHA-bound reference layout and undo its scaling/letterboxing; run the
existing Dino 672/192 Reader on all 3 restored view images.
"""
import argparse,csv,hashlib,json,pathlib
import cv2,numpy as np
from mocap import layout
from mocap.detect import load,run
from mocap.dino_top_camera import camera_from_json,project_top_numpy
from tools.evaluate_dino_full_body import classify_joint

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb")as fd:
        for c in iter(lambda:fd.read(1048576),b""):h.update(c)
    return h.hexdigest()

def summary(a):
    a=np.asarray(a,float)
    a=a[np.isfinite(a)]
    return {"n":int(a.size),"mean":round(float(a.mean()),4)if a.size else None,
            "median":round(float(np.median(a)),4)if a.size else None,
            "p95":round(float(np.percentile(a,95)),4)if a.size else None,
            "max":round(float(a.max()),4)if a.size else None}

def lag_correlation(a,b,window=8):
    data=[]
    for lag in range(-window,window+1):
        if lag>=0:
            x,y=a[:len(a)-lag or None],b[lag:]
        else:
            x,y=a[-lag:],b[:len(b)+lag]
        valid=np.isfinite(x)&np.isfinite(y)
        x,y=x[valid],y[valid]
        if len(x)<15 or x.std()<1.5 or y.std()<1.5:continue
        corr=float(np.corrcoef(x,y)[0,1])
        if np.isfinite(corr):data.append((lag,corr,len(x)))
    best=max(data,key=lambda item:item[1])if data else None
    return {"best_lag_frames":best[0]if best else None,
            "best_corr":round(best[1],4)if best else None,
            "corr_at_zero":round(next((x[1]for x in data if x[0]==0),np.nan),4)if data else None,
            "n":best[2]if best else None}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("ref_manifest")
    p.add_argument("dino_rig")
    p.add_argument("reader_ckpt")
    p.add_argument("top_camera")
    p.add_argument("ref_top_pose")
    p.add_argument("new_output_dir")
    args=p.parse_args(argv)
    for field in ("video","ref_manifest","dino_rig","reader_ckpt","top_camera","ref_top_pose"):
        if not pathlib.Path(getattr(args,field)).is_file():
            raise FileNotFoundError(getattr(args,field))
    out=pathlib.Path(args.new_output_dir)
    if out.exists():raise FileExistsError("refusing existing 3view Reader QA dir")
    ref=json.loads(pathlib.Path(args.ref_manifest).read_text(encoding="utf-8"))
    rig=json.loads(pathlib.Path(args.dino_rig).read_text(encoding="utf-8"))
    names=[x["name"]for x in rig["keypoints"]]
    rest=json.loads(pathlib.Path(args.ref_top_pose).read_text(encoding="utf-8"))
    if (len(names)!=92 or rest["joint_names"]!=names or
            ref["profile"]!="anymove.h3max_dino_front_side_top_reference_smoke_v1"):
        raise ValueError("mismatched Dino 92-joint layout and original Blender rest")
    cam=camera_from_json(args.top_camera)
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():raise FileNotFoundError("video cannot be opened")
    fps=cap.get(cv2.CAP_PROP_FPS)
    T=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if T<30 or not(20<=fps<=60):
        raise ValueError("bad H3 video duration/frame count")
    panels=ref["panels"]
    views={key:[]for key in ("front","side","top")}
    for t in range(T):
        success,bgr=cap.read()
        if not success:raise ValueError("generated video was truncated")
        frame=cv2.cvtColor(cv2.resize(bgr,(1280,720),interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        for key,items in views.items():
            spec=panels[key]
            x,y,w,h=(spec["render_left"],spec["render_top"],
                    spec["render_width"],spec["render_height"])
            crop=frame[y:y+h,x:x+w]
            target=(spec["source_width"],spec["source_height"])
            restored=cv2.resize(crop,target,interpolation=cv2.INTER_CUBIC)
            items.append(restored)
    cap.release()
    model,cknames=load(args.reader_ckpt)
    if [str(v)for v in cknames]!=names:
        raise ValueError("Reader checkpoint does not match Dino rig")
    uv,peak={},{}
    for key,images in views.items():
        detected=run(model,images,bs=2)
        uv[key],peak[key]=np.asarray(detected[0],float),np.asarray(detected[1],float)
        print("JOINT READER 3-VIEW",key,T,"frames","peakmean",round(float(peak[key].mean()),4),flush=True)
    tail=[j for j,n in enumerate(names)if classify_joint(n)=="tail"]
    if len(tail)!=9:raise ValueError("expected nine anatomical Tail joints")
    # From actual Blender camera geometry, in original layout pixels:
    # Front +X => F.u-W_F/2
    # Top +X   => T.v-H/2
    # Side -Y  => S.u-W_S/2
    # Top +Y   => T.u-W_TOP/2
    delta={
        "X_front_minus_top":(uv["front"][...,0]-layout.W_FRONT/2)-
                             (uv["top"][...,1]-360),
        "Y_side_plus_top":(uv["side"][...,0]-layout.W_SIDE/2)+
                             (uv["top"][...,0]-440),
        "Z_front_minus_side":uv["front"][...,1]-uv["side"][...,1],
    }
    inbounds={}
    for key,pair in (("front",(layout.W_FRONT,720)),
                     ("side",(layout.W_SIDE,720)),("top",(880,720))):
        data=uv[key]
        inbounds[key]=(data[...,0]>=0)&(data[...,0]<pair[0])&(
                       data[...,1]>=0)&(data[...,1]<pair[1])&(peak[key]>.2)
    valid={
        "X_front_minus_top":inbounds["front"]&inbounds["top"],
        "Y_side_plus_top":inbounds["side"]&inbounds["top"],
        "Z_front_minus_side":inbounds["front"]&inbounds["side"],
    }
    # Use original FIRST FRAME rest as a one-frame sanity check only.
    xyz=np.asarray(rest["world_joints"],float)
    fgt,sgt=layout.project(xyz)
    tgu=project_top_numpy(xyz,cam)
    frame0gt={"front":fgt,"side":sgt,"top":tgu}
    output={
        "profile":"anymove.h3max_dino_threeview_image_joint_consistency_v1",
        "video_sha256":sha(args.video),"reader_checkpoint_sha256":sha(args.reader_ckpt),
        "frames":T,"fps":fps,"keypoints":len(names),
        "warnings":["Cross-view Reader predictions on AI-generated video have NO animated Blender 3D GT",
                    "Consistency does not prove correct 3D motion, camera models may drift.",
                    "Top Reader is a frozen Front/Side-trained transfer baseline; only Tail was validated in synthetic GT."],
        "raw_2d_gt_first_rest_frame_px":{
            k:{"tail_mean":round(float(np.linalg.norm(uv[k][0,tail]-frame0gt[k][tail],axis=-1).mean()),4),
               "all_mean":round(float(np.linalg.norm(uv[k][0]-frame0gt[k],axis=-1).mean()),4)}
            for k in ("front","side","top")},
        "crossview":{},
        "tail_joint_specific":{},
    }
    for group in delta:
        ids=tail
        a=np.abs(delta[group][:,ids]);v=valid[group][:,ids]
        output["crossview"][group]={"tail_abs_diff_px":summary(a[v]),"valid_coverage":round(float(v.mean()),4)}
    for j in tail:
        name=names[j]
        item={"mean_raw_peak":{key:round(float(peak[key][:,j].mean()),4)for key in views}}
        for group in delta:
            ok=valid[group][:,j]
            item[group]=summary(np.abs(delta[group][ok,j]))
        # Use full trajectory rather than a model confidence proxy for sync
        # (a small occlusion can lower coverage even with correct motion).
        F=uv["front"][:,j,0]-layout.W_FRONT/2
        T_x=uv["top"][:,j,1]-360
        S=uv["side"][:,j,0]-layout.W_SIDE/2
        T_y=-(uv["top"][:,j,0]-440)
        item["lag_X_front_top"]=lag_correlation(F,T_x)
        item["lag_Y_side_top"]=lag_correlation(S,T_y)
        output["tail_joint_specific"][name]=item
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/"reader_threeview.npz",
        names=np.array(names),fps=fps,
        front_uv=uv["front"],side_uv=uv["side"],top_uv=uv["top"],
        front_conf=peak["front"],side_conf=peak["side"],top_conf=peak["top"])
    (out/"crossview_report.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    with (out/"crossview_tail_by_frame.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["frame","joint_name","x_front_minus_top_px","y_side_plus_top_px","z_front_minus_side_px","front_peak","side_peak","top_peak"])
        for t in range(T):
            for j in tail:
                writer.writerow([t,names[j],
                                 round(float(delta["X_front_minus_top"][t,j]),4),
                                 round(float(delta["Y_side_plus_top"][t,j]),4),
                                 round(float(delta["Z_front_minus_side"][t,j]),4),
                                 round(float(peak["front"][t,j]),4),
                                 round(float(peak["side"][t,j]),4),
                                 round(float(peak["top"][t,j]),4)])
    print(json.dumps(output,indent=2))
    return output

if __name__=="__main__":
    main()
