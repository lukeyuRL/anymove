"""Quantify fal 3-panel H3Max video layout and color-marker cross-view consistency.

No 3D truth exists for fal-generated motion. This audit only tests
panel persistence, per-view tail-tip color tracking, and 2D projection
consistency / time offsets. A favorable score does NOT certify physically
identical 3D poses at all instants.
"""
import argparse,csv,json,pathlib,hashlib
import cv2,numpy as np


def filehash(path):
    d=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""):d.update(block)
    return d.hexdigest()


def summary(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    return {"n":int(len(a)),"mean":None if not len(a) else round(float(a.mean()),4),
            "median":None if not len(a) else round(float(np.median(a)),4),
            "p95":None if not len(a) else round(float(np.percentile(a,95)),4)}


def segment_magenta(bgr, x, y, w, h):
    # The original Dino tail tip is hot-pink/purple. Exclude the
    # static gray backdrop and blue/cyan/green limbs with HSV.
    roi=bgr[y:y+h,x:x+w]
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(137,90,95),(179,255,255))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    num,lab,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    ids=[k for k in range(1,num) if stats[k,cv2.CC_STAT_AREA]>=3]
    if not ids:return {"pixels":0,"center":None,"bbox":None}
    k=max(ids,key=lambda c:int(stats[c,cv2.CC_STAT_AREA]))
    count=int(stats[k,cv2.CC_STAT_AREA])
    cx,cy=centroids[k]
    return {"pixels":count,
            "center":[float(cx)+x,float(cy)+y],
            "bbox":[int(z) for z in stats[k,:4]]}


def correlate_shift(a,b,maxlag=12):
    a,b=np.asarray(a,float),np.asarray(b,float)
    results=[]
    for lag in range(-maxlag,maxlag+1):
        if lag>=0:
            x,y=a[:len(a)-lag or None],b[lag:]
        else:
            x,y=a[-lag:],b[:len(b)+lag]
        valid=np.isfinite(x)&np.isfinite(y)
        x,y=x[valid],y[valid]
        if len(x)<12 or x.std()<2 or y.std()<2:
            continue
        r=float(np.corrcoef(x,y)[0,1])
        if np.isfinite(r):results.append((lag,len(x),r))
    results.sort(key=lambda i:-i[2])
    return {"best_lag_frames":results[0][0] if results else None,
            "best_correlation":round(results[0][2],4) if results else None,
            "best_overlap":results[0][1] if results else None,
            "corr_zero_lag":next((round(r,4) for lag,n,r in results if lag==0),None),
            "reason":None if results else "insufficient nonconstant/visible colored tip observations"}


def run(video,reference,layout_path,outdir):
    video,reference,layout_path=map(pathlib.Path,(video,reference,layout_path))
    outdir=pathlib.Path(outdir)
    for p in (video,reference,layout_path):
        if not p.is_file():raise FileNotFoundError(p)
    if outdir.exists():raise FileExistsError("refuse overwrite of existing QA dir")
    manifest=json.loads(layout_path.read_text(encoding="utf-8"))
    if manifest.get("profile")!="anymove.h3max_dino_front_side_top_reference_smoke_v1":
        raise ValueError("unknown triptych panel source geometry")
    img0=cv2.imread(str(reference))
    if img0 is None or img0.shape!=(720,1280,3) or manifest.get("output_sha256")!=filehash(reference):
        raise ValueError("source reference content/geometry SHA mismatch")
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise FileNotFoundError("generated MP4 could not be opened")
    fps=cap.get(cv2.CAP_PROP_FPS)
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w,h=int(cap.get(3)),int(cap.get(4))
    if n<30 or not (20<=fps<=60) or (w,h)!=(1344,768):
        raise ValueError("unexpected H3Max 768P frame geometry/FPS")
    panels=manifest["panels"]
    samples=[]
    outdir.mkdir(parents=True,exist_ok=False)
    overlay=None
    for t in range(n):
        ok,frame=cap.read()
        if not ok:raise ValueError("incomplete fal output video")
        frame=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
        item={"frame":t,"seconds":round(t/fps,5)}
        for view in ("front","side","top"):
            p=panels[view]
            x,y,width,height=(p["render_left"],p["render_top"],
                              p["render_width"],p["render_height"])
            detected=segment_magenta(frame,x,y,width,height)
            item[view+"_tail_tip_pixels"]=detected["pixels"]
            item[view+"_tip_x_original"]=None
            item[view+"_tip_y_original"]=None
            if detected["center"] is not None:
                cx,cy=detected["center"]
                item[view+"_tip_x_original"]=(cx-x)/p["scale_x"]
                item[view+"_tip_y_original"]=(cy-y)/p["scale_y"]
        # Compare actual hot-pink tail-tip 3D X and Y projections:
        # Front u-200 == Top v-360, Side u-440 == -(Top u-440).
        F=item["front_tip_x_original"]
        S=item["side_tip_x_original"]
        TU=item["top_tip_x_original"]
        TV=item["top_tip_y_original"]
        item["crossview_X_residual_px"]=(
            None if F is None or TV is None else (F-200)-(TV-360))
        item["crossview_Y_residual_px"]=(
            None if S is None or TU is None else (S-440)+(TU-440))
        for view in ("front","side","top"):
            x=panels[view]["panel_left"]
            width=panels[view]["panel_width"]
            item[view+"_gray_background_mean"]=round(float(np.mean(frame[3:33,x+8:x+width-8])),4)
        for x in manifest["divider_x"]:
            item["seam_"+str(x)+"_mean_luma"]=round(
                float(cv2.cvtColor(frame[:,x-2:x+3],cv2.COLOR_BGR2GRAY).mean()),3)
        samples.append(item)
    cap.release()
    cols=sorted({key for row in samples for key in row})
    with (outdir/"by_frame.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=cols)
        writer.writeheader();writer.writerows(samples)
    def s(name):
        return [item[name] if item[name] is not None else np.nan for item in samples]
    pred_X=s("front_tip_x_original")
    pred_top_X=s("top_tip_y_original")
    pred_Y=s("side_tip_x_original")
    pred_top_Y=s("top_tip_x_original")
    Xcorr=correlate_shift(pred_X,pred_top_X)
    Ycorr=correlate_shift(pred_Y,-np.asarray(pred_top_Y))
    remainders={v:summary(np.abs(s(v))) for v in
                ("crossview_X_residual_px","crossview_Y_residual_px")}
    initial_frame=cv2.resize(cv2.imread(str(reference)),(1280,720))
    checks={
        "profile":"anymove.fal_h3max_threeview_video_observation_audit_v1",
        "source_video_sha256":filehash(video),
        "source_reference_sha256":filehash(reference),
        "video_pixels":[w,h],"normalized_pixels":[1280,720],
        "fps":fps,"frames":n,"duration_seconds":n/fps,
        "views":{
            view:{
                "tail_tip_detected_frames":sum(row[view+"_tip_x_original"] is not None for row in samples),
                "tail_tip_coverage":round(sum(row[view+"_tip_x_original"] is not None for row in samples)/n,4)
            }for view in ("front","side","top")
        },
        "view_X_front_vs_top":{
            "residual_abs_px":remainders["crossview_X_residual_px"],
            "temporal_correlation":Xcorr},
        "view_Y_side_vs_top":{
            "residual_abs_px":remainders["crossview_Y_residual_px"],
            "temporal_correlation":Ycorr},
        "dividers":{str(x):{
            "dark_luma_median":float(np.median([row["seam_"+str(x)+"_mean_luma"]for row in samples])),
            "dark_luma_p95":float(np.percentile([row["seam_"+str(x)+"_mean_luma"]for row in samples],95)),
        }for x in manifest["divider_x"]},
        "warnings":[
            "This is color-marker tracking of different visible surfaces, NOT 3D ground truth.",
            "Magenta tip centroid can move with occlusion/deformation even for synchronized rigid 3D.",
            "A single three-panel AI-generated video sharing a timeline does NOT prove physical 3D sync.",
            "Only fixed-view camera panels and tail-tip coherence are evaluated here.",
            "Do not apply any motion.json without verifying all-frame joint consistency.",
        ]
    }
    (outdir/"threeview_audit.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))
    return checks


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("video","reference","layout","output_dir"):p.add_argument(n)
    a=p.parse_args()
    run(a.video,a.reference,a.layout,a.output_dir)
