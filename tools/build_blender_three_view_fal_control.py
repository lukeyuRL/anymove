"""Make a 100% Blender-synchronized Dino three-view control in EXACT fal panel layout.

This is NOT AI video. Identical true 3D joints are rendered Front, Side,
Top at each corresponding frame, then downscaled/letterboxed to same
1280x720 three-panel input and encoded as the 1344x768 24fps generated
H3 Max output geometry. Use to estimate 2D Reader and marker-analysis
errors caused by resolution, projection, surface visibility and encoding.
"""
import argparse,hashlib,json,pathlib
import cv2,numpy as np

def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1048576),b""):h.update(c)
    return h.hexdigest()

def run(gt_dir,top_dir,layout_file,new_out):
    gt_dir,top_dir=map(pathlib.Path,(gt_dir,top_dir))
    layout_file=pathlib.Path(layout_file)
    output=pathlib.Path(new_out)
    if output.exists():raise FileExistsError("refuse overwrite existing GT control")
    layout=json.loads(layout_file.read_text(encoding="utf-8"))
    if layout.get("profile")!="anymove.h3max_dino_front_side_top_reference_smoke_v1":
        raise ValueError("unexpected fal triptych layout profile")
    gtpath=gt_dir/"joints.json"
    gt=json.loads(gtpath.read_text(encoding="utf-8"))
    T=len(gt["joints"])
    if T<20 or len(gt["names"])!=92 or abs(float(gt["fps"])-24)>1e-5:
        raise ValueError("expected 24fps Blender 92 joint GT video")
    topaudit=json.loads((top_dir/"top_render_audit.json").read_text(encoding="utf-8"))
    if (topaudit["blender_truth_sha256"]!=digest(gtpath) or topaudit["frames"]!=T):
        raise ValueError("Top renderer and original Front/Side joint GT mismatch")
    output.mkdir(parents=True,exist_ok=False)
    movie=output/"blender_threeview_control.mp4"
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*"mp4v"),24,(1344,768))
    if not writer.isOpened():raise RuntimeError("could not create Blender control MP4")
    bg=np.array([134,130,128],np.float32)
    try:
        for t in range(T):
            canvas=np.empty((720,1280,3),np.uint8)
            canvas[:]=bg.astype(np.uint8)
            for view in ("front","side","top"):
                src=((top_dir/"top")if view=="top" else(gt_dir/view))/f"f_{t:04d}.png"
                rgba=cv2.imread(str(src),cv2.IMREAD_UNCHANGED)
                spec=layout["panels"][view]
                if rgba is None or rgba.shape!=(spec["source_height"],spec["source_width"],4):
                    raise ValueError("missing/misframed Blender view "+str(src))
                x,y,w,h=spec["render_left"],spec["render_top"],spec["render_width"],spec["render_height"]
                r=cv2.resize(rgba,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)
                bgr,alpha=r[...,:3],r[...,3:4]/255.
                canvas[y:y+h,x:x+w]=(bgr*alpha+(1-alpha)*bg).clip(0,255).astype(np.uint8)
            for x in layout["divider_x"]:
                canvas[:,x-2:x+3,:]=(38,34,34)
            frame=cv2.resize(canvas,(1344,768),interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            if t%25==0 or t==T-1:
                print("BLENDER CONTROL frame",t+1,"/",T,flush=True)
    finally:
        writer.release()
    (output/"first_frame_3d_pose.json").write_text(json.dumps({
        "joint_names":gt["names"],"world_joints":gt["joints"][0],
        "profile":"gt_frame0_not_rest_for_true_synchronized_control"
    },indent=2),encoding="utf-8")
    info={
        "profile":"anymove.dino_blender_perfectly_synced_triptych_control_v1",
        "frames":T,"fps":24,"video_pixels":[1344,768],
        "ground_truth_gt_sha256":digest(gtpath),
        "source_layout_sha256":digest(layout_file),
        "top_render_manifest_sha256":digest(top_dir/"top_render_audit.json"),
        "video_sha256":digest(movie),
        "note":"Perfectly synchronized Blender 3D truth; designed to bound Reader/color measurement error after same panel scaling and encoding"
    }
    (output/"control_manifest.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info,indent=2))
    return movie


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("gt_dir","top_dir","layout_file","new_output_dir"):
        p.add_argument(key)
    a=p.parse_args()
    run(a.gt_dir,a.top_dir,a.layout_file,a.new_output_dir)
