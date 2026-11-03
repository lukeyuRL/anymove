"""Build a 2x2 static image reference for fal H3 Max from an EXISTING
single-pose four-view Blender RGBA set. No animation or video is generated.

Crops each view to its actual alpha silhouette and centers it into a
fixed-size equally divided 16:9 panel, preserving original aspect ratio.
Outputs one clean image for I2V and a labeled review-only copy.
"""
from pathlib import Path
import cv2,numpy as np,json,hashlib
SRC=Path(r"F:\anymove\runs\dino\skinmax_camera_aware_v1\test_render_v4")
OUT=Path(r"F:\anymove\runs\dino\fal_h3max_multiview_v1")
POSE=9001
VIEWS=(("front","FRONT"),("side","SIDE"),("back","BACK"),("opposite_side","OPPOSITE SIDE"))
WIDTH,HEIGHT=2048,1152
CW,CH=WIDTH//2,HEIGHT//2
OUT.mkdir(parents=True,exist_ok=True)
clean=OUT/"dino_skinmax41_fourview_2x2_reference_2048x1152.png"
review=OUT/"dino_skinmax41_fourview_2x2_labeled_2048x1152.png"
manifest=OUT/"reference_manifest.json"
for p in (clean,review,manifest):
    if p.exists():raise FileExistsError("refusing to overwrite existing fal preparation output: "+str(p))
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
canvas=np.full((HEIGHT,WIDTH,3),(78,80,84),np.uint8)
labels=canvas.copy()
result=[]
for i,(view,title) in enumerate(VIEWS):
    path=SRC/f"{POSE:05d}_{view}.png"
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim!=3 or im.shape[2]!=4:
        raise ValueError("RGBA input missing: "+str(path))
    h,w=im.shape[:2]
    mask=(im[:,:,3]>8).astype("uint8")
    if not mask.any():raise ValueError("Dino alpha mask empty")
    x,y,boxw,boxh=cv2.boundingRect(mask)
    # Preserve some empty studio space so fins/limbs never meet panel frame.
    pad=max(16,round(max(boxw,boxh)*0.10))
    left=max(0,x-pad);top=max(0,y-pad)
    right=min(w,x+boxw+pad);bottom=min(h,y+boxh+pad)
    crop=im[top:bottom,left:right]
    rw,rh=crop.shape[1],crop.shape[0]
    reserve_x,reserve_y=55,52
    scale=min((CW-2*reserve_x)/rw,(CH-2*reserve_y)/rh)
    new_w=round(rw*scale);new_h=round(rh*scale)
    if new_w<3 or new_h<3:raise ValueError("view crop or scale invalid")
    color=cv2.resize(crop[:,:,:3],(new_w,new_h),interpolation=cv2.INTER_LINEAR)
    alpha=cv2.resize(crop[:,:,3],(new_w,new_h),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
    xo=(i%2)*CW+(CW-new_w)//2
    yo=(i//2)*CH+(CH-new_h)//2
    bg=canvas[yo:yo+new_h,xo:xo+new_w,:].astype(np.float32)
    comp=color.astype(np.float32)*alpha[:,:,None]+bg*(1-alpha[:,:,None])
    canvas[yo:yo+new_h,xo:xo+new_w,:]=np.clip(comp,0,255).astype(np.uint8)
    result.append({"view":view,"source":str(path),"sha256":sha(path),
                   "original_dimensions":[w,h],
                   "alpha_bbox":[x,y,boxw,boxh],
                   "crop_bbox":[left,top,right-left,bottom-top],
                   "composite_bbox":[xo,yo,new_w,new_h],
                   "scale":round(scale,5)})
# Subtle fixed crosshair seam: never obscures the creature silhouettes.
cv2.line(canvas,(CW,0),(CW,HEIGHT),(47,48,52),6,cv2.LINE_AA)
cv2.line(canvas,(0,CH),(WIDTH,CH),(47,48,52),6,cv2.LINE_AA)
labels=canvas.copy()
for i,(view,title) in enumerate(VIEWS):
    px=i%2*CW;py=i//2*CH
    cv2.rectangle(labels,(px+17,py+17),(px+380,py+66),(44,45,49),-1)
    cv2.putText(labels,title,(px+31,py+54),cv2.FONT_HERSHEY_SIMPLEX,1.05,(246,247,248),2,cv2.LINE_AA)
for path,img in ((clean,canvas),(review,labels)):
    if not cv2.imwrite(str(path),img):
        raise RuntimeError("could not write 2x2 collage: "+str(path))
    loaded=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if loaded is None or loaded.shape!=(HEIGHT,WIDTH,3):
        raise RuntimeError("collage dimensions mismatch on disk")
record={"schema":"anymove.fal_h3max_static_quad_reference/1",
    "input_pose_id":POSE,
    "layout":{"top_left":"front","top_right":"side","bottom_left":"back","bottom_right":"opposite_side"},
    "composition_dimensions":[WIDTH,HEIGHT],"aspect_ratio":"16:9",
    "clean_image_to_video_input":str(clean),"clean_sha256":sha(clean),
    "review_labeled_image":str(review),"review_sha256":sha(review),
    "source_views":result,
    "same_pose_all_views":True,
    "no_motion_frames_or_video_generated":True,
    "not_submitted_to_fal":True,
    "output_subject":"Dino original SkinMax41 rig appearance",
    "notes":"This is one static 2x2 image. It is NOT a motion-conditioned video; preserving synchronized four panes is a video-generation hypothesis, not a guarantee."}
manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf8")
print(json.dumps({"clean":str(clean),"review":str(review),"pose":POSE,"width":WIDTH,"height":HEIGHT,
                  "composite_rgb_difference":float(np.mean(np.abs(canvas.astype(float)-labels.astype(float)))),
                  "source_view_sha_match":all(sha(Path(x["source"]))==x["sha256"] for x in result),
                  "video_generated":False},indent=2))
