"""Compose H3 Max four-view Reset Pose static image: no video submission.

All four images must originate from the exact 67-bone identity pose of
original rigged.blend. The complete 2x2 collage is a single image for I2V.
"""
from pathlib import Path
import hashlib,json,cv2
import numpy as np
ROOT=Path(r"F:\anymove\runs\dino\fal_h3max_multiview_v1")
SRC=ROOT/"reset_pose_color41_render"
SOURCE=ROOT/"reset_pose_source/12001_meta.json"
POSE=12001
W,H=2048,1152
TILES=(("front","FRONT"),("side","SIDE"),("back","BACK"),("opposite_side","OPPOSITE SIDE"))
REF=ROOT/"dino_skinmax41_RESET_fourview_reference_2048x1152.png"
REVIEW=ROOT/"dino_skinmax41_RESET_fourview_labeled_2048x1152.png"
MANIFEST=ROOT/"reset_pose_reference_manifest.json"
for p in (REF,REVIEW,MANIFEST):
    if p.exists():raise FileExistsError("Refusing to overwrite existing H3 reset reference: "+str(p))
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()
meta=json.loads(SOURCE.read_text(encoding="utf8"))
if (meta.get("schema")!="anymove.dino_zero_transform_reset_pose_reference/1" or
    len(meta["pose"])!=67 or
    any(any(abs(float(a))>1e-7 for a in xyz) for xyz in meta["pose"].values()) or
    meta["root"]!=[0.,0.,0.] or meta["yaw"]!=0. or
    meta["blur"] or not meta["root_identity"]):
    raise ValueError("not the verified Dino Reset Pose")
frame=np.full((H,W,3),(78,80,84),dtype=np.uint8)
records=[]
for i,(view,label) in enumerate(TILES):
    source=SRC/f"{POSE:05d}_{view}.png"
    manifest=SRC/f"{POSE:05d}_visibility_meta.json"
    m=json.loads(manifest.read_text(encoding="utf8"))
    if (m["sample_index"]!=POSE or m.get("viewaware_color_count")!=41 or
            m["source_meta_sha256"]!=sha(SOURCE)):
        raise ValueError("unexpected pose/color or mismatched source metadata")
    image=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=3 or image.shape[-1]!=4:
        raise ValueError("Dino 4-view Reset Pose requires source RGBA")
    hh,ww=image.shape[:2]
    if (ww,hh)!=((400,720) if view in ("front","back") else (880,720)):
        raise ValueError("source camera dimensions mismatch")
    alpha=image[:,:,3]
    if not np.any(alpha>8):raise ValueError("Reset Pose Dino surface not visible")
    x,y,bw,bh=cv2.boundingRect((alpha>8).astype(np.uint8))
    pad=max(16,round(max(bw,bh)*0.10))
    left,top=max(0,x-pad),max(0,y-pad)
    right,bottom=min(ww,x+bw+pad),min(hh,y+bh+pad)
    crop=image[top:bottom,left:right]
    cw,ch=crop.shape[1],crop.shape[0]
    TILE_W,TILE_H=W//2,H//2
    scale=min((TILE_W-110)/cw,(TILE_H-104)/ch)
    tw,th=max(1,round(cw*scale)),max(1,round(ch*scale))
    rgb=cv2.resize(crop[:,:,:3],(tw,th),interpolation=cv2.INTER_LINEAR)
    a=cv2.resize(crop[:,:,3],(tw,th),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
    ox=i%2*TILE_W+(TILE_W-tw)//2
    oy=i//2*TILE_H+(TILE_H-th)//2
    before=frame[oy:oy+th,ox:ox+tw].astype(np.float32)
    comp=rgb.astype(np.float32)*a[...,None]+before*(1-a[...,None])
    frame[oy:oy+th,ox:ox+tw]=np.clip(comp,0,255).astype(np.uint8)
    records.append({"view":view,"source":str(source),"sha256":sha(source),
                    "alpha_bbox":[x,y,bw,bh],
                    "composite_bbox":[ox,oy,tw,th],
                    "preserved_aspect_ratio":True})
cv2.line(frame,(W//2,0),(W//2,H),(47,48,52),6,cv2.LINE_AA)
cv2.line(frame,(0,H//2),(W,H//2),(47,48,52),6,cv2.LINE_AA)
labeled=frame.copy()
for i,(view,label) in enumerate(TILES):
    x0=i%2*(W//2); y0=i//2*(H//2)
    cv2.rectangle(labeled,(x0+17,y0+17),(x0+380,y0+66),(44,45,49),-1)
    cv2.putText(labeled,label,(x0+31,y0+54),cv2.FONT_HERSHEY_SIMPLEX,1.05,
                (246,247,248),2,cv2.LINE_AA)
for p,img in ((REF,frame),(REVIEW,labeled)):
    if not cv2.imwrite(str(p),img):raise RuntimeError("Could not write Reset Pose collage")
    check=cv2.imread(str(p))
    if check is None or check.shape!=(H,W,3):raise RuntimeError("Wrong resolution")
result={"schema":"anymove.fal_h3max_dino_RESET_quad_reference/1",
    "reset_pose_is_identity":True,
    "source_pose_id":POSE,
    "reset_pose_metadata_path":str(SOURCE),
    "reset_pose_source_sha256":sha(SOURCE),
    "root_translation":[0,0,0],"bone_count":67,"bone_euler_rotations_all_zero":True,
    "camera_layout":{"top_left":"front","top_right":"side","bottom_left":"back",
                     "bottom_right":"opposite_side"},
    "composition_dimensions":[W,H],
    "clean_image_for_h3_image_to_video":str(REF),"clean_sha256":sha(REF),
    "preview_image":str(REVIEW),"preview_sha256":sha(REVIEW),
    "source_images":records,"same_pose_all_views":True,
    "no_animation_or_video_generated":True,"fal_video_not_submitted":True,
    "notes":"Exact rest/reset pose with 41-colour material, not synthetic pose 9001."
}
MANIFEST.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
print(json.dumps({"status":"PASS","clean":str(REF),"labeled":str(REVIEW),
                 "reset_pose_identity":True,"joint_count":len(meta["joints_world"]),
                 "source_sha":sha(SOURCE),"image_width":W,"image_height":H,
                 "no_fal_video_submission":True},indent=2))
