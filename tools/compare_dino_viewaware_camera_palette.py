"""Create static original-vs-camera-aware Dino material comparisons."""
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(r"F:\anymove\runs\dino")
ORIG=ROOT/"p4_blind_fourview_v1"
DEST=ROOT/"skinmax_camera_aware_v1/test_render_v4"
def composite(p):
    rgba=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    if rgba is None or len(rgba.shape)!=3 or rgba.shape[2]!=4:
        raise ValueError("expected RGBA PNG for comparison")
    alpha=rgba[:,:,3:4].astype(np.float32)/255.
    bg=np.full_like(rgba[:,:,:3],65)
    return (rgba[:,:,:3].astype(np.float32)*alpha+bg*(1-alpha)).clip(0,255).astype(np.uint8)
for pose in (9001,9002,9007):
    for view in ("front","side"):
        ref=composite(ORIG/f"{pose:05d}_{view}.png")
        new=composite(DEST/f"{pose:05d}_{view}.png")
        if ref.shape!=new.shape:raise ValueError("camera aspect mismatch")
        h,w=ref.shape[:2]
        strip=np.full((h+52,w*2,3),45,np.uint8)
        strip[52:,:w]=ref
        strip[52:,w:]=new
        cv2.putText(strip,"Original Texture",(14,33),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),2,cv2.LINE_AA)
        cv2.putText(strip,"View-aware 41 SkinMax Colors",(w+14,33),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),2,cv2.LINE_AA)
        f=DEST/f"{pose:05d}_{view}_ab_compare.png"
        if not cv2.imwrite(str(f),strip):raise RuntimeError("could not save composite")
        print("PREVIEW",f,"size",w*2,h+52)
