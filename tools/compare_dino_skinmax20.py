"""Static color20 A/B comparison, no inference or video."""
import cv2
import numpy as np
from pathlib import Path
root=Path(r"F:\anymove\runs\dino")
src=root/"p4_blind_fourview_v1"
dst=root/"color20_candidate_v1/test_render_v1"

def composite(path):
    raw=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if raw is None or raw.shape[2]!=4:
        raise RuntimeError("missing RGBA render "+str(path))
    bg=np.empty_like(raw[:,:,:3]);bg[:]=(65,65,65)
    a=raw[:,:,3:4].astype(np.float32)/255.
    return (raw[:,:,:3]*a+bg*(1-a)).astype(np.uint8)

for idx in (9001,9002):
    for view in ("front","side"):
        left=composite(src/f"{idx:05d}_{view}.png")
        right=composite(dst/f"{idx:05d}_{view}.png")
        h,w=left.shape[:2]
        out=np.empty((h+46,w*2,3),np.uint8);out[:]=(37,37,37)
        out[46:,:w]=left;out[46:,w:]=right
        for x,title in ((12,"Original Material"),(w+12,"SkinMax20 Dominant Bone Colors")):
            cv2.putText(out,title,(x,31),cv2.FONT_HERSHEY_SIMPLEX,.63,(245,245,245),2,cv2.LINE_AA)
        path=dst/f"{idx:05d}_{view}_original_vs_skinmax20.png"
        if not cv2.imwrite(str(path),out):
            raise RuntimeError("Comparison PNG cannot be saved")
        print("A_B_COMPARISON",path,tuple(out.shape))
