"""Synthetic cutout supervision for an optional Reader quality head.

Targets are *synthetically masked keypoints*, not anatomical visibility.
A skeleton joint is inside the creature mesh; raycasting from the camera
directly toward its center would mislabel almost every internal joint as
occluded. Real self-occlusion needs a separate surface/part annotation pass.
"""
import numpy as np

PROFILE = "anymove_synthetic_cutout_visibility_v1"


def synthetic_cutout(img, keypoints, rng, probability=.70):
    """Return RGB image and binary label (1 not artificially masked).

    The input keypoints are already in the affine-warped model image space.
    No image or joint coordinate is modified in place. Supports points that
    project offscreen (their visibility target is ignored downstream).
    """
    image = np.asarray(img)
    kp = np.asarray(keypoints)
    if (image.ndim != 3 or image.shape[2] != 3 or kp.ndim != 2
            or kp.shape[1] != 2 or not np.isfinite(kp).all()):
        raise ValueError("synthetic cutout needs HxWx3 image and finite Jx2 points")
    if not 0 <= probability <= 1:
        raise ValueError("cutout probability outside [0,1]")
    h, w = image.shape[:2]
    valid = ((kp[:, 0] >= 0) & (kp[:, 0] < w) &
             (kp[:, 1] >= 0) & (kp[:, 1] < h))
    visible = np.ones(len(kp), dtype=np.float32)
    result = np.array(image, copy=True)
    if not valid.any() or float(rng.random()) >= probability:
        return result, visible
    ids = np.flatnonzero(valid)
    for _ in range(int(rng.integers(1, 4))):
        j = int(rng.choice(ids))
        size_x = int(rng.uniform(.06, .17) * w)
        size_y = int(rng.uniform(.06, .17) * h)
        cx = kp[j, 0]+rng.uniform(-.15, .15)*size_x
        cy = kp[j, 1]+rng.uniform(-.15, .15)*size_y
        l = max(0, int(round(cx-size_x*.5)))
        r = min(w, int(round(cx+size_x*.5)))
        t = max(0, int(round(cy-size_y*.5)))
        b = min(h, int(round(cy+size_y*.5)))
        if r <= l or b <= t:
            continue
        fill = rng.uniform(50, 195, size=(3,))
        result[t:b, l:r] = fill.astype(result.dtype)
        occluded = (kp[:, 0] >= l) & (kp[:, 0] < r) & \
                   (kp[:, 1] >= t) & (kp[:, 1] < b)
        visible[occluded] = 0
    return result, visible
