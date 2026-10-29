"""Uncalibrated DINOv2 heatmap peak scores -> bounded solver weights.

The joint reader's final layer is linear and trained using MSE against
Gaussian heatmaps. Raw peak amplitudes are NOT calibrated probabilities:
they can legitimately be negative or greater than one. Keep the original
det.npz intact for inspection and normalize only the in-memory weights
used by the pose solver.
"""
import numpy as np


def bounded_peak_weights(peaks, name="joint confidence"):
    """Return clipped [0,1] weights and lossless raw-peak diagnostics.

    Clipping is a weighting convention, NOT probability calibration or
    evidence that detected joint positions are correct. Nonfinite peak
    scores still fail closed; use the downstream reprojection/sync gates
    to determine whether a solved motion is usable.
    """
    raw=np.asarray(peaks)
    if raw.ndim!=2 or raw.size==0 or raw.dtype.kind not in "fiu":
        raise ValueError("%s needs a nonempty real (frames,joints) array"%name)
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite heatmap peaks in "+name)
    bounded=np.clip(raw,0.0,1.0)
    clipped=int(np.count_nonzero((raw<0.0)|(raw>1.0)))
    diagnostics={
        "raw_min":float(raw.min()),
        "raw_max":float(raw.max()),
        "raw_mean":float(raw.mean()),
        "bounded_mean":float(bounded.mean()),
        "clipped_samples":clipped,
        "total_samples":int(raw.size),
        "clipped_fraction":float(clipped/raw.size)
    }
    return bounded,diagnostics
