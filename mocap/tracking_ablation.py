"""Separate Dino P1 coordinate and Solver-weight effects.

Given original det.npz and P1 tracked det.npz:
  UV-only:     tracked coordinates + original Reader peak weights
  weight-only: original coordinates + P1 tracking weight multipliers
  full P1:     tracked det.npz as already produced by mocap.track
All source detections stay immutable; use distinct Solver outputs.
"""
import numpy as np


def make_tracking_ablations(raw, tracked):
    names = [str(n) for n in raw["names"]]
    if names != [str(n) for n in tracked["names"]] or len(set(names)) != len(names):
        raise ValueError("raw/tracked joint order is incompatible")
    if abs(float(raw["fps"]) - float(tracked["fps"])) > 1e-6:
        raise ValueError("raw/tracked FPS mismatch")
    if any(k.endswith("_track_weight") for k in raw):
        raise ValueError("raw detection already includes P1 tracking weights")
    if ("calib" in raw and "calib" in tracked and
            not np.array_equal(raw["calib"], tracked["calib"])):
        raise ValueError("different video calibration")
    uv_only = {k: np.array(v, copy=True) for k, v in raw.items()}
    weight_only = {k: np.array(v, copy=True) for k, v in raw.items()}
    for view in ("front", "side"):
        u = view + "_uv"
        c = view + "_conf"
        w = view + "_track_weight"
        u_orig = view + "_uv_reader"
        if (not all(key in raw for key in (u, c)) or
                not all(key in tracked for key in (u, c, w, u_orig))):
            raise ValueError("missing required P1 detection arrays")
        a = np.asarray(raw[u])
        b = np.asarray(tracked[u])
        peak = np.asarray(raw[c])
        multiplier = np.asarray(tracked[w])
        if (a.ndim != 3 or a.shape[-1] != 2 or
                a.shape[1] != len(names) or b.shape != a.shape or
                peak.shape != a.shape[:2] or
                np.asarray(tracked[c]).shape != peak.shape or
                multiplier.shape != peak.shape or
                not all(np.isfinite(x).all() for x in (
                    a, b, peak, np.asarray(tracked[c]), multiplier)) or
                np.any(multiplier < 0) or np.any(multiplier > 1)):
            raise ValueError("invalid P1 input shape, weights or finite values")
        if not np.array_equal(tracked[c], peak):
            raise ValueError("P1 modified raw Reader peaks")
        if not np.array_equal(tracked[u_orig], a):
            raise ValueError("tracked Reader source does not match original UV")
        uv_only[u] = b.copy()
        weight_only[w] = multiplier.copy()
    return uv_only, weight_only
