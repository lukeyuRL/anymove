"""Experimental P2 Reader localization-error => Solver weight multipliers.

In P2 quality-only training the Dino Reader's existing keypoint/heatmap
weights are frozen. The predicted error is *not* calibrated on real video.
This intentionally uses ONLY expected localization error, NEVER the
synthetic-cutout visibility score (which is not true mesh visibility).
Uncertain keypoints are downweighted rather than moved or dropped.
"""
import numpy as np

PROFILE = "anymove.p2_predicted_error_soft_weight_v1"


def quality_error_weights(predicted_error_px, *, threshold_px=5.0, floor=0.4):
    """Conservative, bounded heuristic; no assumption error == true sigma.

    w = clip(threshold_px/max(error_px,threshold_px), floor, 1).
    If predicted error <= threshold, w=1 and the original Solver is
    unaffected. Does not sharpen or shift any 2D detections.
    """
    err = np.asarray(predicted_error_px, dtype=np.float64)
    if (err.ndim != 2 or not err.size or
            not np.isfinite(err).all() or (err < 0).any()):
        raise ValueError("quality error must be nonnegative finite (T,J)")
    if (not np.isfinite(threshold_px) or threshold_px <= 0 or
            not np.isfinite(floor) or not 0 < floor <= 1):
        raise ValueError("invalid quality weighting threshold or floor")
    w = threshold_px / np.maximum(err, threshold_px)
    w = np.clip(w, floor, 1.)
    return w.astype(np.float32)


def _identical_reader(raw, p2, *, uv_tol_px=0.25, peak_tol=0.02):
    """Require the P2 Reader to preserve raw observations for a fair ablation.

    Low tolerances permit numerical differences across inference invocations.
    Larger differences suggest a changed model, wrong clip, or calibration.
    """
    required = ("front_uv", "side_uv", "front_conf", "side_conf",
                "names", "fps", "calib")
    for key in required:
        if key not in raw or key not in p2:
            raise ValueError("source/P2 detection missing required field: " + key)
    names = [str(n) for n in raw["names"]]
    if not names or len(set(names)) != len(names) or names != [str(n) for n in p2["names"]]:
        raise ValueError("source/P2 Reader keypoint names or order differ")
    fps_a, fps_b = float(raw["fps"]), float(p2["fps"])
    if not (np.isfinite(fps_a) and fps_a > 0 and np.isfinite(fps_b) and
            abs(fps_a-fps_b) < 0.05):
        raise ValueError("source/P2 FPS mismatch")
    ca, cb = np.asarray(raw["calib"]), np.asarray(p2["calib"])
    if (ca.shape != (2, 3) or cb.shape != (2, 3) or
            not np.isfinite(ca).all() or not np.isfinite(cb).all() or
            not np.allclose(ca, cb, rtol=0, atol=0.001)):
        raise ValueError("source/P2 video calibration differs")
    T,J = np.asarray(raw["front_conf"]).shape
    if T < 2 or J != len(names):
        raise ValueError("invalid source Reader observation count")
    report = {"frames": T, "keypoints": J, "fps": fps_a, "views": {}}
    for view in ("front", "side"):
        uv_a = np.asarray(raw[view+"_uv"], dtype=np.float64)
        uv_b = np.asarray(p2[view+"_uv"], dtype=np.float64)
        conf_a = np.asarray(raw[view+"_conf"], dtype=np.float64)
        conf_b = np.asarray(p2[view+"_conf"], dtype=np.float64)
        if (uv_a.shape != (T,J,2) or uv_b.shape != uv_a.shape or
                conf_a.shape != (T,J) or conf_b.shape != conf_a.shape or
                not all(np.isfinite(v).all() for v in (uv_a,uv_b,conf_a,conf_b))):
            raise ValueError(view+": invalid source/P2 observation dimensions")
        max_uv = float(np.linalg.norm(uv_a-uv_b, axis=-1).max())
        max_conf = float(np.abs(conf_a-conf_b).max())
        if max_uv > uv_tol_px or max_conf > peak_tol:
            raise ValueError(
                view+": P2 changed existing Reader observations "
                "(max UV %.3fpx / peak %.4f). Re-run same VIDEO/ref/model; "
                "do not mix Joint Reader training with this weight-only experiment"
                % (max_uv,max_conf))
        report["views"][view] = {
            "max_uv_difference_px": round(max_uv, 5),
            "max_raw_peak_difference": round(max_conf, 6),
        }
    return report


def prepare_quality_weighted_detection(raw, p2, *, threshold_px=5., floor=.4):
    """Return fresh NPZ-ready copy of ORIGINAL Reader observations plus weights.

    The original first-party Solver already recognizes front_track_weight /
    side_track_weight. These are reused as generic multipliers, but this is a
    P2 error-weight ablation, not P1 optical-flow tracking. Never stack
    this experiment onto an existing P1 tracked detection.
    """
    if any(key.endswith("_track_weight") for key in raw):
        raise ValueError("P2 weight-only ablation requires ORIGINAL, non-P1 detection")
    comparable = _identical_reader(raw,p2)
    result = {key: np.array(val, copy=True) for key,val in raw.items()}
    stats = {}
    T,J = len(raw["front_uv"]), len(raw["names"])
    for view in ("front","side"):
        err_key = view+"_expected_error_px"
        if err_key not in p2 or view+"_synthetic_visibility_score" not in p2:
            raise ValueError("P2 checkpoint lacks Quality Head outputs: "+view)
        err = np.asarray(p2[err_key], dtype=float)
        vis = np.asarray(p2[view+"_synthetic_visibility_score"], dtype=float)
        if (err.shape != (T,J) or vis.shape != (T,J) or
                not np.isfinite(vis).all() or (vis < 0).any() or (vis > 1).any()):
            raise ValueError("invalid P2 quality shapes or synthetic score range: "+view)
        weights = quality_error_weights(err,threshold_px=threshold_px,floor=floor)
        result[view+"_track_weight"] = weights
        # Keep predictions as diagnostics, not as true visibility.
        result[err_key] = err.astype(np.float32)
        result[view+"_synthetic_visibility_score"] = vis.astype(np.float32)
        stats[view] = {
            "mean_expected_error_px": round(float(err.mean()),4),
            "p95_expected_error_px": round(float(np.percentile(err,95)),4),
            "mean_solver_multiplier": round(float(weights.mean()),5),
            "downweighted_joint_frames": int((weights < 1).sum()),
            "strongly_downweighted_joint_frames": int((weights <= floor + 1e-6).sum()),
            "total_joint_frames": int(T*J),
        }
    audit = {
        "profile": PROFILE,
        "source_reader_unchanged": comparable,
        "threshold_px": threshold_px,
        "floor": floor,
        "weight_semantics": (
            "Only predicted localization-error heuristic. Not calibrated "
            "real video pixel error or true mesh visibility. Synthetic "
            "visibility score is logged but NEVER used as a Solver weight."),
        "views": stats,
        "experimental": True,
    }
    return result, audit
