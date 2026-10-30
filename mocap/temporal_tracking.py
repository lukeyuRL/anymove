"""Temporal joint tracking fusion on DINOv2 detections (no new model needed).

The optical-flow stage is intentionally conservative. Lucas-Kanade tracks
IMAGE pixels, not anatomical skeleton joints. It can fail on flat, occluded,
or moving surfaces. Backward/forward consistency, a finite outlier gate,
and a maximum flow-only streak keep incorrect tracks from overwriting all
of the Reader's measurements. No claim of learned occlusion awareness.
"""
import numpy as np

PROFILE = "anymove_lk_reader_temporal_fusion_v1"
MAX_FB_ERROR_PX = 2.0
MAX_FLOW_STEP_PX = 45.0
MAX_READER_FLOW_AGREEMENT_PX = 9.0
LOW_PEAK_THRESHOLD = 0.50  # Heuristic raw peak, NOT a probability.
MAX_FLOW_ONLY_STREAK = 3
OUTLIER_WEIGHT = 0.40
FLOW_ONLY_WEIGHT = 0.55


def fuse_tracks(raw_uv, raw_peak, flow_next, flow_valid, fb_error,
                view_width, view_height):
    """Fuse (T,J,2) DINO positions with per-step LK displacement.

    flow_next[t,j] is tracking raw_uv[t-1,j] to the next image, not an
    independent absolute joint estimate. No change to raw heatmap peaks.
    Returns refined_uv, bounded solver weight multiplier, audit.
    """
    uv = np.asarray(raw_uv, dtype=np.float64)
    peaks = np.asarray(raw_peak, dtype=np.float64)
    next_uv = np.asarray(flow_next, dtype=np.float64)
    valid = np.asarray(flow_valid, dtype=bool)
    err = np.asarray(fb_error, dtype=np.float64)
    if (uv.ndim != 3 or uv.shape[-1] != 2 or uv.shape[0] < 2
            or next_uv.shape != uv.shape or peaks.shape != uv.shape[:2]
            or valid.shape != peaks.shape or err.shape != peaks.shape
            or not all(np.isfinite(v).all() for v in (uv, peaks, next_uv, err))
            or not (view_width > 0 and view_height > 0)):
        raise ValueError("invalid temporal tracking arrays")
    T, J = peaks.shape
    adjusted = uv.copy()
    weights = np.ones((T, J), np.float32)
    streak = np.zeros(J, np.int32)
    counts = {"reader_retained": 0, "fused": 0, "flow_only": 0,
              "outlier_flagged": 0, "flow_rejected": 0}
    for t in range(1, T):
        delta = next_uv[t] - uv[t-1]
        step = np.linalg.norm(delta, axis=-1)
        prediction = adjusted[t-1] + delta
        in_bounds = ((prediction[:, 0] >= 0) & (prediction[:, 0] < view_width) &
                     (prediction[:, 1] >= 0) & (prediction[:, 1] < view_height))
        trustworthy = (valid[t] & (err[t] <= MAX_FB_ERROR_PX) &
                       (step <= MAX_FLOW_STEP_PX) & in_bounds)
        discrepancy = np.linalg.norm(prediction - uv[t], axis=-1)
        for j in range(J):
            if not trustworthy[j]:
                streak[j] = 0
                counts["flow_rejected"] += 1
                continue
            if discrepancy[j] <= MAX_READER_FLOW_AGREEMENT_PX:
                # Keep Reader dominant when image flow agrees with detections.
                blend = .15 if discrepancy[j] < 4.0 else .35
                adjusted[t, j] = (1.-blend)*uv[t, j] + blend*prediction[j]
                streak[j] = 0
                counts["fused"] += 1
            elif peaks[t, j] < LOW_PEAK_THRESHOLD and streak[j] < MAX_FLOW_ONLY_STREAK:
                # Only extrapolate briefly when the raw peak is low AND
                # LK's forward/backward check is reliable.
                adjusted[t, j] = prediction[j]
                weights[t, j] = FLOW_ONLY_WEIGHT
                streak[j] += 1
                counts["flow_only"] += 1
            else:
                # High-confidence reader/flow disagreement is unresolved,
                # so keep Reader coordinates but warn/downweight the Solver.
                weights[t, j] = OUTLIER_WEIGHT
                streak[j] = 0
                counts["outlier_flagged"] += 1
                counts["reader_retained"] += 1
    diagnostics = {
        "profile": PROFILE,
        "frames": T,
        "joints": J,
        "counts": counts,
        "mean_reader_to_tracked_px": round(float(
            np.linalg.norm(adjusted-uv, axis=-1).mean()), 4),
        "max_reader_to_tracked_px": round(float(
            np.linalg.norm(adjusted-uv, axis=-1).max()), 4),
        "mean_solver_weight": round(float(weights.mean()), 4),
        "unmodified_reader_peak_scores": True,
        "note": ("LK point tracking is image evidence only, not joint identity "
                 "or a calibrated occlusion/uncertainty estimate.")
    }
    return adjusted.astype(np.float32), weights, diagnostics


def lk_measurements(frames, uv):
    """Pyramidal LK displacement from raw DINO positions for each frame.

    Import OpenCV lazily to keep CI's lightweight NumPy checks working.
    Source frames must already be calibrated/cropped to one view.
    """
    import cv2
    images = np.asarray(frames)
    track = np.asarray(uv, dtype=np.float32)
    if (images.ndim != 4 or images.shape[0] != track.shape[0]
            or images.shape[3] not in (3, 4) or track.ndim != 3
            or track.shape[-1] != 2):
        raise ValueError("LK frames and UV have incompatible shapes")
    T, J = track.shape[:2]
    predicted = np.array(track, copy=True)
    ok = np.zeros((T, J), dtype=bool)
    error = np.full((T, J), 1e6, dtype=np.float32)
    gray = [cv2.cvtColor(f[..., :3], cv2.COLOR_RGB2GRAY)
            for f in images]
    config = {"winSize": (25, 25), "maxLevel": 3,
              "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           30, 0.01),
              "minEigThreshold": 1e-4}
    for t in range(1, T):
        p = np.ascontiguousarray(track[t-1].reshape(-1, 1, 2))
        q, s1, _ = cv2.calcOpticalFlowPyrLK(gray[t-1], gray[t], p,
                                           None, **config)
        if q is None or s1 is None:
            continue
        back, s2, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t-1],
                                              q, None, **config)
        if back is None or s2 is None:
            continue
        q2 = q.reshape(J, 2)
        predicted[t] = q2
        error[t] = np.linalg.norm(back.reshape(J, 2)-track[t-1],
                                  axis=-1)
        ok[t] = (s1.reshape(J) == 1) & (s2.reshape(J) == 1)
    return predicted, ok, error
