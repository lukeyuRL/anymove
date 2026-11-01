"""Temporal joint tracking fusion on DINOv2 detections (no new model needed).

The optical-flow stage is intentionally conservative. Lucas-Kanade tracks
IMAGE pixels, not anatomical skeleton joints. It can fail on flat, occluded,
or moving surfaces. Backward/forward consistency, a finite outlier gate,
and a maximum flow-only streak keep incorrect tracks from overwriting all
of the Reader's measurements. No claim of learned occlusion awareness.
"""
import numpy as np

PROFILE = "anymove_lk_reader_temporal_fusion_v1"
GUARDED_PROFILE = "anymove_lk_reader_temporal_fusion_v2_guarded"
# A conservative engineering guard. Not calibrated to Dino GT.
# An enormous Reader-vs-flow disagreement leaves the joint observation
# unresolved: do not pick either the raw Reader or flow as certain.
MAX_GUARDED_FLOW_READER_GAP_PX = 30.0
UNRESOLVED_TRACK_WEIGHT = 0.10
MAX_FB_ERROR_PX = 2.0
MAX_FLOW_STEP_PX = 45.0
# LK is measured at the RAW previous-frame joint. Once fusion diverges
# from that raw seed, a subsequent raw-seeded displacement is not a valid
# continuation of the corrected trajectory.
MAX_FLOW_SEED_OFFSET_PX = 7.0
MAX_READER_FLOW_AGREEMENT_PX = 9.0
LOW_PEAK_THRESHOLD = 0.50  # Heuristic raw peak, NOT a probability.
MAX_FLOW_ONLY_STREAK = 3
OUTLIER_WEIGHT = 0.40
FLOW_ONLY_WEIGHT = 0.55


def fuse_tracks(raw_uv, raw_peak, flow_next, flow_valid, fb_error,
                view_width, view_height, guard_large_disagreements=False):
    """Fuse (T,J,2) DINO positions with per-step LK displacement.

    flow_next[t,j] is tracking raw_uv[t-1,j] to the next image, not an
    independent absolute joint estimate. No change to raw heatmap peaks.
    When guard_large_disagreements=True (opt-in v2), a huge LOW-PEAK
    raw-vs-flow disagreement is ambiguous: retain raw coordinates but
    strongly downweight their Solver contribution, rather than replacing
    the joint with a 30-170px flow extrapolation. This is abstention,
    not evidence that raw Reader or LK is anatomically correct.
    Returns refined_uv, bounded solver weight multiplier, audit.
    """
    if not isinstance(guard_large_disagreements, (bool, np.bool_)):
        raise ValueError("guard_large_disagreements must be boolean")
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
              "outlier_flagged": 0, "flow_rejected": 0,
              "flow_seed_rejected": 0, "quarantined_large_disagreement": 0}
    for t in range(1, T):
        delta = next_uv[t] - uv[t-1]
        step = np.linalg.norm(delta, axis=-1)
        prediction = adjusted[t-1] + delta
        in_bounds = ((prediction[:, 0] >= 0) & (prediction[:, 0] < view_width) &
                     (prediction[:, 1] >= 0) & (prediction[:, 1] < view_height))
        seed_offset = np.linalg.norm(adjusted[t-1] - uv[t-1], axis=-1)
        usable_seed = seed_offset <= MAX_FLOW_SEED_OFFSET_PX
        trustworthy = (valid[t] & (err[t] <= MAX_FB_ERROR_PX) &
                       (step <= MAX_FLOW_STEP_PX) & in_bounds &
                       usable_seed)
        counts["flow_seed_rejected"] += int(np.sum(valid[t] & ~usable_seed))
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
                # Flow-only recovery is permitted for a short low-peak window,
                # but a 100+ pixel disagreement is *not* proof of reliable flow:
                # a wrongly identified raw landmark can still pass LK F/B.
                if (guard_large_disagreements and
                        discrepancy[j] > MAX_GUARDED_FLOW_READER_GAP_PX):
                    # Decline choosing either candidate. Keep the raw Reader
                    # location to preserve the detection contract, but make
                    # the Solver rely chiefly on other joints and the rig prior.
                    weights[t, j] = UNRESOLVED_TRACK_WEIGHT
                    streak[j] = 0
                    counts["quarantined_large_disagreement"] += 1
                    counts["reader_retained"] += 1
                else:
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
        "profile": GUARDED_PROFILE if guard_large_disagreements else PROFILE,
        "guard_large_disagreements": bool(guard_large_disagreements),
        "quarantine_flow_reader_gap_px": (
            MAX_GUARDED_FLOW_READER_GAP_PX if guard_large_disagreements else None),
        "unresolved_solver_weight": (
            UNRESOLVED_TRACK_WEIGHT if guard_large_disagreements else None),
        "frames": T,
        "joints": J,
        "counts": counts,
        "mean_reader_to_tracked_px": round(float(
            np.linalg.norm(adjusted-uv, axis=-1).mean()), 4),
        "max_reader_to_tracked_px": round(float(
            np.linalg.norm(adjusted-uv, axis=-1).max()), 4),
        "mean_solver_weight": round(float(weights.mean()), 4),
        "unmodified_reader_peak_scores": True,
        "flow_seed_max_offset_px": MAX_FLOW_SEED_OFFSET_PX,
        "note": ("LK point tracking is image evidence only, not joint identity "
                 "or a calibrated occlusion/uncertainty estimate. A large "
                 "disagreement is unresolved, not a guaranteed LK failure; "
                 "the opt-in guard rejects these replacements conservatively.")
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
