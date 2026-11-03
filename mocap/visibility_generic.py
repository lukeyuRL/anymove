"""Creature-generic TWO-VIEW joint observation reliability (opt-in).

This is not mesh raycast visibility, not P(visible), and not a trained
uncertainty model. It detects *geometric reasons to distrust 2D keypoints*:
  1. One projection of a rigid bone exceeds its 3D rest length.
  2. Cross-view shared-height disagreement, plus stronger projection overlap
     against an anatomically separate part in just ONE of the two views.

All correspondence, rest positions, parent graph, and bone segments come from
rig_info.json; no Dragon/Dino names or thresholds are special-cased.
Side is never treated as ground truth. Ambiguous two-view disagreements abstain.
These weights may not be used as a single-view visibility oracle: the
cross-view test *uses both camera observations*.

The current contract has two orthographic front/side cameras sharing height
and the same metres-per-pixel. A future N-view solver requires arbitrary
camera-matrix geometry rather than this 2-view specialization.
"""
from __future__ import annotations

import math

import numpy as np


PROFILE = "anymove.generic_two_view_reliability_proxy/1"
FLOOR = 0.45
GAP_START_PX = 6.0
GAP_FULL_PX = 24.0
OVERLAP_CLOSE_PX = 10.0
OVERLAP_FAR_PX = 36.0
MIN_OVERLAP_ADVANTAGE = 0.15
MEDIAN_WINDOW = 5


def _smooth_median(x: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 != 1:
        raise ValueError("median_window must be a positive odd integer")
    if window == 1:
        return x.copy()
    n = window // 2
    extended = np.pad(x, (n, n), mode="edge")
    return np.median(np.stack(
        [extended[i:i + len(x)] for i in range(window)], axis=0), axis=0)


def _validate(d, names, info, *, floor, gap_start_px, gap_full_px,
              overlap_close_px, overlap_far_px, median_window):
    if not isinstance(info, dict) or not info.get("bones") or not info.get("keypoints"):
        raise ValueError("generic visibility needs rig_info bones and keypoints")
    bones = info["bones"]
    kp = info["keypoints"]
    expected = [p["name"] for p in kp]
    observed = [str(n) for n in names]
    if (len(expected) != len(set(expected)) or observed != expected
            or len(set(observed)) != len(observed)):
        raise ValueError("generic visibility requires exact rig keypoint order/names")
    if not isinstance(info.get("layout"), dict):
        raise ValueError("generic visibility requires a verified two-view camera layout")
    layout = info["layout"]
    for key in ("H", "W_FRONT", "W_SIDE", "M_PER_PX", "Z_CENTER"):
        if key not in layout or not np.isfinite(layout[key]):
            raise ValueError("generic visibility missing finite camera layout: " + key)
    if min(float(layout["H"]), float(layout["W_FRONT"]), float(layout["W_SIDE"]),
           float(layout["M_PER_PX"])) <= 0:
        raise ValueError("generic visibility camera dimensions and pixel scale must be positive")
    if not (0 < float(floor) <= 1) or not all(np.isfinite(x) for x in
            (floor, gap_start_px, gap_full_px, overlap_close_px, overlap_far_px)):
        raise ValueError("generic visibility invalid finite parameters")
    if not (0 <= gap_start_px < gap_full_px and
            0 <= overlap_close_px < overlap_far_px):
        raise ValueError("generic visibility invalid gap/overlap thresholds")
    if not isinstance(median_window, int) or median_window < 1 or median_window % 2 == 0:
        raise ValueError("generic visibility median window must be odd positive")
    views = {}
    shapes = set()
    for view in ("front", "side"):
        key = view + "_uv"
        if key not in d:
            raise ValueError("generic visibility missing " + key)
        uv = np.asarray(d[key], dtype=np.float64)
        if (uv.ndim != 3 or uv.shape[-1] != 2 or uv.shape[0] < 2
                or uv.shape[1] != len(expected) or not np.isfinite(uv).all()):
            raise ValueError("generic visibility requires finite (frames,keypoints,2): "+key)
        views[view] = uv
        shapes.add(uv.shape)
    if len(shapes) != 1:
        raise ValueError("generic visibility views are not frame/keypoint aligned")
    parents = []
    for i, bone in enumerate(bones):
        p = bone.get("parent")
        if not isinstance(p, int) or p >= i or p < -1:
            raise ValueError("generic visibility requires ordered, valid rig parent indices")
        parents.append(p)
    if parents[0] != -1 or sum(p == -1 for p in parents) != 1:
        raise ValueError("generic visibility requires one topological root")
    rest = np.asarray([p["rest"] for p in kp], dtype=np.float64)
    if rest.shape != (len(kp), 3) or not np.isfinite(rest).all():
        raise ValueError("generic visibility requires finite 3D keypoint rest coordinates")
    bone_id = np.asarray([p["bone"] for p in kp])
    if (bone_id.shape != (len(kp),) or not np.issubdtype(bone_id.dtype, np.integer)
            or np.any(bone_id < 0) or np.any(bone_id >= len(bones))):
        raise ValueError("generic visibility requires valid keypoint-to-bone mapping")
    for b in bones:
        if b["head_joint"] not in expected or b["tail_joint"] not in expected:
            raise ValueError("generic visibility bone head/tail missing in keypoints")
    return views, rest, bone_id, parents, layout


def _tree_distance(a, b, parents) -> int:
    if a == b:
        return 0
    dist_a = {}
    n = a
    step = 0
    while n >= 0:
        dist_a[n] = step
        n = parents[n]
        step += 1
    n = b
    step = 0
    while n >= 0:
        if n in dist_a:
            return step + dist_a[n]
        n = parents[n]
        step += 1
    raise ValueError("disconnected rig hierarchy")


def _remote_anatomy_mask(rest, bone_id, parents) -> np.ndarray:
    """Exclude same/nearby chain and near-coincident rest landmarks.

    No side-name, character-name, species, or anatomical vocabulary is used.
    Sibling limbs at >=2 bone hops are allowed when spatially separated.
    """
    J = len(rest)
    extents = np.ptp(rest, axis=0)
    body_size = max(float(np.linalg.norm(extents)), 1e-4)
    min_separation = max(.04 * body_size, 1e-6)
    pair_dist = np.linalg.norm(rest[:, None, :] - rest[None, :, :], axis=-1)
    allowed = (pair_dist > min_separation)
    np.fill_diagonal(allowed, False)
    # Near-connected bones are generally the same moving part. Reject all
    # distance <=1; allow siblings only if rest positions separate enough.
    max_bone = len(parents)
    separated = np.zeros((max_bone, max_bone), dtype=bool)
    for i in range(max_bone):
        for j in range(i+1, max_bone):
            if _tree_distance(i, j, parents) >= 2:
                separated[i, j] = separated[j, i] = True
    allowed &= separated[bone_id[:, None], bone_id[None, :]]
    # The root anchors framing and tracking. Never infer its reliability
    # from overlapping body parts; root is scored only by its own Reader.
    allowed[bone_id == 0, :] = False
    return allowed


def _overlap_strength(uv, compatible, *, close_px, far_px,
                      partner_valid) -> tuple[np.ndarray, np.ndarray]:
    differences = uv[:, :, None, :] - uv[:, None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    valid = compatible[None, :, :] & partner_valid[:, None, :]
    distances = np.where(valid, distances, np.inf)
    closest = distances.min(axis=-1)
    strength = np.clip((far_px - closest) / (far_px - close_px), 0., 1.)
    return strength, closest


def _bone_projection_violation(uv, names, info, layout, *,
                                floor, weights, details, median_window):
    """Hard geometric inequality for ORTHOGRAPHIC cameras:
       sqrt(dx^2 + dz^2) <= length(rest 3D bone).
    Marks distal observed landmark only; never uses other view as truth.
    """
    name_to_idx = {n:i for i,n in enumerate(names)}
    kp = info["keypoints"]
    rest = np.asarray([p["rest"] for p in kp], float)
    px_to_m = float(layout["M_PER_PX"])
    count = 0
    peak = 0.0
    for b in info["bones"]:
        head, tail = name_to_idx[b["head_joint"]], name_to_idx[b["tail_joint"]]
        if head == tail:
            continue
        true_length = float(np.linalg.norm(rest[tail]-rest[head]))
        if true_length <= 1e-6:
            continue
        projected = np.linalg.norm(uv[:,tail,:]-uv[:,head,:], axis=-1) * px_to_m
        exceed = np.maximum(0., projected-true_length)
        tolerance = max(2 * px_to_m, 0.02 * true_length)
        spread = max(4 * px_to_m, 0.08 * true_length)
        risk = np.clip((exceed-tolerance) / spread, 0., 1.)
        risk = _smooth_median(risk, median_window)
        new_w = np.maximum(floor, 1.-(1.-floor)*risk)
        weights[:,tail] = np.minimum(weights[:,tail], new_w)
        count += int(np.count_nonzero(risk > 0))
        peak = max(peak, float(exceed.max()))
    details["bone_segment_violations_frame_pairs"] = count
    details["max_projected_excess_m"] = round(peak, 6)


def generic_view_weights(d, names, info, *, floor=FLOOR,
                         gap_start_px=GAP_START_PX, gap_full_px=GAP_FULL_PX,
                         overlap_close_px=OVERLAP_CLOSE_PX,
                         overlap_far_px=OVERLAP_FAR_PX,
                         min_overlap_advantage=MIN_OVERLAP_ADVANTAGE,
                         median_window=MEDIAN_WINDOW):
    """Return independent front/side (T,J) weight multipliers and JSON audit.

    V1 scores are conservative *risk proxies* and opt-in. They do not estimate
    calibrated P(visible) or 3D truth. Rigid segment overprojection provides
    view-specific evidence; shared-height disagreement chooses a view ONLY
    if that view has substantially more overlapping remote anatomy.
    """
    views, rest, bone_id, parents, layout = _validate(
        d, names, info, floor=floor, gap_start_px=gap_start_px,
        gap_full_px=gap_full_px, overlap_close_px=overlap_close_px,
        overlap_far_px=overlap_far_px, median_window=median_window)
    if not 0 <= min_overlap_advantage < 1 or not np.isfinite(min_overlap_advantage):
        raise ValueError("generic visibility min_overlap_advantage must be in [0,1)")
    J = len(names)
    T = len(views["front"])
    compatible = _remote_anatomy_mask(rest, bone_id, parents)
    confidences = {}
    for view in views:
        raw = d.get(view+"_conf")
        c = np.ones((T,J), dtype=np.float64) if raw is None else np.asarray(raw,dtype=np.float64)
        if c.shape != (T,J) or not np.isfinite(c).all():
            raise ValueError("generic visibility invalid confidence dimensions for "+view)
        # Raw heatmap peaks are NOT probabilities. This weak threshold merely
        # prevents low-response detections acting as overlap *partners*.
        confidences[view] = np.maximum(c,0.) > .35
    overlap = {}
    closest = {}
    for view in views:
        overlap[view],closest[view] = _overlap_strength(
            views[view], compatible, close_px=overlap_close_px,
            far_px=overlap_far_px,partner_valid=confidences[view])
    gap = np.abs(views["front"][...,1] - views["side"][...,1])
    discrepancy = np.clip((gap-gap_start_px)/(gap_full_px-gap_start_px),0.,1.)
    discrepancy = np.stack([_smooth_median(discrepancy[:,j],median_window)
                            for j in range(J)],axis=1)
    weights = {v:np.ones((T,J),dtype=np.float64) for v in views}
    audit_views = {v:{} for v in views}
    for a,b in (("front","side"),("side","front")):
        advantage = overlap[a]-overlap[b]
        # If the two views are equally ambiguous, leave both unchanged.
        asymmetry = np.clip((advantage-min_overlap_advantage)/
                            (1-min_overlap_advantage), 0., 1.)
        risk = discrepancy * overlap[a] * asymmetry
        weights[a] = np.clip(1-(1-floor)*risk, floor, 1.)
        _bone_projection_violation(views[a], names, info, layout,
                                   floor=floor, weights=weights[a],
                                   details=audit_views[a],median_window=median_window)
        if not np.isfinite(weights[a]).all() or not ((weights[a] >= floor) &
                                                     (weights[a] <= 1)).all():
            raise AssertionError("generic visibility nonfinite/out-of-bounds weights")
        changed = weights[a] < .9999
        audit_views[a].update({
            "mean_weight": round(float(weights[a].mean()),6),
            "min_weight": round(float(weights[a].min()),6),
            "joint_frames_downweighted": int(changed.sum()),
            "joints_downweighted": int(changed.any(axis=0).sum()),
            "fraction_joint_frames_downweighted":round(float(changed.mean()),6),
            "risk_from_height_overlap_frames":int((risk > .0001).sum()),
            "weights_by_affected_joint":{
                str(names[j]):np.round(weights[a][:,j],4).tolist()
                for j in range(J) if changed[:,j].any()},
        })
    audit = {
        "schema_version":PROFILE,
        "profile":PROFILE,
        "num_frames":int(T),
        "num_joints":int(J),
        "num_bones":len(parents),
        "camera_assumption":"TWO_ORTHOGRAPHIC_FRONT_SIDE_SAME_Z_AND_METRES_PER_PX",
        "not_true_mesh_occlusion":True,
        "not_calibrated_visibility_probability":True,
        "not_single_view_method":True,
        "parameters":{
            "weight_floor":floor,"gap_start_px":gap_start_px,
            "gap_full_px":gap_full_px,
            "overlap_close_px":overlap_close_px,
            "overlap_far_px":overlap_far_px,
            "min_overlap_advantage":min_overlap_advantage,
            "median_window":median_window,
            "partner_peak_threshold_unbounded":.35},
        "anatomy":{"eligible_distant_keypoint_pairs":int(compatible.sum()),
                   "protected_root_keypoints":int((bone_id == 0).sum())},
        "cross_view_height_gap_px":{
            "mean":round(float(gap.mean()),5),
            "p95":round(float(np.percentile(gap,95)),5)},
        "views":audit_views,
        "note":("Heuristic geometric observation reliability, not true "
                "Visibility learning. Cross-view discrepancy may also be "
                "camera desync, annotation or generative-video inconsistency; "
                "do not substitute for a mesh-based occlusion label."),
    }
    return weights, audit
