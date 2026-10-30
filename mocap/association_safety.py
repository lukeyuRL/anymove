"""Conservative two-view association safety: front occlusion proxy and 3D guard.

Only NumPy is required. Both decisions are hypotheses: image-space proximity
is not an observed visibility mask and 3D continuity is not ground-truth
accuracy. The 3D stage quarantines a risky association, returning the original
baseline solve rather than silently accepting a possibly swapped hind leg.
"""
import numpy as np

HIND = ("hip", "knee", "hock", "hpaw", "htoe")
FORE = ("elbow", "wrist", "fpaw", "ftoe")
MAX_VALIDATION_SEGMENTS = 4
FRONT_MIN_WEIGHT = 0.25
OVERLAP_CLOSE_PX = 12.0
OVERLAP_FAR_PX = 40.0
SIDE_SEPARATION_MARGIN_PX = 8.0
SIDE_SEPARATION_RANGE_PX = 35.0
VISIBILITY_MIN_SEGMENT_MEAN = 0.55

SEAM_STEP_RATIO_LIMIT = 1.8
SEAM_STEP_ADDED_CM = 2.5
SEAM_VECTOR_CHANGE_LIMIT_CM = 10.0
SEAM_VECTOR_RATIO_CONFIRM = 1.3
SEAM_ROOT_ADDED_CM = 1.5
SEAM_ROOT_RATIO_LIMIT = 2.0
SEAM_ACCEL_RATIO_LIMIT = 2.2
SEAM_ACCEL_ADDED_CM = 4.0


def front_hind_reliability(d, names):
    """Return (T,J) weights independent of Front/Side assignment state.

    Front hind joints near front legs *and* distinct from forelegs in Side
    are suspected of being visually ambiguous. The cue is independent of
    which hind-side identity the association algorithm selects: it uses
    each view only internally. No inference from a discrepancy between the
    two possible Side identity assignments is used to compute this weight.

    Raw heatmap peak amplitudes are not calibrated visibility probabilities.
    """
    index = {name: k for k, name in enumerate(names)}
    f = np.asarray(d["front_uv"], dtype=np.float64)
    s = np.asarray(d["side_uv"], dtype=np.float64)
    if f.ndim != 3 or f.shape != s.shape or f.shape[1:] != (len(names), 2):
        raise ValueError("front_hind_reliability needs matching (frames,joints,2)")
    if not np.isfinite(f).all() or not np.isfinite(s).all():
        raise ValueError("front_hind_reliability needs finite UV")
    w = np.ones(f.shape[:2], np.float64)
    per_joint = {}
    for label in ("L", "R"):
        fore = [index[stem + "_" + label] for stem in FORE
                if stem + "_" + label in index]
        if not fore:
            raise ValueError("Dragon front-hind association needs foreleg landmarks for " + label)
        for stem in HIND:
            name = stem + "_" + label
            if name not in index:
                raise ValueError("Dragon front-hind association missing " + name)
            j = index[name]
            front_separation = np.linalg.norm(
                f[:, fore, :] - f[:, j:j+1, :], axis=-1).min(axis=1)
            side_separation = np.linalg.norm(
                s[:, fore, :] - s[:, j:j+1, :], axis=-1).min(axis=1)
            crowded = np.clip((OVERLAP_FAR_PX - front_separation) /
                              (OVERLAP_FAR_PX - OVERLAP_CLOSE_PX), 0, 1)
            side_distinct = np.clip(
                (side_separation-front_separation-SIDE_SEPARATION_MARGIN_PX) /
                SIDE_SEPARATION_RANGE_PX, 0, 1)
            ambiguity = crowded * side_distinct
            # A five-frame median avoids a one-frame false-occlusion spike.
            pad = np.pad(ambiguity, (2, 2), mode="edge")
            ambiguity = np.median(
                np.stack([pad[k:k+len(ambiguity)] for k in range(5)]), axis=0)
            weight = 1.0 - (1.0-FRONT_MIN_WEIGHT)*ambiguity
            w[:, j] = weight
            per_joint[name] = {
                "mean_weight": round(float(weight.mean()), 5),
                "min_weight": round(float(weight.min()), 5),
                "frames_below_0_6": int((weight < 0.6).sum())
            }
    return w, {
        "profile": "front_hind_foreleg_overlap_side_separation_v1",
        "is_true_occlusion_probability": False,
        "parameters": {
            "min_weight": FRONT_MIN_WEIGHT,
            "front_close_px": OVERLAP_CLOSE_PX,
            "front_far_px": OVERLAP_FAR_PX,
            "side_separation_margin_px": SIDE_SEPARATION_MARGIN_PX,
            "side_separation_range_px": SIDE_SEPARATION_RANGE_PX,
            "median_filter_frames": 5,
            "min_acceptable_segment_visibility": VISIBILITY_MIN_SEGMENT_MEAN,
        },
        "joint_summary": per_joint
    }


def _validate_fit(res, names):
    j = np.asarray(res["joints"], dtype=np.float64)
    root = np.asarray(res["trans"], dtype=np.float64)
    rot = np.asarray(res["root_rotmat"], dtype=np.float64)
    if (j.ndim != 3 or j.shape[1:] != (len(names), 3) or j.shape[0] < 3
            or root.shape != (len(j), 3) or rot.shape != (len(j), 3, 3)
            or not all(np.isfinite(x).all() for x in (j, root, rot))):
        raise ValueError("3D continuity guard requires finite joints/root transforms")
    return j, root, rot


def validate_3d_segment(baseline, candidate, names, start, end):
    """Compare candidate and baseline 3D steps for segment [start,end).

    A candidate is quarantined when its *actual temporal 3D step* at a
    label-switch boundary is substantially worse than the same-step baseline,
    or its 3D root step becomes implausibly larger. Difference in pose
    alone is not a rejection criterion: a correct pose repair can be large.
    """
    j0, p0, r0 = _validate_fit(baseline, names)
    j1, p1, r1 = _validate_fit(candidate, names)
    if j0.shape != j1.shape:
        raise ValueError("baseline and candidate shapes differ")
    T = len(j0)
    if not 0 <= start < end <= T:
        raise ValueError("invalid association segment boundaries")
    hind = np.array([i for i, name in enumerate(names)
                     if any(name == stem+"_"+side for stem in HIND for side in ("L", "R"))],
                    dtype=int)
    if len(hind) != 10:
        raise ValueError("Dragon guard needs all ten hind joints")
    boundaries = [t for t in (start, end) if 0 < t < T]
    details = []
    failures = []
    for t in boundaries:
        # metres -> centimetres, t-1 -> t.
        v0 = (j0[t, hind] - j0[t-1, hind])*100.
        v1 = (j1[t, hind] - j1[t-1, hind])*100.
        a = np.linalg.norm(v0, axis=-1)
        b = np.linalg.norm(v1, axis=-1)
        different = np.linalg.norm(v1-v0, axis=-1)
        root_a = float(np.linalg.norm((p0[t]-p0[t-1])*100.))
        root_b = float(np.linalg.norm((p1[t]-p1[t-1])*100.))
        reasons = []
        if (float(b.mean()) > float(a.mean())*SEAM_STEP_RATIO_LIMIT
                and float(b.mean()-a.mean()) > SEAM_STEP_ADDED_CM):
            reasons.append("excessive_hind_step_increase")
        if (float(different.max()) > SEAM_VECTOR_CHANGE_LIMIT_CM
                and float(b.mean()) > float(a.mean())*SEAM_VECTOR_RATIO_CONFIRM):
            reasons.append("large_hind_velocity_vector_discontinuity")
        if (root_b > root_a*SEAM_ROOT_RATIO_LIMIT
                and root_b-root_a > SEAM_ROOT_ADDED_CM):
            reasons.append("excessive_root_translation_step")
        # Three-frame acceleration estimates at each candidate switch.
        accel0 = accel1 = None
        if t >= 2:
            accel0 = float(np.linalg.norm(
                (j0[t, hind]-2*j0[t-1, hind]+j0[t-2, hind])*100.,
                axis=-1).mean())
            accel1 = float(np.linalg.norm(
                (j1[t, hind]-2*j1[t-1, hind]+j1[t-2, hind])*100.,
                axis=-1).mean())
            if (accel1 > accel0*SEAM_ACCEL_RATIO_LIMIT
                    and accel1-accel0 > SEAM_ACCEL_ADDED_CM):
                reasons.append("excessive_hind_acceleration_increase")
        details.append({
            "from_frame": int(t-1), "to_frame": int(t),
            "hind_step_baseline_mean_cm": round(float(a.mean()), 5),
            "hind_step_candidate_mean_cm": round(float(b.mean()), 5),
            "hind_step_candidate_max_cm": round(float(b.max()), 5),
            "hind_step_vector_change_max_cm": round(float(different.max()), 5),
            "root_step_baseline_cm": round(root_a, 5),
            "root_step_candidate_cm": round(root_b, 5),
            "hind_accel_baseline_mean_cm_per_frame2": None if accel0 is None else round(accel0, 5),
            "hind_accel_candidate_mean_cm_per_frame2": None if accel1 is None else round(accel1, 5),
            "failure_reasons": reasons
        })
        failures.extend(reasons)
    # Report pose changes for context; do not use the absolute displacement as
    # a stand-alone guard, since a legitimate reassociation can be very large.
    delta = np.linalg.norm((j1[start:end, hind]-j0[start:end, hind])*100., axis=-1)
    root_delta = np.linalg.norm((p1[start:end]-p0[start:end])*100., axis=-1)
    relative_rotation = np.einsum("tij,tkj->tik", r1[start:end], r0[start:end])
    cos_angle = np.clip((np.trace(relative_rotation, axis1=1, axis2=2)-1.)/2., -1, 1)
    return {
        "status": "quarantined" if failures else "continuity_gate_passed",
        "start_frame": int(start),
        "end_frame_exclusive": int(end),
        "boundary_checks": details,
        "max_hind_pose_delta_cm": round(float(delta.max()), 5),
        "mean_hind_pose_delta_cm": round(float(delta.mean()), 5),
        "max_root_translation_delta_cm": round(float(root_delta.max()), 5),
        "max_root_rotation_delta_deg": round(float(np.degrees(np.arccos(cos_angle)).max()), 5),
        "failure_reasons": sorted(set(failures)),
        "note": "A pass only checks continuity relative to baseline, not anatomical correctness."
    }


def gated_temporal_fit(base_d, candidate_d, association, names, fit_fn):
    """Solve baseline and each proposed segment independently before applying.

    fit_fn(d) returns existing Solver's result (joints in metres, root
    translation and rotation); no changes to model weights or FK. Return
    actual applied detections, actual solved pose, and the safety audit.
    Rejecting all segments returns the *same* baseline result object.
    """
    T = len(base_d["front_uv"])
    if any(np.asarray(base_d[k]).shape != np.asarray(candidate_d[k]).shape
           for k in ("front_uv", "side_uv", "front_conf", "side_conf")):
        raise ValueError("candidate tracks have incompatible shapes")
    states = np.asarray(association.get("state_by_frame", []), dtype=int)
    if states.shape != (T,) or not np.isin(states, [0, 1]).all():
        raise ValueError("association state_by_frame malformed")
    base_fit = fit_fn(base_d)
    _validate_fit(base_fit, names)
    accepted_d = {k: np.array(v, copy=True) for k, v in base_d.items()}
    accepted_fit = base_fit
    new_states = np.zeros(T, dtype=np.int8)
    checks = []
    segments, first = [], None
    for t, value in enumerate(states):
        if value and first is None:
            first = t
        if not value and first is not None:
            segments.append((first, t))
            first = None
    if first is not None:
        segments.append((first, T))
    if len(segments) > MAX_VALIDATION_SEGMENTS:
        # Every rejected interval needs a precise audit entry; never silently
        # drop the remaining segments when the per-clip budget is exceeded.
        checks.extend({
            "status": "quarantined",
            "start_frame": int(start), "end_frame_exclusive": int(end),
            "failure_reasons": ["too_many_segments_for_safe_validation"],
            "candidate_segment_count": len(segments)
        } for start, end in segments)
    else:
        for start, end in segments:
            trial = {k: v.copy() for k, v in accepted_d.items()}
            for key in ("side_uv", "side_conf"):
                trial[key][start:end] = candidate_d[key][start:end]
            trial_fit = fit_fn(trial)
            check = validate_3d_segment(accepted_fit, trial_fit, names, start, end)
            checks.append(check)
            if check["status"] == "continuity_gate_passed":
                accepted_d, accepted_fit = trial, trial_fit
                new_states[start:end] = 1
    return accepted_d, accepted_fit, {
        "profile": "dragon_temporal_3d_continuity_gate_v1",
        "status": "all_candidates_passed" if np.array_equal(new_states, states) else
                  ("all_candidates_quarantined" if not new_states.any() else "partially_applied"),
        "candidate_swapped_frames": int(states.sum()),
        "accepted_swapped_frames": int(new_states.sum()),
        "applied_state_by_frame": new_states.tolist(),
        "checks": checks,
        "parameters": {
            "max_segments": MAX_VALIDATION_SEGMENTS,
            "seam_step_ratio": SEAM_STEP_RATIO_LIMIT,
            "seam_step_added_cm": SEAM_STEP_ADDED_CM,
            "seam_vector_change_cm": SEAM_VECTOR_CHANGE_LIMIT_CM,
            "seam_root_step_ratio": SEAM_ROOT_RATIO_LIMIT,
            "seam_root_added_cm": SEAM_ROOT_ADDED_CM,
            "seam_accel_ratio": SEAM_ACCEL_RATIO_LIMIT,
            "seam_accel_added_cm": SEAM_ACCEL_ADDED_CM
        },
        "warning": "Candidate may be anatomically wrong even if it passes; requires visual and 3D truth QA."
    }
