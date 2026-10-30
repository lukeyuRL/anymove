"""Temporal left/right association for Dragon hind-leg side-view tracks.

Experimental, opt-in, after view synchronization and before pose fitting.
A two-state Viterbi model chooses identity or *whole-chain* left/right swap
for hip/knee/hock/paw/toe. Its observation costs use root-relative joint
heights, temporal height changes, anatomical-chain coherence and bounded
Reader peak weights. State changes carry switching and trajectory costs.

The side view is the candidate for reassignment, not a claim that it is
incorrect: two-view images alone cannot identify the true mislabeled view.
Neither front detections nor source det.npz are changed, and ambiguous
evidence defaults to identity (preserving the painted-limb convention).
"""
import numpy as np

from mocap.association_safety import (
    front_hind_reliability, VISIBILITY_MIN_SEGMENT_MEAN
)

PROFILE = "dragon_temporal_hind_association_v3_transient_visibility_3d_guard"
STEMS = ("hip", "knee", "hock", "hpaw", "htoe")
STEM_WEIGHTS = np.array((0.35, 1.0, 1.2, 1.65, 0.65), dtype=np.float64)
HEIGHT_CLIP_PX = 48.0
DELTA_CLIP_PX = 32.0
HEIGHT_TREND_WEIGHT = 0.40
BONE_WEIGHT = 0.25
SWAP_PRIOR_PER_FRAME = 1.0
SWITCH_PENALTY = 24.0
START_SWAPPED_PENALTY = 24.0
TRAJECTORY_JUMP_ALLOWANCE_PX = 12.0
TRAJECTORY_PENALTY = 0.20
MIN_SWAPPED_SEGMENT_FRAMES = 8
MIN_SEGMENT_GAIN_PER_FRAME_PX = 2.0
MIN_CHAINS_VOTING = 2
STEM_VOTE_MARGIN_PX = 3.0


def _validate(d, names):
    names = list(names)
    required = ("front_uv", "side_uv", "front_conf", "side_conf")
    if len(set(names)) != len(names):
        raise ValueError("temporal association requires unique joint names")
    missing = [f"{stem}_{side}" for side in ("L", "R") for stem in STEMS
               if f"{stem}_{side}" not in names]
    if "hips" not in names:
        missing.insert(0, "hips")
    if missing:
        raise ValueError("Dragon hind association missing required joints: " + ", ".join(missing))
    if any(k not in d for k in required):
        raise ValueError("temporal association requires front/side uv and confidence")
    front, side = np.asarray(d["front_uv"]), np.asarray(d["side_uv"])
    fc, sc = np.asarray(d["front_conf"]), np.asarray(d["side_conf"])
    if (front.ndim != 3 or front.shape[0] < 2 or front.shape[1:] != (len(names), 2)
            or side.shape != front.shape or fc.shape != front.shape[:2]
            or sc.shape != fc.shape
            or any(not np.isfinite(a).all() for a in (front, side, fc, sc))):
        raise ValueError("temporal association needs finite (frames,joints,2) tracks")
    return names


def _segments(states):
    """Half-open intervals of true/swap states."""
    out, start = [], None
    for t, s in enumerate(states):
        if s and start is None:
            start = t
        if not s and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(states)))
    return out


def _correlation(a, b):
    if a.std() <= 0.5 or b.std() <= 0.5:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 5)


def _state_costs(front, side, fc, sc, pairs, hips, visibility):
    # Hips subtraction independently per camera removes body/root vertical
    # translation, which can otherwise make *opposite* limb tracks correlate.
    fy = front[:, pairs, 1] - front[:, hips, 1, None, None]
    sy = side[:, pairs, 1] - side[:, hips, 1, None, None]
    fdelta = np.diff(fy, axis=0, prepend=fy[:1])
    w = STEM_WEIGHTS[None, None, :]
    scores, detail = [], []
    denom = 2.0 * STEM_WEIGHTS.sum()

    for state in (0, 1):
        perm = slice(None) if state == 0 else slice(None, None, -1)
        ids = pairs[perm]
        s = sy[:, perm, :]
        sd = np.diff(s, axis=0, prepend=s[:1])
        peak = np.sqrt(np.clip(fc[:, pairs], 0, 1) *
                       np.clip(sc[:, ids], 0, 1))
        # Downweight ambiguous Front hind landmarks in both candidate states.
        # An occluded landmark must not "prove" a Side L/R swap.
        weights = peak * w * visibility[:, pairs]
        height = np.minimum(np.abs(fy - s), HEIGHT_CLIP_PX)
        change = np.minimum(np.abs(fdelta - sd), DELTA_CLIP_PX)
        # Full-chain agreement: compare vertical span across each adjacent
        # bone as well as joint positions. No per-joint independent swapping.
        fbone = np.diff(fy, axis=-1)
        sbone = np.diff(s, axis=-1)
        bone_err = np.minimum(np.abs(fbone - sbone), HEIGHT_CLIP_PX)
        bone_weight = 0.5 * (weights[:, :, :-1] + weights[:, :, 1:])
        score = ((weights * (height + HEIGHT_TREND_WEIGHT * change)).sum((1, 2)) +
                 BONE_WEIGHT * (bone_weight * bone_err).sum((1, 2))) / denom
        scores.append(score)
        detail.append((height, peak))
    emission = np.stack(scores, axis=-1)
    emission[:, 1] += SWAP_PRIOR_PER_FRAME

    # An isolated paw error should not cause an entire leg swap. Require at
    # least two distinct *anatomical segments* to support the alternative.
    h0, _ = detail[0]
    h1, _ = detail[1]
    votes = (((h0[:, :, 1:] - h1[:, :, 1:]) *
              visibility[:, pairs][:, :, 1:]).mean(axis=1)
             > STEM_VOTE_MARGIN_PX).sum(axis=1)
    emission[:, 1] += np.where(votes >= MIN_CHAINS_VOTING, 0., 10.)
    return emission, votes


def _transition_cost(side, pair, t, previous, current):
    if previous == current:
        return 0.
    before = pair if previous == 0 else pair[::-1]
    after = pair if current == 0 else pair[::-1]
    # Penalize changing assignment if it causes an implausible sudden 2D
    # trajectory jump; a true side-label swap should restore continuity.
    step = np.linalg.norm(side[t, after, :] - side[t-1, before, :], axis=-1)
    extra = np.maximum(step - TRAJECTORY_JUMP_ALLOWANCE_PX, 0.)
    motion_cost = TRAJECTORY_PENALTY * np.mean(np.minimum(extra, 80.))
    return float(SWITCH_PENALTY + motion_cost)


def temporal_hind_association(d, names, visibility_aware=True):
    """Return corrected, copy-owned detection arrays and a JSON-safe audit.

    There are exactly two global hind-chain association states per frame:
    identity and swap L/R across *all five* hind joints. Confidence follows
    its matching UV. No other body part, frame, video or input file is edited.
    """
    names = _validate(d, names)
    idx = {name: i for i, name in enumerate(names)}
    pairs = np.array([[idx[f"{stem}_{side}"] for stem in STEMS]
                      for side in ("L", "R")], dtype=np.intp)
    hips = idx["hips"]
    front = np.asarray(d["front_uv"], dtype=np.float64)
    side = np.asarray(d["side_uv"], dtype=np.float64)
    fc, sc = np.asarray(d["front_conf"]), np.asarray(d["side_conf"])
    if visibility_aware:
        reliability, reliability_audit = front_hind_reliability(d, names)
    else:
        reliability = np.ones(front.shape[:2], dtype=np.float64)
        reliability_audit = {"profile": "disabled_by_caller",
                             "is_true_occlusion_probability": False}
    emission, votes = _state_costs(front, side, fc, sc, pairs, hips,
                                   reliability)
    t_count = len(front)

    # Two-state Viterbi: pay when switching state, not on every joint.
    dp = np.full((t_count, 2), np.inf, dtype=np.float64)
    back = np.zeros((t_count, 2), dtype=np.int8)
    dp[0] = emission[0] + np.array([0., START_SWAPPED_PENALTY])
    for t in range(1, t_count):
        for current in (0, 1):
            options = [dp[t-1, old] + _transition_cost(side, pairs, t, old, current)
                       for old in (0, 1)]
            old = int(np.argmin(options))
            back[t, current] = old
            dp[t, current] = options[old] + emission[t, current]
    states = np.empty(t_count, dtype=np.int8)
    states[-1] = int(np.argmin(dp[-1]))
    for t in range(t_count-1, 0, -1):
        states[t-1] = back[t, states[t]]

    candidate = states.copy()
    rejected = []
    # Conservative post-filter: reject short switches and intervals whose
    # average unpenalized emissions do not clearly favor reassignment.
    for first, last in _segments(states):
        gain = float((emission[first:last, 0] - emission[first:last, 1]).mean())
        # These key joints carry actual hind-leg lifting evidence.
        key_stems = [1, 2, 3]  # knee, hock, paw
        reliability_mean = float(reliability[first:last, pairs][:, :, key_stems].mean())
        reason = None
        if last-first < MIN_SWAPPED_SEGMENT_FRAMES:
            reason = "short_swapped_interval"
        elif gain < MIN_SEGMENT_GAIN_PER_FRAME_PX:
            reason = "insufficient_association_gain"
        elif reliability_mean < VISIBILITY_MIN_SEGMENT_MEAN:
            reason = "front_hind_visibility_ambiguous"
        if reason is not None:
            states[first:last] = 0
            rejected.append({"start_frame": int(first),
                             "end_frame_exclusive": int(last),
                             "mean_gain_px": round(gain, 4),
                             "mean_front_reliability": round(reliability_mean, 4),
                             "reason": reason})

    corrected = {k: np.array(v, copy=True) for k, v in d.items()}
    for t in np.flatnonzero(states):
        # Copy both leg chains atomically, including matching confidence.
        for key in ("side_uv", "side_conf"):
            arr = corrected[key]
            arr[t, pairs[0]] = np.asarray(d[key])[t, pairs[1]]
            arr[t, pairs[1]] = np.asarray(d[key])[t, pairs[0]]

    f = front[:, pairs, 1] - front[:, hips, 1, None, None]
    original_s = side[:, pairs, 1] - side[:, hips, 1, None, None]
    corrected_side = corrected["side_uv"][:, pairs, 1] - \
                     corrected["side_uv"][:, hips, 1, None, None]
    def paw_correlations(s):
        return {name: _correlation(f[:, side_n, 3], s[:, side_n, 3])
                for side_n, name in enumerate(("hpaw_L", "hpaw_R"))}

    segments = []
    for first, last in _segments(states):
        segments.append({
            "start_frame": int(first),
            "end_frame_exclusive": int(last),
            "frame_count": int(last-first),
            "mean_identity_cost_px": round(float(emission[first:last, 0].mean()), 4),
            "mean_swap_cost_px": round(float(emission[first:last, 1].mean()), 4),
            "mean_front_reliability": round(float(
                reliability[first:last, pairs][:, :, [1, 2, 3]].mean()), 4)
        })
    audit = {
        "schema_version": "anymove.temporal_joint_association/1",
        "profile": PROFILE,
        "status": "swapped" if np.any(states) else "identity_retained",
        "candidate_swapped_frames": int(candidate.sum()),
        "accepted_swapped_frames": int(states.sum()),
        "pre_3d_candidate_state_by_frame": states.tolist(),
        "visibility_diagnostics": reliability_audit,
        "accepted_segments": segments,
        "rejected_segments": rejected,
        "state_by_frame": states.tolist(),
        "identity_emission_by_frame": np.round(emission[:, 0], 3).tolist(),
        "swapped_emission_by_frame": np.round(emission[:, 1], 3).tolist(),
        "support_stem_votes_by_frame": votes.tolist(),
        "before_relative_hind_height_mae_px": round(float(np.abs(f-original_s).mean()), 4),
        "after_relative_hind_height_mae_px": round(float(np.abs(f-corrected_side).mean()), 4),
        "before_root_relative_paw_correlation": paw_correlations(original_s),
        "after_root_relative_paw_correlation": paw_correlations(corrected_side),
        "parameters": {
            "hind_stems": list(STEMS),
            "switch_penalty": SWITCH_PENALTY,
            "start_swapped_penalty": START_SWAPPED_PENALTY,
            "minimum_swap_frames": MIN_SWAPPED_SEGMENT_FRAMES,
            "minimum_segment_gain_per_frame_px": MIN_SEGMENT_GAIN_PER_FRAME_PX,
            "root_relative": True,
            "joint_peak_weights": "bounded geometric mean; uncalibrated",
            "visibility_proxy_enabled": bool(visibility_aware),
            "visibility_reference": "within-view front/side foreleg separation, not calibrated occlusion",
            "group_assignment": "both L/R hind chains always switched together",
        },
        "warnings": [
            "This is an experimental correspondence hypothesis, not labeled occlusion or 3D ground truth.",
            "State switches may be wrong if AI-rendered views disagree anatomically.",
            "All unmatched joints and Front detections are preserved; iClone low-paw-agreement gate must not be bypassed.",
            "Passing 2D Viterbi association is NOT permission to apply to 3D; a separate continuity gate may reject it.",
        ],
    }
    if visibility_aware:
        # Diagnostic counterfactual: report how many frames the original
        # unweighted 2D algorithm would have proposed on these same tracks.
        # This reveals accidental global evidence suppression without ever
        # applying those proposals to the Solver or bypassing 3D QA.
        _, unweighted = temporal_hind_association(
            d, names, visibility_aware=False)
        audit["unweighted_2d_reference"] = {
            "profile": unweighted["profile"],
            "candidate_swapped_frames": unweighted["candidate_swapped_frames"],
            "accepted_swapped_frames": unweighted["accepted_swapped_frames"],
            "accepted_segments": unweighted["accepted_segments"],
            "note": ("Diagnostic only; unweighted candidates are not applied "
                     "or passed to the 3D Solver."),
        }
    return corrected, audit
