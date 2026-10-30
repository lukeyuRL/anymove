"""Opt-in Dragon front-view hind-leg reliability proxy for the two-view solver.

This is NOT a measured occlusion mask or calibrated probability. It identifies
frames where front/side observations disagree about a hind-leg joint's height;
proximity to a front-leg landmark strengthens (but cannot alone trigger) the
proxy. The side view and all unrelated joints are left unchanged.

Use only after view alignment and any optional label-swapping decision.
"""
import numpy as np

PROFILE = "dragon_front_hind_height_overlap_v1"
HIND_STEMS = ("knee", "hock", "hpaw", "htoe")
FORE_STEMS = ("elbow", "wrist", "fpaw", "ftoe")
HEIGHT_START_PX = 6.0
HEIGHT_FULL_PX = 24.0
OVERLAP_CLOSE_PX = 10.0
OVERLAP_FAR_PX = 36.0
WEIGHT_FLOOR = 0.35
MEDIAN_WINDOW = 5


def _median_filter(values):
    """Suppress isolated risk spikes; keep sequence length and boundaries."""
    half = MEDIAN_WINDOW // 2
    pad = np.pad(values, (half, half), mode="edge")
    return np.median(np.stack([pad[i:i + len(values)] for i in range(MEDIAN_WINDOW)]), axis=0)


def front_hind_weights(d, names):
    """Return front reliability multiplier (frames, joints) plus JSON-safe audit.

    Weight is in [0.35,1] for Dragon's knee/hock/hpaw/htoe joints only.
    Without a cross-view height discrepancy, even an overlapping limb keeps
    full weight. The output is deterministic and does not mutate det.npz.
    """
    names = list(names)
    if len(names) != len(set(names)):
        raise ValueError("duplicate joint names in visibility input")
    must = [f"{stem}_{side}" for side in ("L", "R")
            for stem in HIND_STEMS + FORE_STEMS]
    missing = [name for name in must if name not in names]
    if missing:
        raise ValueError("Dragon front-hind visibility requires joints: " + ", ".join(missing))
    front = np.asarray(d["front_uv"], dtype=np.float64)
    side = np.asarray(d["side_uv"], dtype=np.float64)
    if (front.ndim != 3 or front.shape[-1] != 2 or front.shape != side.shape
            or front.shape[0] < 2 or front.shape[1] != len(names)
            or not np.isfinite(front).all() or not np.isfinite(side).all()):
        raise ValueError("visibility requires finite front/side (frames,joints,2) arrays")
    index = {name: i for i, name in enumerate(names)}
    weights = np.ones(front.shape[:2], dtype=np.float64)
    per_joint = {}
    frame_weights = {}
    for side_label in ("L", "R"):
        fore = [index[f"{stem}_{side_label}"] for stem in FORE_STEMS]
        candidates = front[:, fore, :]  # (T, F, 2)
        for stem in HIND_STEMS:
            joint = f"{stem}_{side_label}"
            j = index[joint]
            gap = np.abs(front[:, j, 1] - side[:, j, 1])
            gap_risk = np.clip((gap - HEIGHT_START_PX) /
                               (HEIGHT_FULL_PX - HEIGHT_START_PX), 0.0, 1.0)
            separation = np.linalg.norm(candidates - front[:, j:j+1, :], axis=-1)
            nearest_fore = separation.min(axis=1)
            overlap = np.clip((OVERLAP_FAR_PX - nearest_fore) /
                              (OVERLAP_FAR_PX - OVERLAP_CLOSE_PX), 0.0, 1.0)
            # Height conflict is necessary; front-leg overlap strengthens risk.
            risk = _median_filter(gap_risk * (0.55 + 0.45 * overlap))
            weight = np.clip(1.0 - (1.0 - WEIGHT_FLOOR) * risk,
                             WEIGHT_FLOOR, 1.0)
            weights[:, j] = weight
            frame_weights[joint] = np.round(weight, 4).tolist()
            per_joint[joint] = {
                "mean_weight": round(float(weight.mean()), 4),
                "min_weight": round(float(weight.min()), 4),
                "frames_below_0_95": int((weight < 0.95).sum()),
                "frames_below_0_70": int((weight < 0.70).sum()),
                "max_height_disagreement_px": round(float(gap.max()), 3),
                "mean_height_disagreement_px": round(float(gap.mean()), 3),
                "min_front_fore_separation_px": round(float(nearest_fore.min()), 3),
            }
    audit = {
        "schema_version": "anymove.dragon_visibility_proxy/1",
        "profile": PROFILE,
        "algorithm": "front_hind_height_disagreement_gated_by_foreleg_proximity",
        "not_a_true_occlusion_mask": True,
        "num_frames": int(front.shape[0]),
        "num_joints": int(front.shape[1]),
        "parameters": {
            "height_start_px": HEIGHT_START_PX,
            "height_full_px": HEIGHT_FULL_PX,
            "foreleg_overlap_close_px": OVERLAP_CLOSE_PX,
            "foreleg_overlap_far_px": OVERLAP_FAR_PX,
            "minimum_front_multiplier": WEIGHT_FLOOR,
            "temporal_median_window": MEDIAN_WINDOW,
        },
        "joint_summary": per_joint,
        "front_weights_by_joint": frame_weights,
        "affected_joint_count": int(sum(x["frames_below_0_95"] > 0 for x in per_joint.values())),
        "note": "These multipliers bias the solver away from ambiguous front hind detections; they do not correct source-frame view inconsistency or calibrate peak scores.",
    }
    return weights, audit
