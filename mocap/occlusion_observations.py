"""Remove unreliable per-view OBSERVATIONS, never final animation keys.

This is a deliberate opt-in bridge between scored detections and the
unchanged FK motion.json contract. Hidden/invalid 2D coordinates can contain
arbitrarily large outliers: interpolate them ONLY for Solver initialization,
and set that view's Solver confidence to exactly zero in data loss.

When no visible anchor exists for a joint in one view, initialize from the
unanimated Rig rest projection (NOT Blender animation truth). This prior
cannot restore unobserved 3D dimensions; report missing segments separately.
"""
import numpy as np
from mocap import layout

MASK_KEYS = ("front_visible_mask", "side_visible_mask")
PROFILE = "anymove.per_view_masked_observations_v1"


def longest_false_run(observed):
    longest = running = 0
    for yes in np.asarray(observed, dtype=bool):
        running = 0 if yes else running + 1
        longest = max(longest, running)
    return int(longest)


def prepare_masked_solver_input(d, raw, info):
    """Return new initialized observations and a mask audit, no mutation.

    All fields in d and raw are NumPy arrays. The raw Reader UV/peaks are
    preserved in the source detection NPZ (SHA tracked by mocap.solve).
    """
    if not all(k in raw for k in MASK_KEYS):
        raise ValueError("masked Solver requires BOTH per-view visible masks")
    if any(k in raw for k in ("front_track_weight", "side_track_weight")):
        raise ValueError("do not stack occlusion masks with P1/P2 track weights")
    names = [k["name"] for k in info["keypoints"]]
    if "names" not in raw or [str(n) for n in raw["names"]] != names:
        raise ValueError("masked observations and Rig joint order differ")
    T, J = np.asarray(d["front_conf"]).shape
    if J != len(names) or T < 3:
        raise ValueError("unexpected masked Solver frame/joint shape")
    rest = np.asarray([k["rest"] for k in info["keypoints"]], dtype=float)
    if rest.shape != (J,3) or not np.isfinite(rest).all():
        raise ValueError("Dino Rig rest coordinates required for masked initial pose")
    reference_front, reference_side = layout.project(rest)
    result = {k: np.array(v, copy=True) for k,v in d.items()}
    masks = {}
    audit = {"profile":PROFILE,"frames":T,"joint_count":J,
             "interpolation":"piecewise linear in each view at missing frames; REST projected fallback if a view has no valid anchor",
             "meaning":"missing image observations only; final animation still has one pose per frame",
             "views":{}}
    for view,rest_uv in (("front",reference_front),("side",reference_side)):
        mask = np.asarray(raw[view+"_visible_mask"])
        if (mask.shape != (T,J) or mask.dtype != np.dtype("bool")):
            raise ValueError(view+" visible mask must be boolean (T,J), no confidence proxy")
        uv = np.asarray(d[view+"_uv"],dtype=float)
        conf = np.asarray(d[view+"_conf"],dtype=float)
        if (uv.shape != (T,J,2) or conf.shape != (T,J) or
                not np.isfinite(uv).all() or not np.isfinite(conf).all()):
            raise ValueError(view+" invalid UV/confidence")
        masks[view] = mask.copy()
        init_uv = uv.copy()
        t = np.arange(T)
        unanchored=[]
        for j in range(J):
            valid = np.flatnonzero(mask[:,j])
            if valid.size:
                for axis in (0,1):
                    init_uv[:,j,axis] = np.interp(t,valid,uv[valid,j,axis])
            else:
                init_uv[:,j,:] = rest_uv[j]
                unanchored.append(names[j])
        result[view+"_uv"] = init_uv
        # Strict zero weight for a masked observation, not 0.1 / soft-weight.
        result[view+"_conf"] = conf * mask.astype(np.float64)
        audit["views"][view] = {
            "masked_joint_frames":int((~mask).sum()),
            "total_joint_frames":int(mask.size),
            "unanchored_joints":unanchored,
            "largest_missing_run_frames":max(longest_false_run(mask[:,j]) for j in range(J)),
        }
    both_missing = ~(masks["front"] | masks["side"])
    audit["joint_frames_missing_in_both_views"] = int(both_missing.sum())
    audit["no_animation_keys_deleted"] = True
    audit["interpretation"] = (
        "Masked 2D UV ignored by reprojection loss. Imputed UV ONLY initializes "
        "kinematic solve. Smoothness/rig constraints continue to determine "
        "unobserved pose; no real occlusion probability is inferred.")
    return result, masks, audit
