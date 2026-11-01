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


def verify_mask_provenance(det_path, *, allow_oracle=False):
    """Require matching mask-source audit, explicit Oracle GT authorization.

    This is a safety gate against inadvertently applying leaked 2D GT or
    unsupported geometry masks to a normal capture. It does NOT certify
    that a P2 or mesh mask is calibrated for production use.
    """
    import hashlib
    import json
    import pathlib

    source = pathlib.Path(det_path)
    audit_path = source.with_suffix(".occlusion_input.json")
    if not audit_path.is_file():
        raise FileNotFoundError("masked detection requires matching .occlusion_input.json sidecar")
    meta=json.loads(audit_path.read_text(encoding="utf-8"))
    if meta.get("profile") != "anymove.dino_per_view_observation_mask_ablation_v1":
        raise ValueError("unsupported observation mask provenance profile")
    mode=meta.get("mode")
    if mode not in ("mesh","p2_error","mesh_p2","oracle_gt"):
        raise ValueError("unknown observation mask generation mode")
    h=hashlib.sha256()
    with source.open("rb") as fd:
        for block in iter(lambda: fd.read(1<<20),b""):
            h.update(block)
    if h.hexdigest()!=meta.get("masked_detection_sha256"):
        raise ValueError("masked observation NPZ/source sidecar SHA mismatch")
    if mode=="oracle_gt" and not allow_oracle:
        raise PermissionError(
            "Blender GT oracle masks LEAK ground truth. "
            "Use --allow-gt-oracle only for a synthetic GT diagnostic, "
            "never for production or real-video inference.")
    return {"mode":mode,"sha256":h.hexdigest(),
            "profile":meta["profile"],"audit_path":str(audit_path)}
