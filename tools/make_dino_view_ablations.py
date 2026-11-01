"""Controlled Dino two-view reliability ablations (one view per joint group).

The EXISTING Solver accepts optional front_track_weight/side_track_weight
arrays, multiplying the raw heatmap peaks before fitting. Those arrays are
reused ONLY as an opt-in view-weight experiment; they are not optical flow.

Prespecified, fixed 0.5 multipliers (NOT fitted against GT):
  tail_front_half: Front Tail = 0.5; Side unchanged.
  toes_side_half: Side Feet/Toes = 0.5; Front unchanged.
  combined_half: both, on disjoint joints/views.

No 2D coordinates/peaks are changed. The Solver subsequently squares
effective peak confidence in its reprojection loss, so 0.5 input weight is
approximately 0.25 on the weighted data term before robustification.

Lower-confidence views contain UNIQUE coordinates: Side cannot reconstruct
Front's X, and Front cannot reconstruct Side's Y. This experiment does NOT
create missing geometric measurements or calibrated occlusion probabilities.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from tools.evaluate_dino_full_body import classify_joint

PROFILES = ("tail_front_half", "toes_side_half", "combined_half")
PROFILE = "anymove.dino_view_reliability_fixed_half_ablation/1"


def _sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as fd:
        for chunk in iter(lambda: fd.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(raw, rig, profile):
    """Pure NumPy; return NEW NPZ dict and audit, never mutate caller arrays."""
    if profile not in PROFILES:
        raise ValueError("unsupported prespecified view-weight profile: " + str(profile))
    if not isinstance(rig, dict) or not isinstance(rig.get("keypoints"), list):
        raise ValueError("missing Dino rig keypoints")
    required = ("front_uv", "side_uv", "front_conf", "side_conf",
                "names", "fps", "calib")
    if not all(k in raw for k in required):
        raise ValueError("original Reader NPZ missing required detection fields")
    if any(k in raw for k in ("front_track_weight", "side_track_weight")):
        raise ValueError("experiment requires original non-P1/non-P2 raw Reader detections")
    names = [str(n) for n in np.asarray(raw["names"])]
    rig_names = [x.get("name") for x in rig["keypoints"]]
    if (len(names) != 92 or len(names) != len(set(names)) or
            names != rig_names):
        raise ValueError("Dino 92-keypoint original Reader order must match exact rig")
    T,J = np.asarray(raw["front_conf"]).shape
    if T < 3 or J != len(names):
        raise ValueError("expected Dino (T,92) Reader confidence")
    for view in ("front","side"):
        uv = np.asarray(raw[view+"_uv"])
        peak = np.asarray(raw[view+"_conf"])
        if (uv.shape != (T,J,2) or peak.shape != (T,J) or
                not np.isfinite(uv).all() or not np.isfinite(peak).all()):
            raise ValueError(view+" invalid Reader UV/peaks")
    fps = float(raw["fps"])
    calib = np.asarray(raw["calib"])
    if (not np.isfinite(fps) or fps <= 0 or
            calib.shape != (2,3) or not np.isfinite(calib).all()):
        raise ValueError("invalid original Reader fps/calibration")

    labels = [classify_joint(n) for n in names]
    tail = np.asarray([k == "tail" for k in labels])
    feet = np.asarray([k == "feet_toes" for k in labels])
    if not int(tail.sum()) or not int(feet.sum()) or np.any(tail & feet):
        raise ValueError("missing/disjoint Dino Tail and Feet/Toes groups")
    weights = {
        "front": np.ones((T,J), dtype=np.float32),
        "side": np.ones((T,J), dtype=np.float32),
    }
    selected = []
    if profile in ("tail_front_half","combined_half"):
        weights["front"][:,tail] = np.float32(0.5)
        selected.append({"view":"front","group":"tail","joint_count":int(tail.sum())})
    if profile in ("toes_side_half","combined_half"):
        weights["side"][:,feet] = np.float32(0.5)
        selected.append({"view":"side","group":"feet_toes","joint_count":int(feet.sum())})
    output = {key: np.array(value, copy=True) for key,value in raw.items()}
    for view in ("front","side"):
        output[view+"_track_weight"] = weights[view]
    audit = {
        "profile": PROFILE,
        "variant": profile,
        "frames": T,
        "keypoints": J,
        "prespecified_multiplier": 0.5,
        "weighting": "multiply original Reader peak by 0.5 on selected view/joints; effective solver reprojection coefficient squares confidence",
        "experimental": True,
        "selected": selected,
        "views": {
            view: {
                "downweighted_joint_frames": int(np.count_nonzero(weights[view] < 1)),
                "mean_multiplier": round(float(weights[view].mean()), 6),
            } for view in ("front","side")
        },
        "warning": (
            "Fixed anatomical per-view weighting, NOT learned visibility, "
            "not fitted to Blender GT; preserves original 2D Reader UV/peak "
            "and cannot recover missing camera coordinate dimensions."
        ),
    }
    return output, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_detection")
    parser.add_argument("rig_info")
    parser.add_argument("out_detection")
    parser.add_argument("--profile", required=True, choices=PROFILES)
    args = parser.parse_args(argv)
    src = pathlib.Path(args.original_detection).resolve()
    rig_path = pathlib.Path(args.rig_info).resolve()
    dest = pathlib.Path(args.out_detection).resolve()
    sidecar = dest.with_suffix(".view_reliability.json")
    if dest in (src,rig_path) or dest.exists() or sidecar.exists():
        raise FileExistsError("will not overwrite any existing Dino experiment file")
    if not src.is_file() or not rig_path.is_file():
        raise FileNotFoundError("Reader detection or rig_info.json missing")
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    with np.load(src,allow_pickle=False) as z:
        raw = {key:z[key].copy() for key in z.files}
    output, audit = build(raw, rig, args.profile)
    audit["source_detection_sha256"] = _sha256(src)
    audit["rig_info_sha256"] = _sha256(rig_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **output)
    audit["weighted_detection_sha256"] = _sha256(dest)
    try:
        sidecar.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    print(json.dumps(audit, indent=2))
    print("new weighted Dino detections:", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
