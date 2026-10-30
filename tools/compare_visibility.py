"""Compare two solved Dragon motions; this is NOT a ground-truth accuracy test.

python tools/compare_visibility.py baseline_motion.json variant_motion.json
"""
import json
import sys
import numpy as np


def compare(a, b):
    for field in ("fps", "bones", "joint_names"):
        if a.get(field) != b.get(field):
            raise ValueError("incompatible motion field: " + field)
    for field in ("trans", "root_quat", "joints"):
        if len(a[field]) != len(b[field]):
            raise ValueError("different motion frame count: " + field)
    root_a, root_b = np.asarray(a["trans"]), np.asarray(b["trans"])
    joints_a, joints_b = np.asarray(a["joints"]), np.asarray(b["joints"])
    if root_a.shape[1:] != (3,) or joints_a.shape != joints_b.shape:
        raise ValueError("incompatible motion dimensions")
    diff = np.linalg.norm(joints_a-joints_b, axis=-1)
    delta_root = np.linalg.norm(root_a-root_b, axis=-1)
    reproj_a, reproj_b = a.get("reproj_px", {}), b.get("reproj_px", {})
    return {
        "comparison_type": "same-detection solver ablation, NOT 3D truth",
        "frames": len(root_a),
        "joint_position_difference_mean_cm": round(float(diff.mean()*100), 4),
        "joint_position_difference_p95_cm": round(float(np.percentile(diff, 95)*100), 4),
        "root_translation_difference_mean_cm": round(float(delta_root.mean()*100), 4),
        "root_translation_difference_max_cm": round(float(delta_root.max()*100), 4),
        "reprojection_px_baseline": reproj_a,
        "reprojection_px_variant": reproj_b,
        "raw_paw_agreement_baseline": a.get("sync", {}).get("paw_agreement"),
        "raw_paw_agreement_variant": b.get("sync", {}).get("paw_agreement"),
        "conclusion": "Differences measure the effect of weighting only. Without ground-truth joints and visual inspection, smaller reprojection is NOT proof of higher motion accuracy."
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python tools/compare_visibility.py baseline_motion.json variant_motion.json")
    with open(sys.argv[1], encoding="utf-8") as f:
        baseline = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        variant = json.load(f)
    print(json.dumps(compare(baseline, variant), indent=2))


if __name__ == "__main__":
    main()
