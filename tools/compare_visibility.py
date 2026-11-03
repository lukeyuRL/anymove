"""Compare two AnyMove solved motions; this is NOT a ground-truth accuracy test.

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
    hash_a = a.get("source_detections_sha256")
    hash_b = b.get("source_detections_sha256")
    if hash_a and hash_b and hash_a != hash_b:
        comparison_type = "different-detection motion comparison, NOT 3D truth"
        source_note = (
            "Different detection files were used. Tracking can change both 2D "
            "joint coordinates and Solver weights. Each motion's reported "
            "reprojection is against its OWN observations, so the two errors "
            "are not directly comparable without evaluating against the SAME "
            "raw detections.")
    elif hash_a and hash_b and hash_a == hash_b:
        comparison_type = "same-detection solver ablation, NOT 3D truth"
        source_note = (
            "Both motions identify the same source detection file; differing "
            "Solver settings can still alter the fit.")
    else:
        comparison_type = "unverified-source motion comparison, NOT 3D truth"
        source_note = (
            "At least one motion lacks a source detection hash. This tool "
            "cannot establish that both runs used the same detections.")
    return {
        "comparison_type": comparison_type,
        "frames": len(root_a),
        "joint_position_difference_mean_cm": round(float(diff.mean()*100), 4),
        "joint_position_difference_p95_cm": round(float(np.percentile(diff, 95)*100), 4),
        "root_translation_difference_mean_cm": round(float(delta_root.mean()*100), 4),
        "root_translation_difference_max_cm": round(float(delta_root.max()*100), 4),
        "reprojection_px_baseline": reproj_a,
        "reprojection_px_variant": reproj_b,
        "raw_paw_agreement_baseline": a.get("sync", {}).get("paw_agreement"),
        "raw_paw_agreement_variant": b.get("sync", {}).get("paw_agreement"),
        "source_detection_sha256_baseline": hash_a,
        "source_detection_sha256_variant": hash_b,
        "source_comparability": source_note,
        "conclusion": "Motion differences are not ground-truth error. Without known 3D motion and a shared 2D observation reference, smaller per-run reprojection does NOT prove higher accuracy."
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
