"""Apply a verified learned empirical angular prior and recompute FK output.

python -m mocap.denoise_motion PRIOR.json IN_MOTION.json RIG.json DET.npz OUT.json [--strength 1.0]

No FK data is fabricated: after changing Euler, every world joint is
recomputed using the same Rig.forward code as the original Solver, and
reprojection is measured again against the original detections. Root
Translation and Root Quaternion stay unchanged. Never overwrite inputs.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from mocap.motion_prior import denoise_euler, motion_arrays, PROFILE


def quaternion_matrices(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("invalid root quaternions")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if (norm < 1e-6).any():
        raise ValueError("root quaternion has zero length")
    q = q / norm
    w, x, y, z = q.T
    r = np.empty((len(q), 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1-2*(y*y+z*z)
    r[:, 0, 1] = 2*(x*y-z*w)
    r[:, 0, 2] = 2*(x*z+y*w)
    r[:, 1, 0] = 2*(x*y+z*w)
    r[:, 1, 1] = 1-2*(x*x+z*z)
    r[:, 1, 2] = 2*(y*z-x*w)
    r[:, 2, 0] = 2*(x*z-y*w)
    r[:, 2, 1] = 2*(y*z+x*w)
    r[:, 2, 2] = 1-2*(x*x+y*y)
    return r


def _sha(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for x in iter(lambda: f.read(1048576), b""):
            h.update(x)
    return h.hexdigest()


def run(prior_path, source_path, rig_path, det_path, output_path, strength=1.0):
    import torch
    from mocap import layout
    from mocap.layout_guard import verify_layout_for_rig
    from mocap.solve import Rig, project

    paths = [pathlib.Path(x).resolve()
             for x in (prior_path, source_path, rig_path, det_path, output_path)]
    prior_file, src, rig_file, detect, target = paths
    if target in paths[:4] or target.exists():
        raise FileExistsError("do not overwrite prior, input motion or detection")
    prior = json.loads(prior_file.read_text(encoding="utf-8"))
    motion = json.loads(src.read_text(encoding="utf-8"))
    info = json.loads(rig_file.read_text(encoding="utf-8"))
    if prior.get("profile") != PROFILE:
        raise ValueError("unsupported/missing empirical motion prior")
    bones = [b["name"] for b in info["bones"]]
    if prior.get("bones") != bones or motion.get("bones") != bones:
        raise ValueError("motion/prior bone order must match the rig")
    orig, fps = motion_arrays(motion, bones, prior.get("fps"))
    roots = np.asarray(motion.get("trans"), dtype=np.float64)
    q = quaternion_matrices(motion.get("root_quat"))
    if roots.shape != (len(orig), 3) or not np.isfinite(roots).all():
        raise ValueError("invalid root translation")
    active = verify_layout_for_rig(rig_file, layout)
    if active.get("status") != "camera_layout_verified_for_rig":
        raise ValueError("cannot verify Solver/Creature camera layout")
    if motion.get("solver_camera_layout") != active:
        # Absolute path inside provenance may differ between devices, but
        # the exact layout and creature must still match.
        existing = motion.get("solver_camera_layout") or {}
        if (existing.get("active_layout") != active.get("active_layout")
                or existing.get("creature") != active.get("creature")):
            raise ValueError("source motion's camera layout provenance mismatches rig")
    with np.load(str(detect), allow_pickle=False) as f:
        det = {key: f[key].copy() for key in
               ("front_uv", "side_uv", "fps", "names")}
    if (abs(float(det["fps"])-fps) > 1e-6
            or len(det["front_uv"]) != len(orig)
            or [str(n) for n in det["names"]] != motion.get("joint_names")):
        raise ValueError("detections and motion have incompatible frames/names/fps")
    expected_sha = motion.get("source_detections_sha256")
    if expected_sha and expected_sha != _sha(detect):
        raise ValueError("source detection sha256 does not match motion")
    limits = [b["limits_deg"] for b in info["bones"]]
    cleaned = denoise_euler(orig, prior, limits, strength)
    rig = Rig(info, dev="cpu")
    with torch.no_grad():
        orig_fk, _ = rig.forward(torch.tensor(roots, dtype=torch.float64),
                                 torch.tensor(q, dtype=torch.float64),
                                 torch.tensor(orig, dtype=torch.float64))
        new_fk, _ = rig.forward(torch.tensor(roots, dtype=torch.float64),
                                torch.tensor(q, dtype=torch.float64),
                                torch.tensor(cleaned, dtype=torch.float64))
        projected_front, projected_side = project(new_fk)
    reference = np.asarray(motion["joints"], dtype=np.float64)
    baseline_error_cm = float(np.linalg.norm(
        orig_fk.numpy() - reference, axis=-1).max()*100)
    if reference.shape != orig_fk.shape or baseline_error_cm > 1.5:
        raise ValueError("FK reconstructed joints do not match original motion; "
                         "do not emit an inconsistent animation")
    # Legacy original reproj was always computed from unweighted Euclidean
    # distances. Recompute with the actual denoised FK, not stale metadata.
    front = projected_front.numpy()
    side = projected_side.numpy()
    new_reproj = {
        "front": float(np.linalg.norm(front-det["front_uv"], axis=-1).mean()),
        "front_x_only": float(np.abs(front[..., 0]-det["front_uv"][..., 0]).mean()),
        "side": float(np.linalg.norm(side-det["side_uv"], axis=-1).mean()),
    }
    output = dict(motion)
    output["euler"] = np.round(cleaned, 5).tolist()
    output["joints"] = np.round(new_fk.numpy(), 4).tolist()
    output["reproj_px"] = new_reproj
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output), encoding="utf-8")
    angle_delta_deg = np.degrees(np.abs(cleaned-orig))
    diagnostics = {
        "prior_profile": prior["profile"],
        "prior_sha256": _sha(prior_file),
        "source_motion_sha256": _sha(src),
        "source_detection_sha256": _sha(detect),
        "strength": strength, "frames": len(orig),
        "mean_euler_delta_deg": float(angle_delta_deg[:, 1:].mean()),
        "max_euler_delta_deg": float(angle_delta_deg[:, 1:].max()),
        "max_original_fk_discrepancy_cm": baseline_error_cm,
        "root_translation_unchanged": output["trans"] == motion["trans"],
        "root_quaternion_unchanged": output["root_quat"] == motion["root_quat"],
        "reprojection_px_original": motion.get("reproj_px"),
        "reprojection_px_denoised": new_reproj,
        "warning": "Small angular errors and smooth animation do not prove 3D ground-truth correctness.",
    }
    sidecar = target.with_suffix(".denoise.json")
    sidecar.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print("saved", target, "diagnostics", sidecar)
    print("Euler mean/max delta (deg)",
          round(diagnostics["mean_euler_delta_deg"], 4),
          round(diagnostics["max_euler_delta_deg"], 4))
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior")
    parser.add_argument("input_motion")
    parser.add_argument("rig_info")
    parser.add_argument("detections")
    parser.add_argument("output")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args(argv)
    run(args.prior, args.input_motion, args.rig_info,
        args.detections, args.output, args.strength)


if __name__ == "__main__":
    main()
