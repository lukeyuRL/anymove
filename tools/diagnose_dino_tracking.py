"""Compare Dino P1 against ONE common raw 2D observation reference.

python tools/diagnose_dino_tracking.py RAW_DET.npz TRACKED_DET.npz \
  BASE_MOTION.json TRACKED_MOTION.json RIG_INFO.json [--out REPORT.json]

Separate the (1) tracking 2D coordinate delta, (2) bounded Solver weight
change, (3) joint and Root motion displacement, and (4) raw-reference
reprojection. Raw Reader observations are NOT 2D ground truth; this tool
does not claim 3D accuracy improvements or validate anatomical identity.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

PROFILE = "anymove.dino_tracking_diagnostics/1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(d):
    return [str(name) for name in d["names"]]


def _vecstats(values, unit=""):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("diagnostic values must be nonempty and finite")
    return {
        "mean" + unit: round(float(a.mean()), 5),
        "p95" + unit: round(float(np.percentile(a, 95)), 5),
        "max" + unit: round(float(a.max()), 5),
    }


def _project(joints, camera):
    """Same orthographic camera geometry as mocap.layout.project."""
    H = float(camera["H"])
    f_width = float(camera["W_FRONT"])
    s_width = float(camera["W_SIDE"])
    mp = float(camera["M_PER_PX"])
    centre = float(camera["Z_CENTER"])
    if mp <= 0 or not all(np.isfinite(x) for x in (H, f_width, s_width, mp, centre)):
        raise ValueError("invalid camera layout for Dino diagnostics")
    z = H / 2 - (joints[..., 2] - centre) / mp
    front = np.stack([f_width / 2 + joints[..., 0] / mp, z], -1)
    side = np.stack([s_width / 2 - joints[..., 1] / mp, z], -1)
    return front, side


def _root_dynamics(root, fps):
    speed = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    accel = np.linalg.norm(np.diff(root, n=2, axis=0), axis=-1)
    return {
        "speed_cm_per_second": _vecstats(speed * 100 * fps, "_cm_s"),
        "acceleration_cm_per_second_squared": _vecstats(
            accel * 100 * fps * fps, "_cm_s2"),
    }


def compare_tracking(raw, tracked, original, candidate, camera):
    """Pure-NumPy comparison. Input objects are read-only from the caller."""
    required_det = ("front_uv", "side_uv", "front_conf", "side_conf",
                    "names", "fps")
    for k in required_det:
        if k not in raw or k not in tracked:
            raise ValueError("missing required raw/tracked detection field: " + k)
    names = _names(raw)
    if not names or len(names) != len(set(names)) or names != _names(tracked):
        raise ValueError("raw/tracked joint names and order differ")
    fps = float(raw["fps"])
    if (not np.isfinite(fps) or fps <= 0 or
            abs(fps - float(tracked["fps"])) > 1e-6):
        raise ValueError("detection FPS mismatch")
    J = len(names)
    uv_shape = np.asarray(raw["front_uv"]).shape
    if len(uv_shape) != 3 or uv_shape[-2:] != (J, 2) or uv_shape[0] < 3:
        raise ValueError("unexpected raw Dino UV shape")
    T = uv_shape[0]
    for view in ("front", "side"):
        for suffix in ("_uv", "_conf"):
            key = view + suffix
            a = np.asarray(raw[key])
            b = np.asarray(tracked[key])
            expected = (T, J, 2) if suffix == "_uv" else (T, J)
            if (a.shape != expected or b.shape != expected or
                    not np.isfinite(a).all() or not np.isfinite(b).all()):
                raise ValueError("invalid " + key + " shape or finite values")
        if not np.array_equal(raw[view + "_conf"], tracked[view + "_conf"]):
            raise ValueError("P1 tracking must preserve original Reader peaks")
        preserved = view + "_uv_reader"
        if preserved not in tracked or not np.array_equal(
                tracked[preserved], raw[view + "_uv"]):
            raise ValueError("tracked NPZ does not preserve source " + preserved)
        weight = view + "_track_weight"
        if weight not in tracked:
            raise ValueError("P1 NPZ missing independent Solver weights " + weight)
        w = np.asarray(tracked[weight], dtype=float)
        if (w.shape != (T, J) or not np.isfinite(w).all() or
                np.any(w < 0) or np.any(w > 1)):
            raise ValueError("invalid P1 tracking weights: " + weight)

    for field in ("fps", "joint_names", "bones"):
        if original.get(field) != candidate.get(field):
            raise ValueError("incompatible motion contract: " + field)
    if (abs(float(original["fps"]) - fps) > 1e-6 or
            original["joint_names"] != names or
            len(original["bones"]) < 1):
        raise ValueError("motion FPS or joint order differ from detections")
    joints = {}
    roots = {}
    for tag, motion in (("baseline", original), ("tracked", candidate)):
        j = np.asarray(motion["joints"], dtype=np.float64)
        root = np.asarray(motion["trans"], dtype=np.float64)
        quat = np.asarray(motion["root_quat"], dtype=np.float64)
        if (j.shape != (T, J, 3) or root.shape != (T, 3) or
                quat.shape != (T, 4) or not np.isfinite(j).all() or
                not np.isfinite(root).all() or not np.isfinite(quat).all()):
            raise ValueError("invalid motion joints/Root/FPS frame consistency")
        joints[tag] = j
        roots[tag] = root
    if ("calib" in raw and "calib" in tracked and
            not np.array_equal(raw["calib"], tracked["calib"])):
        raise ValueError("tracking changed original video calibration")

    raw_ref = {}
    detector_changes = {}
    for view in ("front", "side"):
        d = np.linalg.norm(np.asarray(raw[view+"_uv"], float) -
                           np.asarray(tracked[view+"_uv"], float), axis=-1)
        weight = np.asarray(tracked[view+"_track_weight"], float)
        detector_changes[view] = {
            "joint_2d_displacement_px": _vecstats(d, "_px"),
            "changed_joint_frames": int(np.count_nonzero(d > 1e-5)),
            "joint_frames_with_weight_below_one": int(np.count_nonzero(weight < 1)),
            "mean_track_weight": round(float(weight.mean()), 5),
            "max_adjustment_frame": int(np.unravel_index(np.argmax(d), d.shape)[0]),
            "max_adjustment_joint": names[int(np.unravel_index(np.argmax(d), d.shape)[1])],
        }
    # Reproject BOTH solved skeletons onto the SAME unmodified Reader UV.
    # This measures raw-Reader agreement, *not* 2D/3D ground-truth accuracy.
    for label in ("baseline", "tracked"):
        pf, ps = _project(joints[label], camera)
        raw_ref[label] = {
            "front": _vecstats(
                np.linalg.norm(pf - np.asarray(raw["front_uv"]), axis=-1), "_px"),
            "front_x_only": _vecstats(
                np.abs(pf[..., 0] - np.asarray(raw["front_uv"])[..., 0]), "_px"),
            "side": _vecstats(
                np.linalg.norm(ps - np.asarray(raw["side_uv"]), axis=-1), "_px"),
        }
    jdelta = np.linalg.norm(joints["tracked"] - joints["baseline"], axis=-1) * 100
    rdelta = np.linalg.norm(roots["tracked"] - roots["baseline"], axis=-1) * 100
    top = sorted(range(J), key=lambda k: float(jdelta[:, k].mean()), reverse=True)
    report = {
        "schema_version": PROFILE,
        "comparison_type": "different-detection P1 tracking, common raw Reader reference; NOT 3D ground truth",
        "frames": T,
        "fps": fps,
        "keypoints": J,
        "detection_changes": detector_changes,
        "shared_raw_reader_reprojection_px": raw_ref,
        "shared_reference_interpretation": (
            "Both motions projected against identical unmodified Reader UV. "
            "The raw Reader is not independent truth, so favoring the original "
            "motion is possible even when tracking corrected a real error."),
        "joint_position_between_solutions_cm": _vecstats(jdelta, "_cm"),
        "root_translation_between_solutions_cm": _vecstats(rdelta, "_cm"),
        "max_root_difference_frame": int(np.argmax(rdelta)),
        "root_dynamics": {
            tag: _root_dynamics(roots[tag], fps)
            for tag in ("baseline", "tracked")
        },
        "worst_joint_mean_deltas_cm": [
            {"joint": names[k],
             "mean_delta_cm": round(float(jdelta[:, k].mean()), 5),
             "max_delta_cm": round(float(jdelta[:, k].max()), 5)}
            for k in top[:10]
        ],
        "paw_agreement": {
            "baseline": original.get("sync", {}).get("paw_agreement"),
            "tracked": candidate.get("sync", {}).get("paw_agreement"),
            "interpretation": "null means not evaluable/missing; NOT score 0",
        },
        "conclusion": (
            "P1 is not proven better without verified Dino 2D/3D motion truth, "
            "occlusion reappearance checks and visual Root/FK comparison."),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_detection")
    parser.add_argument("tracked_detection")
    parser.add_argument("baseline_motion")
    parser.add_argument("tracked_motion")
    parser.add_argument("rig_info")
    parser.add_argument("--out", help="write JSON report, never overwrite an existing file")
    args = parser.parse_args(argv)
    from mocap import layout
    from mocap.layout_guard import verify_layout_for_rig
    camera = verify_layout_for_rig(args.rig_info, layout)
    if camera.get("status") != "camera_layout_verified_for_rig":
        raise ValueError("Dino camera layout is not validated against rig")
    if camera.get("creature") != "dino":
        raise ValueError("Dino P1 diagnostics require a verified Dino rig")
    with np.load(args.raw_detection, allow_pickle=False) as z:
        raw = {k: z[k].copy() for k in z.files}
    with np.load(args.tracked_detection, allow_pickle=False) as z:
        tracked = {k: z[k].copy() for k in z.files}
    base = json.loads(pathlib.Path(args.baseline_motion).read_text(encoding="utf-8"))
    variant = json.loads(pathlib.Path(args.tracked_motion).read_text(encoding="utf-8"))
    for motion, path in ((base, args.raw_detection),
                         (variant, args.tracked_detection)):
        if motion.get("source_detections_sha256") != file_sha256(path):
            raise ValueError("motion references a different detection source SHA-256")
        source = motion.get("solver_camera_layout") or {}
        if (source.get("creature") != camera["creature"] or
                source.get("active_layout") != camera["active_layout"]):
            raise ValueError("motion camera layout differs from current Dino rig")
    audit = pathlib.Path(args.tracked_detection).with_suffix(".tracking.json")
    if audit.exists():
        tracking_audit = json.loads(audit.read_text(encoding="utf-8"))
        if tracking_audit.get("source_detection_sha256") != file_sha256(args.raw_detection):
            raise ValueError("tracking audit source SHA-256 mismatch")
    actual_camera = {k: getattr(layout, k) for k in
                     ("H", "W_FRONT", "W_SIDE", "M_PER_PX", "Z_CENTER")}
    report = compare_tracking(raw, tracked, base, variant, actual_camera)
    report["source_detection_sha256"] = file_sha256(args.raw_detection)
    report["tracked_detection_sha256"] = file_sha256(args.tracked_detection)
    if audit.exists():
        report["tracking_audit"] = tracking_audit
    if args.out:
        target = pathlib.Path(args.out)
        if target.exists() or target.resolve() in {
                pathlib.Path(x).resolve() for x in
                (args.raw_detection, args.tracked_detection,
                 args.baseline_motion, args.tracked_motion, args.rig_info)}:
            raise FileExistsError("diagnostic report must not overwrite existing artifacts")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("saved", target)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
