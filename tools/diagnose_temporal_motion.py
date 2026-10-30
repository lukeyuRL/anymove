"""Frame-by-frame Dragon motion A/B diagnostics with temporal swap-boundary checks.

Usage (from repository root):
  python tools/diagnose_temporal_motion.py BASE.json TEMPORAL.json

Output directory defaults to <temporal_stem>_diagnostics alongside TEMPORAL.json.
The report measures differences between two solutions, NOT 3D ground-truth error.
Only numpy is needed for CSV/JSON; matplotlib adds an optional PNG plot.
"""
import argparse
import csv
import json
import pathlib

import numpy as np


HIND_STEMS = ("hip", "knee", "hock", "hpaw", "htoe")
FILE_STEM = "temporal_3d"


def _read(path):
    with open(str(path), encoding="utf-8") as source:
        return json.load(source)


def _motion_arrays(base, variant):
    """Strictly reject mismatched time, skeleton, camera or source detections."""
    for key in ("fps", "bones", "joint_names"):
        if base.get(key) != variant.get(key):
            raise ValueError("different motion provenance/contract: " + key)
    if ("source_detections_sha256" in base and
            "source_detections_sha256" in variant and
            base["source_detections_sha256"] != variant["source_detections_sha256"]):
        raise ValueError("different source_detections_sha256: not a same-detection A/B")
    if ("solver_camera_layout" in base and "solver_camera_layout" in variant
            and base["solver_camera_layout"] != variant["solver_camera_layout"]):
        raise ValueError("different solver camera layout")
    fps = float(base["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    names = base["joint_names"]
    if not names or len(names) != len(set(names)):
        raise ValueError("missing or duplicate joint names")

    arrays = {}
    for key, expected_last in (("joints", (len(names), 3)),
                               ("trans", (3,)), ("root_quat", (4,))):
        a = np.asarray(base[key], dtype=np.float64)
        b = np.asarray(variant[key], dtype=np.float64)
        if a.shape != b.shape or a.ndim != len(expected_last)+1 or \
                a.shape[1:] != expected_last or a.shape[0] < 3:
            raise ValueError("motion arrays differ in length or shape: " + key)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite motion array: " + key)
        arrays[key] = (a, b)
    frames = arrays["joints"][0].shape[0]
    if any(a.shape[0] != frames for a, b in arrays.values()):
        raise ValueError("motion arrays have inconsistent frame counts")
    return names, fps, arrays


def _association_info(association, frames, source_hash):
    if association is None:
        return np.zeros(frames, dtype=np.int8), [], ["Association report not supplied; boundary analysis unavailable"]
    states = np.asarray(association.get("state_by_frame", []))
    if states.shape != (frames,) or not np.isin(states, [0, 1]).all():
        raise ValueError("association state_by_frame must be 0/1 and match motion frames")
    if (source_hash and association.get("source_detections_sha256") and
            association["source_detections_sha256"] != source_hash):
        raise ValueError("association source detections SHA does not match motion")
    states = states.astype(np.int8)
    segments, beginning = [], None
    for t, value in enumerate(states):
        if value and beginning is None:
            beginning = t
        if not value and beginning is not None:
            segments.append({"start_frame": beginning, "end_frame_exclusive": t,
                             "frame_count": t-beginning})
            beginning = None
    if beginning is not None:
        segments.append({"start_frame": beginning, "end_frame_exclusive": frames,
                         "frame_count": frames-beginning})
    return states, segments, []


def _quantiles(values):
    return {
        "mean_cm": float(np.mean(values)),
        "p95_cm": float(np.percentile(values, 95)),
        "p99_cm": float(np.percentile(values, 99)),
        "max_cm": float(np.max(values)),
    }


def _safe_mean(array):
    return float(np.mean(array)) if np.size(array) else None


def _quaternion_angle_degrees(a, b):
    qa = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    qb = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    dot = np.abs(np.sum(qa * qb, axis=1)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))


def analyze(base, variant, association=None):
    """Return JSON report and detailed CSV-ready rows; no file side effects."""
    names, fps, a = _motion_arrays(base, variant)
    original, temporal = a["joints"]
    root0, root1 = a["trans"]
    q0, q1 = a["root_quat"]
    frames, count = original.shape[:2]
    source_hash = base.get("source_detections_sha256")
    states, segments, warnings = _association_info(association, frames, source_hash)

    hind_ids = np.array([j for j, n in enumerate(names)
                         if any(n == stem + "_" + side
                                for stem in HIND_STEMS for side in ("L", "R"))], dtype=int)
    if hind_ids.size == 0:
        raise ValueError("no hind joints in motion")
    other_ids = np.array([j for j in range(count) if j not in set(hind_ids)], dtype=int)
    cm = 100.0
    joint_delta = np.linalg.norm(temporal-original, axis=-1)*cm
    root_delta = np.linalg.norm(root1-root0, axis=-1)*cm
    root_angle = _quaternion_angle_degrees(q0, q1)

    # A step ending at frame t is displacement from t-1 -> t.
    step0 = np.linalg.norm(np.diff(original, axis=0), axis=-1)*cm
    step1 = np.linalg.norm(np.diff(temporal, axis=0), axis=-1)*cm
    step_change = np.linalg.norm(np.diff(temporal-original, axis=0), axis=-1)*cm
    root_step0 = np.linalg.norm(np.diff(root0, axis=0), axis=-1)*cm
    root_step1 = np.linalg.norm(np.diff(root1, axis=0), axis=-1)*cm
    root_step_change = np.linalg.norm(np.diff(root1-root0, axis=0), axis=-1)*cm
    # Discrete acceleration at frame t uses points t-2, t-1, t.
    accel0 = np.linalg.norm(np.diff(original, n=2, axis=0), axis=-1)*cm
    accel1 = np.linalg.norm(np.diff(temporal, n=2, axis=0), axis=-1)*cm
    accel_change = np.linalg.norm(np.diff(temporal-original, n=2, axis=0), axis=-1)*cm

    per_frame = []
    per_joint_frame = []
    for t in range(frames):
        idx_max = int(np.argmax(joint_delta[t]))
        row = {
            "frame": t,
            "time_s": round(t/fps, 6),
            "swap_state": int(states[t]),
            "mean_joint_delta_cm": float(joint_delta[t].mean()),
            "mean_hind_delta_cm": float(joint_delta[t, hind_ids].mean()),
            "mean_other_delta_cm": _safe_mean(joint_delta[t, other_ids]),
            "max_joint_delta_cm": float(joint_delta[t, idx_max]),
            "max_joint_name": names[idx_max],
            "root_delta_cm": float(root_delta[t]),
            "root_rotation_delta_deg": float(root_angle[t]),
            "root_step_baseline_cm": float(root_step0[t-1]) if t else None,
            "root_step_temporal_cm": float(root_step1[t-1]) if t else None,
            "root_step_vector_change_cm": float(root_step_change[t-1]) if t else None,
            "mean_hind_step_change_cm": float(step_change[t-1, hind_ids].mean()) if t else None,
            "max_hind_step_change_cm": float(step_change[t-1, hind_ids].max()) if t else None,
        }
        per_frame.append(row)
        for j, name in enumerate(names):
            per_joint_frame.append({
                "frame": t, "time_s": round(t/fps, 6),
                "joint": name, "group": "hind" if j in hind_ids else "other",
                "swap_state": int(states[t]),
                "position_delta_cm": float(joint_delta[t, j]),
                "step_baseline_cm": float(step0[t-1, j]) if t else None,
                "step_temporal_cm": float(step1[t-1, j]) if t else None,
                "step_vector_change_cm": float(step_change[t-1, j]) if t else None,
                "accel_baseline_cm_per_frame2": float(accel0[t-2, j]) if t >= 2 else None,
                "accel_temporal_cm_per_frame2": float(accel1[t-2, j]) if t >= 2 else None,
                "accel_vector_change_cm_per_frame2": float(accel_change[t-2, j]) if t >= 2 else None,
            })

    by_joint = []
    for j, name in enumerate(names):
        values = joint_delta[:, j]
        by_joint.append({
            "joint": name, "group": "hind" if j in hind_ids else "other",
            "mean_delta_cm": float(values.mean()),
            "p95_delta_cm": float(np.percentile(values, 95)),
            "p99_delta_cm": float(np.percentile(values, 99)),
            "max_delta_cm": float(values.max()),
            "max_frame": int(values.argmax()),
            "mean_swapped_frames_cm": _safe_mean(values[states == 1]),
            "mean_unswapped_frames_cm": _safe_mean(values[states == 0]),
            "max_step_vector_change_cm": float(step_change[:, j].max()),
            "max_accel_vector_change_cm_per_frame2": float(accel_change[:, j].max()),
        })
    by_joint.sort(key=lambda x: x["max_delta_cm"], reverse=True)

    transitions = np.flatnonzero(states[1:] != states[:-1])+1
    seams = []
    for t in transitions:
        k = t-1
        seams.append({
            "from_frame": int(t-1), "to_frame": int(t),
            "time_s": round(t/fps, 6),
            "transition": "identity_to_swap" if states[t] else "swap_to_identity",
            "hind_step_baseline_mean_cm": float(step0[k, hind_ids].mean()),
            "hind_step_temporal_mean_cm": float(step1[k, hind_ids].mean()),
            "hind_step_temporal_max_cm": float(step1[k, hind_ids].max()),
            "hind_step_vector_change_mean_cm": float(step_change[k, hind_ids].mean()),
            "hind_step_vector_change_max_cm": float(step_change[k, hind_ids].max()),
            "root_step_baseline_cm": float(root_step0[k]),
            "root_step_temporal_cm": float(root_step1[k]),
            "root_step_vector_change_cm": float(root_step_change[k]),
            "largest_hind_step_change_joint": names[hind_ids[np.argmax(step_change[k, hind_ids])]],
            "largest_hind_step_change_cm": float(step_change[k, hind_ids].max()),
            "hind_accel_change_mean_cm_per_frame2": (
                float(accel_change[k-1, hind_ids].mean()) if t >= 2 else None),
        })

    top_frames = sorted(per_frame, key=lambda x: x["max_joint_delta_cm"], reverse=True)[:12]
    report = {
        "type": "same-detection 3D solution comparison; NOT ground-truth error",
        "frames": frames,
        "fps": fps,
        "joint_count": count,
        "hind_joint_count": int(hind_ids.size),
        "source_detections_sha256": source_hash,
        "accepted_swapped_frames": int(states.sum()),
        "swap_segments": segments,
        "joint_position_delta": _quantiles(joint_delta),
        "hind_position_delta": _quantiles(joint_delta[:, hind_ids]),
        "other_position_delta": _quantiles(joint_delta[:, other_ids]) if other_ids.size else None,
        "root_translation_delta": _quantiles(root_delta),
        "max_root_delta_frame": int(root_delta.argmax()),
        "max_root_rotation_delta_degrees": float(root_angle.max()),
        "mean_hind_delta_swapped_frames_cm": _safe_mean(joint_delta[states == 1][:, hind_ids]),
        "mean_hind_delta_outside_swapped_frames_cm": _safe_mean(joint_delta[states == 0][:, hind_ids]),
        "top_joints_by_max_delta": by_joint[:12],
        "top_frames_by_max_joint_delta": top_frames,
        "association_boundary_steps": seams,
        "solver_reprojection_px_baseline": base.get("reproj_px", {}),
        "solver_reprojection_px_temporal": variant.get("reproj_px", {}),
        "warnings": warnings + [
            "Changes outside switched frames are possible because the pose solver optimizes a temporally coupled trajectory.",
            "A large step near a swap boundary is a discontinuity candidate, not proof of an animation defect.",
            "Smaller 2D reprojection or 3D difference DOES NOT establish 3D accuracy without independent ground truth."
        ]
    }
    return report, per_frame, by_joint, per_joint_frame


def _csv(path, rows):
    if not rows:
        return
    with open(str(path), "w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, per_frame, report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    x = [row["frame"] for row in per_frame]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(x, [r["mean_hind_delta_cm"] for r in per_frame], label="Mean hind joint delta")
    axes[0].plot(x, [r["mean_other_delta_cm"] for r in per_frame], label="Mean other joint delta")
    axes[0].plot(x, [r["max_joint_delta_cm"] for r in per_frame], label="Max individual joint delta", alpha=.55)
    axes[0].set_ylabel("Position delta (cm)")
    axes[0].legend(loc="upper right")
    axes[1].plot(x, [r["root_delta_cm"] for r in per_frame], label="Root translation delta")
    axes[1].set_ylabel("Root delta (cm)")
    axes[1].legend(loc="upper right")
    axes[2].plot(x[1:], [r["mean_hind_step_change_cm"] for r in per_frame[1:]],
                 label="Hind step-vector difference")
    axes[2].plot(x[1:], [r["max_hind_step_change_cm"] for r in per_frame[1:]],
                 label="Worst hind step-vector difference", alpha=.6)
    axes[2].set_ylabel("Velocity vector delta (cm/frame)")
    axes[2].set_xlabel("Frame (0-based)")
    axes[2].legend(loc="upper right")
    for ax in axes:
        for seg in report["swap_segments"]:
            ax.axvspan(seg["start_frame"]-.5, seg["end_frame_exclusive"]-.5,
                       alpha=.12)
        for seam in report["association_boundary_steps"]:
            ax.axvline(seam["to_frame"], linestyle="--", alpha=.6)
        ax.grid(alpha=.2)
    fig.suptitle("Dragon temporal association: 3D A/B differences (NOT ground truth)")
    fig.tight_layout()
    fig.savefig(str(path), dpi=145)
    plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=pathlib.Path)
    parser.add_argument("temporal", type=pathlib.Path)
    parser.add_argument("--association", type=pathlib.Path,
                        help="optional association JSON; auto-detected from temporal motion name")
    parser.add_argument("--out-dir", type=pathlib.Path)
    args = parser.parse_args(argv)

    assoc_path = args.association
    if assoc_path is None:
        candidate = args.temporal.with_suffix(".association.json")
        assoc_path = candidate if candidate.is_file() else None
    assoc = _read(assoc_path) if assoc_path is not None else None
    report, per_frame, by_joint, per_joint_frame = analyze(
        _read(args.baseline), _read(args.temporal), assoc)
    out = args.out_dir if args.out_dir is not None else \
          args.temporal.parent / (args.temporal.stem + "_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / (FILE_STEM + "_report.json")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    _csv(out / (FILE_STEM + "_by_frame.csv"), per_frame)
    _csv(out / (FILE_STEM + "_by_joint.csv"), by_joint)
    _csv(out / (FILE_STEM + "_by_joint_frame.csv"), per_joint_frame)
    plot_path = out / (FILE_STEM + "_plot.png")
    plotted = _plot(plot_path, per_frame, report)

    print("Same-detection Motion A/B; NOT 3D truth")
    print("Frames:", report["frames"], "Swap segments:", report["swap_segments"])
    print("3D joint delta mean/p95/max (cm):",
          *(round(report["joint_position_delta"][k], 4)
            for k in ("mean_cm", "p95_cm", "max_cm")))
    print("Hind mean swapped/outside (cm):",
          report["mean_hind_delta_swapped_frames_cm"],
          report["mean_hind_delta_outside_swapped_frames_cm"])
    print("Largest root difference (cm/frame):",
          round(report["root_translation_delta"]["max_cm"], 4),
          "at frame", report["max_root_delta_frame"])
    print("Top 5 joints (by max positional difference):")
    for row in by_joint[:5]:
        print(" ", row["joint"], "max", round(row["max_delta_cm"], 4),
              "cm at frame", row["max_frame"])
    for row in report["association_boundary_steps"]:
        print("BOUNDARY", row["from_frame"], "->", row["to_frame"],
              "hind step-vector difference mean/max",
              round(row["hind_step_vector_change_mean_cm"], 4),
              round(row["hind_step_vector_change_max_cm"], 4), "cm/frame",
              "root step difference", round(row["root_step_vector_change_cm"], 4), "cm/frame")
    print("Results:", out)
    if not plotted:
        print("Plot unavailable (install matplotlib for PNG); CSV and JSON are ready.")


if __name__ == "__main__":
    main()
