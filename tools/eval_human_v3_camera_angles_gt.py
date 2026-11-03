"""V3 human / V6 motion GT camera-angle benchmark (no model inference).

For every existing human V6 reach clip, project EXACT SAME ground truth joint
coordinates into each 2-view arrangement and apply SAME seeded pixel noise.
The results isolate projection/tranigulation and image framing, not Reader
accuracy or mesh occlusion.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from mocap.camera_geometry import pair_condition, project_orthographic, triangulate_orthographic

ROOT = Path("runs/human/lora_synth_A/v6_reach_source")
OUT = Path("runs/human/camera_angle_p0_v3/human_gt_triangulation.json")
ANGLES = {
    "baseline_0_90": (0, 90),
    "symmetric_315_45": (315, 45),
    "oblique_45_135": (45, 135),
}
# Matched to the audited human V6 ortho camera framing (384 x 448).
WIDTH = 384
HEIGHT = 448
M_PER_PX = 2.432802200317383 / 448
Z_CENTER = 0.8688581585884094
SIGMA_PX = 2.0
SEED = 20261103


def run():
    clips = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p / "joints.json").is_file())
    model_path = Path("runs/human/human_skinmax_multicolor_v3.blend")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if not clips:
        raise RuntimeError("Missing V6 human reach GT sequences")
    motions = {}
    for clip in clips:
        source_contract = json.loads((clip / "skeleton_contract.json").read_text(encoding="utf-8"))
        if source_contract.get("source_sha256") != model_sha:
            raise ValueError(f"Human GT source does not match V3 blend: {clip.name}")
        record = json.loads((clip / "joints.json").read_text(encoding="utf-8"))
        xyz = np.asarray(record["joint_position_m"], dtype=np.float64)
        if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[-2] != 55:
            raise ValueError(f"Unexpected human GT shape in {clip}")
        if not np.isfinite(xyz).all():
            raise ValueError(f"Nonfinite human GT {clip}")
        motions[clip.name] = xyz
    rng = np.random.default_rng(SEED)
    # Each angle uses the same numerical random perturbation indexed by
    # clip/frame/joint/view/u-v; no resampling by configuration.
    noise = {name: rng.normal(0, SIGMA_PX, (*points.shape[:2], 2, 2))
             for name, points in motions.items()}

    report = {
        "schema": "anymove.human_v3_camera_azimuth_gt_ab/1",
        "source_blend": str(model_path),
        "source_blend_sha256": model_sha,
        "joint_gt": str(ROOT),
        "provenance": "Existing Blender human V6 3D joint GT; not a trained Reader or generated video",
        "clips": len(motions),
        "frames": sum(len(x) for x in motions.values()),
        "joints_per_frame": 55,
        "noise_px_std": SIGMA_PX,
        "noise_seed": SEED,
        "same_uv_noise_each_configuration": True,
        "width": WIDTH, "height": HEIGHT,
        "metres_per_pixel": M_PER_PX,
        "z_center": Z_CENTER,
        "pairs": {},
    }
    for key, angles in ANGLES.items():
        accurate = []
        noisy = []
        in_frame = []
        for name, points in motions.items():
            f = project_orthographic(points, angles[0], WIDTH, HEIGHT, M_PER_PX, Z_CENTER)
            s = project_orthographic(points, angles[1], WIDTH, HEIGHT, M_PER_PX, Z_CENTER)
            ones = np.ones(points.shape[:2], dtype=np.float64)
            def triang(fv, sv):
                return triangulate_orthographic(
                    fv, sv, ones, ones, front_width=WIDTH,
                    side_width=WIDTH, height=HEIGHT,
                    metres_per_pixel=M_PER_PX, z_center=Z_CENTER,
                    front_angle_deg=angles[0], side_angle_deg=angles[1],
                    front_height_weight=1.0,
                )
            accurate.extend(np.linalg.norm(triang(f, s) - points, axis=-1).ravel())
            n = noise[name]
            noisy.extend(np.linalg.norm(triang(f + n[..., 0, :], s + n[..., 1, :]) - points, axis=-1).ravel())
            ff = (f[..., 0] >= 0) & (f[..., 0] < WIDTH) & (f[..., 1] >= 0) & (f[..., 1] < HEIGHT)
            sf = (s[..., 0] >= 0) & (s[..., 0] < WIDTH) & (s[..., 1] >= 0) & (s[..., 1] < HEIGHT)
            in_frame.append(np.stack((ff.ravel(), sf.ravel()), -1))
        errors = np.asarray(noisy, dtype=float) * 100
        coverage = np.concatenate(in_frame)
        report["pairs"][key] = {
            "angles_deg": list(angles),
            "horizontal_condition_number": pair_condition(*angles),
            "exact_gt_max_error_cm": float(np.max(accurate) * 100),
            "noisy_2px_mpjpe_cm": float(np.mean(errors)),
            "noisy_2px_p95_cm": float(np.percentile(errors, 95)),
            "first_camera_in_frame_fraction": float(coverage[:, 0].mean()),
            "second_camera_in_frame_fraction": float(coverage[:, 1].mean()),
            "both_cameras_in_frame_fraction": float(np.all(coverage, axis=1).mean()),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
