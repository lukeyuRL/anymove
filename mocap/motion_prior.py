"""Train an empirical Gaussian skeletal motion prior; no video model changes.

This is a learnable *statistical* motion prior, NOT a diffusion/generative
model. It estimates per-DOF clean angular acceleration from verified
temporal animations and performs MAP temporal denoising with second-order
regularization. Root Translation and Root Quaternion are untouched.
Independent random Blender poses are NOT suitable prior training data.
"""
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

PROFILE = "anymove_empirical_angular_acceleration_prior_v1"
MIN_SEQUENCES = 2
MIN_FRAMES = 24
SAFE_SOURCE_KINDS = ("verified_blender_animation", "verified_iclone_animation",
                     "measured_mocap", "synthetic_ground_truth_animation")
MIN_STD_RAD_PER_FRAME2 = .0015
MAX_REGULARIZATION = 12.0


def _hash_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def motion_arrays(payload, bones=None, fps=None):
    if bones is None:
        bones = payload.get("bones")
    if payload.get("bones") != bones:
        raise ValueError("motion bone names/order differ from prior skeleton")
    q = np.asarray(payload.get("euler"), np.float64)
    if (q.ndim != 3 or q.shape[1:] != (len(bones), 3)
            or q.shape[0] < 3 or not np.isfinite(q).all()):
        raise ValueError("invalid finite Euler frame data")
    rate = float(payload.get("fps", 0))
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("invalid motion fps")
    if fps is not None and abs(rate-fps) > 1e-6:
        raise ValueError("motion prior fps differs from input animation")
    return q, rate


def train_prior(manifest_path, noise_sigma_rad=.04):
    """Read explicit trusted temporal sequences, never the current AI solve."""
    manifest = pathlib.Path(manifest_path).resolve()
    with open(str(manifest), encoding="utf-8") as stream:
        cfg = json.load(stream)
    if cfg.get("version") != 1 or len(cfg.get("sequences", [])) < MIN_SEQUENCES:
        raise ValueError("motion prior requires manifest version 1 and >=2 verified sequences")
    if not .0001 <= noise_sigma_rad <= .5:
        raise ValueError("noise_sigma_rad outside reasonable [0.0001, 0.5] range")
    root = manifest.parent
    fps = None
    bones = None
    accelerations = []
    sources = []
    for item in cfg["sequences"]:
        source_kind = item.get("source_kind")
        if not item.get("verified", False) or source_kind not in SAFE_SOURCE_KINDS:
            raise ValueError("motion prior requires explicitly verified, typed animation provenance")
        path = pathlib.Path(item["path"])
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        with open(str(path), encoding="utf-8") as stream:
            doc = json.load(stream)
        if bones is None:
            bones = doc["bones"]
        motion, rate = motion_arrays(doc, bones, fps)
        if len(motion) < MIN_FRAMES:
            raise ValueError("motion prior needs at least 24 continuous frames per source")
        fps = rate
        accelerations.append(np.diff(motion, n=2, axis=0))
        sources.append({"path": str(path), "sha256": _hash_file(path),
                        "frames": len(motion), "source_kind": source_kind})
    # Mean squared angular acceleration, not learned from independent poses.
    second = np.concatenate(accelerations, axis=0)
    acceleration_sigma = np.maximum(np.sqrt(np.mean(second**2, axis=0)),
                                    MIN_STD_RAD_PER_FRAME2)
    # Root Euler is ignored by FK (root rotation comes from root_quat).
    acceleration_sigma[0] = 1.0
    return {
        "profile": PROFILE, "version": 1, "bones": bones,
        "fps": fps, "noise_sigma_rad": float(noise_sigma_rad),
        "acceleration_sigma": acceleration_sigma.tolist(),
        "source_sequences": sources,
        "num_training_frames": int(sum(s["frames"] for s in sources)),
        "limit": "Euler angular denoising only; no learned root translation, root quaternion, contact or semantics",
    }


def denoise_euler(observed, prior, limits_deg=None, strength=1.0):
    """MAP: argmin ||x-observed||² + λ||D2 x||² / σ_a².

    Axis-wise gamma incorporates the *learned* clean angular acceleration
    standard deviation. Identity for strength=0, constant sequences and
    the root Euler row; no Root Motion data is touched by this function.
    """
    x = np.asarray(observed, dtype=np.float64)
    names = prior.get("bones")
    if (x.ndim != 3 or x.shape[1:] != (len(names), 3)
            or x.shape[0] < 3 or not np.isfinite(x).all()):
        raise ValueError("invalid input Euler shape/data for prior")
    if not math.isfinite(strength) or not 0 <= strength <= 5:
        raise ValueError("denoising strength must be in [0,5]")
    sigma = np.asarray(prior["acceleration_sigma"], dtype=np.float64)
    if sigma.shape != (len(names), 3) or not np.isfinite(sigma).all() or \
            np.any(sigma <= 0):
        raise ValueError("invalid learned prior acceleration variance")
    noise = float(prior["noise_sigma_rad"])
    if not math.isfinite(noise) or noise <= 0:
        raise ValueError("invalid prior observation noise")
    result = x.copy()
    if strength == 0:
        return result
    T = len(x)
    d2 = np.zeros((T-2, T), dtype=np.float64)
    for t in range(T-2):
        d2[t, t:t+3] = [1., -2., 1.]
    penalty = d2.T @ d2
    eye = np.eye(T)
    for j in range(1, len(names)):
        for axis in range(3):
            gamma = min(MAX_REGULARIZATION,
                        strength*(noise/sigma[j, axis])**2)
            if gamma <= 0:
                continue
            result[:, j, axis] = np.linalg.solve(
                eye + gamma*penalty, x[:, j, axis])
    if limits_deg is not None:
        limits = np.radians(np.asarray(limits_deg, np.float64))
        if limits.shape != (len(names), 3, 2):
            raise ValueError("invalid Blender joint limit shape")
        result[:, 1:] = np.maximum(
            np.minimum(result[:, 1:], limits[None, 1:, :, 1]),
            limits[None, 1:, :, 0])
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) == 3 and args[0] == "train":
        prior = train_prior(args[1])
        target = pathlib.Path(args[2])
        if target.exists():
            raise FileExistsError("do not overwrite an existing motion prior")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(prior, indent=2), encoding="utf-8")
        print("trained empirical acceleration prior", target,
              "from", len(prior["source_sequences"]), "verified sequences")
    else:
        raise SystemExit("usage: python -m mocap.motion_prior train manifest.json prior.json")


if __name__ == "__main__":
    main()
