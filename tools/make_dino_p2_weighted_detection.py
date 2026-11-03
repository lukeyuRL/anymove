"""Bridge a P2 Dino Quality Reader to the existing Solver (opt-in only).

python -m tools.make_dino_p2_weighted_detection RAW_DET.npz P2_DET.npz     OUT_P2_WEIGHTED.npz [--checkpoint P2_READER.pt]

- RAW must be the original, NON-tracked Reader NPZ for the same video.
- P2 must be produced with the frozen baseline Dino Reader + Quality Head.
- Refuses any frame/joint/calibration/UV/peak mismatch above tolerance.
- Keeps exact raw original joint UV/peak and adds separate bounded Solver
  multipliers derived from predicted localization error only.
- Does NOT use synthetic visibility predictions as mesh visibility.
- DOES NOT overwrite the Reader, original NPZ, motion.json or prior outputs.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

from mocap.quality_weighting import prepare_quality_weighted_detection


def sha256_file(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for x in iter(lambda: f.read(1024*1024), b""):
            h.update(x)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original_reader")
    p.add_argument("p2_reader")
    p.add_argument("output")
    p.add_argument("--checkpoint", help="optional P2 Reader snapshot; provenance SHA only")
    p.add_argument("--threshold-px", type=float, default=5.,
                   help="do not modify weight if predicted error <= 5 layout px")
    p.add_argument("--floor", type=float, default=.4,
                   help="minimum multiplier applied by existing Solver")
    a = p.parse_args(argv)
    source = pathlib.Path(a.original_reader).resolve()
    quality = pathlib.Path(a.p2_reader).resolve()
    target = pathlib.Path(a.output).resolve()
    sidecar = target.with_suffix(".p2_weights.json")
    if source == quality:
        raise ValueError("P2 detection must be separate from baseline")
    if target in (source, quality) or target.exists() or sidecar.exists():
        raise FileExistsError("refuse overwrite of P2 inputs or previous outputs")
    if not source.is_file() or not quality.is_file():
        raise FileNotFoundError("missing original/P2 detection NPZ")
    with np.load(source, allow_pickle=False) as z:
        raw = {k: z[k].copy() for k in z.files}
    with np.load(quality, allow_pickle=False) as z:
        p2 = {k: z[k].copy() for k in z.files}
    output, audit = prepare_quality_weighted_detection(
        raw, p2, threshold_px=a.threshold_px, floor=a.floor)
    audit["source_detection_sha256"] = sha256_file(source)
    audit["quality_detection_sha256"] = sha256_file(quality)
    if a.checkpoint:
        ck = pathlib.Path(a.checkpoint)
        if not ck.is_file():
            raise FileNotFoundError("P2 checkpoint snapshot missing")
        audit["quality_checkpoint_sha256"] = sha256_file(ck)
        audit["quality_checkpoint"] = str(ck.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **output)
    audit["weighted_detection_sha256"] = sha256_file(target)
    try:
        sidecar.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    print(json.dumps(audit, indent=2))
    print("P2 weight-only detection saved:", target)
    print("Run: python -m mocap.solve", target, "assets/dino/rigged/rig_info.json NEW_motion_p2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
