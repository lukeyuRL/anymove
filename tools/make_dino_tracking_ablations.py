"""Create non-destructive P1 2x2-factor ablation detections.

python tools/make_dino_tracking_ablations.py original_det.npz tracked_det.npz OUT_DIR
Produces OUT_DIR/det_p1_uv_only.npz and det_p1_weight_only.npz.
Use mocap.solve separately on each to isolate UV vs weight contributions.
"""
import argparse
import json
import pathlib

import numpy as np

from mocap.tracking_ablation import make_tracking_ablations
from tools.diagnose_dino_tracking import file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_detection")
    parser.add_argument("tracked_detection")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    original = pathlib.Path(args.raw_detection).resolve()
    refined = pathlib.Path(args.tracked_detection).resolve()
    outdir = pathlib.Path(args.output_dir).resolve()
    if original == refined:
        raise ValueError("P1 tracked and source detections must be distinct")
    targets = (
        outdir/"det_p1_uv_only.npz",
        outdir/"det_p1_weight_only.npz",
        outdir/"p1_ablation_inputs.json",
    )
    for target in targets:
        if target.exists() or target in (original, refined):
            raise FileExistsError("P1 ablation output must not replace source or previous outputs: " + str(target))
    audit_path = refined.with_suffix(".tracking.json")
    if audit_path.is_file():
        tracking = json.loads(audit_path.read_text(encoding="utf-8"))
        if tracking.get("source_detection_sha256") != file_sha256(original):
            raise ValueError("tracking audit references a different raw detection SHA")
    with np.load(original, allow_pickle=False) as pack:
        raw = {k: pack[k].copy() for k in pack.files}
    with np.load(refined, allow_pickle=False) as pack:
        tracked = {k: pack[k].copy() for k in pack.files}
    uv_only, weight_only = make_tracking_ablations(raw, tracked)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(targets[0], **uv_only)
    np.savez_compressed(targets[1], **weight_only)
    metadata = {
        "profile": "anymove.p1_dino_factorial_ablation/1",
        "raw_detection_sha256": file_sha256(original),
        "tracked_detection_sha256": file_sha256(refined),
        "uv_only": str(targets[0]),
        "weight_only": str(targets[1]),
        "meaning": {
            "baseline": "original Reader coordinates and original weights",
            "uv_only": "tracked UV coordinates but original Reader weights",
            "weight_only": "original Reader coordinates with P1 tracking weight multipliers",
            "full_p1": "tracked coordinates and P1 multipliers",
        },
        "no_ground_truth": True,
    }
    targets[2].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
