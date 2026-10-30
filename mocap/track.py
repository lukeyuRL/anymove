"""Apply conservative optical-flow tracking to an existing AnyMove detection.

python -m mocap.track VIDEO.mp4 original_det.npz tracked_det.npz

Does NOT rerun DINOv2, change the source detection, or infer a new identity.
Produces an opt-in det.npz with the same existing Reader/Solver keys, plus
track_weights and a separate JSON audit. The extra keys are optional.
"""
import hashlib
import json
import pathlib
import sys

import cv2
import numpy as np

from mocap import layout
from mocap.temporal_tracking import lk_measurements, fuse_tracks


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def video_frames(video, transform, expected):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("unable to open video: " + str(video))
    images = []
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        images.append(cv2.warpAffine(frame[..., ::-1], transform,
                                    (layout.W, layout.H),
                                    flags=cv2.INTER_AREA))
    cap.release()
    if len(images) != expected:
        raise ValueError("video/detection frame mismatch: %d versus %d" %
                         (len(images), expected))
    return np.asarray(images), fps


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 3:
        raise SystemExit("usage: python -m mocap.track VIDEO.mp4 IN_DET.npz OUT_DET.npz")
    video, source, target = [pathlib.Path(v) for v in args]
    if source.resolve() == target.resolve():
        raise ValueError("must not overwrite source detections")
    if target.exists():
        raise FileExistsError("target exists: " + str(target))
    with np.load(str(source), allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    for view in ("front", "side"):
        for field in ("_uv", "_conf"):
            if view+field not in data:
                raise ValueError("missing source detection field: " + view+field)
    if "calib" not in data:
        raise ValueError("source det.npz lacks source-video calibration")
    affine = np.asarray(data["calib"], np.float32)
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError("invalid calibration affine")
    T = len(data["front_uv"])
    if T < 2 or len(data["side_uv"]) != T:
        raise ValueError("invalid detection frame count")
    images, fps = video_frames(video, affine, T)
    if abs(float(data["fps"]) - fps) > 0.05:
        raise ValueError("video fps differs from detection fps")
    details = {"profile": "anymove_lk_reader_temporal_fusion_v1",
               "source_detection_sha256": _sha256(source),
               "source_video": str(video),
               "fps": fps, "views": {}}
    for view, x0, width in (("front", 0, layout.W_FRONT),
                            ("side", layout.W_FRONT, layout.W_SIDE)):
        raw_uv = np.asarray(data[view+"_uv"], np.float64)
        frames = images[:, :, x0:x0+width, :]
        flow, ok, error = lk_measurements(frames, raw_uv)
        uv, weight, summary = fuse_tracks(
            raw_uv, data[view+"_conf"], flow, ok, error, width, layout.H)
        data[view+"_uv_reader"] = data[view+"_uv"].copy()
        data[view+"_uv"] = uv
        data[view+"_track_weight"] = weight
        details["views"][view] = summary
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(target), **data)
    audit = target.with_suffix(".tracking.json")
    audit.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print("saved", target, "audit", audit)
    for view, report in details["views"].items():
        print(view, report["counts"], "mean UV adjustment",
              report["mean_reader_to_tracked_px"], "px")


if __name__ == "__main__":
    main()
