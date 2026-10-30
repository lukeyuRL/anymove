"""Read every joint in both halves of a two-camera video with a trained joint reader.

python -m mocap.detect VIDEO.mp4 MODEL.pt REF.png OUT.npz [overlay.mp4]
The clip is first aligned to the reference frame (the video model may resize or crop the canvas), then each frame is
split into its front and side views. Writes pixel positions in layout coordinates
and RAW, UNCALIBRATED heatmap peak amplitudes (not probabilities) per view.
The solver clips these peak values to [0,1] for confidence weighting.
"""
import os, sys
from contextlib import nullcontext

import cv2
import numpy as np
import torch

from mocap import layout
from mocap.model import JointReader, decode, quality_at
from mocap.reader_config import LEGACY_INPUT_SIZE, spec_from_checkpoint
from mocap.reader_geometry import Letterbox


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[..., ::-1])
    return np.array(frames), fps


def calibrate(first, ref):
    """Affine map from video pixels to layout pixels, found by aligning the first frame to the reference image."""
    h, w = first.shape[:2]
    S = np.array([[layout.W / w, 0, 0], [0, layout.H / h, 0]], np.float32)  # plain stretch to start
    a = cv2.cvtColor(cv2.resize(ref, (640, 360)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(cv2.warpAffine(first, S, (layout.W, layout.H)), (640, 360)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        print("ECC did not converge; using a plain stretch")
    # warp maps reference (half-size) coords to frame coords; we want frame -> layout
    W2 = np.vstack([warp, [0, 0, 1]])
    W2[:2, 2] *= 2  # half-size -> full-size translation
    A = np.linalg.inv(W2) @ np.vstack([S, [0, 0, 1]])
    return A[:2].astype(np.float32)


def prep(view, size=LEGACY_INPUT_SIZE):
    h, w = view.shape[:2]
    geo = Letterbox(w, h, size)
    fill = tuple(float(x) for x in view[2, 2])
    k = size / geo.side
    out = cv2.warpAffine(view, np.array(geo.affine, np.float32), (size, size),
                         flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return out, geo


@torch.no_grad()
def run(model, views, bs=None, with_quality=False):
    size, hm_size = model.input_size, model.heatmap_size
    if bs is None:
        bs = int(os.environ.get("MOCAP_DETECT_BATCH", 1 if size > 448 else 32))
    if bs < 1:
        raise ValueError("MOCAP_DETECT_BATCH must be positive")
    device = next(model.parameters()).device
    uv_all, conf_all = [], []
    visibility_all, expected_error_all = [], []
    for i in range(0, len(views), bs):
        batch = [prep(v, size) for v in views[i:i + bs]]
        x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float().to(device) / 255
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with context:
            if with_quality:
                pred, quality_map = model(x, return_quality=True)
                pred, quality_map = pred.float(), quality_map.float()
            else:
                pred = model(x).float()
        uv_hm, conf = decode(pred)
        if with_quality:
            vis_logit, log_error = quality_at(quality_map, uv_hm)
            vis = torch.sigmoid(vis_logit).cpu().numpy()
            error_hm = torch.expm1(log_error.clamp(0, 6)).cpu().numpy()
        uv = uv_hm.cpu().numpy() * size / hm_size
        for index, ((_, geo), u) in enumerate(zip(batch, uv)):
            uv_all.append(u * geo.side / size - np.array([geo.ox, geo.oy]))
            if with_quality:
                visibility_all.append(vis[index])
                # The head predicts an uncalibrated localization residual
                # in heatmap pixels. Undo geometry back to view layout pixels.
                expected_error_all.append(error_hm[index] * geo.side / hm_size)
        conf_all.append(conf.cpu().numpy())
    if with_quality:
        return (np.array(uv_all), np.concatenate(conf_all),
                np.array(visibility_all), np.array(expected_error_all))
    return np.array(uv_all), np.concatenate(conf_all)


def load(model_path):
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    spec = spec_from_checkpoint(ck)
    raw_quality = ck.get("quality_spec")
    uses_quality = isinstance(raw_quality, dict) and raw_quality.get("version") == 1
    if raw_quality is not None and not uses_quality:
        raise ValueError("unsupported quality_spec checkpoint")
    if bool((ck.get("reader_spec") or {}).get("quality_head", False)) != uses_quality:
        raise ValueError("reader_spec quality_head mismatch")
    model = JointReader(len(ck["names"]), input_size=spec.input_size,
                        heatmap_size=spec.heatmap_size,
                        quality_head=uses_quality).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    print(f"loaded reader {spec.input_size}x{spec.input_size} / heatmap {spec.heatmap_size}", flush=True)
    return model, ck["names"]


def overlay(frames, fuv, suv, fconf, names, path, fps):
    h, w = layout.H, layout.W
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    col = lambda n: (209, 60, 245) if n.endswith("_L") else (255, 107, 47) if n.endswith("_R") else (255, 255, 255)  # BGR
    for t, f in enumerate(frames):
        img = np.ascontiguousarray(f[..., ::-1])
        for j, n in enumerate(names):
            for (u, v), x0 in ((fuv[t, j], 0), (suv[t, j], layout.W_FRONT)):
                cv2.circle(img, (int(u + x0), int(v)), 4, col(n), -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    video, model_path, ref_path, out = args[:4]
    frames, fps = read_frames(video)
    ref = cv2.imread(ref_path)[..., ::-1]
    if "--no-calib" in sys.argv:  # already a layout frame (the exam)
        A = np.array([[layout.W / frames.shape[2], 0, 0], [0, layout.H / frames.shape[1], 0]], np.float32)
    else:
        A = calibrate(frames[0], ref)
    print("calibration (video px -> layout px):", A.round(4).tolist())
    frames = np.array([cv2.warpAffine(f, A, (layout.W, layout.H), flags=cv2.INTER_AREA) for f in frames])
    model, names = load(model_path)
    front_result = run(model, [f[:, :layout.W_FRONT] for f in frames],
                       with_quality=model.quality_head)
    side_result = run(model, [f[:, layout.W_FRONT:] for f in frames],
                      with_quality=model.quality_head)
    fuv, fconf = front_result[:2]
    suv, sconf = side_result[:2]
    data = dict(front_uv=fuv, side_uv=suv, front_conf=fconf,
                side_conf=sconf, fps=fps, names=np.array(names), calib=A)
    if model.quality_head:
        data.update({
            "front_synthetic_visibility_score": front_result[2],
            "side_synthetic_visibility_score": side_result[2],
            "front_expected_error_px": front_result[3],
            "side_expected_error_px": side_result[3],
        })
        print("optional synthetic-cutout quality head present; scores are "
              "NOT calibrated real occlusion probabilities or validated "
              "solver confidence weights", flush=True)
    np.savez(out, **data)
    print(f"{len(frames)} frames at {fps:.2f} fps; raw heatmap peak means "
          f"front {fconf.mean():.2f} side {sconf.mean():.2f}; "
          f"ranges front [{fconf.min():.3g},{fconf.max():.3g}] "
          f"side [{sconf.min():.3g},{sconf.max():.3g}]")
    if len(args) > 4:
        overlay(frames, fuv, suv, fconf, names, args[4], fps)


if __name__ == "__main__":
    main()
