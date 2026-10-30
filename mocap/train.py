"""Fine-tune the DINOv2 joint reader on two views with configurable resolution.

python -m mocap.train DATASET_DIR rig_info.json OUT.pt [epochs] [batch]
    [--input-size 672] [--heatmap-size 192] [--workers 4]
Renders may be at a different (known) scale from the 1280x720 camera layout;
all validation errors are reported back in the unscaled layout pixel space.
"""
import argparse, json, math, pathlib, time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mocap import layout
from mocap.model import JointReader, decode, quality_at
from mocap.quality_targets import synthetic_cutout, PROFILE as QUALITY_TARGET_PROFILE
from mocap.reader_config import HIGH_RES_INPUT_SIZE, HIGH_RES_HEATMAP_SIZE, ReaderSpec
from mocap.reader_geometry import Letterbox
from mocap.reader_schedule import accumulation_group_size, optimizer_step_count


def backdrop(h, w, ground, rng):
    g = rng.uniform(95, 175)
    tint = rng.uniform(-6, 6, 3)
    img = np.empty((h, w, 3), np.float32)
    img[:] = g + tint
    img += np.linspace(-1, 1, h)[:, None, None] * rng.uniform(-10, 10)
    img[ground:] += rng.uniform(5, 35) * (1 if rng.random() < 0.85 else -1)
    return img


def photometric(img, rng):
    img = img.astype(np.float32)
    if rng.random() < 0.8:  # brightness, contrast, saturation, hue
        img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
        m = img.mean()
        img = (img - m) * rng.uniform(0.75, 1.25) + m
        hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-5, 5)) % 180
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.3:
        f = rng.uniform(0.4, 0.8)
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.2:
        img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return img


class Views(Dataset):
    def __init__(self, metas, train, input_size=HIGH_RES_INPUT_SIZE,
                 quality_targets=False):
        self.metas, self.train, self.input_size = metas, train, input_size
        self.quality_targets = quality_targets

    def __len__(self):
        return 2 * len(self.metas)

    def __getitem__(self, i):
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        m = self.metas[i // 2]
        view = ("front", "side")[i % 2]
        meta = json.loads(m.read_text())
        s = float(meta["scale"])
        if not np.isfinite(s) or s <= 0:
            raise ValueError(f"invalid render scale in {m}")
        path = str(m).replace("_meta.json", f"_{view}.png")
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
            raise ValueError(f"missing or invalid RGBA dataset image: {path}")
        h, w = raw.shape[:2]
        expected = (round(getattr(layout, f"W_{view.upper()}") * s), round(layout.H * s))
        if (w, h) != expected:
            raise ValueError(f"{path}: image {(w,h)} != expected {expected}; check render scale and MOCAP_CREATURE")
        rgba = raw[..., [2, 1, 0, 3]].astype(np.float32)
        kp = np.array(meta[f"{view}_uv"], np.float32) * s
        if self.train:
            bg = backdrop(h, w, round(layout.GROUND_V * s), rng)
        else:
            bg = np.empty((h, w, 3), np.float32); bg[:] = layout.BG
            bg[round(layout.GROUND_V * s):] = layout.FLOOR
        a = rgba[..., 3:] / 255
        img = rgba[..., :3] * a + bg * (1 - a)
        if self.train and rng.random() < 0.7:
            img[:, -2:] = 34 if view == "front" else img[:, -2:]
            img[:, :2] = 34 if view == "side" else img[:, :2]
        geo = Letterbox(w, h, self.input_size)
        ox, oy, side = geo.ox, geo.oy, geo.side
        k = self.input_size / side
        A = np.array(geo.affine, np.float32)
        if self.train:
            sc, rot = rng.uniform(0.85, 1.15), math.radians(rng.uniform(-4, 4))
            c = self.input_size / 2
            R = cv2.getRotationMatrix2D((c, c), math.degrees(rot), sc)
            R[:, 2] += rng.uniform(-0.05, 0.05, 2) * self.input_size
            A = R @ np.vstack([A, [0, 0, 1]])
        out = cv2.warpAffine(img, A.astype(np.float32), (self.input_size, self.input_size),
                             flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=tuple(float(x) for x in bg[0, 0]))
        kp = kp @ A[:, :2].T + A[:, 2]
        if self.train:
            out = photometric(out, rng)
        visible = None
        if self.quality_targets:
            if self.train:
                out, visible = synthetic_cutout(out, kp, rng)
            else:
                visible = np.ones(len(kp), dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float() / 255
        record = (x, torch.from_numpy(kp.astype(np.float32)),
                  torch.tensor([ox, oy, side, s], dtype=torch.float32))
        if self.quality_targets:
            return record + (torch.from_numpy(visible),)
        return record


def heatmaps(kp, input_size, hm_size, sigma=2.0):
    """Gaussian targets at hm_size resolution for keypoints in input_size pixels."""
    b, j, _ = kp.shape
    g = (kp * hm_size / input_size).view(b, j, 1, 1, 2)
    ys, xs = torch.meshgrid(torch.arange(hm_size, device=kp.device),
                            torch.arange(hm_size, device=kp.device), indexing="ij")
    d2 = (xs - g[..., 0]) ** 2 + (ys - g[..., 1]) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("rig_info")
    parser.add_argument("output")
    parser.add_argument("epochs", type=int, nargs="?", default=20)
    parser.add_argument("batch", type=int, nargs="?", default=8)
    parser.add_argument("--input-size", type=int, default=HIGH_RES_INPUT_SIZE)
    parser.add_argument("--heatmap-size", type=int, default=HIGH_RES_HEATMAP_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="microbatches per optimizer step, e.g. batch=1 + accum=8")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="trade compute for lower DINOv2 activation memory")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="fallback: train only the heatmap head; may reduce accuracy")
    parser.add_argument("--train-quality", action="store_true",
                        help="new checkpoint with supervised synthetic cutout / localization error quality maps")
    args = parser.parse_args()
    spec = ReaderSpec(args.input_size, args.heatmap_size)
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.grad_accum < 1:
        parser.error("epochs/batch/grad-accum must be positive and workers >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("DINOv2 fine-tuning requires a CUDA GPU")

    info = json.load(open(args.rig_info))
    names = [k["name"] for k in info["keypoints"]]

    def split():
        metas = sorted(pathlib.Path(args.dataset_dir).glob("*_meta.json"))
        val = [m for m in metas if int(m.name[:5]) % 25 == 0]
        train = [m for m in metas if int(m.name[:5]) % 25 != 0]
        return train, val

    train, val = split()
    if not train or not val:
        raise ValueError("dataset must include both train and held-out samples")
    scales, profiles = set(), set()
    for m in train + val:
        metadata = json.loads(m.read_text())
        scales.add(float(metadata["scale"]))
        # Legacy datasets remain trainable, but must not mix with new soft-blur
        # samples: otherwise "rerender 30" can silently create mixed domains.
        profiles.add(metadata.get("blur_profile", "legacy_unversioned"))
    if len(scales) != 1:
        raise ValueError(f"mixed dataset render scales {sorted(scales)}; use a fresh dataset_hr directory")
    if len(profiles) != 1:
        raise ValueError(f"mixed dataset blur profiles {sorted(profiles)}; rerender all poses with tools/dataset.sh")
    print(f"{len(train)} train / {len(val)} val poses; {len(names)} joints; "
          f"render scale {next(iter(scales))}; blur profile {next(iter(profiles))}; "
          f"reader {spec.input_size}/{spec.heatmap_size}", flush=True)
    vl = DataLoader(Views(val, False, spec.input_size,
                          quality_targets=args.train_quality),
                    batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    model = JointReader(len(names), input_size=spec.input_size,
                        heatmap_size=spec.heatmap_size,
                        quality_head=args.train_quality).cuda()
    # Prefer memory-efficient PyTorch scaled dot-product attention.
    model.backbone.set_attn_implementation("sdpa")
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    elif args.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    groups = [{"params": head, "lr": 1e-3}]
    if not args.freeze_backbone:
        groups.insert(0, {"params": model.backbone.parameters(), "lr": 2e-5})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    base = [g["lr"] for g in opt.param_groups]
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    print(f"CUDA {torch.cuda.get_device_name(0)}; AMP {amp_dtype}; "
          f"batch {args.batch} x accumulation {args.grad_accum} = "
          f"effective batch {args.batch * args.grad_accum}; "
          f"gradient_checkpointing {args.gradient_checkpointing}; "
          f"freeze_backbone {args.freeze_backbone}", flush=True)
    step, t0, warm = 0, time.time(), 300
    for ep in range(args.epochs):
        train, _ = split()
        dl = DataLoader(Views(train, True, spec.input_size,
                              quality_targets=args.train_quality),
                        batch_size=args.batch,
                        shuffle=True, num_workers=args.workers, pin_memory=True)
        if not len(dl):
            raise RuntimeError("empty training DataLoader")
        n_updates = optimizer_step_count(len(dl), args.grad_accum)
        print(f"epoch {ep}: {len(train)} train poses; {n_updates} optimizer steps", flush=True)
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        for bi_, sample in enumerate(dl):
            if args.train_quality:
                x, kp, _, visible = sample
            else:
                x, kp, _ = sample
                visible = None
            progress = (ep + (bi_ // args.grad_accum) / n_updates) / args.epochs
            f = 0.5 * (1 + math.cos(math.pi * progress)) * min(1, (step + 1) / warm)
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            x, kp = x.cuda(non_blocking=True), kp.cuda(non_blocking=True)
            if visible is not None:
                visible = visible.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                if args.train_quality:
                    pred, quality = model(x, return_quality=True)
                else:
                    pred = model(x)
            loss = F.mse_loss(pred.float(), heatmaps(kp, spec.input_size, spec.heatmap_size)) * 1000
            if args.train_quality:
                kp_hm = kp * spec.heatmap_size / spec.input_size
                vis_logits, log_error = quality_at(quality.float(), kp_hm)
                # This is *synthetic patch visibility*, not ground truth
                # self-occlusion. Negative labels are upweighted due to rarity.
                # Rendered joints may project beyond the crop. Such
                # targets cannot provide a valid visibility or pixel-error
                # supervision signal in the Reader's input frame.
                in_frame = ((kp_hm[..., 0] >= 0) &
                            (kp_hm[..., 0] < spec.heatmap_size) &
                            (kp_hm[..., 1] >= 0) &
                            (kp_hm[..., 1] < spec.heatmap_size)).float()
                vis_loss = F.binary_cross_entropy_with_logits(
                    vis_logits, visible, reduction="none")
                vis_loss = (vis_loss * torch.where(
                    visible < .5, 3.0, 1.0) * in_frame).sum() / in_frame.sum().clamp(min=1)
                with torch.no_grad():
                    predicted_uv, _ = decode(pred.float().detach())
                    residual = (predicted_uv-kp_hm).norm(dim=-1)
                    log_target = torch.log1p(residual.clamp(max=128.0))
                # The residual target is a training error proxy, not a
                # confidence calibration. Do not use this prediction as a
                # solver weight until validation against known 2D labels.
                sigma_loss = (F.smooth_l1_loss(
                    log_error.clamp(-1.0, 6.0), log_target,
                    reduction="none") * in_frame).sum() / in_frame.sum().clamp(min=1)
                loss = loss + .05 * (vis_loss + sigma_loss)
            divisor = accumulation_group_size(bi_, len(dl), args.grad_accum)
            scaler.scale(loss / divisor).backward()
            if (bi_ + 1) % args.grad_accum == 0 or bi_ + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    used = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(f"ep {ep} optimizer_step {step} loss {loss.item():.4f} "
                          f"lr_factor {f:.3f} peak_vram_allocated {used:.2f}GiB "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)
        print(f"epoch {ep} training peak GPU allocated "
              f"{torch.cuda.max_memory_allocated() / (1024**3):.2f}GiB; "
              f"peak reserved {torch.cuda.max_memory_reserved() / (1024**3):.2f}GiB",
              flush=True)
        model.eval()
        errs = {"front": [], "side": []}
        with torch.no_grad():
            for bi, sample in enumerate(vl):
                x, kp, geo = sample[:3]
                with torch.autocast("cuda", dtype=amp_dtype):
                    pred = model(x.cuda(non_blocking=True)).float()
                uv, _ = decode(pred)
                uv = uv.cpu() * spec.input_size / spec.heatmap_size
                # Undo letterbox and render scale; report in 1280x720 layout pixels.
                lay = lambda p: (p * geo[:, None, 2:3] / spec.input_size - geo[:, None, :2]) / geo[:, None, 3:4]
                e = (lay(uv) - lay(kp)).norm(dim=-1)
                for r in range(len(x)):
                    errs[("front", "side")[(bi * args.batch + r) % 2]].append(e[r].numpy())
        msg = {v: float(np.mean(errs[v])) for v in errs}
        print(f"epoch {ep} val mean px error (layout pixels) front {msg['front']:.2f} side {msg['side']:.2f}", flush=True)
        torch.save({"state_dict": model.state_dict(), "names": names, "epoch": ep,
                    "val": msg,
                    "reader_spec": dict(spec.checkpoint_metadata(),
                                        quality_head=bool(args.train_quality)),
                    "quality_spec": ({"version": 1,
                                      "target": QUALITY_TARGET_PROFILE,
                                      "error_unit": "heatmap_pixels",
                                      "visibility_semantics": "synthetic_cutout_only"}
                                     if args.train_quality else None),
                    "train_setup": {"batch": args.batch, "grad_accum": args.grad_accum,
                                    "gradient_checkpointing": args.gradient_checkpointing,
                                    "freeze_backbone": args.freeze_backbone,
                                    "amp_dtype": str(amp_dtype)}}, args.output)
    print("saved", args.output)




if __name__ == "__main__":
    main()
