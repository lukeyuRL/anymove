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
from mocap.model import JointReader, decode
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
    def __init__(self, metas, train, input_size=HIGH_RES_INPUT_SIZE):
        self.metas, self.train, self.input_size = metas, train, input_size

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
        x = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float() / 255
        return x, torch.from_numpy(kp.astype(np.float32)), torch.tensor([ox, oy, side, s], dtype=torch.float32)


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
    scales = set()
    for m in train + val:
        metadata = json.loads(m.read_text())
        scales.add(float(metadata["scale"]))
    if len(scales) != 1:
        raise ValueError(f"mixed dataset render scales {sorted(scales)}; use a fresh dataset_hr directory")
    print(f"{len(train)} train / {len(val)} val poses; {len(names)} joints; "
          f"render scale {next(iter(scales))}; reader {spec.input_size}/{spec.heatmap_size}", flush=True)
    vl = DataLoader(Views(val, False, spec.input_size),
                    batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    model = JointReader(len(names), input_size=spec.input_size,
                        heatmap_size=spec.heatmap_size).cuda()
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
        dl = DataLoader(Views(train, True, spec.input_size), batch_size=args.batch,
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
        for bi_, (x, kp, _) in enumerate(dl):
            progress = (ep + (bi_ // args.grad_accum) / n_updates) / args.epochs
            f = 0.5 * (1 + math.cos(math.pi * progress)) * min(1, (step + 1) / warm)
            for g, b0 in zip(opt.param_groups, base):
                g["lr"] = b0 * f
            x, kp = x.cuda(non_blocking=True), kp.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                pred = model(x)
            loss = F.mse_loss(pred.float(), heatmaps(kp, spec.input_size, spec.heatmap_size)) * 1000
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
            for bi, (x, kp, geo) in enumerate(vl):
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
                    "val": msg, "reader_spec": spec.checkpoint_metadata(),
                    "train_setup": {"batch": args.batch, "grad_accum": args.grad_accum,
                                    "gradient_checkpointing": args.gradient_checkpointing,
                                    "freeze_backbone": args.freeze_backbone,
                                    "amp_dtype": str(amp_dtype)}}, args.output)
    print("saved", args.output)




if __name__ == "__main__":
    main()
