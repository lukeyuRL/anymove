"""The joint reader: DINOv2 (ViT-B/14 with registers) plus a small upsampling head that draws one heatmap per joint.

Both views go through the same network one at a time. Input and heatmap sizes are
specified by the checkpoint (legacy 448/128, high-res default 672/192).
Positional embeddings are interpolated automatically by transformers 5.20.x.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from mocap.reader_config import ReaderSpec, LEGACY_INPUT_SIZE, LEGACY_HEATMAP_SIZE

BACKBONE = "facebook/dinov2-with-registers-base"
# Kept for callers outside this repository; not the high-res training defaults.
SIZE, HM = LEGACY_INPUT_SIZE, LEGACY_HEATMAP_SIZE
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class JointReader(nn.Module):
    def __init__(self, n_joints, backbone=BACKBONE, input_size=SIZE, heatmap_size=HM):
        super().__init__()
        self.spec = ReaderSpec(input_size, heatmap_size)
        self.input_size = self.spec.input_size
        self.heatmap_size = self.spec.heatmap_size
        self.backbone = AutoModel.from_pretrained(backbone)
        self.n_reg = getattr(self.backbone.config, "num_register_tokens", 0)
        c = self.backbone.config.hidden_size

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(16, o), nn.GELU())

        self.neck = block(c, 256)
        self.up1 = block(256, 128)
        self.up2 = block(128, 128)
        self.out = nn.Conv2d(128, n_joints, 1)

    def forward(self, x):
        b = x.shape[0]
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            raise ValueError(f"reader expects {self.input_size}x{self.input_size}, got {tuple(x.shape[-2:])}")
        # Dinov2WithRegistersEmbeddings in pinned transformers 5.20.2
        # automatically interpolates pretrained positional embeddings.
        tokens = self.backbone(pixel_values=(x - MEAN.to(x)) / STD.to(x)).last_hidden_state
        g = self.input_size // 14
        patches = tokens[:, 1 + self.n_reg:]
        if patches.shape[1] != g * g:
            raise ValueError(f"reader expected {g*g} patch tokens, got {patches.shape[1]}")
        f = patches.transpose(1, 2).reshape(b, -1, g, g)
        f = self.neck(f)
        f = self.up1(F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False))
        f = self.up2(F.interpolate(f, size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False))
        return self.out(f)


def square(img, fill):
    """Pad an H x W x 3 view to a centred square; return it and the (offset_x, offset_y) of the view inside."""
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.empty((s, s, 3), img.dtype)
    out[:] = fill
    ox, oy = (s - w) // 2, (s - h) // 2
    out[oy:oy + h, ox:ox + w] = img
    return out, ox, oy, s


def decode(hm):
    """Heatmaps (B, J, H, W) to sub-pixel peaks in heatmap units and raw values."""
    b, j, h, w = hm.shape
    flat = hm.reshape(b, j, -1)
    conf, idx = flat.max(-1)
    y, x = (idx // w).float(), (idx % w).float()
    # quadratic refinement from the neighbours of the peak
    xi, yi = x.long().clamp(1, w - 2), y.long().clamp(1, h - 2)
    gather = lambda dy, dx: hm.gather(-1, (xi + dx).view(b, j, 1, 1).expand(b, j, h, 1)).squeeze(-1) \
        .gather(-1, (yi + dy).view(b, j, 1)).squeeze(-1)
    c, l, r, u, d = gather(0, 0), gather(0, -1), gather(0, 1), gather(-1, 0), gather(1, 0)
    dx = 0.5 * (r - l) / (2 * c - l - r).clamp(min=1e-6)
    dy = 0.5 * (d - u) / (2 * c - u - d).clamp(min=1e-6)
    x = x + dx.clamp(-0.5, 0.5)
    y = y + dy.clamp(-0.5, 0.5)
    return torch.stack([x, y], -1), conf
