"""Native 672px / 192px heatmap DINOv2 Reader V2.

This is deliberately separate from the legacy 448/128 JointReader. Supports
the pretrained DINOv2-with-registers-base positional interpolation for a
48x48 token grid. Frozen backbone; train only the compact heatmap head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

BACKBONE = "facebook/dinov2-with-registers-base"
INPUT_SIZE = 672
HEATMAP_SIZE = 192
PATCH_GRID = INPUT_SIZE // 14
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


class HumanReaderHRV2(nn.Module):
    def __init__(self, joints=67, backbone=BACKBONE, freeze_backbone=True):
        super().__init__()
        if joints <= 0:
            raise ValueError("Positive joint count required")
        self.num_joints = int(joints)
        self.backbone = AutoModel.from_pretrained(backbone, local_files_only=True)
        self.registers = int(getattr(self.backbone.config, "num_register_tokens", 0))
        c = int(self.backbone.config.hidden_size)
        self.register_buffer("mean", torch.tensor(IMAGE_MEAN).view(1,3,1,1),persistent=False)
        self.register_buffer("std", torch.tensor(IMAGE_STD).view(1,3,1,1),persistent=False)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        self.neck = nn.Sequential(
            nn.Conv2d(c, 192, kernel_size=3,padding=1),
            nn.GroupNorm(16,192),nn.GELU(),
            nn.Conv2d(192,128,kernel_size=3,padding=1),
            nn.GroupNorm(16,128),nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.GroupNorm(16,128),nn.GELU(),
        )
        self.output = nn.Conv2d(128,self.num_joints,kernel_size=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images):
        if images.ndim!=4 or images.shape[1:]!=(3,INPUT_SIZE,INPUT_SIZE):
            raise ValueError("Expected NCHW RGB tensors of shape (B,3,672,672)")
        normalized=(images-self.mean.to(images))/self.std.to(images)
        if self.freeze_backbone:
            with torch.no_grad():
                tokens=self.backbone(pixel_values=normalized).last_hidden_state
        else:
            tokens=self.backbone(pixel_values=normalized).last_hidden_state
        b=tokens.shape[0]
        patch=tokens[:,1+self.registers:,:]
        if patch.shape[1]!=PATCH_GRID*PATCH_GRID:
            raise RuntimeError("Unexpected pretrained DINOv2 patch grid for 672 input")
        patch=patch.transpose(1,2).reshape(b,-1,PATCH_GRID,PATCH_GRID)
        head=self.neck(patch)
        head=self.refine(F.interpolate(head,size=(HEATMAP_SIZE,HEATMAP_SIZE),
                                        mode="bilinear",align_corners=False))
        return self.output(head)

def decode_hr(heatmaps):
    """Argmax plus bounded 3x3 peak subpixel correction, native 672 px."""
    if heatmaps.ndim!=4 or heatmaps.shape[2:]!=(HEATMAP_SIZE,HEATMAP_SIZE):
        raise ValueError("Expected 192x192 heatmaps")
    b,j,h,w=heatmaps.shape
    flat=heatmaps.reshape(b,j,-1)
    peak,index=flat.max(-1)
    yy=index//w
    xx=index%w
    def neighbor(dx,dy):
        x=(xx+dx).clamp(0,w-1)
        y=(yy+dy).clamp(0,h-1)
        return flat.gather(-1,(y*w+x).unsqueeze(-1)).squeeze(-1)
    left,right=neighbor(-1,0),neighbor(1,0)
    up,down=neighbor(0,-1),neighbor(0,1)
    def correction(a,b):
        denom=2*peak-a-b
        safe=torch.where(denom.abs()>1e-4,denom,torch.ones_like(denom))
        return ((b-a)/(2*safe)).clamp(-.5,.5)
    x=(xx.float()+correction(left,right)).clamp(0,w-1)
    y=(yy.float()+correction(up,down)).clamp(0,h-1)
    positions=torch.stack((x,y),-1)*(INPUT_SIZE/HEATMAP_SIZE)
    return positions,peak
