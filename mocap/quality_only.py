"""Freeze a warm-started AnyMove Reader while learning only P2 quality maps.

This prevents P2 experiments from silently degrading an already validated
Dino 672/192 keypoint detector. The original weights remain identical in the
new quality checkpoint; only quality_out is learnable.
"""


def configure_quality_only(model):
    if not getattr(model, "quality_head", False) or not hasattr(model, "quality_out"):
        raise ValueError("quality-only training requires a model with quality_head=True")
    model.requires_grad_(False)
    model.quality_out.requires_grad_(True)
    trainable = [name for name, param in model.named_parameters()
                 if param.requires_grad]
    if (not trainable or
            any(not name.startswith("quality_out.") for name in trainable)):
        raise RuntimeError("quality-only mode must train only quality_out parameters")
    return trainable
