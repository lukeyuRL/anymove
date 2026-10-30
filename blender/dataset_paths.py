"""Deterministic paths for synthetic pose images and their JSON labels.

Blender's render path handling is not guaranteed to share pathlib's relative
base after loading a .blend, especially on Windows. Resolve before loading the
file, pass an absolute image path to the renderer, then verify it was written.

This module deliberately imports only stdlib so it can be regression tested
without installing Blender.
"""
from pathlib import Path


def resolve_render_inputs(blend, rig_info, output):
    """Call exactly once before bpy.ops.wm.open_mainfile changes context."""
    return (Path(blend).expanduser().resolve(),
            Path(rig_info).expanduser().resolve(),
            Path(output).expanduser().resolve())


def sample_file(out, index, kind):
    """Both PNGs and metadata live beside one another in an absolute root."""
    if kind not in ("front", "side", "meta"):
        raise ValueError(f"invalid dataset file kind: {kind!r}")
    if type(index) is not int or index < 0:
        raise ValueError("dataset sample index must be >= 0")
    root = Path(out)
    if not root.is_absolute():
        raise ValueError(f"dataset output root must be absolute: {root}")
    ext = "json" if kind == "meta" else "png"
    return root / f"{index:05d}_{kind}.{ext}"


def is_written(path):
    """Do not mistake an absent/zero-byte PNG for a completed sample."""
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def require_written(path):
    if not is_written(path):
        raise RuntimeError(
            f"Blender did not write expected dataset image: {path}. "
            "Check the Blender render log and Blender output path configuration."
        )
