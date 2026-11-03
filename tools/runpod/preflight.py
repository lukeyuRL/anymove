"""Standard-library preflight for the two-view AnyMove training dataset.

This deliberately runs before installing CUDA packages or allocating training work.
"""
import argparse
import json
import math
import pathlib
import re
import struct
import sys

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
META_NAME = re.compile(r"^(\d+)_meta\.json$")


def png_dimensions(path):
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_MAGIC or header[12:16] != b"IHDR":
        raise ValueError("invalid PNG header: " + str(path))
    width, height = struct.unpack(">II", header[16:24])
    if not width or not height:
        raise ValueError("invalid PNG dimensions: " + str(path))
    return width, height


def check_dataset(dataset, rig_info, creature, min_poses=25):
    dataset, rig_info, creature = map(pathlib.Path, (dataset, rig_info, creature))
    if not dataset.is_dir():
        raise ValueError("dataset not found: " + str(dataset))
    if not rig_info.is_file() or not creature.is_file():
        raise ValueError("rig_info.json and creature.json are both required")
    rig = json.loads(rig_info.read_text(encoding="utf-8"))
    config = json.loads(creature.read_text(encoding="utf-8"))
    joint_count = len(rig["keypoints"])
    layout = config.get("layout", {})
    width = {"front": int(layout.get("W_FRONT", 420)), "side": int(layout.get("W_SIDE", 860))}
    height = 720
    metas = sorted(dataset.glob("*_meta.json"))
    if len(metas) < min_poses:
        raise ValueError("only %d poses; need at least %d" % (len(metas), min_poses))
    scales, profiles, indices = set(), set(), set()
    for meta_file in metas:
        match = META_NAME.fullmatch(meta_file.name)
        if not match:
            raise ValueError("unexpected metadata filename: " + meta_file.name)
        index = int(match.group(1))
        if index in indices:
            raise ValueError("duplicate pose index: %d" % index)
        indices.add(index)
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        scale = float(meta["scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid render scale in " + meta_file.name)
        scales.add(scale)
        profiles.add(str(meta.get("blur_profile", "legacy_unversioned")))
        stem = meta_file.name[: -len("_meta.json")]
        for view in ("front", "side"):
            keypoints = meta[view + "_uv"]
            if len(keypoints) != joint_count:
                raise ValueError("%s: %s has %d joints; expected %d" %
                                 (meta_file.name, view, len(keypoints), joint_count))
            image = dataset / (stem + "_" + view + ".png")
            if not image.is_file():
                raise ValueError("missing image: " + str(image))
            got = png_dimensions(image)
            expected = (round(width[view] * scale), round(height * scale))
            if got != expected:
                raise ValueError("%s: dimensions %r; expected %r" % (image, got, expected))
    if len(scales) != 1 or len(profiles) != 1:
        raise ValueError("mixed render scale or blur_profile; regenerate a consistent dataset")
    train_count = sum(index % 25 != 0 for index in indices)
    val_count = len(indices) - train_count
    if not train_count or not val_count:
        raise ValueError("dataset must contain both index-modulo-25 train and validation poses")
    return {
        "ready": True, "poses": len(indices), "training_poses": train_count,
        "validation_poses": val_count, "images": 2 * len(indices),
        "joints": joint_count, "scale": next(iter(scales)),
        "blur_profile": next(iter(profiles)), "dataset": str(dataset.resolve())
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--rig-info", default="assets/dragon/rigged/rig_info.json")
    p.add_argument("--creature", default="assets/dragon/creature.json")
    p.add_argument("--min-poses", type=int, default=25)
    a = p.parse_args(argv)
    if a.min_poses < 2:
        p.error("--min-poses must be >= 2")
    try:
        print(json.dumps(check_dataset(a.dataset, a.rig_info, a.creature, a.min_poses),
                         indent=2, sort_keys=True))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print("DATASET PREFLIGHT FAILED: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
