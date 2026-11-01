"""Verify the LOCAL Dino static dataset against its exact rig and camera framing.

No CUDA, Blender runtime or image decoder required.
python -m tools.audit_dino_dataset \
  --dataset runs/dino/dataset_hr_v1 --rig assets/dino/rigged/rig_info.json \
  --creature assets/dino/creature.json --min-poses 25

Checks a render-signature hash of the entire rig, joint labels, camera layout
and yaw policy. A correct image count alone is NOT evidence of rig identity.
Does not assert actual mesh occlusion or temporal motion ground truth.
"""
import argparse
import json
import math
import pathlib
import re
import struct

from blender.dataset_blur import BLUR_PROFILE
from blender.dataset_framing import framing_signature

PNG_HEADER = b"\x89PNG\r\n\x1a\n"
POSE_META = re.compile(r"^(\d+)_meta\.json$")


def png_size(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != PNG_HEADER or header[12:16] != b"IHDR":
        raise ValueError("missing/invalid PNG signature or IHDR: " + str(path))
    w, h = struct.unpack(">II", header[16:24])
    if min(w, h) <= 0:
        raise ValueError("nonpositive PNG dimensions: " + str(path))
    return int(w), int(h)


def _finite_joints(value, n_joints, label):
    if not isinstance(value, list) or len(value) != n_joints:
        raise ValueError(label + " joint count differs from rig")
    if any(not isinstance(row, list) or len(row) != 2 or
           any(type(coord) not in (int, float) or not math.isfinite(coord)
               for coord in row) for row in value):
        raise ValueError(label + " has malformed/nonfinite UV labels")


def audit_dataset(dataset, rig_file, creature_file, min_poses=25):
    dataset = pathlib.Path(dataset)
    if not dataset.is_dir():
        raise ValueError("Dino dataset directory missing: " + str(dataset))
    rig = json.loads(pathlib.Path(rig_file).read_text(encoding="utf-8"))
    creature = json.loads(pathlib.Path(creature_file).read_text(encoding="utf-8"))
    joints = rig["keypoints"]
    bone_names = [b["name"] for b in rig["bones"]]
    names = [j["name"] for j in joints]
    if (not joints or not bone_names or len(names) != len(set(names)) or
            len(bone_names) != len(set(bone_names))):
        raise ValueError("Dino rig lacks unique bones/keypoint names")
    override = creature.get("layout", {})
    h = 720
    f = int(override.get("W_FRONT", 420))
    s = int(override.get("W_SIDE", 860))
    metres = float(override.get("M_PER_PX", 1.9/h))
    ground = int(override.get("GROUND_V", 630))
    if min(f, s, ground) <= 0 or metres <= 0 or not math.isfinite(metres):
        raise ValueError("invalid creature camera layout")
    lay = {"H": h, "W_FRONT": f, "W_SIDE": s, "M_PER_PX": metres,
           "GROUND_V": ground, "Z_CENTER": (ground-h/2)*metres}
    signature = framing_signature(
        rig, lay, creature.get("dataset", {}).get("yaw_max_deg"))
    files = sorted(dataset.glob("*_meta.json"))
    if len(files) < min_poses:
        raise ValueError(f"insufficient poses: {len(files)}; required {min_poses}")
    indices = set()
    scales = set()
    profiles = set()
    for path in files:
        match = POSE_META.fullmatch(path.name)
        if match is None:
            raise ValueError("unexpected pose filename " + path.name)
        index = int(match.group(1))
        if index in indices:
            raise ValueError("duplicate numeric pose index " + str(index))
        indices.add(index)
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("framing_signature") != signature:
            raise ValueError(path.name + ": stale/mismatched rig, keypoints or camera signature")
        scale = float(meta["scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(path.name + ": invalid scale")
        scales.add(scale)
        profiles.add(meta.get("blur_profile"))
        if meta.get("blur_profile") != BLUR_PROFILE:
            raise ValueError(path.name + ": unsupported/stale blur policy")
        if set(meta.get("pose", {})) != set(bone_names):
            raise ValueError(path.name + ": pose bone names differ from rig")
        for view, width in (("front", f), ("side", s)):
            _finite_joints(meta[view+"_uv"], len(joints), path.name + " "+view)
            image = dataset / (match.group(1) + "_" + view + ".png")
            if not image.is_file():
                raise ValueError("missing rendered view: " + str(image))
            expected = (round(width*scale), round(h*scale))
            if png_size(image) != expected:
                raise ValueError(f"{image}: image dimension does not match {expected}")
    if len(scales) != 1 or len(profiles) != 1:
        raise ValueError("mixed render scale/blur policy in Dino dataset")
    val = sum(i % 25 == 0 for i in indices)
    if val == 0 or val == len(indices):
        raise ValueError("Dino dataset has no held-out validation/train split")
    return {
        "status": "static_reader_dataset_preflight_pass",
        "creature": creature.get("name"),
        "poses": len(files), "train_poses": len(files)-val,
        "val_poses": val, "images": len(files)*2,
        "keypoints": len(names), "bones": len(bone_names),
        "render_scale": next(iter(scales)),
        "blur_profile": next(iter(profiles)),
        "framing_signature": signature,
        "note": "valid still poses; does not establish a continuous motion corpus",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="runs/dino/dataset_hr_v1")
    p.add_argument("--rig", default="assets/dino/rigged/rig_info.json")
    p.add_argument("--creature", default="assets/dino/creature.json")
    p.add_argument("--min-poses", type=int, default=25)
    args = p.parse_args(argv)
    if args.min_poses < 2:
        p.error("--min-poses must be >= 2")
    try:
        report = audit_dataset(args.dataset, args.rig, args.creature, args.min_poses)
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        p.exit(2, "DINO DATASET AUDIT FAILED: " + str(exc) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()
