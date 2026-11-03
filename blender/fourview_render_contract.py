"""Blender-independent contracts for the Dino SkinMax41 four-view renderer.

This module deliberately does not import bpy; it can be tested by ordinary
Python/CI and helps prevent mixing incompatible partial dataset renders.
"""
import hashlib
import json
import math
import pathlib
import re
import struct

VIEWS = ("front", "side", "back", "opposite_side")
PNG_HEADER = b"\x89PNG\r\n\x1a\n"
POSE_NAME = re.compile(r"^(\d+)_meta\.json$")


def eevee_identifier(blender_version):
    """Blender 5.0 renamed BLENDER_EEVEE_NEXT to BLENDER_EEVEE."""
    if tuple(blender_version[:2]) >= (5, 0):
        return "BLENDER_EEVEE"
    if tuple(blender_version[:2]) >= (4, 2):
        return "BLENDER_EEVEE_NEXT"
    raise ValueError("Four-view Eevee renderer needs Blender 4.2+ (5.0 recommended)")


def engine_identifier(engine, blender_version):
    if engine == "cycles":
        return "CYCLES"
    if engine == "eevee":
        return eevee_identifier(blender_version)
    raise ValueError("render-engine must be 'eevee' or 'cycles'")


def sha256_file(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def contract_sha256(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def png_dimensions(path):
    with pathlib.Path(path).open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != PNG_HEADER or header[12:16] != b"IHDR":
        raise ValueError("Invalid PNG: " + str(path))
    return tuple(int(x) for x in struct.unpack(">II", header[16:24]))


def pose_indices(source_dataset, requested="all"):
    src = pathlib.Path(source_dataset)
    if not src.is_dir():
        raise ValueError("Source dataset does not exist: " + str(src))
    available = {}
    for path in src.glob("*_meta.json"):
        match = POSE_NAME.fullmatch(path.name)
        if match is not None:
            ix = int(match.group(1))
            if ix in available:
                raise ValueError(f"Duplicate numeric pose ID {ix}")
            available[ix] = path
    if not available:
        raise ValueError("No source *_meta.json pose files found")
    if requested.strip().lower() == "all":
        ids = sorted(available)
    else:
        ids = []
        for part in requested.split(","):
            part = part.strip()
            if not part.isdecimal():
                raise ValueError("Invalid pose ID: " + repr(part))
            idx = int(part)
            if idx not in available:
                raise ValueError(f"Requested pose {idx} has no source metadata")
            if idx in ids:
                raise ValueError("Duplicate requested pose " + str(idx))
            ids.append(idx)
    return [(i, available[i]) for i in ids]


def validate_pose(meta, bone_names, num_joints):
    pose = meta.get("pose")
    if not isinstance(pose, dict) or set(pose) != set(bone_names):
        raise ValueError("Source pose does not match the exact target rig bones")
    for name, angles in pose.items():
        if (not isinstance(angles, list) or len(angles) != 3 or
                not all(isinstance(x, (int, float)) and math.isfinite(x)
                        for x in angles)):
            raise ValueError(f"Invalid source Euler rotation for {name}")
    root = meta.get("root")
    if (not isinstance(root, list) or len(root) != 3 or
            not all(isinstance(x, (int, float)) and math.isfinite(x) for x in root)):
        raise ValueError("Source root translation must be finite xyz")
    if not isinstance(meta.get("yaw"), (int, float)) or not math.isfinite(meta["yaw"]):
        raise ValueError("Source yaw must be finite")
    joints = meta.get("joints_world")
    if not isinstance(joints, list) or len(joints) != num_joints:
        raise ValueError("Source 3D joint count mismatch")
    for xyz in joints:
        if (not isinstance(xyz, list) or len(xyz) != 3 or
                not all(isinstance(x, (int, float)) and math.isfinite(x) for x in xyz)):
            raise ValueError("Source 3D ground truth must contain finite xyz")
    return True


def sample_file(folder, pose_id, view=None, suffix="png"):
    if view is not None and view not in VIEWS:
        raise ValueError("Unsupported view: " + str(view))
    stem = f"{pose_id:05d}"
    name = stem + ("_" + view if view else "")
    return pathlib.Path(folder) / (name + "." + suffix)


def sample_is_complete(folder, pose_id, source_sha, render_contract_sha, image_sizes):
    """Only skip a pose when EVERY expected artifact and hash verifies."""
    folder = pathlib.Path(folder)
    report = folder / f"{pose_id:05d}_visibility_meta.json"
    if not report.is_file():
        return False
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("source_pose_sha256") != source_sha:
        raise ValueError("Existing pose source hash changed; use a NEW dataset folder")
    if data.get("render_contract_sha256") != render_contract_sha:
        raise ValueError("Existing pose render contract changed; use a NEW dataset folder")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    for view in VIEWS:
        for suffix in ("png", "npz"):
            filename = f"{pose_id:05d}_{view}" + (
                "_labels.npz" if suffix == "npz" else ".png"
            )
            path = folder / filename
            if not path.is_file() or artifacts.get(filename) != sha256_file(path):
                return False
            if suffix == "png" and png_dimensions(path) != tuple(image_sizes[view]):
                return False
    return True
