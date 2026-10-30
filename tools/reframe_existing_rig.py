"""Recalculate Front/Side camera framing without regenerating or editing a rig.

Usage from repo root:
    py -3.14 tools/reframe_existing_rig.py assets/dino

Reads the existing rig_import_report.json bounding box and updates only the
two JSON camera layouts plus a capped Dataset yaw policy when needed.
A timestamped backup of all 3 files is completed before any edits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

# Standalone execution ('python tools/reframe_existing_rig.py') and package
# imports in unit tests must both find tools.import_existing_rig.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.import_existing_rig import camera_layout, atomic_json


def _load(path):
    if not path.is_file():
        raise FileNotFoundError("Required metadata not found: " + str(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Metadata root must be an object: " + str(path))
    return obj


def reframe(asset_dir):
    root = Path(asset_dir).expanduser().resolve()
    cpath = root / "creature.json"
    ipath = root / "rigged" / "rig_info.json"
    rpath = root / "rigged" / "rig_import_report.json"
    paths = [cpath, ipath, rpath]
    creature, info, report = [_load(p) for p in paths]

    minimum, maximum = report["bounds_min"], report["bounds_max"]
    if (len(minimum) != 3 or len(maximum) != 3
            or any(a > b for a, b in zip(minimum, maximum))):
        raise ValueError("Invalid Mesh bounds in rig_import_report.json")
    layout = camera_layout([minimum, maximum])
    camera = {k: layout[k] for k in
              ("W_FRONT", "W_SIDE", "M_PER_PX", "GROUND_V")}
    dataset = creature.get("dataset")
    if dataset is None:
        dataset = {}
    if not isinstance(dataset, dict):
        raise ValueError("creature.json dataset must be an object")
    dataset = dict(dataset)
    if layout["W_FRONT"] != 640:
        dataset["yaw_max_deg"] = 10.0
        dataset["camera_policy"] = "elongated_creature_v1"

    if (creature.get("layout") == camera and info.get("layout") == layout
            and creature.get("dataset", {}) == dataset):
        return {"status": "unchanged", "layout": camera}

    creature["layout"] = camera
    if dataset:
        creature["dataset"] = dataset
    info["layout"] = layout
    report["framing"] = {
        "front_width_px": layout["W_FRONT"],
        "side_width_px": layout["W_SIDE"],
        "estimated_rest_height_px": (
            maximum[2] - minimum[2]) / layout["M_PER_PX"],
        "dataset_yaw_max_deg": dataset.get("yaw_max_deg"),
        "source": "existing_rig_import_report_bounds",
    }
    report["framing_updated"] = True
    if isinstance(report.get("mesh_names"), list):
        # The renderer now resolves single-Mesh Armatures by modifier,
        # independent of original FBX Object names.
        report["current_dataset_loader_compatible"] = (
            len(report["mesh_names"]) == 1)
    report.setdefault("warnings", [])
    if not isinstance(report["warnings"], list):
        raise ValueError("rig_import_report.json warnings must be a list")
    report["warnings"] = [
        text for text in report["warnings"]
        if not text.startswith("Existing render_dataset.py needs ONE Mesh 'body'")
    ]
    warning = ("Camera layout changed: previous Dataset images, reference "
               "frames, detections and trained Reader are not compatible. "
               "Use a NEW dataset folder and regenerate all dependent artifacts.")
    if warning not in report["warnings"]:
        report["warnings"].append(warning)

    # Make backups before touching anything; do not remove earlier backups.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backups = [p.with_name(p.name + ".before_reframe_" + stamp) for p in paths]
    for path, backup in zip(paths, backups):
        shutil.copy2(path, backup)
    for path, data in zip(paths, (creature, info, report)):
        atomic_json(path, data)
    return {
        "status": "framing_updated",
        "creature": root.name,
        "layout": camera,
        "yaw_max_deg": dataset.get("yaw_max_deg"),
        "backups": [str(p) for p in backups],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Update camera framing only; preserve rigged.blend and existing Skin Weights")
    parser.add_argument("asset_dir", help="Existing assets/dino folder")
    args = parser.parse_args(argv)
    result = reframe(args.asset_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
