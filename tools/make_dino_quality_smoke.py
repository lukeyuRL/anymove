"""Prepare a small P2 Reader smoke dataset from the verified Dino 2,200 poses.

Copies exact original RGBA and pose metadata, does NOT change labels or
pretend that these independently generated stills form motion sequences.

python -m tools.make_dino_quality_smoke \
  runs/dino/dataset_hr_v1 runs/dino/dataset_hr_v1_p2_smoke --poses 30
"""
import argparse
import json
import pathlib
import re
import shutil
import tempfile

POSE_META = re.compile(r"^(\d+)_meta\.json$")


def make_smoke(source, target, poses=30):
    src = pathlib.Path(source).resolve()
    dst = pathlib.Path(target).resolve()
    if not src.is_dir():
        raise ValueError("source Dino pose dataset does not exist")
    if dst == src or src in dst.parents:
        # A nested folder could be accidentally read as a sample source by
        # future tooling; keep smoke dataset a sibling and independent.
        raise ValueError("P2 smoke output must not be inside source dataset")
    if dst.exists():
        raise FileExistsError("P2 smoke destination exists; refuse overwrite: " + str(dst))
    if type(poses) is not int or poses < 26:
        raise ValueError("P2 smoke requires >=26 poses to exercise 0/25 held-out split")
    metas = sorted(src.glob("*_meta.json"))
    if len(metas) < poses:
        raise ValueError("source has fewer poses than requested")
    chosen = metas[:poses]
    index = []
    for path in chosen:
        matched = POSE_META.fullmatch(path.name)
        if not matched:
            raise ValueError("unsupported source pose name: " + path.name)
        index.append(int(matched.group(1)))
    if len(set(index)) != len(index):
        raise ValueError("duplicate numeric sample ids")
    nval = sum(i % 25 == 0 for i in index)
    if nval == 0 or nval == len(index):
        raise ValueError("chosen Dino smoke subset lacks train and held-out samples")
    sources = []
    for meta in chosen:
        stem = meta.name[:-len("_meta.json")]
        for suffix in ("meta.json", "front.png", "side.png"):
            path = src / (stem + "_" + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("missing/incomplete source sample: " + str(path))
            sources.append(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = pathlib.Path(tempfile.mkdtemp(prefix=".dino_p2_smoke_", dir=str(dst.parent)))
    try:
        for p in sources:
            shutil.copy2(p, stage / p.name)
        stage.rename(dst)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"poses": len(chosen), "train_poses": len(chosen)-nval,
            "val_poses": nval, "images": len(chosen)*2,
            "source": str(src), "output": str(dst),
            "note": "Copied unchanged static pose metadata/PNGs; NOT continuous motion."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dataset")
    parser.add_argument("output_dataset")
    parser.add_argument("--poses", type=int, default=30)
    args = parser.parse_args(argv)
    print(json.dumps(make_smoke(
        args.source_dataset, args.output_dataset, args.poses),
        indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
