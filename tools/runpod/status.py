"""Summarize a Runpod training run without importing torch.

python tools/runpod/status.py /workspace/anymove-results/<run-id>
"""
import argparse
import json
import pathlib
import re
import sys

STEP = re.compile(
    r"ep (?P<epoch>\d+) optimizer_step (?P<step>\d+) "
    r"loss (?P<loss>[0-9.eE+-]+) lr_factor (?P<lr>[0-9.eE+-]+) "
    r"peak_vram_allocated (?P<vram>[0-9.]+)GiB elapsed (?P<elapsed>\d+)s"
)
VAL = re.compile(
    r"epoch (?P<epoch>\d+) val mean px error \(layout pixels\) "
    r"front (?P<front>[0-9.]+) side (?P<side>[0-9.]+)"
)


def summarize(run_dir):
    run_dir = pathlib.Path(run_dir)
    status_path = run_dir / "status.json"
    log_path = run_dir / "train.log"
    if not status_path.exists():
        raise ValueError("missing status.json: " + str(status_path))
    report = json.loads(status_path.read_text(encoding="utf-8"))
    result = dict(report)
    result["log_exists"] = log_path.is_file()
    result["checkpoint_exists"] = (run_dir / "reader.pt").is_file()
    if log_path.is_file():
        steps, vals = [], []
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                step = STEP.search(line)
                if step:
                    d = step.groupdict()
                    steps.append({
                        "epoch": int(d["epoch"]), "optimizer_step": int(d["step"]),
                        "loss": float(d["loss"]), "lr_factor": float(d["lr"]),
                        "peak_vram_gib": float(d["vram"]), "elapsed_seconds": int(d["elapsed"])
                    })
                val = VAL.search(line)
                if val:
                    d = val.groupdict()
                    vals.append({
                        "epoch": int(d["epoch"]), "front_px": float(d["front"]),
                        "side_px": float(d["side"])
                    })
        result["latest_step"] = steps[-1] if steps else None
        result["latest_validation"] = vals[-1] if vals else None
        result["validation_epochs"] = len(vals)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    args = p.parse_args(argv)
    try:
        print(json.dumps(summarize(args.run_dir), indent=2, sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("STATUS ERROR: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
