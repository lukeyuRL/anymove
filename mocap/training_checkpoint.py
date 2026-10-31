"""Optional epoch-complete checkpoint/resume for the AnyMove joint reader.

MOCAP_RESUME_DIR points to persistent storage. Runpod Global Volumes are
object-backed and do NOT support atomic rename, so persistent checkpoints
are immutable epoch-named objects with a completion sidecar. The model-only
checkpoint stays on a normal local filesystem. Legacy RUNPOD_DINO_RESUME_V1
checkpoints (without sidecars) remain loadable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import uuid

import torch


def metadata_fingerprint(metas):
    """Fingerprint annotation JSON and render settings, not image pixels."""
    digest = hashlib.sha256()
    for path in sorted(map(pathlib.Path, metas)):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_torch_save(payload, destination):
    """Save model weights on a local/POSIX disk and publish via os.replace."""
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, stage)
        os.replace(stage, destination)
    finally:
        stage.unlink(missing_ok=True)


def _sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_path(path):
    return pathlib.Path(str(path) + ".complete.json")


def _publish_object(source, target):
    """Publish an immutable object then a completion marker; no remote rename."""
    source, target = pathlib.Path(source), pathlib.Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = _completion_path(target)
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite completed checkpoint: {target}")
    shutil.copyfile(source, target)
    size = source.stat().st_size
    if target.stat().st_size != size:
        raise OSError(f"Incomplete persistent checkpoint copy: {target}")
    marker.write_text(json.dumps({
        "version": 1,
        "size_bytes": size,
        "sha256": _sha256_file(source),
    }, sort_keys=True), encoding="utf-8")


def _same_dataset(a, b):
    """Permit the same relative dataset to be mounted under another pod root."""
    if a == b:
        return True
    a_parts, b_parts = pathlib.Path(str(a)).parts, pathlib.Path(str(b)).parts
    return len(a_parts) >= 3 and len(b_parts) >= 3 and a_parts[-3:] == b_parts[-3:]


def _check_config(saved, current):
    if not isinstance(saved, dict):
        raise ValueError("Resume checkpoint has no training config")
    required = (
        "dataset", "names", "epochs", "batch", "grad_accum",
        "input_size", "heatmap_size", "freeze_backbone",
        "gradient_checkpointing", "train_quality", "amp_dtype",
    )
    for name in required:
        if name not in saved or name not in current:
            raise ValueError(f"Resume configuration missing {name}")
        matches = (_same_dataset(saved[name], current[name])
                   if name == "dataset" else saved[name] == current[name])
        if not matches:
            raise ValueError(f"Resume configuration mismatch: {name}")
    # The temporary Runpod implementation lacks this newer field.
    if ("metadata_sha256" in saved and
            saved["metadata_sha256"] != current.get("metadata_sha256")):
        raise ValueError("Resume dataset metadata fingerprint mismatch")


class EpochCheckpoints:
    """Full training-state backups. Disabled if directory is empty/unset."""

    def __init__(self, directory, model_path, keep_last=3):
        self.directory = pathlib.Path(directory) if directory else None
        self.model_path = pathlib.Path(model_path)
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        self.keep_last = keep_last
        self.best_score = math.inf
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _epochs(self):
        if not self.directory:
            return []
        found = []
        for path in self.directory.glob("epoch_*_full.pt"):
            match = re.fullmatch(r"epoch_(\d+)_full\.pt", path.name)
            if match:
                found.append((int(match.group(1)), path))
        return sorted(found, reverse=True)

    def restore(self, model, optimizer, scaler, config):
        """Return (next_epoch, optimizer_step), newest valid checkpoint first."""
        for number, source in self._epochs():
            local = self.model_path.with_name(f".resume-{uuid.uuid4().hex}.pt")
            try:
                marker = _completion_path(source)
                info = json.loads(marker.read_text()) if marker.exists() else None
                shutil.copyfile(source, local)
                if info is not None:
                    if local.stat().st_size != info["size_bytes"]:
                        raise ValueError("Persistent checkpoint size mismatch")
                    if _sha256_file(local) != info["sha256"]:
                        raise ValueError("Persistent checkpoint SHA256 mismatch")
                state = torch.load(local, map_location="cpu", weights_only=True)
                if not isinstance(state, dict) or not all(
                        key in state for key in ("model", "optimizer", "scaler", "config", "epoch", "step")):
                    raise ValueError("Missing required checkpoint fields")
                if type(state["epoch"]) is not int or state["epoch"] + 1 != number:
                    raise ValueError("Checkpoint epoch and filename differ")
                if type(state["step"]) is not int or state["step"] < 0:
                    raise ValueError("Invalid optimizer step")
            except Exception as exc:
                print(f"Skipping invalid resume checkpoint {source}: {exc}", flush=True)
                continue
            finally:
                local.unlink(missing_ok=True)

            # A valid but incompatible checkpoint MUST fail, never restart silently.
            _check_config(state["config"], config)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            if number > config["epochs"]:
                raise ValueError("Checkpoint exceeds requested epochs")
            best = state.get("best_validation")
            if best is not None and math.isfinite(float(best)):
                self.best_score = float(best)
            print(f"RESUMED: completed {number} epochs, optimizer step {state['step']}", flush=True)
            return number, state["step"]
        return 0, 0

    def save_epoch(self, epoch, step, model, optimizer, scaler, config, validation):
        """Save one immutable checkpoint; retain last N plus best model."""
        if not self.directory:
            return
        if not all(key in validation for key in ("front", "side")):
            raise ValueError("Missing validation metrics")
        values = [float(validation[key]) for key in ("front", "side")]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite validation metrics")
        score = sum(values) / 2
        better = score < self.best_score or not any(
            self.directory.glob(f"{self.model_path.stem}_best_epoch_*.pt"))
        new_best = score if better else self.best_score

        payload = {
            "format_version": 1,
            "epoch": int(epoch), "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": dict(config),
            "validation": dict(validation),
            "best_validation": new_best,
        }
        local = self.model_path.with_name(f".fullstate-{uuid.uuid4().hex}.pt")
        try:
            torch.save(payload, local)
            _publish_object(local, self.directory / f"epoch_{epoch + 1:03d}_full.pt")
        finally:
            local.unlink(missing_ok=True)

        if better:
            best_path = self.directory / (
                f"{self.model_path.stem}_best_epoch_{epoch + 1:03d}.pt")
            _publish_object(self.model_path, best_path)
            for stale in self.directory.glob(f"{self.model_path.stem}_best_epoch_*.pt"):
                if stale != best_path:
                    stale.unlink()
                    _completion_path(stale).unlink(missing_ok=True)
        self.best_score = new_best

        for _, stale in self._epochs()[self.keep_last:]:
            stale.unlink()
            _completion_path(stale).unlink(missing_ok=True)
        print(f"PERSISTENT CHECKPOINT SAVED: epoch {epoch + 1} (best {self.best_score:.2f}px)", flush=True)
