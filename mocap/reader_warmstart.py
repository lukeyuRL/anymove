"""Validate an existing creature Reader before warm-starting quality training.

No implicit architecture, skeleton or quality-head migration is permitted.
Existing inference Reader weights remain immutable; the new output is separate.
"""
import hashlib
from pathlib import Path

from mocap.reader_config import spec_from_checkpoint


def validate_base_reader(checkpoint, names, spec):
    """Return a regular Reader state_dict only when its contract matches."""
    if not isinstance(checkpoint, dict):
        raise ValueError("initial Reader checkpoint must be a mapping")
    if len(names) != len(set(names)) or not names:
        raise ValueError("initial Reader requires unique nonempty joint names")
    saved_names = checkpoint.get("names")
    if not isinstance(saved_names, (tuple, list)) or list(saved_names) != list(names):
        raise ValueError("initial Reader joint names/order mismatch the Dino rig")
    if spec_from_checkpoint(checkpoint) != spec:
        raise ValueError("initial Reader resolution / heatmap does not match training")
    if checkpoint.get("quality_spec") is not None or (checkpoint.get("reader_spec") or {}).get("quality_head", False):
        raise ValueError("initial Reader must NOT contain a quality head")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state or any(
            not isinstance(k, str) for k in state):
        raise ValueError("initial Reader state_dict is missing or malformed")
    if any(k.startswith("quality_out.") for k in state):
        raise ValueError("initial Reader already contains quality-head weights")
    return state


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()
