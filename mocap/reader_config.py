"""Reader resolution and checkpoint contract, independent of torch/transformers.

Legacy reader.pt files contain no reader_spec and must always load as 448/128.
High-resolution training checkpoints store the image and heatmap sizes explicitly.
"""
from dataclasses import dataclass

PATCH_SIZE = 14
LEGACY_INPUT_SIZE = 448
LEGACY_HEATMAP_SIZE = 128
HIGH_RES_INPUT_SIZE = 672
HIGH_RES_HEATMAP_SIZE = 192
READER_SPEC_VERSION = 1


@dataclass(frozen=True)
class ReaderSpec:
    input_size: int = LEGACY_INPUT_SIZE
    heatmap_size: int = LEGACY_HEATMAP_SIZE

    def __post_init__(self):
        if (type(self.input_size) is not int or self.input_size < PATCH_SIZE
                or self.input_size % PATCH_SIZE):
            raise ValueError("reader input_size must be a positive multiple of 14")
        if (type(self.heatmap_size) is not int or self.heatmap_size < 3):
            raise ValueError("reader heatmap_size must be an integer >= 3")

    def checkpoint_metadata(self):
        return {"version": READER_SPEC_VERSION,
                "input_size": self.input_size,
                "heatmap_size": self.heatmap_size}


def spec_from_checkpoint(checkpoint):
    """Use saved resolution; only metadata-free legacy checkpoints use 448/128."""
    if not isinstance(checkpoint, dict):
        raise ValueError("reader checkpoint must be a dictionary")
    raw = checkpoint.get("reader_spec")
    if raw is None:
        return ReaderSpec()  # original reader.pt: never infer 672 from file name
    if not isinstance(raw, dict) or raw.get("version") != READER_SPEC_VERSION:
        raise ValueError("unsupported reader_spec in checkpoint")
    return ReaderSpec(raw.get("input_size"), raw.get("heatmap_size"))
