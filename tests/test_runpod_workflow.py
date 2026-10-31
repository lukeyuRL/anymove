"""Offline tests for Runpod preflight and log status (no cloud account or GPU)."""
import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load("runpod_preflight", "tools/runpod/preflight.py")
status = load("runpod_status", "tools/runpod/status.py")


class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.rig = self.root / "rig_info.json"
        self.creature = self.root / "creature.json"
        self.rig.write_text(json.dumps({"keypoints": [{"name": "a"}, {"name": "b"}]}))
        self.creature.write_text(json.dumps({"layout": {"W_FRONT": 2, "W_SIDE": 3}}))
        for i in range(26):
            self.write_pose(i)

    def write_pose(self, i, scale=1.0, profile="v1"):
        stem = "%05d" % i
        (self.dataset / (stem + "_meta.json")).write_text(
            json.dumps({"scale": scale, "blur_profile": profile,
                        "front_uv": [[0, 0], [1, 1]],
                        "side_uv": [[0, 0], [1, 1]]}))
        for view, width in (("front", 2), ("side", 3)):
            path = self.dataset / (stem + "_" + view + ".png")
            path.write_bytes(
                preflight.PNG_MAGIC + struct.pack(">I", 13) +
                b"IHDR" + struct.pack(">II", round(width * scale), round(720 * scale))
            )

    def test_ready(self):
        result = preflight.check_dataset(self.dataset, self.rig, self.creature)
        self.assertEqual(result["poses"], 26)
        self.assertEqual(result["training_poses"], 24)
        self.assertEqual(result["validation_poses"], 2)
        self.assertEqual(result["images"], 52)

    def test_missing_image_rejected(self):
        (self.dataset / "00000_side.png").unlink()
        with self.assertRaisesRegex(ValueError, "missing image"):
            preflight.check_dataset(self.dataset, self.rig, self.creature)

    def test_mixed_scale_rejected(self):
        self.write_pose(1, scale=2.0)
        with self.assertRaisesRegex(ValueError, "mixed render scale"):
            preflight.check_dataset(self.dataset, self.rig, self.creature)

    def test_profile_mismatch_rejected(self):
        self.write_pose(1, profile="v2")
        with self.assertRaisesRegex(ValueError, "blur_profile"):
            preflight.check_dataset(self.dataset, self.rig, self.creature)


class StatusTest(unittest.TestCase):
    def test_latest_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            (folder / "status.json").write_text('{"state":"running","run_id":"demo"}')
            (folder / "train.log").write_text(
                "ep 0 optimizer_step 50 loss 0.5291 lr_factor 0.167 "
                "peak_vram_allocated 5.36GiB elapsed 16s\n"
                "epoch 0 val mean px error (layout pixels) front 3.21 side 4.56\n"
            )
            report = status.summarize(folder)
            self.assertEqual(report["latest_step"]["optimizer_step"], 50)
            self.assertEqual(report["latest_validation"]["side_px"], 4.56)
            self.assertFalse(report["checkpoint_exists"])


if __name__ == "__main__":
    unittest.main()
