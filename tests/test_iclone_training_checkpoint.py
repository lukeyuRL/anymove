"""CPU-only regression checks for optional epoch checkpoint/resume."""
import math
import pathlib
import tempfile
import unittest
from unittest import mock

import importlib.util

if importlib.util.find_spec("torch") is not None:
    import torch
    from mocap.training_checkpoint import (
        EpochCheckpoints, _check_config, atomic_torch_save, metadata_fingerprint,
    )
else:
    torch = None


@unittest.skipUnless(torch is not None, "CPU PyTorch required; covered by checkpoint-cpu CI job")
class TrainingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.weights = self.root / "reader_hr_v1.pt"
        self.storage = self.root / "persistent"
        self.config = {
            "dataset": "/workspace/anymove/runs/dino/dataset_hr_v1",
            "names": ["head", "tail"], "epochs": 20, "batch": 4,
            "grad_accum": 1, "input_size": 672, "heatmap_size": 192,
            "freeze_backbone": False, "gradient_checkpointing": False,
            "train_quality": False, "amp_dtype": "torch.bfloat16",
            "metadata_sha256": "ab" * 32,
        }
        self.model = torch.nn.Linear(2, 2)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)

    def snapshot(self, val):
        atomic_torch_save({
            "state_dict": self.model.state_dict(), "val": val,
        }, self.weights)

    def train_step(self):
        self.opt.zero_grad()
        loss = self.model(torch.ones(1, 2)).sum()
        loss.backward()
        self.opt.step()

    def test_save_restore_optimizer_step_and_best_model(self):
        self.train_step()
        self.snapshot({"front": 10, "side": 14})
        checkpoint = EpochCheckpoints(self.storage, self.weights)
        checkpoint.save_epoch(0, 12, self.model, self.opt, self.scaler,
                              self.config, {"front": 10, "side": 14})
        self.assertTrue((self.storage / "epoch_001_full.pt").is_file())
        self.assertTrue((self.storage / "reader_hr_v1_best_epoch_001.pt").is_file())
        self.assertTrue((self.storage / "epoch_001_full.pt.complete.json").is_file())
        for p in self.model.parameters():
            p.data.zero_()
        new_opt = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        new_scaler = torch.amp.GradScaler("cpu", enabled=False)
        restored = EpochCheckpoints(self.storage, self.weights)
        self.assertEqual(restored.restore(self.model, new_opt, new_scaler, self.config), (1, 12))
        self.assertAlmostEqual(restored.best_score, 12)
        self.assertTrue(any(torch.any(p != 0).item() for p in self.model.parameters()))
        self.assertTrue(new_opt.state)

    def test_retains_only_last_three_and_best(self):
        checkpointer = EpochCheckpoints(self.storage, self.weights, keep_last=3)
        for ep in range(5):
            val = {"front": 8 + ep, "side": 12 + ep}
            self.snapshot(val)
            checkpointer.save_epoch(ep, ep+1, self.model, self.opt, self.scaler, self.config, val)
        self.assertEqual(sorted(p.name for p in self.storage.glob("epoch_*_full.pt")),
                         ["epoch_003_full.pt", "epoch_004_full.pt", "epoch_005_full.pt"])
        best = torch.load(self.storage / "reader_hr_v1_best_epoch_001.pt", weights_only=True)
        self.assertEqual(best["val"]["front"], 8)

    def test_corrupt_or_missing_marker_falls_back_to_previous(self):
        self.train_step()
        cp = EpochCheckpoints(self.storage, self.weights)
        for ep in (0, 1):
            self.snapshot({"front": 10, "side": 20})
            cp.save_epoch(ep, ep+1, self.model, self.opt, self.scaler,
                          self.config, {"front": 10, "side": 20})
        (self.storage / "epoch_002_full.pt.complete.json").write_text("bad json")
        self.assertEqual(EpochCheckpoints(self.storage, self.weights).restore(
            self.model, self.opt, self.scaler, self.config), (1, 1))

    def test_corrupt_newest_falls_back_to_previous(self):
        self.train_step()
        cp = EpochCheckpoints(self.storage, self.weights)
        for ep in (0, 1):
            self.snapshot({"front": 10 - ep, "side": 20 - ep})
            cp.save_epoch(ep, 30 * (ep+1), self.model, self.opt, self.scaler,
                          self.config, {"front": 10 - ep, "side": 20 - ep})
        (self.storage / "epoch_002_full.pt").write_bytes(b"broken")
        self.assertEqual(EpochCheckpoints(self.storage, self.weights).restore(
            self.model, self.opt, self.scaler, self.config), (1, 30))

    def test_incompatible_config_fails_instead_of_silent_restart(self):
        self.snapshot({"front": 4, "side": 5})
        cp = EpochCheckpoints(self.storage, self.weights)
        cp.save_epoch(0, 1, self.model, self.opt, self.scaler,
                      self.config, {"front": 4, "side": 5})
        bad = dict(self.config, names=["wrong"])
        with self.assertRaisesRegex(ValueError, "names"):
            cp.restore(self.model, self.opt, self.scaler, bad)

    def test_original_runpod_checkpoint_backward_compatible(self):
        # The temporary Runpod script used these keys and no format_version.
        legacy = {
            "model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(), "epoch": 2, "step": 55,
            "validation": {"front": 15.0, "side": 20.0},
            "config": {k: v for k, v in self.config.items() if k != "metadata_sha256"},
        }
        self.storage.mkdir()
        torch.save(legacy, self.storage / "epoch_003_full.pt")
        moved = dict(self.config, dataset="/other/pod/runs/dino/dataset_hr_v1")
        self.assertEqual(EpochCheckpoints(self.storage, self.weights).restore(
            self.model, self.opt, self.scaler, moved), (3, 55))

    def test_atomic_save_preserves_prior_file_if_replace_fails(self):
        atomic_torch_save({"value": 1}, self.weights)
        with mock.patch("mocap.training_checkpoint.os.replace", side_effect=OSError("volume unavailable")):
            with self.assertRaises(OSError):
                atomic_torch_save({"value": 2}, self.weights)
        self.assertEqual(torch.load(self.weights, weights_only=True)["value"], 1)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_completed_remote_backup_avoids_atomic_rename(self):
        # Global Volume has no POSIX atomic rename; local save uses one.
        self.snapshot({"front": 1.0, "side": 2.0})
        cp = EpochCheckpoints(self.storage, self.weights)
        with mock.patch("mocap.training_checkpoint.os.replace", side_effect=AssertionError("rename on global")):
            cp.save_epoch(0, 1, self.model, self.opt, self.scaler,
                          self.config, {"front": 1.0, "side": 2.0})
        self.assertTrue((self.storage / "epoch_001_full.pt.complete.json").exists())

    def test_partial_checkpoint_without_marker_falls_back(self):
        self.snapshot({"front": 8.0, "side": 10.0})
        cp = EpochCheckpoints(self.storage, self.weights)
        cp.save_epoch(0, 1, self.model, self.opt, self.scaler,
                      self.config, {"front": 8.0, "side": 10.0})
        (self.storage / "epoch_002_full.pt").write_bytes(b"partially copied checkpoint")
        self.assertEqual(EpochCheckpoints(self.storage, self.weights).restore(
            self.model, self.opt, self.scaler, self.config), (1, 1))

    def test_fingerprint_changes_with_annotations(self):
        a = self.root / "00000_meta.json"
        a.write_text('{"scale":1.0}')
        first = metadata_fingerprint([a])
        a.write_text('{"scale":0.6}')
        self.assertNotEqual(first, metadata_fingerprint([a]))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            _check_config(dict(self.config, metadata_sha256=first), self.config)

    def test_disabled_checkpointing_does_not_write(self):
        cp = EpochCheckpoints(None, self.weights)
        self.assertEqual(cp.restore(self.model, self.opt, self.scaler, self.config), (0, 0))
        cp.save_epoch(0, 0, self.model, self.opt, self.scaler, self.config,
                      {"front": 10, "side": 20})
        self.assertFalse(self.storage.exists())


if __name__ == "__main__":
    unittest.main()
