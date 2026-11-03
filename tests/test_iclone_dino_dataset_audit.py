"""Dino pose preflight checks without Blender, image decoding or CUDA."""
import json
import pathlib
import struct
import tempfile
import unittest

from blender.dataset_framing import framing_signature
from blender.dataset_blur import BLUR_PROFILE
from tools.audit_dino_dataset import audit_dataset, PNG_HEADER


class DinoDatasetAuditTests(unittest.TestCase):
    def fixture(self, folder, count=30):
        root = pathlib.Path(folder)
        data = root/"dataset"
        data.mkdir()
        rig = root/"rig.json"
        creature = root/"creature.json"
        info = {
            "bones": [{"name": "root", "parent": -1}],
            "keypoints": [{"name": "head", "bone": 0, "end": "head"},
                          {"name": "tail", "bone": 0, "end": "tail"}],
        }
        config = {"name": "dino", "layout": {"W_FRONT": 640, "W_SIDE": 640,
                    "M_PER_PX": 0.004, "GROUND_V": 630},
                  "dataset": {"yaw_max_deg": 45}}
        rig.write_text(json.dumps(info))
        creature.write_text(json.dumps(config))
        lay = {"H": 720, "W_FRONT": 640, "W_SIDE": 640,
               "M_PER_PX": .004, "GROUND_V": 630,
               "Z_CENTER": (630-360)*.004}
        signature = framing_signature(info, lay, 45)
        for i in range(count):
            stem = f"{i:05d}"
            meta = {
                "framing_signature": signature, "scale": 1.,
                "blur_profile": BLUR_PROFILE,
                "pose": {"root": [0, 0, 0]},
                "front_uv": [[100, 100], [200, 100]],
                "side_uv": [[110, 110], [210, 110]],
            }
            (data/(stem+"_meta.json")).write_text(json.dumps(meta))
            for view in ("front", "side"):
                header = PNG_HEADER + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 640, 720)
                (data/(stem+"_"+view+".png")).write_bytes(header)
        return data, rig, creature

    def test_valid_dataset_and_correct_train_val_split(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset, rig, creature = self.fixture(folder)
            report = audit_dataset(dataset, rig, creature, 25)
            self.assertEqual(report["poses"], 30)
            self.assertEqual(report["val_poses"], 2)
            self.assertEqual(report["train_poses"], 28)
            self.assertEqual(report["keypoints"], 2)

    def test_rejects_changed_joint_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            data, rig, creature = self.fixture(folder)
            record = json.loads(rig.read_text())
            record["keypoints"][0]["name"] = "head_changed"
            rig.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "signature"):
                audit_dataset(data, rig, creature, 25)

    def test_rejects_missing_png_and_wrong_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            data, rig, creature = self.fixture(folder)
            image = data/"00000_front.png"
            image.unlink()
            with self.assertRaisesRegex(ValueError, "missing rendered"):
                audit_dataset(data, rig, creature, 25)
            image.write_bytes(PNG_HEADER + struct.pack(">I", 13) + b"IHDR" +
                              struct.pack(">II", 999, 720))
            with self.assertRaisesRegex(ValueError, "dimension"):
                audit_dataset(data, rig, creature, 25)

    def test_rejects_static_pose_renaming_as_new_rig(self):
        with tempfile.TemporaryDirectory() as folder:
            data, rig, creature = self.fixture(folder)
            config = json.loads(creature.read_text())
            config["dataset"]["yaw_max_deg"] = 30
            creature.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "signature"):
                audit_dataset(data, rig, creature, 25)


if __name__ == "__main__":
    unittest.main()
