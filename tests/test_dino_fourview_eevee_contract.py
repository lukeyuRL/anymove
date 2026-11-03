"""Pure Python regression tests for Dino SkinMax41 four-view Eevee contracts.

No Blender, CUDA, model weights, or rendered dataset required.
python -m unittest discover -s tests -p test_dino_fourview_eevee_contract.py
"""
import json
import pathlib
import struct
import tempfile
import unittest

from blender.fourview_render_contract import (
    VIEWS, contract_sha256, eevee_identifier, engine_identifier,
    png_dimensions, pose_indices, sample_is_complete, sha256_file,
    validate_pose,
)

PNG = b"\x89PNG\r\n\x1a\n"


class FourViewEeveeContractTests(unittest.TestCase):
    def test_blender_five_engine_name(self):
        self.assertEqual(eevee_identifier((5, 0, 0)), "BLENDER_EEVEE")
        self.assertEqual(engine_identifier("eevee", (5, 0, 0)), "BLENDER_EEVEE")
        self.assertEqual(eevee_identifier((4, 5, 0)), "BLENDER_EEVEE_NEXT")
        self.assertEqual(engine_identifier("cycles", (5, 0, 0)), "CYCLES")
        with self.assertRaises(ValueError):
            eevee_identifier((4, 1, 0))
        with self.assertRaises(ValueError):
            engine_identifier("unknown", (5, 0, 0))

    def test_selected_existing_pose_ids(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            for idx in (1, 2, 3, 25, 50, 2199):
                (root / f"{idx:05d}_meta.json").write_text("{}", encoding="utf-8")
            self.assertEqual([i for i, _ in pose_indices(root)], [1, 2, 3, 25, 50, 2199])
            self.assertEqual([i for i, _ in pose_indices(root, "50,1,3")], [50, 1, 3])
            with self.assertRaises(ValueError):
                pose_indices(root, "1,1")
            with self.assertRaises(ValueError):
                pose_indices(root, "999")
            with self.assertRaises(ValueError):
                pose_indices(root, "1,not-int")

    def test_source_pose_validation(self):
        pose = {
            "pose": {"A": [0, 0, 0], "B": [1.0, -0.5, 0.2]},
            "root": [0, 0, 0], "yaw": 0.0,
            "joints_world": [[1, 2, 3], [2, 3, 4]],
        }
        self.assertTrue(validate_pose(pose, ("A", "B"), 2))
        bad = dict(pose, root=[0, float("nan"), 1])
        with self.assertRaises(ValueError):
            validate_pose(bad, ("A", "B"), 2)
        with self.assertRaises(ValueError):
            validate_pose(pose, ("A", "C"), 2)
        with self.assertRaises(ValueError):
            validate_pose(pose, ("A", "B"), 3)

    def test_manifest_digest_deterministic(self):
        self.assertEqual(contract_sha256({"a": 1, "b": 2}),
                         contract_sha256({"b": 2, "a": 1}))
        self.assertNotEqual(contract_sha256({"a": 1}),
                            contract_sha256({"a": 2}))

    def test_png_header_and_resume_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ids = 1
            artifacts = {}
            dims = {
                "front": (400, 720), "side": (880, 720),
                "back": (400, 720), "opposite_side": (880, 720),
            }
            for view in VIEWS:
                w, h = dims[view]
                # Only the 24-byte IHDR signature is consumed by the preflight;
                # actual Blender renders have valid PNG contents beyond this.
                png = root / f"{ids:05d}_{view}.png"
                png.write_bytes(PNG + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", w, h))
                label = root / f"{ids:05d}_{view}_labels.npz"
                label.write_bytes(b"npz-test-labels")
                artifacts[png.name] = sha256_file(png)
                artifacts[label.name] = sha256_file(label)
                self.assertEqual(png_dimensions(png), (w, h))
            meta = root / "00001_visibility_meta.json"
            report = {"source_pose_sha256": "source-sha",
                      "render_contract_sha256": "contract-sha",
                      "artifacts": artifacts}
            meta.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(sample_is_complete(root, ids, "source-sha", "contract-sha", dims))
            self.assertFalse(sample_is_complete(
                root, ids, "source-sha", "contract-sha",
                dict(dims, front=(401, 720))))
            with self.assertRaises(ValueError):
                sample_is_complete(root, ids, "changed-source", "contract-sha", dims)
            with self.assertRaises(ValueError):
                sample_is_complete(root, ids, "source-sha", "changed-contract", dims)
            (root / "00001_back_labels.npz").write_bytes(b"changed-labels")
            self.assertFalse(sample_is_complete(root, ids, "source-sha", "contract-sha", dims))


if __name__ == "__main__":
    unittest.main()
