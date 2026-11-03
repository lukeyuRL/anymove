"""Dino Reader warm-start contracts; pure Python, no GPU required."""
import pathlib
import tempfile
import unittest

from mocap.reader_config import ReaderSpec
from mocap.reader_warmstart import validate_base_reader, sha256_file


class ReaderWarmstartTests(unittest.TestCase):
    def setUp(self):
        self.names = ["root", "head", "tail"]
        self.spec = ReaderSpec(672, 192)
        self.checkpoint = {
            "names": list(self.names),
            "reader_spec": self.spec.checkpoint_metadata(),
            "state_dict": {"out.weight": "stub", "out.bias": "stub"},
        }

    def test_matching_inference_reader_can_initialize_quality_head(self):
        weights = validate_base_reader(self.checkpoint, self.names, self.spec)
        self.assertIs(weights, self.checkpoint["state_dict"])

    def test_rejects_wrong_creature_order_or_resolution(self):
        with self.assertRaisesRegex(ValueError, "names/order"):
            validate_base_reader(self.checkpoint, list(reversed(self.names)), self.spec)
        with self.assertRaisesRegex(ValueError, "resolution"):
            validate_base_reader(self.checkpoint, self.names, ReaderSpec(448, 128))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_base_reader(self.checkpoint, ["root", "root"], self.spec)

    def test_rejects_already_trained_quality_head(self):
        self.checkpoint["quality_spec"] = {"version": 1}
        with self.assertRaisesRegex(ValueError, "quality head"):
            validate_base_reader(self.checkpoint, self.names, self.spec)
        self.checkpoint.pop("quality_spec")
        self.checkpoint["state_dict"]["quality_out.bias"] = "stub"
        with self.assertRaisesRegex(ValueError, "already contains"):
            validate_base_reader(self.checkpoint, self.names, self.spec)

    def test_hash_is_path_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            first = pathlib.Path(folder) / "reader.pt"
            second = pathlib.Path(folder) / "reader_copy.pt"
            first.write_bytes(b"same original Dino checkpoint")
            second.write_bytes(first.read_bytes())
            self.assertEqual(sha256_file(first), sha256_file(second))
            second.write_bytes(b"different checkpoint")
            self.assertNotEqual(sha256_file(first), sha256_file(second))


if __name__ == "__main__":
    unittest.main()
