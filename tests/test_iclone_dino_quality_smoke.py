"""P2 Dino quality smoke copying does not touch its source data."""
import pathlib
import tempfile
import unittest

from tools.make_dino_quality_smoke import make_smoke


class DinoQualitySmokeTests(unittest.TestCase):
    def create_data(self, parent, poses=35):
        source = pathlib.Path(parent)/"dataset_hr_v1"
        source.mkdir()
        for n in range(poses):
            stem = f"{n:05d}"
            (source/(stem+"_meta.json")).write_text('{"pose": "independent"}')
            (source/(stem+"_front.png")).write_bytes(b"front"+stem.encode())
            (source/(stem+"_side.png")).write_bytes(b"side"+stem.encode())
        return source

    def test_copy_30_verified_still_poses_with_held_out_split(self):
        with tempfile.TemporaryDirectory() as root:
            src = self.create_data(root)
            dst = pathlib.Path(root)/"dataset_hr_v1_p2_smoke"
            result = make_smoke(src, dst)
            self.assertEqual(result["train_poses"], 28)
            self.assertEqual(result["val_poses"], 2)
            self.assertEqual(result["images"], 60)
            self.assertEqual(len(list(dst.glob("*"))), 90)
            self.assertEqual((dst/"00025_meta.json").read_bytes(),
                             (src/"00025_meta.json").read_bytes())
            with self.assertRaisesRegex(FileExistsError, "exists"):
                make_smoke(src, dst)

    def test_missing_source_image_or_wrong_subset_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            src = self.create_data(root)
            dst = pathlib.Path(root)/"smoke"
            (src/"00002_side.png").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                make_smoke(src, dst)
            self.assertFalse(dst.exists())
            with self.assertRaisesRegex(ValueError, ">=26"):
                make_smoke(src, dst, 15)

    def test_refuse_nested_output(self):
        with tempfile.TemporaryDirectory() as root:
            src = self.create_data(root)
            with self.assertRaisesRegex(ValueError, "inside source"):
                make_smoke(src, src/"smoke", 30)


if __name__ == "__main__":
    unittest.main()
