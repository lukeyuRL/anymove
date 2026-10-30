"""Dataset image and metadata output must always share a Windows-safe root."""
import os
import pathlib
import tempfile
import unittest

from blender.dataset_paths import (resolve_render_inputs, sample_file,
                                   is_written, require_written)


class DatasetPathTests(unittest.TestCase):
    def test_absolute_image_and_json_paths_share_directory(self):
        cwd = pathlib.Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chdir(td)
                blend, info, out = resolve_render_inputs(
                    "assets/dragon/rigged/rigged.blend",
                    "assets/dragon/rigged/rig_info.json",
                    "runs/dragon/dataset_hr_smoke")
                self.assertTrue(blend.is_absolute())
                self.assertTrue(info.is_absolute())
                self.assertTrue(out.is_absolute())
                # Simulate Blender changing its relative path context after
                # opening the rigged.blend; saved files must remain under td.
                os.mkdir("other")
                os.chdir("other")
                paths = [sample_file(out, 0, k) for k in ("front", "side", "meta")]
                self.assertEqual({p.parent for p in paths}, {out})
                self.assertEqual([p.name for p in paths],
                                 ["00000_front.png", "00000_side.png", "00000_meta.json"])
                for p in paths:
                    self.assertTrue(p.is_absolute())
            finally:
                os.chdir(cwd)

    def test_render_missing_or_empty_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            path = sample_file(pathlib.Path(td), 0, "front")
            self.assertFalse(is_written(path))
            with self.assertRaisesRegex(RuntimeError, "did not write expected"):
                require_written(path)
            path.write_bytes(b"")
            self.assertFalse(is_written(path))
            path.write_bytes(b"png test fixture")
            self.assertTrue(is_written(path))
            require_written(path)

    def test_reject_invalid_file_kind_and_relative_root(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                sample_file(pathlib.Path(td), 0, "top")
            with self.assertRaises(ValueError):
                sample_file(pathlib.Path("relative"), 0, "front")
            with self.assertRaises(ValueError):
                sample_file(pathlib.Path(td), -1, "front")


if __name__ == "__main__":
    unittest.main()
