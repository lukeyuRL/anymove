"""High-resolution reader shape and coordinate contract: stdlib only (CI-ready)."""
import unittest

from mocap.reader_config import (
    ReaderSpec, LEGACY_INPUT_SIZE, LEGACY_HEATMAP_SIZE,
    HIGH_RES_INPUT_SIZE, HIGH_RES_HEATMAP_SIZE, spec_from_checkpoint,
)
from mocap.reader_geometry import Letterbox


class ReaderMetadataTests(unittest.TestCase):
    def test_legacy_checkpoint_is_448_128(self):
        spec = spec_from_checkpoint({"state_dict": {}, "names": ["root"]})
        self.assertEqual((spec.input_size, spec.heatmap_size),
                         (LEGACY_INPUT_SIZE, LEGACY_HEATMAP_SIZE))

    def test_new_checkpoint_preserves_high_res(self):
        spec = ReaderSpec(HIGH_RES_INPUT_SIZE, HIGH_RES_HEATMAP_SIZE)
        loaded = spec_from_checkpoint({"state_dict": {}, "reader_spec": spec.checkpoint_metadata()})
        self.assertEqual(loaded, spec)

    def test_reader_size_must_match_patch_grid(self):
        for bad in (0, 512, 673, 896.0, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ReaderSpec(bad, 192)

    def test_bad_heatmap_or_unknown_checkpoint_version_rejected(self):
        with self.assertRaises(ValueError):
            ReaderSpec(672, 0)
        with self.assertRaises(ValueError):
            spec_from_checkpoint({"reader_spec": {"version": 2,
                "input_size": 672, "heatmap_size": 192}})
        with self.assertRaises(ValueError):
            spec_from_checkpoint({"reader_spec": {"version": 1, "input_size": 672}})


class LetterboxCoordinateTests(unittest.TestCase):
    def test_full_res_dragon_roundtrip(self):
        for size in (448, 672, 896):
            with self.subTest(size=size):
                geo = Letterbox(640, 720, size)
                for uv in ((0., 0.), (320., 360.), (639., 719.), (-25., 812.)):
                    result = geo.to_layout(*geo.to_reader(*uv))
                    self.assertAlmostEqual(result[0], uv[0], places=6)
                    self.assertAlmostEqual(result[1], uv[1], places=6)

    def test_legacy_06_render_scale_roundtrip(self):
        geo = Letterbox(384, 432, 672)
        for uv in ((12., 36.), (320., 360.), (639., 720.)):
            result = geo.to_layout(*geo.to_reader(*uv, render_scale=0.6), render_scale=0.6)
            self.assertAlmostEqual(result[0], uv[0], places=6)
            self.assertAlmostEqual(result[1], uv[1], places=6)

    def test_reader_coordinates_are_size_dependent_but_layout_is_not(self):
        physical_joint = (211.5, 492.25)
        for size in (448, 672, 896):
            geo = Letterbox(640, 720, size)
            model_uv = geo.to_reader(*physical_joint)
            self.assertAlmostEqual(geo.to_layout(*model_uv)[0], physical_joint[0])
            self.assertAlmostEqual(geo.to_layout(*model_uv)[1], physical_joint[1])

    def test_padded_side_layout(self):
        geo = Letterbox(420, 720, 672)
        self.assertEqual(geo.ox, 150)
        self.assertEqual(geo.oy, 0)
        self.assertAlmostEqual(geo.to_reader(210, 360)[0], 336)
        self.assertAlmostEqual(geo.to_layout(336, 336)[0], 210)

    def test_bad_geometry_and_scales_rejected(self):
        for dims in ((0, 720, 672), (640, -1, 672), (640, 720, 0)):
            with self.assertRaises(ValueError):
                Letterbox(*dims)
        geo = Letterbox(640, 720, 672)
        for scale in (0, -0.6, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                geo.to_layout(1, 2, scale)


if __name__ == "__main__":
    unittest.main()
