"""P2 synthetic cutout targets are unambiguous and leave legacy images intact."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.quality_targets import synthetic_cutout, PROFILE
    from mocap.reader_config import ReaderSpec, spec_from_checkpoint


@unittest.skipUnless(np is not None, "NumPy optional in lightweight CI")
class QualityTargetTests(unittest.TestCase):
    def test_new_training_target_marks_occluded_joint_without_modifying_source(self):
        image = np.full((192, 192, 3), 132, dtype=np.uint8)
        points = np.array([[60., 60.], [90., 100.], [160., 150.]], np.float32)
        original = image.copy()
        out, labels = synthetic_cutout(
            image, points, np.random.default_rng(1), probability=1.)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(labels.shape, (3,))
        self.assertGreater(int(np.sum(labels == 0)), 0)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(PROFILE, "anymove_synthetic_cutout_visibility_v1")

    def test_unchanged_when_disabled_or_all_points_offscreen(self):
        image = np.full((100, 100, 3), 99, np.uint8)
        points = np.array([[22., 32.], [72., 84.]], np.float32)
        out, y = synthetic_cutout(
            image, points, np.random.default_rng(42), probability=0)
        np.testing.assert_array_equal(out, image)
        np.testing.assert_array_equal(y, 1)
        off = np.array([[-60., 12.], [300., 22.]], np.float32)
        result, v = synthetic_cutout(
            image, off, np.random.default_rng(42), probability=1)
        np.testing.assert_array_equal(result, image)
        np.testing.assert_array_equal(v, 1)

    def test_invalid_label_inputs_fail_closed(self):
        img = np.zeros((15, 15, 3), np.uint8)
        pts = np.ones((5, 2))
        for bad in (-.1, 1.1):
            with self.assertRaisesRegex(ValueError, "probability"):
                synthetic_cutout(img, pts, np.random.default_rng(0), bad)
        with self.assertRaisesRegex(ValueError, "finite"):
            pts[0, 0] = np.nan
            synthetic_cutout(img, pts, np.random.default_rng(0))

    def test_quality_head_checkpoint_uses_same_672_resolution_contract(self):
        ck = {"reader_spec": dict(ReaderSpec(672, 192).checkpoint_metadata(),
                                   quality_head=True),
              "quality_spec": {"version": 1,
                               "target": "anymove_synthetic_cutout_visibility_v1"}}
        self.assertEqual(spec_from_checkpoint(ck), ReaderSpec(672, 192))
        self.assertEqual(spec_from_checkpoint({"state_dict": {}, "names": []}),
                         ReaderSpec(448, 128))


if __name__ == "__main__":
    unittest.main()
