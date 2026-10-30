"""OpenCV synthetic optical-flow integration smoke, no neural reader weights."""
import unittest

try:
    import numpy as np
    import cv2
except ImportError:
    np = None
    cv2 = None

if np is not None and cv2 is not None:
    from mocap.temporal_tracking import lk_measurements, fuse_tracks


@unittest.skipUnless(np is not None and cv2 is not None,
                     "requires NumPy+OpenCV separate CI job")
class OpticalFlowTests(unittest.TestCase):
    def test_image_point_moves_with_textured_patch(self):
        T, H, W = 9, 100, 120
        rng = np.random.default_rng(30)
        pattern = rng.integers(60, 230, (25, 25), dtype=np.uint8)
        frames = np.full((T, H, W, 3), 75, dtype=np.uint8)
        uv = np.zeros((T, 1, 2), np.float32)
        for t in range(T):
            x, y = 30+2*t, 40+t
            frames[t, y-12:y+13, x-12:x+13, :] = pattern[..., None]
            uv[t, 0] = [float(x), float(y)]
        predicted, valid, fb = lk_measurements(frames, uv)
        self.assertTrue(valid[1:, 0].all(), "forward/backward LK should track translated texture")
        self.assertLess(float(np.median(fb[1:, 0])), 1.)
        self.assertLess(float(np.mean(np.linalg.norm(
            predicted[1:, 0]-uv[1:, 0], axis=-1))), 1.)
        refined, weight, report = fuse_tracks(
            uv, np.full((T, 1), .8), predicted, valid, fb, W, H)
        np.testing.assert_allclose(refined, uv, atol=1.)
        self.assertEqual(report["counts"]["flow_only"], 0)
        self.assertTrue((weight >= 0).all())


if __name__ == "__main__":
    unittest.main()
