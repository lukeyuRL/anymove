"""Pure NumPy tests for P1 temporal joint tracking. No OpenCV/GPU required."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.temporal_tracking import fuse_tracks


@unittest.skipUnless(np is not None, "NumPy is optional for lightweight CI")
class TemporalTrackingTests(unittest.TestCase):
    def fixture(self, T=12):
        uv = np.zeros((T, 2, 2), np.float32)
        for t in range(T):
            uv[t, 0] = [20+t*2, 30+t*1.5]
            uv[t, 1] = [40+t*1.5, 70+t*.5]
        pk = np.full((T, 2), .9, np.float32)
        # Flow from the Reader's raw t-1 position to the next frame.
        flow = uv.copy()
        ok = np.ones((T, 2), bool)
        err = np.zeros((T, 2), np.float32)
        return uv, pk, flow, ok, err

    def test_perfect_detections_stay_unchanged(self):
        uv, pk, flow, ok, err = self.fixture()
        actual, weights, report = fuse_tracks(uv, pk, flow, ok, err, 640, 720)
        np.testing.assert_allclose(actual, uv, atol=1e-5)
        np.testing.assert_array_equal(weights, 1.0)
        self.assertEqual(report["counts"]["fused"], 22)

    def test_low_peak_reader_outlier_can_use_trusted_flow(self):
        uv, pk, flow, ok, err = self.fixture()
        base = uv.copy()
        # Unreliable Reader spontaneously assigns a hind joint to wrong
        # body region. The flow from the previous visible frame is stable.
        uv[4, 0] += np.array([20., 22.])
        pk[4, 0] = .3
        flow[4, 0] = base[4, 0]
        refined, w, report = fuse_tracks(uv, pk, flow, ok, err, 640, 720)
        self.assertLess(np.linalg.norm(refined[4, 0]-base[4, 0]), 1.)
        self.assertAlmostEqual(w[4, 0], .55)
        self.assertEqual(report["counts"]["flow_only"], 1)
        np.testing.assert_allclose(uv[4, 0], base[4, 0]+[20, 22])

    def test_high_peak_outlier_not_forcibly_replaced(self):
        uv, pk, flow, ok, err = self.fixture()
        base = uv.copy()
        uv[4, 0] += np.array([20., 22.])
        flow[4, 0] = base[4, 0]
        refined, w, report = fuse_tracks(uv, pk, flow, ok, err, 640, 720)
        np.testing.assert_allclose(refined[4, 0], uv[4, 0])
        self.assertAlmostEqual(w[4, 0], .40)
        self.assertEqual(report["counts"]["outlier_flagged"], 1)

    def test_bad_bidirectional_flow_never_drives_coordinates(self):
        uv, pk, flow, ok, err = self.fixture()
        orig = uv.copy()
        flow[6, 1] += np.array([40., 40.])
        err[6, 1] = 8.5
        refined, weights, report = fuse_tracks(
            uv, pk, flow, ok, err, 640, 720)
        np.testing.assert_allclose(refined, orig, atol=1e-5)
        self.assertEqual(weights[6, 1], 1.)
        self.assertTrue(report["counts"]["flow_rejected"] > 0)

    def test_out_of_bounds_predicted_flow_rejected(self):
        uv, pk, flow, ok, err = self.fixture()
        flow[2, 0] = [-1., 15.]
        refined, _, report = fuse_tracks(uv, pk, flow, ok, err, 640, 720)
        np.testing.assert_array_equal(refined[2, 0], uv[2, 0])
        self.assertGreaterEqual(report["counts"]["flow_rejected"], 1)

    def test_invalid_shapes_and_nonfinite_fail_closed(self):
        uv, pk, flow, ok, err = self.fixture()
        with self.assertRaisesRegex(ValueError, "invalid"):
            fuse_tracks(uv, pk[:, :1], flow, ok, err, 640, 720)
        bad = uv.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid"):
            fuse_tracks(bad, pk, flow, ok, err, 640, 720)
        with self.assertRaisesRegex(ValueError, "invalid"):
            fuse_tracks(uv, pk, flow, ok, err, 0, 720)


if __name__ == "__main__":
    unittest.main()
