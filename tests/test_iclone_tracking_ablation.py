"""P1 Dino 2x2 factor test: UV and Solver weights must stay separable."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.tracking_ablation import make_tracking_ablations


@unittest.skipUnless(np is not None, "NumPy optional in lightweight CI")
class TrackingAblationTests(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "names": np.array(["hips", "tail"]),
            "fps": np.array(24.),
            "calib": np.eye(2, 3),
            "front_uv": np.ones((6, 2, 2)),
            "side_uv": np.ones((6, 2, 2))*2,
            "front_conf": np.ones((6, 2))*.8,
            "side_conf": np.ones((6, 2))*.75,
        }
        self.tracked = {k: v.copy() for k, v in self.raw.items()}
        self.tracked["front_uv_reader"] = self.raw["front_uv"].copy()
        self.tracked["side_uv_reader"] = self.raw["side_uv"].copy()
        self.tracked["front_uv"][2, 1, 0] += 3
        self.tracked["side_uv"][3, 1, 1] -= 2
        self.tracked["front_track_weight"] = np.ones((6, 2))
        self.tracked["side_track_weight"] = np.ones((6, 2))
        self.tracked["front_track_weight"][2, 1] = .55
        self.tracked["side_track_weight"][3, 1] = .4

    def test_uv_only_and_weight_only_do_not_mutate_raw(self):
        saved = {k: v.copy() for k, v in self.raw.items()}
        uv, weight = make_tracking_ablations(self.raw, self.tracked)
        for view in ("front", "side"):
            self.assertNotIn(view+"_track_weight", uv)
            np.testing.assert_array_equal(uv[view+"_uv"], self.tracked[view+"_uv"])
            np.testing.assert_array_equal(uv[view+"_conf"], self.raw[view+"_conf"])
            np.testing.assert_array_equal(weight[view+"_uv"], self.raw[view+"_uv"])
            np.testing.assert_array_equal(weight[view+"_track_weight"], self.tracked[view+"_track_weight"])
        for k, v in saved.items():
            np.testing.assert_array_equal(self.raw[k], v)

    def test_bad_source_calibration_or_weights_rejected(self):
        self.tracked["calib"][0, 0] = 2
        with self.assertRaisesRegex(ValueError, "calibration"):
            make_tracking_ablations(self.raw, self.tracked)
        self.tracked["calib"] = self.raw["calib"].copy()
        self.tracked["front_track_weight"][2, 1] = 1.3
        with self.assertRaisesRegex(ValueError, "weights"):
            make_tracking_ablations(self.raw, self.tracked)

    def test_wrong_reader_source_rejected(self):
        self.tracked["side_uv_reader"][0, 0, 0] += 9
        with self.assertRaisesRegex(ValueError, "source"):
            make_tracking_ablations(self.raw, self.tracked)


if __name__ == "__main__":
    unittest.main()
