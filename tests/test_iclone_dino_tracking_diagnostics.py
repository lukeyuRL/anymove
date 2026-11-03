"""Dino P1 diagnostic regression on a shared raw Reader reference."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from tools.diagnose_dino_tracking import compare_tracking, _project


@unittest.skipUnless(np is not None, "NumPy unavailable; visibility CI installs it")
class DinoTrackingDiagnosticsTests(unittest.TestCase):
    def setup_case(self, frames=12):
        names = np.array(["hips", "left_foot", "right_foot"])
        camera = {"H": 720, "W_FRONT": 640, "W_SIDE": 640,
                  "M_PER_PX": 0.006875, "Z_CENTER": 1.85625}
        f = np.arange(frames, dtype=float)
        points = np.zeros((frames, 3, 3))
        points[:, :, 0] = np.array([0., -.3, .3])[None, :]
        points[:, :, 1] = np.array([0., .1, .1])[None, :]
        points[:, :, 2] = 1. + .05*np.sin(f[:, None]*.2)
        front, side = _project(points, camera)
        raw = {
            "front_uv": front.copy(),
            "side_uv": side.copy(),
            "front_conf": np.ones((frames, 3), dtype=float),
            "side_conf": np.ones((frames, 3), dtype=float),
            "fps": np.asarray(24.),
            "names": names,
            "calib": np.array([[1., 0., 0.], [0., 1., 0.]]),
        }
        tracked = {k: v.copy() for k, v in raw.items()}
        tracked["front_uv_reader"] = front.copy()
        tracked["side_uv_reader"] = side.copy()
        tracked["front_uv"][4, 1, 0] += 4
        tracked["side_uv"][7, 2, 1] += 3
        tracked["front_track_weight"] = np.ones((frames, 3))
        tracked["side_track_weight"] = np.ones((frames, 3))
        tracked["front_track_weight"][4, 1] = .55
        variant = points.copy()
        variant[:, 1, 0] += .02
        variant[:, 2, 2] += .015
        root = np.column_stack([np.zeros(frames), np.zeros(frames),
                                .02*np.sin(f*.1)])
        original_motion = {
            "fps": 24., "bones": ["root", "tail"],
            "joint_names": names.tolist(), "joints": points.tolist(),
            "trans": root.tolist(),
            "root_quat": np.tile([1., 0., 0., 0.], (frames, 1)).tolist(),
        }
        candidate_motion = dict(original_motion)
        candidate_motion["joints"] = variant.tolist()
        candidate_motion["trans"] = (root + np.array([.01, 0, 0])).tolist()
        return raw, tracked, original_motion, candidate_motion, camera

    def test_detects_both_uv_and_weight_changes_and_common_raw_residual(self):
        case = self.setup_case()
        report = compare_tracking(*case)
        self.assertEqual(report["frames"], 12)
        self.assertEqual(report["keypoints"], 3)
        self.assertEqual(report["detection_changes"]["front"]["changed_joint_frames"], 1)
        self.assertEqual(
            report["detection_changes"]["front"]["joint_frames_above_displacement_px"]["3"], 1)
        self.assertEqual(
            report["detection_changes"]["front"]["joint_frames_above_displacement_px"]["30"], 0)
        self.assertEqual(
            report["detection_changes"]["front"]["max_adjustment_joint"], "left_foot")
        self.assertEqual(
            report["detection_changes"]["front"]["max_adjustment_solver_track_weight"], .55)
        self.assertEqual(
            report["detection_changes"]["front"]["max_adjustment_raw_reader_peak"], 1.)
        self.assertEqual(report["detection_changes"]["side"]["changed_joint_frames"], 1)
        self.assertEqual(report["detection_changes"]["front"]["joint_frames_with_weight_below_one"], 1)
        self.assertAlmostEqual(report["root_translation_between_solutions_cm"]["mean_cm"], 1., places=4)
        self.assertAlmostEqual(report["shared_raw_reader_reprojection_px"]["baseline"]["front"]["mean_px"], 0., places=5)
        self.assertGreater(report["shared_raw_reader_reprojection_px"]["tracked"]["front"]["mean_px"], 0)
        self.assertEqual(len(report["worst_joint_mean_deltas_cm"]), 3)
        self.assertIn("NOT 3D ground truth", report["comparison_type"])

    def test_does_not_mutate_source_data(self):
        raw, tracked, original, candidate, camera = self.setup_case()
        before = {k: v.copy() for k, v in raw.items()}
        compare_tracking(raw, tracked, original, candidate, camera)
        for key, val in before.items():
            np.testing.assert_array_equal(raw[key], val)

    def test_rejects_wrong_reader_coordinates_and_peak_mutation(self):
        raw, tracked, original, candidate, camera = self.setup_case()
        tracked["front_uv_reader"][0, 1, 0] += 1
        with self.assertRaisesRegex(ValueError, "preserve"):
            compare_tracking(raw, tracked, original, candidate, camera)
        tracked["front_uv_reader"] = raw["front_uv"].copy()
        tracked["side_conf"][0, 0] += .1
        with self.assertRaisesRegex(ValueError, "peaks"):
            compare_tracking(raw, tracked, original, candidate, camera)

    def test_rejects_wrong_fps_joint_order_or_weight(self):
        raw, tracked, original, candidate, camera = self.setup_case()
        tracked["fps"] = np.asarray(30.)
        with self.assertRaisesRegex(ValueError, "FPS"):
            compare_tracking(raw, tracked, original, candidate, camera)
        tracked["fps"] = np.asarray(24.)
        tracked["side_track_weight"][0, 0] = 1.2
        with self.assertRaisesRegex(ValueError, "weights"):
            compare_tracking(raw, tracked, original, candidate, camera)
        tracked["side_track_weight"][0, 0] = 1.
        candidate["joint_names"] = list(reversed(original["joint_names"]))
        with self.assertRaisesRegex(ValueError, "contract"):
            compare_tracking(raw, tracked, original, candidate, camera)


if __name__ == "__main__":
    unittest.main()
