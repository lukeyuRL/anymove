"""Regression for detailed Dragon 3D motion A/B diagnostics.

No Blender/Torch required. Requires NumPy, installed in the dedicated CI job.
"""
import json
import pathlib
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from tools.diagnose_temporal_motion import analyze, main


@unittest.skipUnless(np is not None, "NumPy missing in lightweight CI")
class TemporalMotionDiagnosticTests(unittest.TestCase):
    def fixture(self):
        frames = 8
        names = ["hips", "knee_L", "hpaw_L", "hpaw_R", "hock_R", "head"]
        pos = np.zeros((frames, len(names), 3))
        for t in range(frames):
            pos[t, :, 1] = t * 0.005
        # Only one hind joint differs during associated frames, starting
        # just after the first identity->swap transition.
        changed = pos.copy()
        changed[2:4, names.index("hpaw_L"), 2] = .10
        root = np.zeros((frames, 3))
        root_var = root.copy()
        root_var[2:4, 2] = .015
        quat = np.tile(np.array([[1.0, 0, 0, 0]]), (frames, 1))
        def make(joints, trans):
            return {
                "fps": 24., "bones": ["root"], "joint_names": names,
                "joints": joints.tolist(), "trans": trans.tolist(),
                "root_quat": quat.tolist(),
                "source_detections_sha256": "same",
                "solver_camera_layout": {"W_FRONT": 640, "W_SIDE": 640},
            }
        association = {
            "state_by_frame": [0, 0, 1, 1, 0, 0, 0, 0],
            "source_detections_sha256": "same",
            "accepted_segments": [{"start_frame": 2, "end_frame_exclusive": 4,
                                   "frame_count": 2}]
        }
        return make(pos, root), make(changed, root_var), association

    def test_report_finds_localized_joint_and_root_difference(self):
        base, temporal, association = self.fixture()
        report, per_frame, by_joint, detailed = analyze(base, temporal, association)
        self.assertEqual(report["frames"], 8)
        self.assertEqual(report["accepted_swapped_frames"], 2)
        self.assertEqual(report["swap_segments"][0]["start_frame"], 2)
        self.assertEqual(report["swap_segments"][0]["end_frame_exclusive"], 4)
        self.assertEqual(report["top_joints_by_max_delta"][0]["joint"], "hpaw_L")
        self.assertAlmostEqual(report["top_joints_by_max_delta"][0]["max_delta_cm"], 10)
        self.assertEqual(report["max_root_delta_frame"], 2)
        self.assertAlmostEqual(report["root_translation_delta"]["max_cm"], 1.5)
        self.assertEqual(len(per_frame), 8)
        self.assertEqual(len(by_joint), 6)
        self.assertEqual(len(detailed), 8 * 6)
        self.assertGreater(report["mean_hind_delta_swapped_frames_cm"],
                           report["mean_hind_delta_outside_swapped_frames_cm"])
        self.assertEqual(per_frame[0]["mean_hind_step_change_cm"], None)

    def test_seam_velocity_differences_detect_switch_boundaries(self):
        base, temporal, association = self.fixture()
        report, _, _, _ = analyze(base, temporal, association)
        seams = report["association_boundary_steps"]
        self.assertEqual([(s["from_frame"], s["to_frame"]) for s in seams],
                         [(1, 2), (3, 4)])
        self.assertGreater(seams[0]["hind_step_vector_change_max_cm"], 9)
        self.assertGreater(seams[1]["hind_step_vector_change_max_cm"], 9)
        self.assertAlmostEqual(seams[0]["root_step_vector_change_cm"], 1.5)
        self.assertAlmostEqual(seams[1]["root_step_vector_change_cm"], 1.5)

    def test_no_association_is_safe_and_warns(self):
        a, b, _ = self.fixture()
        r, _, _, _ = analyze(a, b)
        self.assertEqual(r["accepted_swapped_frames"], 0)
        self.assertEqual(r["association_boundary_steps"], [])
        self.assertTrue(any("not supplied" in x for x in r["warnings"]))

    def test_provenance_mismatch_rejected(self):
        a, b, assoc = self.fixture()
        b["source_detections_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "same-detection"):
            analyze(a, b, assoc)
        b["source_detections_sha256"] = "same"
        b["fps"] = 30.
        with self.assertRaisesRegex(ValueError, "contract"):
            analyze(a, b, assoc)
        b["fps"] = 24.
        assoc["source_detections_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "association source"):
            analyze(a, b, assoc)

    def test_joint_order_mismatch_and_nonfinite_rejected(self):
        a, b, assoc = self.fixture()
        b["joint_names"] = list(reversed(a["joint_names"]))
        with self.assertRaisesRegex(ValueError, "joint_names"):
            analyze(a, b, assoc)
        b["joint_names"] = list(a["joint_names"])
        b["joints"][3][1][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            analyze(a, b, assoc)

    def test_report_written_to_separate_folder_with_utf8_csv(self):
        a, b, assoc = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            a_path = folder / "motion.json"
            b_path = folder / "motion_temporal.json"
            assoc_path = folder / "motion_temporal.association.json"
            a_path.write_text(json.dumps(a), encoding="utf-8")
            b_path.write_text(json.dumps(b), encoding="utf-8")
            assoc_path.write_text(json.dumps(assoc), encoding="utf-8")
            main([str(a_path), str(b_path)])
            dest = folder / "motion_temporal_diagnostics"
            self.assertTrue((dest / "temporal_3d_report.json").is_file())
            self.assertTrue((dest / "temporal_3d_by_frame.csv").is_file())
            self.assertTrue((dest / "temporal_3d_by_joint.csv").is_file())
            self.assertTrue((dest / "temporal_3d_by_joint_frame.csv").is_file())
            summary = json.loads((dest / "temporal_3d_report.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["accepted_swapped_frames"], 2)
            self.assertEqual(len((dest / "temporal_3d_by_joint_frame.csv").read_text(encoding="utf-8-sig").splitlines()), 49)
            self.assertTrue(a_path.is_file())
            self.assertTrue(b_path.is_file())


if __name__ == "__main__":
    unittest.main()
