"""NumPy-only regression tests for visibility + 3D continuity gates."""
import json
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.association_safety import (
        front_hind_reliability, validate_3d_segment, gated_temporal_fit, HIND
    )
    from mocap.temporal_association import temporal_hind_association


@unittest.skipUnless(np is not None, "NumPy unavailable in lightweight CI")
class AssociationSafetyTests(unittest.TestCase):
    def fixture(self, T=65):
        names = ["hips", "fpaw_L", "fpaw_R", "elbow_L", "elbow_R"] + [
            stem + "_" + side for side in ("L", "R") for stem in HIND]
        k = {name: i for i, name in enumerate(names)}
        front = np.zeros((T, len(names), 2), np.float64)
        side = np.zeros_like(front)
        for name, i in k.items():
            front[:, i, 0] = 50 + i * 18
            side[:, i, 0] = 80 + i * 20
            front[:, i, 1] = 250 + i * .7
            side[:, i, 1] = 260 + i * .7
        d = {
            "front_uv": front, "side_uv": side,
            "front_conf": np.full(front.shape[:2], .9, np.float64),
            "side_conf": np.full(front.shape[:2], .9, np.float64),
        }
        return d, names

    def fitted(self, names, T=65):
        j = np.zeros((T, len(names), 3), np.float64)
        root = np.zeros((T, 3), np.float64)
        root_rotmat = np.repeat(np.eye(3)[None], T, axis=0)
        return {"joints": j, "trans": root, "root_rotmat": root_rotmat,
                "reproj_px": {"front": 4.0, "side": 5.0},
                "bones": ["root"], "joint_names": names}

    def test_visibility_hypothesis_cues_downweight_but_never_alters_input(self):
        d, names = self.fixture()
        i = {name: j for j, name in enumerate(names)}
        target, fore = i["hpaw_L"], i["fpaw_L"]
        baseline = {k: v.copy() for k, v in d.items()}
        d["front_uv"][10:40, target] = d["front_uv"][10:40, fore]
        d["side_uv"][10:40, target, 0] = 500.
        prior = {k: v.copy() for k, v in d.items()}
        weights, report = front_hind_reliability(d, names)
        self.assertLess(weights[20, target], .4)
        self.assertEqual(weights[20, fore], 1.0)
        self.assertEqual(weights[2, target], 1.0)
        self.assertFalse(report["is_true_occlusion_probability"])
        for name in d:
            np.testing.assert_array_equal(d[name], prior[name])
        self.assertNotEqual(float(prior["front_uv"][20, target, 0]),
                            float(baseline["front_uv"][20, target, 0]))

    def test_persistent_front_anatomical_overlap_is_not_occlusion(self):
        # Regression against the 124-frame field report in which 109-124
        # frames were downweighted for six rear knee/hock/paw joints.
        # Anatomical/camera projection proximity is not per-frame
        # evidence of a transient visual obstruction.
        d, names = self.fixture()
        idx = {name: k for k, name in enumerate(names)}
        rear, fore = idx["hpaw_L"], idx["fpaw_L"]
        d["front_uv"][:, rear, :] = d["front_uv"][:, fore, :]
        d["side_uv"][:, rear, 0] = 500.0
        weights, report = front_hind_reliability(d, names)
        self.assertTrue(np.all(weights[:, rear] == 1.0))
        self.assertEqual(report["joint_summary"]["hpaw_L"][
            "transient_overlap_active_frames"], 0)
        self.assertTrue(report["joint_summary"]["hpaw_L"][
            "persistent_overlap_not_treated_as_occlusion"])

    def test_same_position_in_both_views_does_not_fake_occlusion(self):
        d, names = self.fixture()
        i = {name: j for j, name in enumerate(names)}
        j, f = i["hpaw_L"], i["fpaw_L"]
        for view in ("front_uv", "side_uv"):
            d[view][10:40, j, :] = d[view][10:40, f, :]
        w, _ = front_hind_reliability(d, names)
        self.assertTrue((w[:, j] == 1).all())

    def test_default_temporal_association_writes_visibility_audit(self):
        d, names = self.fixture()
        adjusted, report = temporal_hind_association(d, names)
        self.assertIn("visibility_diagnostics", report)
        self.assertEqual(report["visibility_diagnostics"]["profile"],
                         "front_hind_transient_foreleg_overlap_v2")
        self.assertEqual(report["accepted_swapped_frames"], 0)
        self.assertIn("unweighted_2d_reference", report)
        self.assertEqual(report["unweighted_2d_reference"][
            "accepted_swapped_frames"], 0)
        np.testing.assert_array_equal(adjusted["side_uv"], d["side_uv"])

    def test_real_report_like_3d_seam_is_quarantined(self):
        # Controlled numerical reconstruction of the user's *boundary*
        # observation (not the original full-body 3D joint array).
        _, names = self.fixture()
        base = self.fitted(names)
        variant = self.fitted(names)
        hind = [i for i, n in enumerate(names) if n.split("_")[0] in HIND]
        start = 41
        for j in hind:
            base["joints"][start:, j, 0] = .030585
            variant["joints"][start:, j, 0] = .077792
        result = validate_3d_segment(base, variant, names, 41, 53)
        self.assertEqual(result["status"], "quarantined")
        self.assertIn("excessive_hind_step_increase", result["failure_reasons"])
        self.assertAlmostEqual(result["boundary_checks"][0]["hind_step_baseline_mean_cm"],
                               3.0585, places=3)
        self.assertAlmostEqual(result["boundary_checks"][0]["hind_step_candidate_mean_cm"],
                               7.7792, places=3)
        self.assertFalse("max_hind_pose_delta_cm" in result["failure_reasons"])
        json.dumps(result, allow_nan=False)

    def test_large_but_smooth_pose_repair_not_blocked_by_position_delta_alone(self):
        _, names = self.fixture()
        base = self.fitted(names)
        candidate = self.fitted(names)
        hind = [i for i, n in enumerate(names) if n.split("_")[0] in HIND]
        for j in hind:
            candidate["joints"][:, j, 0] = .50  # 50cm but zero step/jerk
        result = validate_3d_segment(base, candidate, names, 15, 45)
        self.assertEqual(result["status"], "continuity_gate_passed")
        self.assertAlmostEqual(result["max_hind_pose_delta_cm"], 50.)

    def test_rejected_segment_returns_exact_baseline_fit_and_detections(self):
        d, names = self.fixture()
        candidate = {k: v.copy() for k, v in d.items()}
        candidate["side_uv"][41:53, :, 1] += 90.
        states = [0]*len(d["front_uv"])
        states[41:53] = [1]*12
        report = {"state_by_frame": states}
        calls = []
        original = {k: v.copy() for k, v in d.items()}
        def solve(data):
            res = self.fitted(names)
            if np.any(data["side_uv"] != d["side_uv"]):
                for j, name in enumerate(names):
                    if name.split("_")[0] in HIND:
                        res["joints"][41:, j, 0] = .20
            calls.append(res)
            return res
        actual_d, actual_fit, gate = gated_temporal_fit(d, candidate, report, names, solve)
        self.assertEqual(gate["status"], "all_candidates_quarantined")
        self.assertEqual(gate["accepted_swapped_frames"], 0)
        self.assertEqual(len(calls), 2)
        self.assertIs(actual_fit, calls[0])  # exact baseline object, no interpolation
        for key in original:
            np.testing.assert_array_equal(actual_d[key], original[key])
            np.testing.assert_array_equal(d[key], original[key])

    def test_accepted_smooth_candidate_retains_side_uv_conf_and_original_input(self):
        d, names = self.fixture()
        candidate = {k: v.copy() for k, v in d.items()}
        candidate["side_uv"][12:40, :, 1] += 90.
        candidate["side_conf"][12:40, :] = .7
        old = {k: v.copy() for k, v in d.items()}
        states = [int(12 <= i < 40) for i in range(len(d["front_uv"]))]
        def smooth(data):
            result = self.fitted(names)
            if np.any(data["side_uv"] != d["side_uv"]):
                for j, name in enumerate(names):
                    if name.split("_")[0] in HIND:
                        result["joints"][:, j, 0] = .25  # smooth and nonzero
            return result
        actual, fit_result, gate = gated_temporal_fit(
            d, candidate, {"state_by_frame": states}, names, smooth)
        self.assertEqual(gate["status"], "all_candidates_passed")
        self.assertEqual(gate["accepted_swapped_frames"], 28)
        np.testing.assert_array_equal(actual["side_uv"][12:40],
                                      candidate["side_uv"][12:40])
        np.testing.assert_array_equal(actual["side_conf"][12:40],
                                      candidate["side_conf"][12:40])
        np.testing.assert_array_equal(actual["side_uv"][:12], old["side_uv"][:12])
        np.testing.assert_array_equal(actual["front_uv"], old["front_uv"])
        for key in d:
            np.testing.assert_array_equal(d[key], old[key])
        self.assertAlmostEqual(float(fit_result["joints"][20, -1, 0]), .25)

    def test_no_candidate_calls_solver_exactly_once(self):
        d, names = self.fixture()
        counter = []
        def solve(d):
            counter.append(1)
            return self.fitted(names)
        _, _, report = gated_temporal_fit(d, d, {
            "state_by_frame": [0]*len(d["front_uv"])
        }, names, solve)
        self.assertEqual(len(counter), 1)
        self.assertEqual(report["accepted_swapped_frames"], 0)
        self.assertEqual(report["status"], "not_run_no_candidates")
        self.assertEqual(report["continuity_checks_run"], 0)
        self.assertEqual(report["validation_skipped_reason"], "no_2d_candidates")
        self.assertEqual(report["checks"], [])

    def test_root_rotation_jump_at_swap_boundary_is_quarantined(self):
        _, names = self.fixture()
        base = self.fitted(names)
        candidate = self.fitted(names)
        theta = np.radians(12.)
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0.],
            [np.sin(theta), np.cos(theta), 0.],
            [0., 0., 1.]
        ])
        candidate["root_rotmat"][41:] = rotation
        report = validate_3d_segment(base, candidate, names, 41, 53)
        self.assertEqual(report["status"], "quarantined")
        self.assertIn("excessive_root_rotation_step", report["failure_reasons"])
        self.assertAlmostEqual(
            report["boundary_checks"][0]["root_angular_step_candidate_deg"],
            12., places=4)

    def test_bad_3d_solution_fails_closed(self):
        _, names = self.fixture()
        a, b = self.fitted(names), self.fitted(names)
        b["joints"][2, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_3d_segment(a, b, names, 10, 20)


if __name__ == "__main__":
    unittest.main()
