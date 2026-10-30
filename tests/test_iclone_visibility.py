"""NumPy-based regression for opt-in Dragon front-hind weighting."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.visibility import front_hind_weights, PROFILE, HIND_STEMS, FORE_STEMS
    from tools.compare_visibility import compare


@unittest.skipUnless(np is not None, "NumPy not installed in lightweight CI")
class VisibilityWeightsTests(unittest.TestCase):
    def fixture(self, frames=35):
        names = ["hips", "head"] + [
            f"{stem}_{side}" for side in ("L", "R")
            for stem in HIND_STEMS + FORE_STEMS
        ]
        a = np.full((frames, len(names), 2), 100.0)
        a[:, :, 0] = np.arange(len(names))[None, :] * 20 + 50
        return {"front_uv": a, "side_uv": a.copy()}, names

    def test_valid_data_stays_unchanged(self):
        d, names = self.fixture()
        weight, report = front_hind_weights(d, names)
        np.testing.assert_array_equal(weight, 1.0)
        self.assertEqual(report["affected_joint_count"], 0)
        self.assertEqual(len(report["front_weights_by_joint"]), 8)

    def test_mismatched_front_paw_cloned_from_foreleg_is_downweighted(self):
        d, names = self.fixture()
        j, k = names.index("hpaw_L"), names.index("fpaw_L")
        d["front_uv"][8:22, j, :] = d["front_uv"][8:22, k, :]
        d["side_uv"][8:22, j, 1] = d["front_uv"][8:22, j, 1] + 32
        before = {key: val.copy() for key, val in d.items()}
        weight, report = front_hind_weights(d, names)
        self.assertLess(weight[15, j], 0.4)
        self.assertEqual(weight[15, k], 1)
        self.assertTrue((weight >= 0.35).all() and (weight <= 1).all())
        for i in range(weight.shape[1]):
            if i != j:
                self.assertTrue((weight[:, i] == 1).all())
        for key in d:
            np.testing.assert_array_equal(d[key], before[key])
        self.assertEqual(report["profile"], PROFILE)

    def test_front_overlap_without_height_conflict_does_not_change_weight(self):
        d, names = self.fixture()
        j, k = names.index("hpaw_L"), names.index("fpaw_L")
        d["front_uv"][5:15, j, :] = d["front_uv"][5:15, k, :]
        d["side_uv"][5:15, j, 1] = d["front_uv"][5:15, j, 1]
        w, _ = front_hind_weights(d, names)
        self.assertTrue((w == 1).all())

    def test_nonoverlapping_hind_height_disagreement_is_less_penalized(self):
        d, names = self.fixture()
        j = names.index("hock_R")
        d["side_uv"][5:20, j, 1] += 30
        w, _ = front_hind_weights(d, names)
        self.assertGreater(w[12, j], 0.6)
        self.assertLess(w[12, j], 1.0)

    def test_corrupted_front_measurement_has_smaller_initial_height_effect(self):
        d, names = self.fixture()
        j, k = names.index("hpaw_L"), names.index("fpaw_L")
        d["front_uv"][7:24, j, :] = d["front_uv"][7:24, k, :]
        d["side_uv"][7:24, j, 1] = d["front_uv"][7:24, j, 1] + 35
        weight, _ = front_hind_weights(d, names)
        # Same relative weighting used by triangulate(). Side is treated as
        # synthetic ground truth for this controlled corruption case only.
        f, s = d["front_uv"][15, j, 1], d["side_uv"][15, j, 1]
        baseline = (0.25*f + s) / 1.25
        variant = (0.25*weight[15, j]*f + s) / (1 + 0.25*weight[15, j])
        self.assertLess(abs(variant-s), abs(baseline-s))

    def test_bad_names_and_nonfinite_inputs_fail_closed(self):
        d, names = self.fixture()
        with self.assertRaisesRegex(ValueError, "requires joints"):
            front_hind_weights(d, names[:-1])
        d["front_uv"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            front_hind_weights(d, names)

    def test_report_is_json_serializable(self):
        import json
        d, names = self.fixture()
        _, report = front_hind_weights(d, names)
        self.assertIn("front_weights_by_joint", json.loads(json.dumps(report)))


@unittest.skipUnless(np is not None, "NumPy not installed in lightweight CI")
class ComparisonTests(unittest.TestCase):
    def motion(self):
        return {
            "fps": 24.0, "bones": ["pelvis"], "joint_names": ["hips"],
            "root_quat": [[1, 0, 0, 0], [1, 0, 0, 0]],
            "trans": [[0, 0, 0], [0, 0, 0]],
            "joints": [[[0, 0, 0]], [[0, 0, 0]]],
            "reproj_px": {"front": 5.0, "side": 3.0},
            "sync": {"paw_agreement": {"score": 0.36}},
        }

    def test_identical_motions_do_not_claim_accuracy_gain(self):
        result = compare(self.motion(), self.motion())
        self.assertEqual(result["joint_position_difference_mean_cm"], 0)
        self.assertIn("NOT 3D truth", result["comparison_type"])

    def test_motion_delta_and_reject_invalid_pair(self):
        a, b = self.motion(), self.motion()
        b["trans"] = [[0, 0, .10], [0, 0, .20]]
        b["joints"] = [[[0, 0, .10]], [[0, 0, .20]]]
        r = compare(a, b)
        self.assertEqual(r["root_translation_difference_mean_cm"], 15)
        b["fps"] = 25.
        with self.assertRaisesRegex(ValueError, "fps"):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()
