"""No-ML/GPU tests for held-out Dino localization uncertainty audits."""
import unittest
try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.quality_metrics import error_calibration_report, synthetic_visibility_report, PROFILE


@unittest.skipUnless(np is not None, "NumPy optional in lightweight CI")
class QualityMetricsTests(unittest.TestCase):
    def test_perfect_mean_calibration_has_zero_gap(self):
        expected = np.arange(1., 41.).reshape(10, 4)
        report = error_calibration_report(expected, expected)
        self.assertEqual(report["profile"], PROFILE)
        self.assertEqual(report["samples"], 40)
        self.assertEqual(report["mean_error_calibration_gap_px"], 0.)
        self.assertLess(report["lowest_50pct_predicted_risk_px"],
                        report["all_sample_risk_px"])
        self.assertEqual(sum(row["count"] for row in report["bins"]), 40)

    def test_biased_quality_model_is_not_called_calibrated(self):
        expected = np.ones((100, 2)) * 1.5
        actual = np.ones((100, 2)) * 9.
        report = error_calibration_report(expected, actual)
        self.assertAlmostEqual(report["mean_error_calibration_gap_px"], 7.5)
        self.assertIsNone(report["predicted_vs_actual_pearson"])
        self.assertIn("not calibrated real visibility", report["interpretation"])

    def test_perfect_cutout_visibility_auc_and_brier(self):
        scores = np.array([.1, .15, .82, .95, .13, .93], dtype=float)
        visible = np.array([0, 0, 1, 1, 0, 1], dtype=float)
        r = synthetic_visibility_report(scores, visible)
        self.assertEqual(r["roc_auc"], 1.0)
        self.assertEqual(r["artificially_masked"], 3)
        self.assertLess(r["brier"], .04)
        self.assertIn("NOT true mesh", r["meaning"])

    def test_cutout_visibility_ties_and_single_class(self):
        r = synthetic_visibility_report(
            np.full(20, .5), np.tile([0, 1], 10))
        self.assertEqual(r["roc_auc"], .5)
        no_mask = synthetic_visibility_report(np.ones(10), np.ones(10))
        self.assertIsNone(no_mask["roc_auc"])
        self.assertEqual(no_mask["artificially_masked"], 0)
        with self.assertRaisesRegex(ValueError, "binary"):
            synthetic_visibility_report([.1, .8], [0, .6])
        with self.assertRaisesRegex(ValueError, "probabilities"):
            synthetic_visibility_report([-.2, .8], [0, 1])

    def test_invalid_predictions_fail_closed(self):
        a = np.ones((5, 3))
        with self.assertRaisesRegex(ValueError, "matching"):
            error_calibration_report(a, np.ones(5))
        b = a.copy()
        b[2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            error_calibration_report(a, b)
        b = a.copy()
        b[2, 1] = -.5
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            error_calibration_report(b, a)
        with self.assertRaisesRegex(ValueError, "bins"):
            error_calibration_report(a, a, bins=1)


if __name__ == "__main__":
    unittest.main()
