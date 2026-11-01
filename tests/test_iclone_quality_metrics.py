"""No-ML/GPU tests for held-out Dino localization uncertainty audits."""
import unittest
import numpy as np

from mocap.quality_metrics import error_calibration_report, PROFILE


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
