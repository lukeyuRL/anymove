"""Held-out 2D localization error diagnostics for the experimental quality head.

The Reader's 'expected_error_px' is an uncalibrated point estimate. These
metrics compare it against known Blender 2D labels; they are NOT evidence
of mesh self-occlusion calibration or real-video uncertainty calibration.
"""
import numpy as np

PROFILE = "anymove_reader_error_calibration_audit_v1"


def error_calibration_report(expected_error_px, actual_error_px, bins=8):
    """Summarize mean-error calibration and selective risk, without fitting.

    Input arrays may be (N,J) or any matching shape. Quantile bins are
    ordered by predicted error, so each bin gets similar sample count.
    No thresholds or solver weights are learned from these same labels.
    """
    pred = np.asarray(expected_error_px, dtype=np.float64)
    actual = np.asarray(actual_error_px, dtype=np.float64)
    if (pred.shape != actual.shape or pred.size < 2 or
            not np.isfinite(pred).all() or not np.isfinite(actual).all() or
            np.any(pred < 0) or np.any(actual < 0)):
        raise ValueError("calibration requires matching finite nonnegative predicted/actual errors")
    if type(bins) is not int or bins < 2:
        raise ValueError("calibration bins must be an integer >=2")
    pred = pred.reshape(-1)
    actual = actual.reshape(-1)
    order = np.argsort(pred, kind="stable")
    groups = [g for g in np.array_split(order, min(bins, len(order))) if len(g)]
    table = []
    weighted_gap = 0.
    for ids in groups:
        expected = float(pred[ids].mean())
        observed = float(actual[ids].mean())
        weighted_gap += len(ids) * abs(expected-observed)
        table.append({
            "count": len(ids),
            "predicted_mean_px": round(expected, 5),
            "observed_mean_px": round(observed, 5),
            "absolute_gap_px": round(abs(expected-observed), 5),
        })
    k = max(1, len(order)//2)
    standard_deviation = float(pred.std() * actual.std())
    corr = float(np.corrcoef(pred, actual)[0, 1]) if standard_deviation > 1e-12 else None
    return {
        "profile": PROFILE,
        "samples": int(len(pred)),
        "expected_error_mean_px": round(float(pred.mean()), 5),
        "actual_error_mean_px": round(float(actual.mean()), 5),
        "mean_error_calibration_gap_px": round(weighted_gap / len(pred), 5),
        "predicted_vs_actual_pearson": None if corr is None else round(corr, 5),
        "lowest_50pct_predicted_risk_px": round(float(actual[order[:k]].mean()), 5),
        "all_sample_risk_px": round(float(actual.mean()), 5),
        "bins": table,
        "interpretation": "held-out synthetic 2D label residuals; not calibrated real visibility",
    }
