"""Toe-tip physical plausibility gate for level orthographic human Reader output.

Calibrates ONLY on train-split 2D GT foot/toe geometry. At inference, rejects
toe-tip detections that violate a conservative anatomical screen-space bound.
It reconstructs a rejected tip from the detected foot and toe-base joints
using a train-only learned linear rest-geometry transform. Produces an explicit
reliability mask and never silently claims the reconstructed tip was observed.

This P0 guard does not solve toe visibility, perspective, temporal tracking,
camera motion or general humanoid foot anatomy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToeTipGuardConfig:
    toe_name: str
    toe_base_name: str
    foot_name: str
    slot: str
    angles_deg: tuple[float, float]
    offset_matrix: tuple[tuple[float, float], tuple[float, float]]
    max_train_segment_px: float
    max_train_foot_segment_px: float
    allowed_tip_segment_px: float
    allowed_fallback_residual_px: float
    allowed_foot_segment_px: float
    train_rows: int
    fit_residual_median_px: float

    def to_dict(self):
        return {
            "toe_name": self.toe_name,
            "toe_base_name": self.toe_base_name,
            "foot_name": self.foot_name,
            "slot": self.slot,
            "angles_deg": list(self.angles_deg),
            "offset_matrix": [list(row) for row in self.offset_matrix],
            "max_train_segment_px": self.max_train_segment_px,
            "max_train_foot_segment_px": self.max_train_foot_segment_px,
            "allowed_tip_segment_px": self.allowed_tip_segment_px,
            "allowed_fallback_residual_px": self.allowed_fallback_residual_px,
            "allowed_foot_segment_px": self.allowed_foot_segment_px,
            "train_rows": self.train_rows,
            "fit_residual_median_px": self.fit_residual_median_px,
        }

    @classmethod
    def from_dict(cls, x):
        return cls(
            toe_name=x["toe_name"], toe_base_name=x["toe_base_name"],
            foot_name=x["foot_name"], slot=x["slot"],
            angles_deg=tuple(float(v) for v in x["angles_deg"]),
            offset_matrix=tuple(tuple(float(v) for v in r) for r in x["offset_matrix"]),
            max_train_segment_px=float(x["max_train_segment_px"]),
            max_train_foot_segment_px=float(x["max_train_foot_segment_px"]),
            allowed_tip_segment_px=float(x["allowed_tip_segment_px"]),
            allowed_fallback_residual_px=float(x["allowed_fallback_residual_px"]),
            allowed_foot_segment_px=float(x["allowed_foot_segment_px"]),
            train_rows=int(x["train_rows"]),
            fit_residual_median_px=float(x["fit_residual_median_px"]),
        )


def calibrate_from_train(manifest: dict) -> dict[str, ToeTipGuardConfig]:
    """Derive a separate 2x2 toe segment matrix per side/view from TRAIN rows.

    No validation or test coordinates influence its thresholds/coefficients.
    The matrices operate on projected foot-to-toe-base displacement and therefore
    can be applied on reader-detected pixels at the same camera calibration.
    """
    names = manifest["joint_names"]
    index = {n: i for i, n in enumerate(names)}
    if len(index) != len(names) or len(names) != 67:
        raise ValueError("Expected exact 67-point human fullbody marker contract")
    angles = tuple(float(v) % 360 for v in manifest["angles_deg"])
    if len(angles) != 2 or not all(math.isfinite(x) for x in angles):
        raise ValueError("Invalid configured camera pair")
    pairs = {}
    for side in ("L", "R"):
        foot = "CC_Base_" + side + "_Foot"
        base = "CC_Base_" + side + "_ToeBase"
        tip = "CC_Base_" + side + "_ToeTip"
        if not all(n in index for n in (foot, base, tip)):
            raise ValueError("Missing required foot or toe marker")
        for slot in ("front", "side"):
            rows = [r for r in manifest["samples"]
                    if r["split"] == "train" and r["slot"] == slot
                    and int(r["frame"]) not in (0, 106)]
            if len(rows) < 20:
                raise ValueError("Insufficient independent TRAIN poses to calibrate toe guard")
            if any(float(r["camera_azimuth_deg"]) % 360 != angles[0 if slot == "front" else 1] for r in rows):
                raise ValueError("Camera angle/slot mismatch in toe train calibration")
            all_uv = np.asarray([r["uv"] for r in rows], dtype=np.float64)
            if all_uv.ndim != 3 or all_uv.shape[1:] != (len(names), 2):
                raise ValueError("TRAIN keypoint UV shape mismatch")
            if not np.isfinite(all_uv).all():
                raise ValueError("Non-finite TRAIN positions")
            base_delta = all_uv[:, index[base]] - all_uv[:, index[foot]]
            tip_delta = all_uv[:, index[tip]] - all_uv[:, index[base]]
            matrix, *_ = np.linalg.lstsq(base_delta, tip_delta, rcond=None)
            if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
                raise ValueError("Invalid toe segment calibration")
            tip_len = np.linalg.norm(tip_delta, axis=-1)
            base_len = np.linalg.norm(base_delta, axis=-1)
            residual = np.linalg.norm(base_delta @ matrix - tip_delta, axis=-1)
            mx = float(np.max(tip_len))
            foot_mx = float(np.max(base_len))
            # Broad, TRAIN-defined bounds deliberately reject catastrophic
            # errors (>100px) while leaving ordinary small distortions alone.
            tip_gate = max(12.0, 2.5 * mx + 3.0)
            residual_gate = max(12.0, 2.5 * mx + 3.0)
            foot_gate = max(35.0, 2.5 * foot_mx + 10.0)
            key = side + "_" + slot
            pairs[key] = ToeTipGuardConfig(
                toe_name=tip,
                toe_base_name=base,
                foot_name=foot,
                slot=slot, angles_deg=angles,
                offset_matrix=tuple(tuple(float(v) for v in row) for row in matrix),
                max_train_segment_px=mx, max_train_foot_segment_px=foot_mx,
                allowed_tip_segment_px=tip_gate,
                allowed_fallback_residual_px=residual_gate,
                allowed_foot_segment_px=foot_gate,
                train_rows=len(rows),
                fit_residual_median_px=float(np.median(residual)),
            )
    return pairs


def guard_view(
    uv: np.ndarray, names: list[str], slot: str, calibration: dict[str, ToeTipGuardConfig],
    pair_angles_deg: tuple[float, float]
) -> tuple[np.ndarray, dict]:
    """Repair obvious toe-tip outliers, returning *explicit* repair/missing flags.

    uv may be (J,2) or (B,J,2), in original image pixel coordinates.
    Returns corrected copy plus per-tip booleans with the same leading shape.
    """
    if slot not in ("front", "side"):
        raise ValueError("Unknown camera slot")
    src = np.asarray(uv, dtype=np.float64)
    if src.ndim not in (2, 3) or src.shape[-2:] != (len(names), 2):
        raise ValueError("Reader UV shape mismatch")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate joint names")
    if not np.isfinite(src).all():
        raise ValueError("Non-finite Reader keypoints")
    angles = tuple(float(x) % 360 for x in pair_angles_deg)
    if len(angles) != 2:
        raise ValueError("Camera angles must have two entries")
    original_2d = src.ndim == 2
    work = src[None, ...].copy() if original_2d else src.copy()
    result = {
        "slot": slot, "angles_deg": list(angles),
        "tip_flags": {},
        "total_replaced": 0,
        "total_unreliable": 0,
    }
    index = {n: i for i, n in enumerate(names)}
    for side in ("L", "R"):
        cfg = calibration.get(side + "_" + slot)
        if cfg is None:
            raise ValueError("Missing train-calibrated toe guard")
        if cfg.slot != slot or any(abs((got-want+180) % 360 - 180)>1e-5 for got,want in zip(cfg.angles_deg, angles)):
            raise ValueError("Toe guard camera angle mismatch")
        t, b, f = (index[cfg.toe_name], index[cfg.toe_base_name], index[cfg.foot_name])
        delta = work[:, b] - work[:, f]
        expected = work[:, b] + delta @ np.asarray(cfg.offset_matrix, dtype=np.float64)
        observed_delta = work[:, t] - work[:, b]
        exceed_bone = np.linalg.norm(observed_delta, axis=-1) > cfg.allowed_tip_segment_px
        exceed_residual = np.linalg.norm(work[:, t] - expected, axis=-1) > cfg.allowed_fallback_residual_px
        anomalous = exceed_bone | exceed_residual
        base_unreliable = np.linalg.norm(delta, axis=-1) > cfg.allowed_foot_segment_px
        recoverable = anomalous & ~base_unreliable
        untrusted = anomalous & base_unreliable
        work[recoverable, t] = expected[recoverable]
        result["tip_flags"][cfg.toe_name] = {
            "replaced": bool(recoverable[0]) if original_2d else recoverable.tolist(),
            "untrusted_no_repair": bool(untrusted[0]) if original_2d else untrusted.tolist(),
            "observed_segment_gt_limit": bool(exceed_bone[0]) if original_2d else exceed_bone.tolist(),
            "residual_gt_limit": bool(exceed_residual[0]) if original_2d else exceed_residual.tolist(),
        }
        result["total_replaced"] += int(recoverable.sum())
        result["total_unreliable"] += int(untrusted.sum())
    return (work[0] if original_2d else work), result

@dataclass(frozen=True)
class ToeBaseGuardConfig:
    toe_base_name: str
    foot_name: str
    calf_name: str
    slot: str
    angles_deg: tuple[float, float]
    base_offset_matrix: tuple[tuple[float, float], tuple[float, float]]
    allowed_base_segment_px: float
    allowed_fallback_residual_px: float
    allowed_calf_segment_px: float
    train_rows: int
    median_fit_error_px: float

    def to_dict(self):
        return {
            "toe_base_name": self.toe_base_name,
            "foot_name": self.foot_name,
            "calf_name": self.calf_name,
            "slot": self.slot,
            "angles_deg": list(self.angles_deg),
            "base_offset_matrix": [list(x) for x in self.base_offset_matrix],
            "allowed_base_segment_px": self.allowed_base_segment_px,
            "allowed_fallback_residual_px": self.allowed_fallback_residual_px,
            "allowed_calf_segment_px": self.allowed_calf_segment_px,
            "train_rows": self.train_rows,
            "median_fit_error_px": self.median_fit_error_px,
        }

    @classmethod
    def from_dict(cls, record):
        return cls(
            toe_base_name=record["toe_base_name"],
            foot_name=record["foot_name"],
            calf_name=record["calf_name"],
            slot=record["slot"],
            angles_deg=tuple(float(x) for x in record["angles_deg"]),
            base_offset_matrix=tuple(tuple(float(x) for x in row)
                                     for row in record["base_offset_matrix"]),
            allowed_base_segment_px=float(record["allowed_base_segment_px"]),
            allowed_fallback_residual_px=float(record["allowed_fallback_residual_px"]),
            allowed_calf_segment_px=float(record["allowed_calf_segment_px"]),
            train_rows=int(record["train_rows"]),
            median_fit_error_px=float(record["median_fit_error_px"]),
        )


def calibrate_base_from_train(manifest: dict) -> dict[str, ToeBaseGuardConfig]:
    """Fit calf→foot→toe-base projected geometry on Train only."""
    names = manifest["joint_names"]
    indices = {n: i for i, n in enumerate(names)}
    if len(indices) != len(names) or len(names) != 67:
        raise ValueError("Requires the 67-point human rig schema")
    angles = tuple(float(a) % 360 for a in manifest["angles_deg"])
    result = {}
    for side in ("L", "R"):
        calf = "CC_Base_" + side + "_Calf"
        foot = "CC_Base_" + side + "_Foot"
        base = "CC_Base_" + side + "_ToeBase"
        for slot in ("front", "side"):
            rows = [row for row in manifest["samples"]
                    if row["split"] == "train" and row["slot"] == slot
                    and int(row["frame"]) not in (0, 106)]
            if len(rows) < 20:
                raise ValueError("Not enough Train foot observations")
            if any(float(row["camera_azimuth_deg"]) % 360 != angles[0 if slot=="front" else 1] for row in rows):
                raise ValueError("Camera calibration mismatch in Train foot observations")
            all_uv = np.asarray([row["uv"] for row in rows], dtype=np.float64)
            if all_uv.shape != (len(rows), len(names), 2) or not np.isfinite(all_uv).all():
                raise ValueError("Invalid Train foot marker coordinates")
            v_calf = all_uv[:, indices[foot]] - all_uv[:, indices[calf]]
            v_base = all_uv[:, indices[base]] - all_uv[:, indices[foot]]
            matrix = np.linalg.lstsq(v_calf, v_base, rcond=None)[0]
            residual = np.linalg.norm(v_calf @ matrix - v_base, axis=-1)
            result[side + "_" + slot] = ToeBaseGuardConfig(
                toe_base_name=base, foot_name=foot, calf_name=calf,
                slot=slot, angles_deg=angles,
                base_offset_matrix=tuple(tuple(float(v) for v in row) for row in matrix),
                allowed_base_segment_px=max(25.0, 2.2 * float(np.max(np.linalg.norm(v_base, axis=-1))) + 5),
                allowed_fallback_residual_px=max(15.0, 2.5 * float(np.percentile(residual, 99)) + 5),
                allowed_calf_segment_px=max(100.0, 1.8 * float(np.max(np.linalg.norm(v_calf, axis=-1))) + 10),
                train_rows=len(rows), median_fit_error_px=float(np.median(residual)),
            )
    return result


def guard_foot_chain(
    uv: np.ndarray, names: list[str], slot: str,
    base_calibration: dict[str, ToeBaseGuardConfig],
    tip_calibration: dict[str, ToeTipGuardConfig],
    pair_angles_deg: tuple[float, float]
) -> tuple[np.ndarray, dict]:
    """Validate calf/foot/toe-base first; recover tip only from plausible anchors.

    Returns corrected coordinates plus diagnostic flags. Any unrecoverable
    observation is explicitly flagged, not silently interpolated.
    """
    points = np.asarray(uv, dtype=np.float64)
    unbatched = points.ndim == 2
    if points.ndim not in (2, 3) or points.shape[-2:] != (len(names), 2):
        raise ValueError("Invalid fullbody Reader UV dimensions")
    arr = points[None, ...].copy() if unbatched else points.copy()
    if not np.isfinite(arr).all() or len(set(names)) != len(names):
        raise ValueError("Invalid Reader positions or marker names")
    angles = tuple(float(x) % 360 for x in pair_angles_deg)
    index = {name: i for i, name in enumerate(names)}
    base_diag = {}
    unreliable_upstream = {}
    fixed = 0
    failed = 0
    for side in ("L", "R"):
        cfg = base_calibration.get(side + "_" + slot)
        if cfg is None or cfg.slot != slot or cfg.angles_deg != angles:
            raise ValueError("Missing or mismatched toe-base guard camera angle / slot calibration")
        j_calf = index[cfg.calf_name]
        j_foot = index[cfg.foot_name]
        j_base = index[cfg.toe_base_name]
        ankle_vec = arr[:, j_foot] - arr[:, j_calf]
        expected = arr[:, j_foot] + ankle_vec @ np.asarray(cfg.base_offset_matrix)
        bone_vec = arr[:, j_base] - arr[:, j_foot]
        bad = ((np.linalg.norm(bone_vec, axis=-1) > cfg.allowed_base_segment_px) |
               (np.linalg.norm(arr[:, j_base] - expected, axis=-1) > cfg.allowed_fallback_residual_px))
        unusable = (np.linalg.norm(ankle_vec, axis=-1) > cfg.allowed_calf_segment_px)
        repair = bad & ~unusable
        fail = bad & unusable
        arr[repair, j_base] = expected[repair]
        fixed += int(repair.sum())
        failed += int(fail.sum())
        unreliable_upstream[side] = fail
        base_diag[cfg.toe_base_name] = {
            "repaired": bool(repair[0]) if unbatched else repair.tolist(),
            "untrusted_no_repair": bool(fail[0]) if unbatched else fail.tolist(),
        }
    corrected, tip_diag = guard_view(arr[0] if unbatched else arr, names, slot,
                                    tip_calibration, angles)
    # A broken Foot/Calf anchor invalidates all downstream toe inference,
    # even when ToeTip stays close to the equally incorrect ToeBase.
    total_tip_untrusted = 0
    for side in ("L", "R"):
        name = "CC_Base_" + side + "_ToeTip"
        mask = np.asarray(tip_diag["tip_flags"][name]["untrusted_no_repair"], dtype=bool)
        combined = np.atleast_1d(mask) | unreliable_upstream[side]
        tip_diag["tip_flags"][name]["untrusted_no_repair"] = (bool(combined[0]) if unbatched else combined.tolist())
        total_tip_untrusted += int(combined.sum())
    result = {
        "slot": slot, "angles_deg": list(angles),
        "toe_base_flags": base_diag,
        "toe_tip_flags": tip_diag["tip_flags"],
        "base_repaired": fixed,
        "base_untrusted": failed,
        "tip_repaired": tip_diag["total_replaced"],
        "tip_untrusted": total_tip_untrusted,
    }
    return corrected, result

def confidence_scales_from_diagnostics(
    names: list[str], diagnostics: dict, inferred_scale: float = 0.25
) -> np.ndarray:
    """Conservative reliability modifiers for a downstream weighted Solver.

    Uncorrected joints get 1.0. A model-derived replacement is *not* a
    direct observation and receives inferred_scale. Unrecoverable chain
    coordinates receive 0.0 and must not be used as image evidence.
    The default inferred_scale is a cautious heuristic, NOT a calibrated
    probability; consumers may supply a validation-calibrated value.
    """
    weight = float(inferred_scale)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("inferred_scale must be in [0,1]")
    index = {n: i for i, n in enumerate(names)}
    if len(index) != len(names):
        raise ValueError("Duplicate joint names")
    all_flags = {}
    for key in ("toe_base_flags", "toe_tip_flags"):
        all_flags.update(diagnostics.get(key, {}))
    if not all_flags:
        raise ValueError("No toe-chain reliability flags")
    first = next(iter(all_flags.values()))
    first_rep = first.get("repaired", first.get("replaced", False))
    batch_count = len(first_rep) if isinstance(first_rep, list) else 1
    scales = np.ones((batch_count, len(names)), dtype=np.float64)
    for name, flags in all_flags.items():
        if name not in index:
            raise ValueError("Reliability flags reference an unknown marker: "+name)
        replaced = np.atleast_1d(np.asarray(flags.get("repaired", flags.get("replaced", False)), dtype=bool))
        missing = np.atleast_1d(np.asarray(flags.get("untrusted_no_repair", False), dtype=bool))
        if replaced.shape != (batch_count,) or missing.shape != (batch_count,):
            raise ValueError("Inconsistent batch size in toe-chain flags")
        scales[replaced, index[name]] = np.minimum(scales[replaced, index[name]], weight)
        scales[missing, index[name]] = 0.0
    return scales[0] if batch_count == 1 and not isinstance(first_rep, list) else scales
