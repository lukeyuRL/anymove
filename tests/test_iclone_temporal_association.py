"""Synthetic temporal association regressions for Dragon (NumPy-only).

The tests deliberately do not claim that an AI clip's true side labeling is
known. They verify paired five-joint association, temporal stability,
confidence propagation, root invariance and safe abstention.
"""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.temporal_association import (
        STEMS, PROFILE, temporal_hind_association,
        root_relative_paw_correlations, MIN_SWAPPED_SEGMENT_FRAMES
    )


@unittest.skipUnless(np is not None, "NumPy not available")
class TemporalAssociationTests(unittest.TestCase):
    def fixture(self, T=100):
        names = ["hips", "chest", "head", "fpaw_L", "fpaw_R"] + [
            f"{stem}_{s}" for s in ("L", "R") for stem in STEMS
        ] + ["wing_L", "wing_R"]
        index = {n: k for k, n in enumerate(names)}
        times = np.arange(T, dtype=np.float64)
        root = 320. + 40. * np.sin(times * .07)
        root_side = root + 11. * np.cos(times * .06)
        front = np.zeros((T, len(names), 2), np.float32)
        side = np.zeros_like(front)
        for j, name in enumerate(names):
            front[:, j, 0] = 200 + j * 4
            side[:, j, 0] = 180 + j * 5
            front[:, j, 1] = root + 6 + j * .9
            side[:, j, 1] = root_side + 6 + j * .9
        front[:, index["hips"], 1] = root
        side[:, index["hips"], 1] = root_side
        for s, sign in (("L", -1), ("R", 1)):
            for k, stem in enumerate(STEMS):
                name = f"{stem}_{s}"
                j = index[name]
                # Different relative paw cycles, distinguish left and right.
                # Deliberately strong root motion so root subtraction matters.
                relative = 19 + k * 7 + 21 * np.sin(.19 * times +
                             (0.3 if s == "L" else 2.55) + .11*k)
                front[:, j, 1] = root + relative
                side[:, j, 1] = root_side + relative + .3*np.sin(times*.37+k)
                front[:, j, 0] = 160 + 40 * sign + k * 3
                side[:, j, 0] = 180 + 10 * sign + k * 6
        data = {
            "front_uv": front, "side_uv": side,
            "front_conf": np.full(front.shape[:2], .93, np.float32),
            "side_conf": np.full(front.shape[:2], .91, np.float32),
        }
        return data, names

    def swap_side(self, d, names, begin, end, stems=None):
        if stems is None:
            stems = STEMS
        ix = {n: i for i, n in enumerate(names)}
        for stem in stems:
            L, R = ix[f"{stem}_L"], ix[f"{stem}_R"]
            for key in ("side_uv", "side_conf"):
                a = d[key]
                saved = a[begin:end, L].copy()
                a[begin:end, L] = a[begin:end, R].copy()
                a[begin:end, R] = saved

    def test_clean_tracks_preserved_without_paw_swaps(self):
        data, names = self.fixture()
        original = {k: a.copy() for k, a in data.items()}
        corrected, report = temporal_hind_association(data, names)
        self.assertEqual(report["status"], "identity_retained")
        self.assertEqual(report["accepted_swapped_frames"], 0)
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])
            np.testing.assert_array_equal(corrected[key], original[key])

    def test_entire_clip_pair_swap_restored(self):
        d, names = self.fixture()
        unswapped = {k: a.copy() for k, a in d.items()}
        self.swap_side(d, names, 0, 100)
        raw = {k: a.copy() for k, a in d.items()}
        corrected, report = temporal_hind_association(d, names)
        self.assertEqual(report["accepted_swapped_frames"], 100)
        self.assertEqual(report["accepted_segments"][0]["frame_count"], 100)
        self.assertEqual(report["profile"], PROFILE)
        for key in raw:
            np.testing.assert_array_equal(d[key], raw[key])
            np.testing.assert_allclose(corrected[key], unswapped[key], atol=0)
        self.assertLess(report["after_relative_hind_height_mae_px"],
                        report["before_relative_hind_height_mae_px"])

    def test_middle_interval_only_and_trajectory_stable(self):
        d, names = self.fixture()
        baseline = {k: a.copy() for k, a in d.items()}
        self.swap_side(d, names, 28, 74)
        fixed, report = temporal_hind_association(d, names)
        states = report["state_by_frame"]
        self.assertTrue(all(s == 1 for s in states[36:65]))
        self.assertTrue(all(s == 0 for s in states[:15]+states[85:]))
        self.assertLessEqual(len(report["accepted_segments"]), 2)
        # The 2D tracks should be closer to the clean side trajectories.
        j = names.index("hpaw_L")
        before = np.mean(np.abs(d["side_uv"][:, j, 1] -
                                baseline["side_uv"][:, j, 1]))
        after = np.mean(np.abs(fixed["side_uv"][:, j, 1] -
                               baseline["side_uv"][:, j, 1]))
        self.assertLess(after, before)

    def test_short_or_isolated_swap_cannot_flicker(self):
        d, names = self.fixture()
        self.swap_side(d, names, 38, 38+MIN_SWAPPED_SEGMENT_FRAMES-2)
        _, report = temporal_hind_association(d, names)
        self.assertEqual(report["accepted_swapped_frames"], 0)

    def test_one_bad_paw_does_not_swap_whole_leg(self):
        d, names = self.fixture()
        original = d["side_uv"].copy()
        self.swap_side(d, names, 20, 75, stems=("hpaw",))
        _, report = temporal_hind_association(d, names)
        self.assertEqual(report["accepted_swapped_frames"], 0)
        self.assertNotEqual(float(original[:, names.index("hpaw_L"), 1].sum()),
                            float(d["side_uv"][:, names.index("hpaw_L"), 1].sum()))

    def test_low_confidence_only_cannot_force_swapping(self):
        d, names = self.fixture()
        self.swap_side(d, names, 20, 78)
        for k in ("front_conf", "side_conf"):
            d[k][:] = .005
        _, report = temporal_hind_association(d, names)
        self.assertEqual(report["accepted_swapped_frames"], 0)

    def test_rest_of_skeleton_and_front_view_never_changed(self):
        d, names = self.fixture()
        self.swap_side(d, names, 12, 90)
        original = {k: a.copy() for k, a in d.items()}
        fixed, report = temporal_hind_association(d, names)
        self.assertTrue(report["accepted_swapped_frames"] > 0)
        for key in ("front_uv", "front_conf"):
            np.testing.assert_array_equal(fixed[key], original[key])
        for name in ("hips", "chest", "head", "fpaw_L", "fpaw_R", "wing_L", "wing_R"):
            j = names.index(name)
            np.testing.assert_array_equal(fixed["side_uv"][:, j],
                                          original["side_uv"][:, j])
            np.testing.assert_array_equal(fixed["side_conf"][:, j],
                                          original["side_conf"][:, j])

    def test_invalid_data_fails_closed(self):
        d, names = self.fixture()
        with self.assertRaisesRegex(ValueError, "required joints"):
            temporal_hind_association(d, [n for n in names if n != "hpaw_R"])
        with self.assertRaisesRegex(ValueError, "unique"):
            temporal_hind_association(d, names[:-1]+[names[-2]])
        broken = dict(d)
        broken["side_uv"] = d["side_uv"].copy()
        broken["side_uv"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            temporal_hind_association(broken, names)

    def test_real_repo_example_npz_smoke_without_model_training(self):
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        sample = repo / "examples" / "dragon_roar" / "det.npz"
        if not sample.is_file():
            self.skipTest("original example detections are absent")
        with np.load(str(sample), allow_pickle=False) as z:
            data = {k: z[k].copy()
                    for k in ("front_uv", "side_uv", "front_conf", "side_conf")}
            names = [str(x) for x in z["names"].tolist()]
        prior = {k: v.copy() for k, v in data.items()}
        fixed, report = temporal_hind_association(data, names)
        self.assertEqual(fixed["side_uv"].shape, data["side_uv"].shape)
        self.assertEqual(report["num_frames"] if "num_frames" in report
                         else len(report["state_by_frame"]), len(data["front_uv"]))
        self.assertGreater(len(data["front_uv"]), 20)
        self.assertTrue(np.isfinite(fixed["side_uv"]).all())
        self.assertTrue(np.isfinite(fixed["side_conf"]).all())
        for key, value in prior.items():
            np.testing.assert_array_equal(data[key], value)
        print("EXAMPLE_ASSN_REPORT",
              {"frames": len(data["front_uv"]),
               "swapped": report["accepted_swapped_frames"],
               "segments": report["accepted_segments"],
               "raw_mae": report["before_relative_hind_height_mae_px"],
               "new_mae": report["after_relative_hind_height_mae_px"]})

    def test_root_relative_paw_report_uses_final_applied_tracks(self):
        # Reproduces the real v3 diagnostic bug: a candidate is rejected,
        # but the final 'after' metric incorrectly described the candidate.
        baseline, names = self.fixture(124)
        original = {key: val.copy() for key, val in baseline.items()}
        ref = root_relative_paw_correlations(baseline, names)
        variant = {key: val.copy() for key, val in baseline.items()}
        self.swap_side(variant, names, 42, 53)
        candidate = root_relative_paw_correlations(variant, names)
        self.assertNotEqual(ref, candidate)
        # The continuity gate rejected all swaps, so finalized tracks are
        # the untouched baseline; candidate metrics remain audit-only.
        actual = root_relative_paw_correlations(baseline, names)
        self.assertEqual(actual, ref)
        self.assertNotEqual(actual, candidate)
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], original[key])

    def test_audit_json_and_state_contract(self):
        import json
        d, names = self.fixture(124)
        self.swap_side(d, names, 10, 90)
        fixed, report = temporal_hind_association(d, names)
        encoded = json.dumps(report, allow_nan=False)
        self.assertEqual(len(json.loads(encoded)["state_by_frame"]), 124)
        self.assertEqual(fixed["side_uv"].shape, (124, len(names), 2))
        self.assertTrue(all(s in (0, 1) for s in report["state_by_frame"]))
        self.assertEqual(sum(report["state_by_frame"]),
                         report["accepted_swapped_frames"])


if __name__ == "__main__":
    unittest.main()
