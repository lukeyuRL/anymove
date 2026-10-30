"""Regression tests for the high-res Blender Dataset blur policy.

No Blender, NumPy, or PyTorch required for CI.
"""
import random
import unittest

from blender.dataset_blur import (
    BLUR_PROFILE, BLUR_PROBABILITY, SHUTTER_MIN, SHUTTER_MAX,
    choose_blur, profile_matches, scale_matches,
)


class BlurProfileTests(unittest.TestCase):
    def test_new_policy_matches_requested_settings(self):
        self.assertEqual(BLUR_PROBABILITY, .20)
        self.assertEqual((SHUTTER_MIN, SHUTTER_MAX), (.20, .40))
        self.assertTrue(BLUR_PROFILE.startswith("soft_blur_"))

    def test_all_blurred_shutters_are_light_and_clear_samples_have_none(self):
        states = [choose_blur(random.Random(500 + i)) for i in range(2000)]
        blurred = [(on, shutter) for on, shutter in states if on]
        self.assertTrue(0.17 <= len(blurred) / len(states) <= 0.23)
        self.assertTrue(all(SHUTTER_MIN <= shutter <= SHUTTER_MAX
                            for on, shutter in blurred))
        self.assertTrue(all(shutter is None for on, shutter in states if not on))

    def test_per_index_randomness_is_stable_across_resumes(self):
        a = [choose_blur(random.Random(500 + i)) for i in range(50)]
        b = {i: choose_blur(random.Random(500 + i))
             for i in range(49, -1, -1) if i % 3 == 0}
        self.assertEqual({i: a[i] for i in b}, b)

    def test_old_metadata_does_not_silently_resume(self):
        self.assertFalse(profile_matches({"scale": 1.0, "blur": True}))
        self.assertFalse(profile_matches({"blur_profile": "old_50pct_strong"}))
        self.assertTrue(profile_matches({"blur_profile": BLUR_PROFILE}))

    def test_scale_guard_preserves_existing_contract(self):
        self.assertTrue(scale_matches({"scale": 1.0}, 1.0))
        self.assertFalse(scale_matches({"scale": 0.6}, 1.0))
        self.assertFalse(scale_matches({"scale": float("nan")}, 1.0))
        self.assertFalse(scale_matches({}, 1.0))


if __name__ == "__main__":
    unittest.main()
