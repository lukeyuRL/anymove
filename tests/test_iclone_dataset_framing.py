"""Dataset framing policy tests, no Blender or torch required."""
import copy
import random
import unittest

from blender.dataset_framing import (
    choose_yaw_degrees, framing_signature, same_framing, validate_yaw_limit,
)


class FramingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.layout = {"H":720, "W_FRONT":400, "W_SIDE":880,
                       "M_PER_PX":.025, "GROUND_V":630, "Z_CENTER":6.75}
        self.rig = {"bones":[{"name":"root","limits_deg":[[-20,20]]*3}],
                    "keypoints":[{"name":"hip","rest":[0,0,1],"bone":0,"end":"head"}]}

    def test_capped_yaw_never_reaches_extreme_70_deg(self):
        yaws = [choose_yaw_degrees(random.Random(500+i), 10)
                for i in range(1000)]
        self.assertLessEqual(max(map(abs,yaws)), 10.0)
        self.assertTrue(any(v == 0 for v in yaws))

    def test_legacy_yaw_keeps_original_extremes(self):
        values = [choose_yaw_degrees(random.Random(500+i))
                  for i in range(1000)]
        self.assertTrue(any(abs(v) > 30 for v in values))
        self.assertLessEqual(max(map(abs, values)), 70)

    def test_invalid_yaw_values_rejected(self):
        for bad in (-1, 71, True, float("nan"), float("inf"), "10"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_yaw_limit(bad)

    def test_framing_signature_requires_exact_camera_and_rig(self):
        signature = framing_signature(self.rig, self.layout, 10)
        self.assertTrue(same_framing({"framing_signature": signature}, signature))
        self.assertFalse(same_framing({}, signature))
        changed = dict(self.layout, W_FRONT=420, W_SIDE=860)
        self.assertNotEqual(signature, framing_signature(self.rig, changed, 10))
        self.assertNotEqual(signature, framing_signature(self.rig, self.layout, 12))
        different_rig = copy.deepcopy(self.rig)
        different_rig["keypoints"][0]["rest"][2] = 1.1
        self.assertNotEqual(signature, framing_signature(different_rig, self.layout, 10))
        self.assertNotEqual(signature, framing_signature(self.rig, self.layout, None))


if __name__ == "__main__":
    unittest.main()
