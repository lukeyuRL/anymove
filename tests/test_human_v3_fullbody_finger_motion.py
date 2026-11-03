"""Pure-python regression for V3 fullbody/finger procedural GT contracts."""
import math
import unittest

from tools.human_v3_fullbody_motion import (
    FAMILIES,BASE,FINGER_NAMES,pose_changes,finger_tip_names,toe_tip_names
)

class HumanV3FullBodyFingerMotionTests(unittest.TestCase):
    def test_endpoint_pose_is_identical_at_start_and_end(self):
        for family in FAMILIES:
            for frame in (0,106):
                delta,root=pose_changes(family,4001,frame)
                self.assertLess(max(abs(x) for triple in delta.values() for x in triple),1e-10)
                self.assertLess(max(abs(x) for x in root),1e-10)
    def test_every_family_has_active_legs_arms_and_all_finger_bones(self):
        for family in FAMILIES:
            sweep={}
            for f in range(107):
                change,root=pose_changes(family,5000+len(family),f)
                self.assertTrue(all(math.isfinite(x) for vec in change.values() for x in vec))
                for side in ("L","R"):
                    for digit in FINGER_NAMES:
                        for segment in (1,2,3):
                            name="CC_Base_"+side+"_"+digit+str(segment)
                            self.assertIn(name,change)
                            sweep.setdefault(name,[]).append(change[name][0])
                for side in ("L","R"):
                    for part in ("Thigh","Calf","Foot","Upperarm","Forearm","Hand"):
                        self.assertIn("CC_Base_"+side+"_"+part,change)
            for name,x in sweep.items():
                with self.subTest(family=family,joint=name):
                    self.assertGreater(max(x)-min(x),.01)
    def test_marker_names_are_unique_ten_finger_and_two_toe_tips(self):
        finger=finger_tip_names()
        toe=toe_tip_names()
        self.assertEqual(len(finger),10)
        self.assertEqual(len(toe),2)
        self.assertEqual(len(set(finger+toe)),12)
        self.assertEqual(len(FAMILIES),8)
        self.assertEqual(len(BASE),2)

if __name__=="__main__":
    unittest.main()
