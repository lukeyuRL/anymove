"""Unit tests for balanced Human V3 Nose HR Dataset V2 procedural GT.

These do not render, train, or require Blender.
"""
import math
import unittest

import numpy as np
from tools import human_v3_fullbody_motion_v2 as v2

class FullBodyV2Contracts(unittest.TestCase):
    def test_twelve_motion_families_are_distinct(self):
        self.assertEqual(len(v2.FAMILIES),12)
        self.assertEqual(len(set(v2.FAMILIES)),12)
        self.assertEqual(len(v2.finger_tip_names()),10)
        self.assertEqual(len(v2.toe_tip_names()),2)

    def test_no_cross_split_seed_leakage(self):
        rows=[]
        for family_i,family in enumerate(v2.FAMILIES):
            for j in range(16):
                split="train" if j<12 else ("validation" if j<14 else "test")
                seed=72000+family_i*1901+j*791
                rows.append((family,split,seed))
        self.assertEqual(len(rows),192)
        self.assertEqual(len({x[2] for x in rows}),192)
        self.assertEqual(sum(x[1]=="train" for x in rows),144)
        self.assertEqual(sum(x[1]=="validation" for x in rows),24)
        self.assertEqual(sum(x[1]=="test" for x in rows),24)

    def test_each_family_has_leg_hand_finger_motion(self):
        for family in v2.FAMILIES:
            with self.subTest(family=family):
                seed=72000+v2.FAMILIES.index(family)*1901
                trajectories={}
                for frame in range(107):
                    changes,root=v2.pose_changes(family,seed,frame)
                    for side in ("L","R"):
                        for part in ("Thigh","Calf","Foot","ToeBase","Upperarm","Forearm","Hand"):
                            self.assertIn("CC_Base_"+side+"_"+part,changes)
                        for digit in v2.FINGER_NAMES:
                            for segment in (1,2,3):
                                joint="CC_Base_"+side+"_"+digit+str(segment)
                                self.assertIn(joint,changes)
                                trajectories.setdefault(joint,[]).append(changes[joint][0])
                    self.assertTrue(all(math.isfinite(x) for p in changes.values() for x in p))
                    self.assertTrue(all(math.isfinite(x) for x in root))
                self.assertEqual(len(trajectories),30)
                self.assertGreater(min(np.ptp(angles) for angles in trajectories.values()),0.01)

    def test_nonrest_keyframes_unique(self):
        frames=np.linspace(8,98,14).round().astype(int)
        self.assertEqual(len(frames),14)
        self.assertEqual(len(set(frames)),14)
        self.assertGreater(min(frames),0)
        self.assertLess(max(frames),106)

    def test_input_motion_starts_and_ends_at_same_pose(self):
        for fam in v2.FAMILIES:
            seed=72000+v2.FAMILIES.index(fam)*1901
            for frame in (0,106):
                changes,root=v2.pose_changes(fam,seed,frame)
                self.assertLess(max(abs(x) for vals in changes.values() for x in vals),1e-9)
                self.assertLess(max(abs(x) for x in root),1e-9)

if __name__=="__main__":
    unittest.main()
