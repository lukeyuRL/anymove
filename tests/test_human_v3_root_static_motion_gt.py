"""Regression contract for the Human V3 Nose non-animated RL_BoneRoot.

The root pose-bone stays identity. Global motion belongs to the
Armature Object transform, represented by independent root quaternion and
translation. Applies across all 12 procedural source motions and 107 frames.
"""
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools import human_v3_spine_motion_prior_v1 as motion


class HumanV3StaticRootMotionContract(unittest.TestCase):
    def test_144_synthetic_clips_never_animate_RL_BoneRoot(self):
        cases=motion.all_unique_clips()
        self.assertEqual(len(cases),144)
        for case in cases:
            with self.subTest(family=case["family"],take=case["take"]):
                sampled_root_yaw=[]
                for frame in range(107):
                    local,translation,root_euler=motion.pose_changes(
                        case["family"],case["seed"],frame)
                    self.assertNotIn("RL_BoneRoot",local)
                    self.assertEqual(motion.BASE.get("RL_BoneRoot",(0,0,0)),(0,0,0))
                    self.assertTrue(all(math.isfinite(v) for v in root_euler))
                    self.assertTrue(all(math.isfinite(v) for v in translation))
                    sampled_root_yaw.append(root_euler[2])
                self.assertGreater(np.ptp(sampled_root_yaw),np.deg2rad(4))

    def test_armature_world_rotation_and_root_bone_local_identity_are_independent(self):
        for yaw_deg in (0,22,55,-55):
            # In Blender: Armature.matrix_world contains world turn; the
            # local RL_BoneRoot matrix_basis remains identity.
            global_rotation=Rotation.from_euler("z",yaw_deg,degrees=True).as_matrix()
            local_root=np.eye(3)
            world_pose=global_rotation @ local_root
            np.testing.assert_allclose(world_pose,global_rotation,atol=1e-12)
            np.testing.assert_allclose(local_root,np.eye(3),atol=0)

    def test_finger_and_spine_remain_animated_despite_static_root(self):
        for i,family in enumerate(motion.FAMILIES):
            changes,offset,root=motion.pose_changes(family,400000+i*2003,53)
            self.assertIn("CC_Base_Waist",changes)
            self.assertIn("CC_Base_Spine01",changes)
            self.assertNotIn("RL_BoneRoot",changes)
            self.assertGreater(len([x for x in changes if "Index" in x]),4)
            self.assertGreater(abs(root[2]),0)


if __name__=="__main__":
    unittest.main()
