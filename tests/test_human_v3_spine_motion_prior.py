"""SO(3) angular prior correctness and train-motion diversity contracts."""
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mocap.so3_motion_prior import (
    relative_rotvec,angular_motion,excess_angular_motion_loss,
)
from mocap.solve import euler_xyz
from tools import human_v3_spine_motion_prior_v1 as source


def synthetic_profile():
    return {
        "schema":"anymove.human_v3_so3_prior/1",
        "spine_bone_names":["CC_Base_Waist"],
        "bone_velocity_threshold_rad_per_frame":[.055],
        "bone_acc_threshold_rad_per_frame2":[.015],
        "root_velocity_threshold_rad_per_frame":.07,
        "root_acc_threshold_rad_per_frame2":.015,
        "bone_axis_lower_rad":[[-.45,-.45,-.45]],
        "bone_axis_upper_rad":[[.45,.45,.45]],
        "weights":{
            "bone_velocity":120.0,"bone_acceleration":800.0,
            "root_velocity":60.0,"root_acceleration":1000.0,
            "bone_axis_range":150.0,
        },
    }


class SO3MotionPriorContracts(unittest.TestCase):
    def test_identity_and_linear_turn_do_not_generate_false_acceleration(self):
        t=np.arange(21,dtype=float)
        R=torch.tensor(Rotation.from_euler("z",(t*.035)[:,None]).as_matrix(),dtype=torch.float64)
        v,a=angular_motion(R,torch.ones(20,dtype=torch.float64))
        self.assertEqual(tuple(v.shape),(20,3))
        self.assertEqual(tuple(a.shape),(19,3))
        torch.testing.assert_close(v[:,2],torch.full((20,),.035,dtype=torch.float64),atol=1e-10,rtol=0)
        self.assertLess(float(a.abs().max()),1e-10)
        identity=torch.eye(3,dtype=torch.float64).repeat(21,1,1)
        self.assertLess(float(relative_rotvec(identity).abs().max()),1e-10)

    def test_nonunit_frame_intervals_preserve_angular_acceleration(self):
        frames=np.array([8,13,22,34,55],dtype=float)
        a=Rotation.from_euler("z",(.008*frames)[:,None]).as_matrix()
        R=torch.as_tensor(a,dtype=torch.float64)
        vel,acc=angular_motion(R,torch.as_tensor(np.diff(frames),dtype=torch.float64))
        torch.testing.assert_close(vel[:,2],torch.full((4,),.008,dtype=torch.float64),atol=1e-11,rtol=0)
        self.assertLess(float(acc.abs().max()),1e-10)

    def test_unpenalized_large_constant_root_heading_turn(self):
        T=23
        times=torch.arange(T,dtype=torch.float64)
        angles=torch.zeros(T,1,3,dtype=torch.float64)
        matrices=euler_xyz(angles.reshape(-1,3)).reshape(T,1,3,3)
        root=torch.as_tensor(Rotation.from_euler("z",(times.numpy()*.046)[:,None]).as_matrix(),dtype=torch.float64)
        profile=synthetic_profile()
        score,terms=excess_angular_motion_loss(
            matrices,root,angles,torch.ones(T-1,dtype=torch.float64),profile)
        self.assertLess(float(score.abs()),1e-10)
        self.assertLess(float(terms["root_ang_vel"].abs()),1e-10)

    def test_spine_jitter_has_penalty_and_backpropagates(self):
        T=24
        local=torch.zeros(T,1,3,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():
            local[:,0,1]=torch.tensor([.20 if i%2 else -.20 for i in range(T)],dtype=torch.float64)
        R=euler_xyz(local.reshape(-1,3)).reshape(T,1,3,3)
        root=torch.eye(3,dtype=torch.float64).repeat(T,1,1)
        loss,terms=excess_angular_motion_loss(R,root,local,torch.ones(T-1,dtype=torch.float64),synthetic_profile())
        self.assertGreater(float(loss.detach()),100)
        self.assertGreater(float(terms["bone_ang_acc"].detach()),10)
        loss.backward()
        self.assertTrue(torch.isfinite(local.grad).all())
        self.assertGreater(float(local.grad.abs().max()),1e-4)

    def test_invalid_schema_and_degenerate_time_are_rejected(self):
        T=3
        angles=torch.zeros(T,1,3,dtype=torch.float64)
        rot=torch.eye(3,dtype=torch.float64).repeat(T,1,1,1)
        root=torch.eye(3,dtype=torch.float64).repeat(T,1,1)
        with self.assertRaises(ValueError):
            excess_angular_motion_loss(rot,root,angles,torch.ones(2),{"schema":"unknown"})
        with self.assertRaisesRegex(ValueError,"Strictly positive"):
            angular_motion(rot,torch.tensor([1.,0.]))

    def test_spine_motion_gt_family_and_clip_disjointness(self):
        defs=source.all_unique_clips()
        self.assertEqual(len(defs),144)
        self.assertEqual(len(set(d["clip"] for d in defs)),144)
        self.assertEqual(len(set(d["seed"] for d in defs)),144)
        for split,count in (("train",96),("validation",24),("test",24)):
            self.assertEqual(sum(d["split"]==split for d in defs),count)
        self.assertEqual(len(source.FAMILIES),12)

    def test_all_motion_families_turn_root_and_animate_fingers(self):
        for i,family in enumerate(source.FAMILIES):
            with self.subTest(family=family):
                frames=[0,26,53,80,106]
                seq=[source.pose_changes(family,400000+2003*i,frame) for frame in frames]
                yaw=[r[2][2] for r in seq]
                self.assertGreater(np.rad2deg(np.ptp(yaw)),4.0)
                fingers=[key for key in seq[2][0] if any(token in key for token in ("Index","Thumb","Pinky","Ring","Mid"))]
                self.assertGreaterEqual(len(fingers),30)
                self.assertIn("CC_Base_Waist",seq[2][0])
                self.assertTrue(np.isfinite(np.asarray(yaw)).all())


if __name__=="__main__":
    unittest.main()
