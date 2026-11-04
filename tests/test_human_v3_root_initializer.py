"""Root Kabsch and SO(3) Human V3 orientation recovery contracts."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from mocap.human_root_initializer import (
    ANCHORS,_orthonormal_factor,estimate_human_v3_root_axis_angle
)


class HumanRootInitializerTests(unittest.TestCase):
    def setUp(self):
        self.names=list(ANCHORS)+["Extra_%02d"%i for i in range(63)]
        self.rest=np.zeros((67,3))
        self.rest[:4]=[
            [0.0,0.0,.945],
            [0.0,-.018,1.022],
            [.082,-.004,.933],
            [-.082,-.004,.933],
        ]
        self.rest[4:]=np.random.default_rng(777).normal(size=(63,3))*.1
        self.frames=17

    def _world_pose(self,rotation,translate=(.09,-.03,.015)):
        # Use synthetic known world-space root, no GT required by estimator.
        pts=(rotation@self.rest.T).T+np.asarray(translate)
        return np.tile(pts[None],(self.frames,1,1))

    def test_kabsch_recovers_root_yaw_and_translation_invariant(self):
        world=Rotation.from_euler("z",47,degrees=True).as_matrix()
        xyz=self._world_pose(world)
        out=estimate_human_v3_root_axis_angle(
            xyz,self.rest,self.names,np.eye(3),smoothing_window=3)
        actual=Rotation.from_rotvec(out).as_matrix()
        error=(Rotation.from_matrix(world).inv()*Rotation.from_matrix(actual)).magnitude()
        self.assertLess(np.rad2deg(error.max()),.00001)
        self.assertEqual(tuple(out.shape),(17,3))

    def test_local_root_conjugacy_with_nonidentity_rest_basis(self):
        Rbasis=Rotation.from_euler("xyz",[.18,.13,-.2]).as_matrix()
        world=Rotation.from_euler("xyz",[.1,.04,.5]).as_matrix()
        observed=self._world_pose(world)
        got=estimate_human_v3_root_axis_angle(
            observed,self.rest,self.names,Rbasis,smoothing_window=1)
        expected=Rbasis.T@world@Rbasis
        err=(Rotation.from_matrix(expected).inv()*
             Rotation.from_rotvec(got)).magnitude()
        self.assertLess(np.rad2deg(err.max()),.00001)

    def test_known_noise_recovers_within_five_degrees(self):
        theta=np.linspace(0,np.deg2rad(55),self.frames)
        xyz=np.stack([self._world_pose(Rotation.from_euler("z",angle).as_matrix())[0]
                      for angle in theta])
        rng=np.random.default_rng(511)
        xyz+=rng.normal(0,.002,xyz.shape)
        estimate=estimate_human_v3_root_axis_angle(
            xyz,self.rest,self.names,np.eye(3),smoothing_window=3)
        expected=Rotation.from_euler("z",theta[:,None]).as_matrix()
        deviation=(Rotation.from_matrix(expected).inv()*
                   Rotation.from_rotvec(estimate)).magnitude()
        self.assertLess(np.rad2deg(np.percentile(deviation,95)),5.0)

    def test_rejects_invalid_rig_and_confidence(self):
        xyz=self._world_pose(np.eye(3))
        with self.assertRaisesRegex(ValueError,"Invalid 3D observation|67-marker"):
            estimate_human_v3_root_axis_angle(xyz,self.rest,self.names[:4],np.eye(3))
        with self.assertRaisesRegex(ValueError,"confidence"):
            estimate_human_v3_root_axis_angle(xyz,self.rest,self.names,np.eye(3),np.ones((17,55)))
        with self.assertRaises(ValueError):
            estimate_human_v3_root_axis_angle(xyz,self.rest,self.names,np.eye(3),smoothing_window=4)


if __name__=="__main__":
    unittest.main()
