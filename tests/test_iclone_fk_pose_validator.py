"""Numerical FK axis validation; no iClone runtime necessary."""
import math
import unittest

from iclone_adapter.fk_pose_validator import compare_fk_local_pose, _sample_frames, _from_quat
from iclone_adapter.anymove_bridge import from_euler_xyz

IDENT=[1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.]


def as_mat3(m):
    return [[m[4*i+j] for j in range(3)] for i in range(3)]


def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def mul(a,b):
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def flat(a,tx=0):
    return [a[0][0],a[0][1],a[0][2],tx,
            a[1][0],a[1][1],a[1][2],0.,
            a[2][0],a[2][1],a[2][2],0.,
            0.,0.,0.,1.]


class PoseValidatorTests(unittest.TestCase):
    def test_identity_and_wrong_axes(self):
        move=[.3,-.2,.1]
        expected=_from_quat(from_euler_xyz(move))
        good=compare_fk_local_pose(IDENT,flat(expected),move)
        self.assertLess(good["post_rest_error_deg"],0.001)
        bad=compare_fk_local_pose(IDENT,flat(transpose(expected)),move)
        self.assertGreater(bad["post_rest_error_deg"],5.)

    def test_nontrivial_rest_pose_and_100x_scale(self):
        rest=_from_quat(from_euler_xyz([.0,.6,-.3]))
        move=[.22, .17, .08]
        posed=mul(rest,_from_quat(from_euler_xyz(move)))
        a=flat(rest,123.)
        b=flat(posed,456.)
        for i in range(3):
            for j in range(3):
                a[4*i+j]*=100.
                b[4*i+j]*=100.
        good=compare_fk_local_pose(a,b,move)
        self.assertLess(good["post_rest_error_deg"],1e-4)
        self.assertGreater(good["pre_rest_error_deg"],1.)

    def test_sampling_includes_final_and_peak(self):
        m={"frame_count":100,"tracks":{
            "a":[[.3 if i==61 else 0.,0.,0.] for i in range(100)],
            "b":[[.2 if i==42 else 0.,0.,0.] for i in range(100)]}}
        frames=_sample_frames(m)
        self.assertEqual(frames[0],0)
        self.assertIn(99,frames)
        self.assertIn(42,frames)
        self.assertIn(61,frames)


if __name__=="__main__":
    unittest.main()
