"""Regression tests for frame-cadence-aware AnyMove Solver loss."""
import unittest
import torch

from mocap.solve import time_normalized_joint_acceleration

class SolverTimeDerivativeTests(unittest.TestCase):
    def test_original_unit_cadence_compatibility(self):
        x=torch.tensor([0.,.2,.9,1.1,1.5],dtype=torch.float64).reshape(5,1,1)
        actual=time_normalized_joint_acceleration(x,torch.ones(4,dtype=torch.float64))
        expected=x[2:]-2*x[1:-1]+x[:-2]
        torch.testing.assert_close(actual,expected,atol=1e-14,rtol=0)
    def test_sparse_constant_velocity_is_zero(self):
        times=torch.tensor([8.,15.,22.,31.,44.],dtype=torch.float64)
        data=(.015*times+.75).reshape(-1,1,1).repeat(1,67,3)
        actual=time_normalized_joint_acceleration(data,times.diff())
        torch.testing.assert_close(actual,torch.zeros_like(actual),atol=1e-12,rtol=0)
    def test_second_derivative_is_physically_invariant_to_cadence(self):
        for times in ([0.,1.,2.,3.,4.],[0.,7.,14.,21.,28.],[0.,2.,5.,10.,21.]):
            t=torch.tensor(times,dtype=torch.float64)
            values=(.5*t**2).reshape(-1,1,1)
            measured=time_normalized_joint_acceleration(values,t.diff())
            torch.testing.assert_close(measured,torch.ones_like(measured),atol=1e-11,rtol=0)
    def test_too_short_clip_no_acceleration(self):
        x=torch.zeros(2,67,3,dtype=torch.float64)
        a=time_normalized_joint_acceleration(x,torch.ones(1,dtype=torch.float64))
        self.assertEqual(tuple(a.shape),(0,67,3))
    def test_reject_nonpositive_deltas(self):
        with self.assertRaisesRegex(ValueError,"Invalid"):
            time_normalized_joint_acceleration(torch.zeros(4,1,3),torch.tensor([1.,0.,1.]))
if __name__=="__main__":
    unittest.main()
