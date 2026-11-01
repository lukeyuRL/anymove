"""Dino Top camera geometry and GT-oracle nonleakage gates.

These tests cannot prove real Top Reader inference: the oracle source
is intentionally synthetic Blender joint ground truth and is fail-closed.
"""
import unittest

try:
    import numpy as np
except ImportError:
    np=None

if np is not None:
    from mocap import layout
    from mocap.dino_top_camera import (
        project_top_numpy, validate_top_camera, TOP_WIDTH, TOP_HEIGHT)
    from tools.make_dino_top_oracle import make


@unittest.skipUnless(np is not None,"NumPy required in visibility CI")
class DinoTopOracleContractTests(unittest.TestCase):
    def setUp(self):
        self.names=(
            ["joint:CC_Dinosaur_Tail%02d:tail"%i for i in range(9)] +
            ["joint:CC_Dinosaur_L_LeftToe%02d:tail"%i for i in range(34)] +
            ["joint:CC_Dinosaur_Spine%02d:head"%i for i in range(49)])
        T=8
        self.rig={"keypoints":[{"name":n} for n in self.names]}
        J=np.zeros((T,92,3),float)
        # Scale the fabricated NumPy test joints to whichever default
        # creature layout is active in CI; no real Dino assets required.
        J[:,:,0]=np.arange(T)[:,None]*layout.M_PER_PX*4
        J[:,:,1]=(np.arange(92)[None,:]-46)*layout.M_PER_PX*7
        J[:,:,2]=2.0
        self.truth={"names":self.names,"fps":24.0,
                    "joints":J.tolist()}
        self.camera={
            "type":"ORTHO","res":[TOP_WIDTH,TOP_HEIGHT],
            "R":np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]).tolist(),"t":[0.,0.,-30.],
            "f":1/layout.M_PER_PX,
            "cx":TOP_WIDTH/2,"cy":TOP_HEIGHT/2
        }
        self.raw={
            "names":np.asarray(self.names),
            "fps":np.asarray(24.),
            "calib":np.eye(2,3,dtype=np.float32),
            "front_conf":np.ones((T,92),np.float32)*.8,
            "side_conf":np.ones((T,92),np.float32)*.8,
            "front_uv":np.ones((T,92,2),np.float32)*75,
            "side_uv":np.ones((T,92,2),np.float32)*95,
        }

    def test_top_camera_projection_is_independent_x_y(self):
        J=np.asarray(self.truth["joints"],float)
        uv=project_top_numpy(J,self.camera)
        self.assertEqual(uv.shape,(8,92,2))
        np.testing.assert_allclose(
            uv[...,0],TOP_WIDTH/2+J[...,1]/layout.M_PER_PX,
            rtol=0,atol=1e-9)
        np.testing.assert_allclose(
            uv[...,1],TOP_HEIGHT/2+J[...,0]/layout.M_PER_PX,
            rtol=0,atol=1e-9)

    def test_oracle_tail_only_preserves_front_side_raw(self):
        source={key:v.copy() for key,v in self.raw.items()}
        out,meta=make(source,self.rig,self.truth,self.camera,"tail_only")
        self.assertEqual(meta["selected_joint_count"],9)
        for key,v in self.raw.items():
            np.testing.assert_array_equal(out[key],v)
            np.testing.assert_array_equal(self.raw[key],v)
        self.assertEqual(int((out["top_conf"]>0).sum()),8*9)
        self.assertTrue(np.all(out["top_conf"][:,9:]==0))
        self.assertTrue(np.all(out["top_uv"][:,9:]==0))
        self.assertTrue(np.all(np.isfinite(out["top_uv"])))

    def test_oracle_all_joints_is_upper_bound_not_reader(self):
        out,meta=make(self.raw,self.rig,self.truth,self.camera,"all_joints")
        self.assertEqual(meta["selected_joint_count"],92)
        self.assertEqual(int((out["top_conf"]>0).sum()),8*92)
        self.assertIn("Blender truth",meta["leakage_warning"])

    def test_reject_camera_mismatch_and_different_gt_joint_order(self):
        shifted=dict(self.camera,t=[.1,0,-30.])
        with self.assertRaisesRegex(ValueError,"target"):
            make(self.raw,self.rig,self.truth,shifted,"tail_only")
        flip=dict(self.camera,R=np.eye(3).tolist())
        with self.assertRaisesRegex(ValueError,"rotated"):
            validate_top_camera(flip)
        other=dict(self.truth,names=list(reversed(self.names)))
        with self.assertRaisesRegex(ValueError,"order"):
            make(self.raw,self.rig,other,self.camera,"tail_only")
        other=dict(self.truth,fps=30.0)
        with self.assertRaisesRegex(ValueError,"FPS"):
            make(self.raw,self.rig,other,self.camera,"tail_only")

    def test_differentiable_torch_projection_matches_numpy(self):
        try:
            import torch
            from mocap.dino_top_camera import project_top_torch
        except ImportError:
            self.skipTest("Torch not installed for NumPy CI")
        points=torch.tensor(self.truth["joints"],dtype=torch.float64,
                            requires_grad=True)
        uv=project_top_torch(points,self.camera)
        np.testing.assert_allclose(
            uv.detach().cpu().numpy(),
            project_top_numpy(np.asarray(self.truth["joints"]),self.camera),
            atol=1e-8,rtol=0)
        uv[...,0].sum().backward()
        self.assertTrue(np.allclose(
            points.grad[...,1].detach().cpu().numpy(),
            1/layout.M_PER_PX))


if __name__=="__main__":
    unittest.main()
