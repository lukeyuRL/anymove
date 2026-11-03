"""P1 Front-only BVH solver unit tests, including poisoned heldout Side."""
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot

SRC=Path(__file__).resolve().parents[1]/"tools"/"mocapanything_v2_front_solver.py"
spec=importlib.util.spec_from_file_location("p1_front_only",SRC)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestV2FrontOnlySolver(unittest.TestCase):
    def test_euler_xyz_matches_scipy_rotation(self):
        angles=np.array([[.2,-.3,.6],[-.1,.1,.2]],np.float32)
        computed=m.euler_xyz(torch.tensor(angles)).numpy()
        expected=SciRot.from_euler("xyz",angles).as_matrix()
        np.testing.assert_allclose(computed,expected,atol=2e-7)

    def test_fk_lengths_and_root_no_stretch(self):
        parents=[-1,0,1]
        off=torch.tensor([[0,0,0],[0,0,2],[1,0,0]],dtype=torch.float64)
        rp=torch.tensor([[.1,.2,.3],[1.,2.,3.]],dtype=torch.float64)
        R=torch.eye(3,dtype=torch.float64).reshape(1,1,3,3).expand(2,3,3,3)
        pos=m.fk_torch(rp,off,parents,R).numpy()
        np.testing.assert_allclose(pos[:,0],rp.numpy(),atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(pos[:,1]-pos[:,0],axis=1),2)
        np.testing.assert_allclose(np.linalg.norm(pos[:,2]-pos[:,1],axis=1),1)

    def test_side_poison_has_zero_effect_on_solver_output(self):
        # No 'side_uv' may enter optimize(); change it by huge amounts and
        # require identical BVH-world output for the same Front inputs.
        T,J=4,3
        E=np.broadcast_to(np.eye(3),(T,J,3,3)).copy()
        off=np.array([[0,0,0],[0,0,1],[0,0,1]],float)
        root=np.tile(np.array([0,0,0.1]),(T,1))
        fu=np.tile(np.array([[200,359],[200,350],[200,340]],float),(T,1,1))
        fu[:,2,0]+=np.array([0,5,11,3],float)
        base=dict(T=T,J=J,active=np.array([0,1]),
            v2_rot=E,rest_rot=np.eye(3),root_positions=root,joint_offsets=off,
            original_bvh=SimpleNamespace(parents=np.array([-1,0,1],dtype=int)),
            basis=np.eye(3),scale=1.0,mapping=np.array([0,1,2]),
            layout={"M_PER_PX":0.1,"W_FRONT":400,"H":720,"Z_CENTER":0.},
            obs={"front_uv":fu,"front_conf":np.ones((T,J),float),
                 "side_uv":np.ones_like(fu)})
        changed=base.copy()
        changed["obs"]=base["obs"].copy()
        changed["obs"]["side_uv"]=np.full_like(fu,1e7)
        x=m.optimize(base,"rest",steps=12,prior_strength=2,temporal_strength=.01,
                     lr=.01,device="cpu")
        y=m.optimize(changed,"rest",steps=12,prior_strength=2,temporal_strength=.01,
                     lr=.01,device="cpu")
        np.testing.assert_array_equal(x["rot_matrices"],y["rot_matrices"])
        np.testing.assert_array_equal(x["world"],y["world"])
        self.assertEqual(x["best_loss"],y["best_loss"])

    def test_evaluation_side_reacts_to_poison(self):
        T=3;obs={"front_uv":np.zeros((T,1,2),float),
            "side_uv":np.zeros((T,1,2),float)}
        world=np.zeros((T,1,3),float)
        layout={"M_PER_PX":1.0,"H":720,"W_FRONT":400,"W_SIDE":800,"Z_CENTER":0.}
        a,_,_=m.eval_world(world,obs,layout,[0])
        changed={"front_uv":obs["front_uv"],"side_uv":np.ones((T,1,2))*999}
        b,_,_=m.eval_world(world,changed,layout,[0])
        self.assertEqual(a["front"],b["front"])
        self.assertNotEqual(a["heldout_side"],b["heldout_side"])


if __name__=="__main__":
    unittest.main()
