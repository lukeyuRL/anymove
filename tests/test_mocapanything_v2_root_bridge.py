"""Contract tests for V2 Root bridge; no downloaded assets or GPU needed."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "tools" / "mocapanything_v2_root_bridge.py"
spec=importlib.util.spec_from_file_location("v2_root_bridge",MODULE)
bridge=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class TestV2RootBridge(unittest.TestCase):
    def test_similarity_known_basis(self):
        q=np.array([[1,0,0],[0,0,1],[0,-1,0]],dtype=float)
        rng=np.random.default_rng(72)
        a=rng.normal(size=(67,3))
        scale=1.0
        translate=np.array([0,-0.2353,4.15])
        target=scale*a@q+translate
        s,r,t,err=bridge.similarity_from_rest(a,target)
        self.assertAlmostEqual(s,1,places=11)
        self.assertLess(float(np.max(np.abs(r-q))),1e-10)
        self.assertLess(float(np.max(np.abs(t-translate))),1e-10)
        self.assertLess(float(np.max(err)),1e-10)

    def test_front_root_only_holds_depth(self):
        dt={"names":np.array(["joint:Hip:head"]),
            "front_uv":np.array([[[202,460]],[[210,440]]],float)}
        l={"root_bone":"Hip","M_PER_PX":0.02,"W_FRONT":400,
           "H":720,"Z_CENTER":6.5}
        rest=np.array([0.,-0.2,4.5])
        target,uv,idx=bridge.compute_front_root(dt,l,rest)
        self.assertEqual(idx,0)
        np.testing.assert_allclose(target[:,1],rest[1])
        projected_u=200+target[:,0]/0.02
        projected_v=360-(target[:,2]-6.5)/0.02
        np.testing.assert_allclose(np.stack([projected_u,projected_v],1),uv)
        self.assertFalse(np.allclose(target[0],target[1]))

    def test_bvh_root_only_patched(self):
        bvh=("HIERARCHY\nROOT Hip\n{\n OFFSET 0 0 0\n"
             " CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
             " JOINT Spine\n {\n OFFSET 0 1 0\n CHANNELS 3 Zrotation Yrotation Xrotation\n"
             " End Site\n {\n OFFSET 0 1 0\n }\n }\n}\nMOTION\nFrames: 2\n"
             "Frame Time: 0.041667\n"
             "0 0 0 10 20 30 11 12 13\n"
             "1 2 3 40 50 60 14 15 16\n")
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/"src.bvh"
            output=Path(td)/"out.bvh"
            source.write_text(bvh,encoding="utf8")
            lines,first,channels,name=bridge.read_motion_lines(source,2)
            self.assertEqual(name,"Hip")
            shift=np.array([[0.15,0.2,-0.3],[0.1,-0.1,0.4]])
            bridge.write_root_positions(lines,first,channels,shift,output)
            origlines=source.read_text().splitlines()
            out_lines=output.read_text().splitlines()
            self.assertEqual(origlines[:first],out_lines[:first])
            for i in range(2):
                old=origlines[first+i].split()
                new=out_lines[first+i].split()
                np.testing.assert_allclose(np.array(new[:3],float),
                                           np.array(old[:3],float)+shift[i])
                self.assertEqual(new[3:],old[3:])

    def test_fail_closed_on_missing_channels(self):
        invalid=("HIERARCHY\nROOT Root\n{\n OFFSET 0 0 0\n"
                 " CHANNELS 3 Zrotation Yrotation Xrotation\n}\n"
                 "MOTION\nFrames: 1\nFrame Time: 0.04\n1 2 3\n")
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/"source.bvh";src.write_text(invalid)
            with self.assertRaisesRegex(ValueError,"root motion requires"):
                bridge.read_motion_lines(src,1)


if __name__=="__main__":
    unittest.main()
