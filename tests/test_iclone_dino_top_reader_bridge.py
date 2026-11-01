"""The actual-image Dino Top Reader must not alter original Front/Side UV."""
import unittest

try:
    import numpy as np
except ImportError:
    np=None

if np is not None:
    from tools.make_dino_top_reader_detection import build


@unittest.skipUnless(np is not None,"NumPy optional for minimal CI")
class DinoTopReaderBridgeTests(unittest.TestCase):
    def make_fixture(self):
        names=(
            ["joint:CC_Dinosaur_Tail%02d:tail"%i for i in range(9)] +
            ["joint:CC_Dinosaur_L_LeftToe%02d:tail"%i for i in range(34)] +
            ["joint:CC_Dinosaur_Spine%02d:head"%i for i in range(49)]
        )
        T,J=6,len(names)
        rig={"keypoints":[{"name":n}for n in names]}
        raw={
            "names":np.asarray(names),
            "fps":np.asarray(24.0),
            "calib":np.asarray([[1,0,0],[0,1,0]],np.float32),
            "front_uv":np.ones((T,J,2),np.float32)*101.,
            "side_uv":np.ones((T,J,2),np.float32)*240.,
            "front_conf":np.ones((T,J),np.float32)*.9,
            "side_conf":np.ones((T,J),np.float32)*.85,
        }
        top={
            "names":np.asarray(names),
            "fps":np.asarray(24.0),
            "top_uv":np.tile(np.array([440.,360.],np.float32),(T,J,1)),
            "top_conf":np.full((T,J),.7,np.float32),
        }
        return rig,raw,top

    def test_only_tail_top_enabled_and_original_reader_immutable(self):
        rig,raw,top=self.make_fixture()
        preserved={k:v.copy() for k,v in raw.items()}
        result,report=build(raw,top,rig)
        self.assertEqual(report["tail_joint_count"],9)
        self.assertEqual(report["active_top_joint_frames"],54)
        for key,val in preserved.items():
            np.testing.assert_array_equal(result[key],val)
            np.testing.assert_array_equal(raw[key],val)
        self.assertEqual(result["top_uv"].shape,(6,92,2))
        np.testing.assert_array_equal(result["top_conf"][:,:9],.7)
        np.testing.assert_array_equal(result["top_conf"][:,9:],0)
        np.testing.assert_array_equal(result["top_uv"][:,9:,:],0)

    def test_out_of_image_predictions_must_not_be_used(self):
        rig,raw,top=self.make_fixture()
        top["top_uv"][2,8]=[-20.,100.]
        top["top_conf"][0,7]=2.5
        top["top_conf"][1,7]=-.3
        result,report=build(raw,top,rig)
        self.assertEqual(result["top_conf"][2,8],0.)
        self.assertEqual(result["top_conf"][0,7],1.)
        self.assertEqual(result["top_conf"][1,7],0.)
        self.assertEqual(report["out_of_frame_tail_joint_frames"],1)

    def test_reject_bad_joint_order_frame_or_nonfinite(self):
        rig,raw,top=self.make_fixture()
        top["names"]=top["names"][::-1].copy()
        with self.assertRaisesRegex(ValueError,"order"):
            build(raw,top,rig)
        _,raw,top=self.make_fixture()
        top["fps"]=np.asarray(30.)
        with self.assertRaisesRegex(ValueError,"FPS"):
            build(raw,top,rig)
        _,raw,top=self.make_fixture()
        top["top_uv"][3,1,0]=np.nan
        with self.assertRaisesRegex(ValueError,"invalid original Top"):
            build(raw,top,rig)

    def test_refuses_preweighted_or_top_decorated_inputs(self):
        rig,raw,top=self.make_fixture()
        raw["top_uv"]=np.zeros((6,92,2),np.float32)
        with self.assertRaisesRegex(ValueError,"already contains Top"):
            build(raw,top,rig)


if __name__=="__main__":
    unittest.main()
