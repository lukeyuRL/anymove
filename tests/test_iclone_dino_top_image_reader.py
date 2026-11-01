"""Regression for actual Top-image Reader bridge (NOT Blender GT oracle).

No 3D ground truth or oracle metadata is used to build observations.
"""
import unittest

try:
    import numpy as np
except ImportError:
    np=None

if np is not None:
    from tools.make_dino_top_reader_detection import build


@unittest.skipUnless(np is not None,"NumPy is required")
class DinoTopActualImageReaderBridgeTests(unittest.TestCase):
    def fixture(self):
        names=(
            ["joint:CC_Dinosaur_Tail%02d:tail"%i for i in range(9)] +
            ["joint:CC_Dinosaur_L_LeftToe%02d:tail"%i for i in range(34)] +
            ["joint:CC_Dinosaur_Hip%02d:head"%i for i in range(49)])
        T,J=6,92
        raw={
            "names":np.array(names),
            "fps":np.asarray(24.),
            "calib":np.eye(2,3,dtype=np.float32),
            "front_uv":np.ones((T,J,2),np.float32)*100.,
            "side_uv":np.ones((T,J,2),np.float32)*200.,
            "front_conf":np.ones((T,J),np.float32)*.8,
            "side_conf":np.ones((T,J),np.float32)*.72,
        }
        top={"names":np.array(names),"fps":np.asarray(24.),
             "top_uv":np.ones((T,J,2),np.float32)*360.,
             "top_conf":np.ones((T,J),np.float32)*.9}
        rig={"keypoints":[{"name":n} for n in names]}
        return raw,top,rig

    def test_tail_only_has_actual_top_values_but_untouched_original_reader(self):
        raw,top,rig=self.fixture()
        save={k:v.copy() for k,v in raw.items()}
        out,audit=build(raw,top,rig)
        for k in raw:
            np.testing.assert_array_equal(raw[k],save[k])
            np.testing.assert_array_equal(out[k],save[k])
        self.assertEqual(audit["tail_joint_count"],9)
        self.assertEqual(audit["active_top_joint_frames"],6*9)
        np.testing.assert_array_equal(out["top_uv"][:,:9,:],
                                      top["top_uv"][:,:9,:])
        self.assertTrue(np.all(out["top_uv"][:,9:]==0))
        self.assertTrue(np.all(out["top_conf"][:,9:]==0))
        np.testing.assert_allclose(out["top_conf"][:,:9],.9,atol=1e-7)

    def test_offscreen_reader_predictions_are_not_used_as_solver_evidence(self):
        raw,top,rig=self.fixture()
        top["top_uv"][3,7,0]=-10
        top["top_uv"][0,2,1]=770
        out,audit=build(raw,top,rig)
        self.assertEqual(audit["out_of_frame_tail_joint_frames"],2)
        self.assertEqual(out["top_conf"][3,7],0.)
        self.assertEqual(out["top_conf"][0,2],0.)
        self.assertEqual(audit["active_top_joint_frames"],6*9-2)

    def test_bad_source_order_and_nonfinite_reader_predictions_fail(self):
        raw,top,rig=self.fixture()
        top["names"]=top["names"][::-1]
        with self.assertRaisesRegex(ValueError,"order"):
            build(raw,top,rig)
        top["names"]=raw["names"].copy()
        top["top_conf"][2,5]=np.nan
        with self.assertRaisesRegex(ValueError,"invalid original Top"):
            build(raw,top,rig)
        top["top_conf"][2,5]=.9
        raw["top_uv"]=np.zeros((6,92,2))
        with self.assertRaisesRegex(ValueError,"already contains Top"):
            build(raw,top,rig)


if __name__=="__main__":
    unittest.main()
