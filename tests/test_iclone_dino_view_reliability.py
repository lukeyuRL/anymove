"""Dino preregistered joint-specific Front/Side observation reliability tests."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from tools.make_dino_view_ablations import PROFILES, build


@unittest.skipUnless(np is not None, "NumPy optional outside NumPy CI")
class DinoJointViewReliabilityTests(unittest.TestCase):
    def fixture(self):
        names = (
            ["joint:CC_Dinosaur_Tail%02d:tail" % i for i in range(9)] +
            ["joint:CC_Dinosaur_L_LeftToe%02d:tail" % i for i in range(34)] +
            ["joint:CC_Dinosaur_Hip%02d:head" % i for i in range(49)]
        )
        rig = {"keypoints":[{"name":n} for n in names]}
        T,J=5,len(names)
        raw={
            "names":np.array(names),
            "fps":np.array(24.0),
            "calib":np.array([[1,0,0],[0,1,0]],np.float32),
            "front_conf":np.ones((T,J),np.float32)*.85,
            "side_conf":np.ones((T,J),np.float32)*.80,
            "front_uv":np.ones((T,J,2),np.float32)*128,
            "side_uv":np.ones((T,J,2),np.float32)*300,
        }
        return raw,rig

    def test_each_profile_is_fixed_to_one_view_and_anatomical_region(self):
        raw,rig=self.fixture()
        saved={k:v.copy() for k,v in raw.items()}
        for variant in PROFILES:
            with self.subTest(variant=variant):
                out,report=build(raw,rig,variant)
                self.assertEqual(report["profile"],
                                 "anymove.dino_view_reliability_fixed_half_ablation/1")
                for k,v in saved.items():
                    np.testing.assert_array_equal(out[k],v)
                    np.testing.assert_array_equal(raw[k],v)
                front=out["front_track_weight"]
                side=out["side_track_weight"]
                fcount=int((front<1).sum())
                scount=int((side<1).sum())
                self.assertEqual(fcount,
                                 45 if variant in ("tail_front_half","combined_half") else 0)
                self.assertEqual(scount,
                                 170 if variant in ("toes_side_half","combined_half") else 0)
                self.assertTrue(np.all(np.isin(front,[.5,1.])))
                self.assertTrue(np.all(np.isin(side,[.5,1.])))
                self.assertEqual(fcount,report["views"]["front"]["downweighted_joint_frames"])
                self.assertEqual(scount,report["views"]["side"]["downweighted_joint_frames"])

    def test_reject_wrong_rig_names_or_non_dino_and_double_weighting(self):
        raw,rig=self.fixture()
        bad={"keypoints":list(reversed(rig["keypoints"]))}
        with self.assertRaisesRegex(ValueError,"order"):
            build(raw,bad,"combined_half")
        raw["front_track_weight"]=np.ones((5,92),np.float32)
        with self.assertRaisesRegex(ValueError,"non-P1/non-P2"):
            build(raw,rig,"tail_front_half")
        raw.pop("front_track_weight")
        with self.assertRaisesRegex(ValueError,"unsupported"):
            build(raw,rig,"oracle_best")

    def test_reject_invalid_calibration_or_nonfinite_reader_observation(self):
        raw,rig=self.fixture()
        raw["front_uv"][1,4,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"invalid Reader"):
            build(raw,rig,"tail_front_half")
        raw["front_uv"][1,4,0]=128.
        raw["calib"]=np.zeros((3,3))
        with self.assertRaisesRegex(ValueError,"calibration"):
            build(raw,rig,"toes_side_half")

    def test_one_view_weight_never_deletes_unique_axis_observation(self):
        raw,rig=self.fixture()
        out,report=build(raw,rig,"combined_half")
        self.assertEqual(out["front_track_weight"][:,0].min(),.5)
        self.assertEqual(out["side_track_weight"][:,0].min(),1.)
        self.assertEqual(out["front_track_weight"][:,9].min(),1.)
        self.assertEqual(out["side_track_weight"][:,9].min(),.5)
        self.assertEqual(report["prespecified_multiplier"],.5)


if __name__ == "__main__":
    unittest.main()
