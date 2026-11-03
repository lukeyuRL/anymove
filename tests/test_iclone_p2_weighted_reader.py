"""P2 Reader-to-Solver error-only ablation, independently unit-tested."""
import unittest

try:
    import numpy as np
except ImportError:
    np=None

if np is not None:
    from mocap.quality_weighting import (
        quality_error_weights, prepare_quality_weighted_detection)
    from tools.compare_dino_p2_motion import compare as compare_motions


@unittest.skipUnless(np is not None, "NumPy optional in base CI")
class DinoP2WeightingTests(unittest.TestCase):
    def fixture(self,frames=12):
        names=np.array(["joint:A:head","joint:Tail07:tail","joint:Toe03:tail"])
        t=np.arange(frames,dtype=float)
        uv=np.zeros((frames,len(names),2),dtype=np.float32)
        uv[:,:,0]=np.array([100.,200.,300.])
        uv[:,:,1]=150.+t[:,None]
        raw={
            "names":names,
            "fps":np.array(24.),
            "calib":np.array([[1.,0,0],[0,1.,0]],dtype=np.float32),
            "front_uv":uv.copy(),"side_uv":uv.copy()+10,
            "front_conf":np.ones((frames,len(names)),np.float32)*.85,
            "side_conf":np.ones((frames,len(names)),np.float32)*.80,
        }
        p2={k:v.copy() for k,v in raw.items()}
        for view in ("front","side"):
            p2[view+"_expected_error_px"]=np.ones((frames,3),np.float32)*2.
            p2[view+"_synthetic_visibility_score"]=np.ones((frames,3),np.float32)*.9
        p2["front_expected_error_px"][4,2]=20.
        p2["side_expected_error_px"][5,1]=10.
        return raw,p2

    def test_threshold_soft_weights_bounded_and_monotone(self):
        err=np.array([[0.,2.,5.,10.,20.,100.]])
        w=quality_error_weights(err)
        np.testing.assert_allclose(w,[[1.,1.,1.,.5,.4,.4]],atol=1e-7)
        self.assertTrue(np.all(np.diff(w[0])<=0))

    def test_error_only_ablation_preserves_all_original_joint_observations(self):
        raw,p2=self.fixture()
        saved={k:v.copy() for k,v in raw.items()}
        out,audit=prepare_quality_weighted_detection(raw,p2)
        self.assertEqual(audit["profile"],"anymove.p2_predicted_error_soft_weight_v1")
        for k in saved:
            np.testing.assert_array_equal(out[k],saved[k])
            np.testing.assert_array_equal(raw[k],saved[k])
        self.assertAlmostEqual(out["front_track_weight"][4,2],.4)
        self.assertAlmostEqual(out["side_track_weight"][5,1],.5)
        self.assertEqual(audit["views"]["front"]["downweighted_joint_frames"],1)
        # Artificial-cutout score must not change error-derived Solver weight.
        p2["front_synthetic_visibility_score"][:]=0
        a,_=prepare_quality_weighted_detection(raw,p2)
        np.testing.assert_array_equal(out["front_track_weight"],a["front_track_weight"])

    def test_reject_nonmatching_reader_or_calibration(self):
        raw,p2=self.fixture()
        p2["front_uv"][3,1,0]+=2.
        with self.assertRaisesRegex(ValueError,"changed existing Reader observations"):
            prepare_quality_weighted_detection(raw,p2)
        p2["front_uv"][3,1,0]-=2.
        p2["calib"][0,0]=.9
        with self.assertRaisesRegex(ValueError,"calibration"):
            prepare_quality_weighted_detection(raw,p2)
        p2["calib"]=raw["calib"].copy()
        p2["names"]=np.array(list(reversed(p2["names"])))
        with self.assertRaisesRegex(ValueError,"names or order"):
            prepare_quality_weighted_detection(raw,p2)

    def test_reject_tracked_or_nonfinite_inputs(self):
        raw,p2=self.fixture()
        raw["front_track_weight"]=np.ones((12,3))
        with self.assertRaisesRegex(ValueError,"non-P1"):
            prepare_quality_weighted_detection(raw,p2)
        raw.pop("front_track_weight")
        p2["front_expected_error_px"][1,1]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            prepare_quality_weighted_detection(raw,p2)

    def test_solver_comparison_uses_same_coordinates(self):
        raw,p2=self.fixture()
        weighted,_=prepare_quality_weighted_detection(raw,p2)
        T,J=raw["front_uv"].shape[:2]
        camera={"H":720,"W_FRONT":640,"W_SIDE":640,
                "M_PER_PX":.006875,"Z_CENTER":1.85625}
        j=np.zeros((T,J,3),float)
        q=np.tile([1.,0.,0.,0.],(T,1))
        motion={"fps":24.,"bones":["root","tail"],
                "joint_names":[str(n) for n in raw["names"]],
                "trans":np.zeros((T,3)).tolist(),
                "root_quat":q.tolist(),"joints":j.tolist()}
        variant=dict(motion)
        variant["trans"]=(np.zeros((T,3))+.01).tolist()
        report=compare_motions(raw,weighted,motion,variant,camera)
        self.assertEqual(report["frames"],12)
        self.assertEqual(report["p2_solver_multipliers"]["front"]["downweighted_count"],1)
        self.assertIn("same raw Reader",report["comparison_type"])
        self.assertAlmostEqual(report["between_solution_root_cm"]["mean_cm"],1.73205,places=4)
        weighted["side_uv"][1,1,0]+=2
        with self.assertRaisesRegex(ValueError,"changed original Reader"):
            compare_motions(raw,weighted,motion,variant,camera)


if __name__=="__main__":
    unittest.main()
