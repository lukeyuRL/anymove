"""P2 N-camera FK Solver invariants: no GT leakage, no hidden views, and
independent static poses without temporal prior.
"""
import copy
import json
import unittest
from pathlib import Path
import numpy as np

from mocap.multiview import load_packet
from mocap.solve_multiview import fit_multiview

ROOT=Path(__file__).resolve().parents[1]


class NViewFKSolverTests(unittest.TestCase):
    def fixtures(self,creature="dino"):
        root=ROOT/"runs"/creature/"four_view_visibility_p1_validation"
        p=root/"p2_synthetic_noisy.npz"
        rig=ROOT/"assets"/creature/"rigged/rig_info.json"
        if not p.is_file() or not rig.is_file():
            self.skipTest("requires local P2 smoke assets")
        obs,meta=load_packet(p)
        info=json.loads(rig.read_text(encoding="utf8"))
        return obs,info,meta

    def test_two_view_optimizer_ignores_poisoned_unused_cameras(self):
        p,info,meta=self.fixtures()
        v2=["front","side"]
        out=fit_multiview(p,info,view_ids=v2,
                  iterations=14,device="cpu",verbose=False)
        poison={k:v.copy() for k,v in p.items()}
        poison["uv"][2:]+=np.float32(100000)
        poison["confidence"][2:]=np.float32(10000)
        out2=fit_multiview(poison,info,view_ids=v2,
                  iterations=14,device="cpu",verbose=False)
        np.testing.assert_allclose(out["world"],out2["world"],atol=0,rtol=0)
        self.assertEqual(out["best_loss"],out2["best_loss"])

    def test_solver_refuses_parallel_view_geometry(self):
        p,info,meta=self.fixtures()
        with self.assertRaisesRegex(ValueError,"too few rank-valid"):
            fit_multiview(p,info,view_ids=["front","back"],
                          iterations=2,device="cpu",verbose=False)

    def test_reader_untriangulable_minor_joint_falls_back_to_rig_rest(self):
        path=ROOT/"runs/dino/four_view_visibility_p1_validation/p2_reader_ood.npz"
        if not path.is_file():
            self.skipTest("Reader input not available")
        p,_=load_packet(path)
        info=json.loads((ROOT/"assets/dino/rigged/rig_info.json").read_text())
        # The previously observed 177/184 points requires a safe fallback.
        fit=fit_multiview(p,info,view_ids=["front","side"],
             iterations=3,device="cpu",verbose=False)
        self.assertEqual(fit["triangulation_count"],177)
        self.assertTrue(np.isfinite(fit["world"]).all())

    def test_uncalibrated_quality_maps_are_not_used_as_solver_weights(self):
        p,info,_=self.fixtures()
        v=['front','side']
        baseline=fit_multiview(p,info,view_ids=v,
            iterations=8,device='cpu',verbose=False)
        altered={k:x.copy() for k,x in p.items()}
        altered['quality_semantics']=np.asarray('synthetic_cutout_only_uncalibrated')
        altered['visibility_score']=np.full(p['confidence'].shape,.001,dtype=np.float32)
        altered['expected_error_px']=np.full(p['confidence'].shape,999.,dtype=np.float32)
        check=fit_multiview(altered,info,view_ids=v,
            iterations=8,device='cpu',verbose=False)
        np.testing.assert_array_equal(baseline['world'],check['world'])
        self.assertEqual(baseline['best_loss'],check['best_loss'])

    def test_p2_results_keep_3d_gt_independent_and_quality_negative(self):
        for name in ("dino","dragon"):
            path=ROOT/"runs"/name/"four_view_visibility_p1_validation"
            synth=path/"p2_synth_2v4v_3d_eval_v1.json"
            real=path/"p2_reader_2v4v_3d_eval_v1.json"
            if not synth.exists() or not real.exists():
                self.skipTest("P2 saved 3D GT result not present")
            with self.subTest(creature=name):
                a=json.loads(synth.read_text())
                b=json.loads(real.read_text())
                self.assertTrue(a["independent_stills"])
                self.assertEqual(a["frames"],2)
                self.assertTrue(a["paired_3d_comparison"]["four_view_improves_this_small_sample"])
                self.assertFalse(b["paired_3d_comparison"]["four_view_improves_this_small_sample"])
                self.assertIn("reader_inference",b["source_kind"])
                self.assertTrue(
                    a["two_vs_four"]["two_view"]["solver_fk_roundtrip_max_m"]<.0003)


if __name__=="__main__":
    unittest.main()
