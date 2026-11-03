"""P3 hybrid Reader ablation: exact selected-view provenance and GT separation."""
import json
import unittest
from pathlib import Path
import numpy as np

from mocap.multiview import load_packet
from tools.blend_fourview_reader_p3 import sha

ROOT=Path(__file__).resolve().parents[1]


class HybridReaderP3Tests(unittest.TestCase):
    def fixtures(self,creature):
        base=ROOT/"runs"/creature/"dataset_fourview_p3_material_fullres_v1"
        names=("p3_material_baseline_heldout.npz",
               "p3_material_trained_heldout.npz",
               "p3_material_hybrid_heldout.npz")
        if not all((base/name).exists() for name in names):
            self.skipTest("local fullres hybrid not finished")
        old,_=load_packet(base/names[0])
        tuned,_=load_packet(base/names[1])
        merged,_=load_packet(base/names[2])
        return base,old,tuned,merged

    def test_hybrid_sources_have_exact_observation_fidelity(self):
        for creature in ("dino","dragon"):
            with self.subTest(creature=creature):
                base,old,tuned,merged=self.fixtures(creature)
                for key in ("uv","confidence","valid"):
                    np.testing.assert_array_equal(merged[key][:2],old[key][:2])
                    np.testing.assert_array_equal(merged[key][2:],tuned[key][2:])
                for key in ("world_to_camera","image_size_px","meters_per_px","names","frame_ids"):
                    np.testing.assert_array_equal(merged[key],old[key])
                    np.testing.assert_array_equal(merged[key],tuned[key])

    def test_hybrid_packet_contains_no_ground_truth_or_uncalibrated_quality_weights(self):
        for creature in ("dino","dragon"):
            base,old,tuned,merged=self.fixtures(creature)
            for field in ("joints_world","gt_world",
                          "joint_surface_part_visible",
                          "visibility_score","expected_error_px","quality_semantics"):
                self.assertNotIn(field,merged)
            self.assertEqual(merged["reader_source_by_view"].tolist(),[
                "original_front_side","original_front_side",
                "new_four_view_back","new_four_view_opposite_side"])
            manifest=json.loads((base/"p3_material_hybrid_heldout.manifest.json").read_text())
            self.assertEqual(manifest["original_input_sha256"],
                             sha(base/"p3_material_baseline_heldout.npz"))
            self.assertEqual(manifest["finetuned_input_sha256"],
                             sha(base/"p3_material_trained_heldout.npz"))
            self.assertTrue(manifest["no_gt_leakage"])

    def test_two_view_hybrid_is_unchanged_from_original_two_view(self):
        for creature in ("dino","dragon"):
            base,old,tuned,merged=self.fixtures(creature)
            with np.load(base/"p3_material_baseline_fk_2v.npz",allow_pickle=False) as a:
                before=a["world_joints"]
            with np.load(base/"p3_material_hybrid_fk_2v.npz",allow_pickle=False) as b:
                after=b["world_joints"]
            np.testing.assert_allclose(before,after,atol=1e-9,rtol=0)

    def test_hybrid_3d_is_an_ablation_not_production_promotion(self):
        for creature in ("dino","dragon"):
            base,_,_,_=self.fixtures(creature)
            report=base/"p3_material_hybrid_fk_3d_eval_v1.json"
            if not report.exists():self.skipTest("hybrid 3D result not available")
            r=json.loads(report.read_text())
            self.assertTrue(r["independent_stills"])
            self.assertEqual(r["frame_ids"],[25,50,75,100])
            self.assertTrue(r["paired_3d_comparison"]["four_view_improves_this_small_sample"])
            self.assertTrue(r["paired_3d_comparison"]["not_proof_of_generalization"])
            gate=json.loads((ROOT/"runs"/creature/"fourview_p3_material_fullres_acceptance_v1.json").read_text())
            self.assertFalse(gate["production_promotion_allowed"])


if __name__=="__main__":
    unittest.main()
