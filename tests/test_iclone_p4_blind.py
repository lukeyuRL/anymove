"""P4 blind evaluation and experiment-provenance regression."""
import json,unittest
from pathlib import Path
import numpy as np
from tools.eval_p4_blind_hybrid import compare
from tools.audit_p4_joint_regressions import run as audit_regions
from tools.blend_fourview_reader_p3 import sha
from mocap.multiview import load_packet

ROOT=Path(__file__).resolve().parents[1]


class BlindFourViewP4Tests(unittest.TestCase):
    def test_independent_3d_dataset_is_not_original_reader_training(self):
        for cr in ('dino','dragon'):
            root=ROOT/'runs'/cr
            split=json.loads((root/'p4_source_material_split_v1.json').read_text())
            ids=split['test_ids']
            self.assertEqual(ids,list(range(9001,9013)))
            self.assertTrue(set(ids).isdisjoint(split['train_ids']+split['val_ids']))
            audit=json.loads((root/'p4_blind_fourview_v1/blind_audit_v1.json').read_text())
            self.assertEqual(audit['status'],'PASS')
            self.assertEqual(audit['source_pose_count'],12)
            self.assertEqual(audit['view_count'],48)
            self.assertEqual(audit['source_front_side_reused_image_count'],0)
            self.assertTrue(audit['new_synthetic_blind_poses'])
            self.assertLess(audit['max_replay_3d_gt_error_m'],.001)

    def test_hybrid_uses_exact_original_front_side_without_oracle(self):
        for cr in ('dino','dragon'):
            base=ROOT/'runs'/cr/'p4_blind_fourview_v1'
            merged=base/'p4_hybrid_blind_reader.npz'
            if not merged.exists():self.skipTest('P4 hybrid blind not finished yet')
            origin,_=load_packet(base/'p4_original_blind_reader.npz')
            hybrid,_=load_packet(merged)
            adapted,_=load_packet(base/'p4_adapted_blind_reader.npz')
            for k in ('uv','valid','confidence'):
                np.testing.assert_array_equal(hybrid[k][:2],origin[k][:2])
                np.testing.assert_array_equal(hybrid[k][2:],adapted[k][2:])
            for k in ('joints_world','gt_world','joint_surface_part_visible',
                      'visibility_score','expected_error_px'):
                self.assertNotIn(k,hybrid)
            self.assertEqual(hybrid['frame_ids'].tolist(),list(range(9001,9013)))
            self.assertTrue((base/'p4_original_blind_reader.ground_truth.npz').exists())

    def test_paired_bootstrap_report_reproduces_from_saved_3d(self):
        for cr in ('dino','dragon'):
            result=ROOT/'runs'/cr/'p4_blind_acceptance_v1.json'
            if not result.exists():self.skipTest('P4 blind FK experiment not complete')
            stored=json.loads(result.read_text())
            calculated=compare(ROOT,cr,boot_count=4000,seed=41111)
            self.assertEqual(stored,calculated)
            self.assertEqual(stored['blind_synthetic_static_pose_count'],12)
            self.assertTrue(stored['hybrid_original_front_side_2d_unchanged'])
            self.assertEqual(stored['original_two_vs_hybrid_two_max_joint_difference_mm'],0.)
            self.assertEqual(stored['paired_95pct_pose_bootstrap_CI_mm'],
                sorted(stored['paired_95pct_pose_bootstrap_CI_mm']))
            self.assertFalse(stored['temporal_motion_evaluation_performed'])

    def test_region_audit_consistent_with_3d_fits(self):
        for cr in ('dino','dragon'):
            path=ROOT/'runs'/cr/'p4_blind_region_audit_v2.json'
            if not path.exists():self.skipTest('P4 ancestry-aware 3D region audit not yet done')
            report=json.loads(path.read_text())
            self.assertEqual(report,audit_regions(ROOT,cr))
            self.assertEqual(report['poses'],12)
            self.assertEqual(sum(v['keypoint_count'] for v in report['region_error'].values()),
                             report['keypoints'])
            if cr=='dragon':
                self.assertEqual(report['region_error']['wings']['keypoint_count'],12)
                self.assertGreater(report['region_error']['wings']['delta_mean_mm'],0)
                self.assertLess(report['region_error']['body_root']['delta_mean_mm'],0)

    def test_pilot_model_is_separate_from_original_reader(self):
        for cr,source in (('dino','reader_hr_v1.pt'),('dragon','reader_hr.pt')):
            root=ROOT/'runs'/cr
            report=root/'reader_backopp_p4_pilot_v1.validation.json'
            if not report.exists():self.skipTest('P4 64/12 pilot model not ready')
            j=json.loads(report.read_text())
            self.assertEqual(j['train_views'],['back','opposite_side'])
            self.assertEqual(j['checkpoint_selection_metric'],'back_opposite')
            self.assertEqual(j['train_poses'],64)
            self.assertEqual(j['val_poses'],12)
            self.assertEqual(j['input_checkpoint_sha256'],sha(root/source))
            self.assertEqual(j['trained_checkpoint_sha256'],sha(root/'reader_backopp_p4_pilot_v1.pt'))
            self.assertTrue(set(j['train_pose_ids']).isdisjoint(j['validation_pose_ids']))


if __name__=='__main__':unittest.main()
