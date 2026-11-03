"""Dino controlled 41-color vs original-material matched A/B regression.

Tests are local and read-only, leveraging immutable data from new-seed Blender
poses 11001..11012. Must never leak evaluation GT into Reader or FK input.
"""
import json,unittest
from pathlib import Path
import numpy as np
from tools.eval_dino_color41_ab import analysis,report_metrics,TRAIN_IDS,VAL_IDS,TEST_IDS,hash_file

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/"runs/dino/skinmax_camera_aware_v1"
class DinoColorABTests(unittest.TestCase):
    def test_new_seed_is_not_palette_tuning_or_training(self):
        for pose in TEST_IDS:
            p=E/"ab_blind_new_pose_v1"/f"{pose:05d}_meta.json"
            if not p.exists():self.skipTest("new-seed pose generation missing")
            j=json.loads(p.read_text(encoding="utf8"))
            self.assertEqual(j["sample_seed"],20261221+pose)
            self.assertEqual(j["schema"],"anymove.p4_independent_3d_blender_pose/1")
            self.assertTrue(j["not_model_training"])
            self.assertFalse(j["blur"])
        self.assertTrue(set(TEST_IDS).isdisjoint(TRAIN_IDS+VAL_IDS))
        self.assertTrue(set(TEST_IDS).isdisjoint({9001,9002,9007}))
    def test_paired_material_data_same_rig_camera_gt_and_pose(self):
        path=E/"ab_dataset_contract_v1.json"
        if not path.exists():self.skipTest("same-material dataset not yet rendered")
        data=json.loads(path.read_text())
        self.assertEqual(data,analysis(ROOT))
        self.assertTrue(data["training"]["RGB_difference_detected"])
        self.assertTrue(data["new_blind"]["RGB_difference_detected"])
        self.assertEqual(data["training"]["pose_count"],28)
        self.assertEqual(data["new_blind"]["pose_count"],12)
        self.assertTrue(data["test_pose_not_used_in_palette_optimizer"])
    def test_both_trainings_start_from_exact_same_checkpoint(self):
        reports=[]
        for label,kind in (("A","original"),("B","color41")):
            path=E/f"reader_{label}_{kind}_5ep.validation.json"
            if not path.exists():self.skipTest("both trained A/B checkpoints not yet saved")
            r=json.loads(path.read_text())
            reports.append(r)
            self.assertEqual(r["train_pose_ids"],list(TRAIN_IDS))
            self.assertEqual(r["validation_pose_ids"],list(VAL_IDS))
            self.assertEqual(r["train_poses"],24)
            self.assertEqual(r["val_poses"],4)
            self.assertEqual(r["input_size"],672)
            self.assertEqual(r["heatmap_size"],192)
            self.assertEqual(r["train_views"],["front","side","back","opposite_side"])
            self.assertTrue(r["frozen_backbone"])
            self.assertEqual(r["input_checkpoint_sha256"],hash_file(ROOT/"runs/dino/reader_hr_v1.pt"))
        self.assertEqual(reports[0]["input_checkpoint_sha256"],reports[1]["input_checkpoint_sha256"])
        self.assertEqual(reports[0]["train_pose_ids"],reports[1]["train_pose_ids"])
        self.assertEqual(reports[0]["validation_pose_ids"],reports[1]["validation_pose_ids"])
    def test_2d_3d_blind_reports_separate_gt_and_fk(self):
        path=E/"ab_result_v1.json"
        if not path.exists():self.skipTest("Reader 2D + 3D A/B not complete")
        obj=json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(obj,report_metrics(ROOT))
        self.assertTrue(obj["checkpoint_training_matched"])
        self.assertEqual(obj["pose_count"],12)
        for label in ("A","B"):
            self.assertEqual(set(obj["reader_2d_per_view"][label]),{"front","side","back","opposite_side"})
            self.assertEqual(set(obj["3d"][label]),{"2v","4v"})
            self.assertGreater(obj["groups"]["toes"],0)
            self.assertGreater(obj["groups"]["forelimbs_hands"],0)
            self.assertGreater(obj["groups"]["tail"],0)
            path=E/f"ab_blind_{label}_{'original' if label=='A' else 'color41'}_v1"/f"reader_{label}_blind.npz"
            with np.load(path,allow_pickle=False) as p:
                self.assertNotIn("joints_world",p.files)
                self.assertNotIn("gt_world",p.files)
                self.assertNotIn("joint_surface_part_visible",p.files)

if __name__=="__main__":unittest.main()
