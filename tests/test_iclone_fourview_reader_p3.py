"""P3 four-view Reader dataset/quality supervision contract regression."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from mocap.fourview_data import (
    FourViewDataset, split_poses, sha256,
    expected_visibility_and_error_summary, VIEWS)
from mocap.reader_config import spec_from_checkpoint

ROOT=Path(__file__).resolve().parents[1]


class FourViewReaderTrainingTests(unittest.TestCase):
    def dataset(self,creature):
        return ROOT/"runs"/creature/"dataset_fourview_p3_smoke_v1"

    def test_validation_is_pose_disjoint_and_all_four_views_stay_together(self):
        d=self.dataset("dino")
        if not d.exists():self.skipTest("P3 smoke not rendered on this runner")
        train,val=split_poses(d)
        self.assertEqual(len(train),24)
        self.assertEqual(len(val),4)
        self.assertFalse(set(train)&set(val))
        info=ROOT/"assets/dino/rigged/rig_info.json"
        tr=FourViewDataset(d,info,train,train=True)
        vl=FourViewDataset(d,info,val,train=False)
        self.assertEqual(len(tr),4*len(train))
        self.assertEqual(len(vl),4*len(val))
        self.assertFalse(set(tr.sample_keys)&set(vl.sample_keys))
        for pose in val:
            self.assertEqual([v for p,v in vl.sample_keys if p==pose],list(VIEWS))

    def test_source_material_appearance_preserves_dino_leg_colours(self):
        from mocap.fourview_data import FourViewDataset
        import cv2
        old=ROOT/'runs/dino/dataset_fourview_p3_smoke_v1'
        fixed=ROOT/'runs/dino/four_view_material_fix_fullres_v1'
        rig=ROOT/'assets/dino/rigged/rig_info.json'
        if not (old/'00001_front.png').is_file() or not (fixed/'00001_front.png').is_file():
            self.skipTest('local original gray and source-material smoke required')
        previous=FourViewDataset(old,rig,[1],train=False)
        repaired=FourViewDataset(fixed,rig,[1],train=False)
        self.assertEqual(previous.appearance_mode,'neutral-geometry')
        self.assertEqual(repaired.appearance_mode,'source-material')
        def paint_fraction(img):
            hsv=cv2.cvtColor(img[:,:,:3],cv2.COLOR_BGR2HSV)
            fg=img[:,:,3]>64
            bright=fg&(hsv[:,:,1]>110)
            red=bright&((hsv[:,:,0]<11)|(hsv[:,:,0]>170))
            blue=bright&(hsv[:,:,0]>95)&(hsv[:,:,0]<138)
            return bright.sum()/max(fg.sum(),1),int(red.sum()),int(blue.sum())
        gray=cv2.imread(str(old/'00001_front.png'),cv2.IMREAD_UNCHANGED)
        coloured=cv2.imread(str(fixed/'00001_front.png'),cv2.IMREAD_UNCHANGED)
        a,ar,ab=paint_fraction(gray)
        b,br,bb=paint_fraction(coloured)
        self.assertLess(a,.01)
        self.assertGreater(b,.30)
        self.assertGreater(br,300)
        self.assertGreater(bb,300)

    def test_mesh_visibility_unknown_labels_and_camera_geometry(self):
        d=self.dataset("dino")
        if not d.exists():self.skipTest("local P3 dataset required")
        v=FourViewDataset(d,ROOT/"assets/dino/rigged/rig_info.json",[25],
                            train=False)
        for j in range(4):
            sample=v[j]
            self.assertEqual(tuple(sample["image"].shape),(3,672,672))
            self.assertEqual(tuple(sample["keypoints"].shape),(92,2))
            self.assertEqual(tuple(sample["visibility"].shape),(92,))
            self.assertTrue(np.isin(sample["visibility"].numpy(),[-1,0,1]).all())
            self.assertTrue(sample["inside"].dtype==torch.bool)
            self.assertTrue(torch.isfinite(sample["image"]).all())
            self.assertEqual(int(sample["view_index"]),j)

    def test_smoke_acceptance_disallows_production_promotion(self):
        from tools.eval_four_view_reader_p3 import summarize
        for creature in ('dino','dragon'):
            report=ROOT/'runs'/creature/'fourview_reader_p3_acceptance_v1.json'
            if not report.exists():self.skipTest('P3 acceptance report not yet produced')
            actual=summarize(creature,ROOT)
            self.assertEqual(actual,json.loads(report.read_text()))
            self.assertFalse(actual['production_promotion_allowed'])
            self.assertFalse(actual['gates']['no_two_view_3d_regression'])
            self.assertEqual(actual['train_pose_count'],24)
            self.assertEqual(actual['validation_pose_count'],4)

    def test_frozen_backbone_unchanged_bitwise(self):
        for creature in ('dino','dragon'):
            original=ROOT/'runs'/creature/('reader_hr_v1.pt' if creature=='dino' else 'reader_hr.pt')
            tuned=ROOT/'runs'/creature/'reader_fourview_p3_smoke_v1.pt'
            if not tuned.exists():self.skipTest('P3 checkpoint not yet trained')
            b=torch.load(original,map_location='cpu',weights_only=True)['state_dict']
            t=torch.load(tuned,map_location='cpu',weights_only=True)['state_dict']
            backbone=[key for key in b if key.startswith('backbone.')]
            self.assertGreater(len(backbone),100)
            for key in backbone:
                self.assertTrue(torch.equal(b[key],t[key]),f'{creature} backbone changed at {key}')
            self.assertIn('quality_out.weight',t)
            self.assertFalse(torch.equal(b['out.weight'],t['out.weight']))

    def test_source_dataset_not_updated_by_new_trainer(self):
        trained=ROOT/"runs/dino/reader_fourview_p3_smoke_v1.validation.json"
        if not trained.is_file():self.skipTest("local trained checkpoint missing")
        report=json.loads(trained.read_text())
        base=Path(report["input_checkpoint"])
        self.assertEqual(sha256(base),report["input_checkpoint_sha256"])
        self.assertEqual(set(report["train_pose_ids"])&set(report["validation_pose_ids"]),set())
        self.assertEqual(report["train_poses"],24)
        self.assertEqual(report["val_poses"],4)
        original=torch.load(base,map_location="cpu",weights_only=True)
        assert original.get("quality_spec") is None
        final=torch.load(report["trained_checkpoint"],map_location="cpu",weights_only=True)
        self.assertEqual(final["quality_spec"]["version"],2)
        self.assertEqual(final["quality_spec"]["target"],"mesh_surface_part_visibility")
        self.assertEqual(list(final["names"]),list(original["names"]))
        self.assertEqual(spec_from_checkpoint(final),spec_from_checkpoint(original))
        self.assertTrue(final["reader_spec"]["quality_head"])
        self.assertTrue(final["four_view_spec"]["frozen_backbone"])

    def test_unknown_mesh_surface_labels_not_calculated_as_visible(self):
        scores=np.asarray([1.,0.,0.5])
        y=np.asarray([1,0,-1])
        estimated=np.asarray([1.,2.,10.])
        actual=np.asarray([1.,2.,4.])
        report=expected_visibility_and_error_summary(scores,y,estimated,actual)
        self.assertEqual(report["known_surface_part_samples"],2)
        self.assertEqual(report["unknown_surface_part_samples"],1)
        self.assertAlmostEqual(report["brier"],0.)
        self.assertAlmostEqual(report["auc"],1.)
        self.assertEqual(report["occluded_surface_part_samples"],1)

    def test_p3_reader_packet_keeps_mesh_visibility_training_gt_separate(self):
        path=ROOT/'runs/dino/dataset_fourview_p3_smoke_v1/p3_trained_heldout_views.npz'
        if not path.exists():self.skipTest('P3 inference packet not generated')
        from mocap.multiview import load_packet
        p,meta=load_packet(path)
        self.assertEqual(meta['source_kind'],'reader_inference')
        self.assertEqual(p['quality_semantics'].item(),
                         'mesh_surface_part_visibility_v1_uncalibrated')
        self.assertEqual(p['visibility_score'].shape,(4,4,92))
        self.assertTrue((p['visibility_score']>=0).all())
        self.assertTrue((p['visibility_score']<=1).all())
        self.assertTrue((p['expected_error_px']>=0).all())
        self.assertNotIn('joint_surface_part_visible',p)
        self.assertNotIn('joints_world',p)

    def test_quality_calibration_is_a_report_not_assumed_real_probability(self):
        file=ROOT/"runs/dino/reader_fourview_p3_smoke_v1.validation.json"
        if not file.exists():self.skipTest("P3 validation report not produced yet")
        data=json.loads(file.read_text(encoding="utf8"))
        self.assertIn("NOT H3-calibrated",data["visibility_semantics"])
        best=data["history"][-1]["validation"]
        for view in VIEWS:
            q=best[view]["quality"]
            self.assertGreater(q["known_surface_part_samples"],0)
            self.assertGreater(q["occluded_surface_part_samples"],0)
            self.assertTrue(0<=q["brier"]<=1)
            self.assertTrue(0<=q["ece_8bins"]<=1)
            self.assertIn("mean_error_calibration_gap_px",
                          q["localization_error_calibration"])


if __name__=="__main__":
    unittest.main()
