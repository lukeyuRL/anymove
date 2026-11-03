"""P3 material correction regression: exact pose split, true colour data, checkpoints.

These tests use immutable local Blender outputs and fail if gray Workbench
images masquerade as a legitimate material-preserving four-view Reader set.
"""
import json
import unittest
from pathlib import Path

import numpy as np
from mocap.fourview_data import FourViewDataset,split_poses,sha256
from tools.audit_p3_source_material_dataset import audit

ROOT=Path(__file__).resolve().parents[1]
VIEWS=("front","side","back","opposite_side")


class SourceMaterialFourViewTests(unittest.TestCase):
    def fixture(self,cr):
        base=ROOT/"runs"/cr/"dataset_fourview_p3_material_fullres_v1"
        rig=ROOT/"assets"/cr/"rigged/rig_info.json"
        if len(list(base.glob("*_visibility_meta.json")))<28:
            self.skipTest("P3 full-res source-material Dataset not fully rendered")
        return base,rig

    def test_all_four_views_per_pose_holdout_is_disjoint(self):
        for cr in ("dino","dragon"):
            base,rig=self.fixture(cr)
            tr,val=split_poses(base)
            self.assertEqual(tr,tuple(range(1,25)))
            self.assertEqual(val,(25,50,75,100))
            ds=FourViewDataset(base,rig,[25],train=False)
            self.assertEqual(ds.appearance_mode,"source-material")
            self.assertEqual(len(ds),4)
            for i in range(4):
                r=ds[i]
                self.assertEqual(int(r["view_index"]),i)
                self.assertTrue(np.isin(r["visibility"].numpy(),[-1,0,1]).all())

    def test_true_material_colors_and_depth_integrity(self):
        for cr in ("dino","dragon"):
            base,rig=self.fixture(cr)
            result=audit(base,rig)
            self.assertEqual(result["status"],"PASS")
            self.assertEqual(result["source_poses"],28)
            self.assertEqual(result["rendered_views"],112)
            self.assertEqual(result["pose_ids"]["validation"],[25,50,75,100])
            self.assertLess(result["max_replay_error_m"],.001)
            self.assertLess(result["max_depth_raycast_p95_m"],.03)
            self.assertTrue(result["all_appearance_source_material"])
            if cr=="dino":
                # Source baked colour markers normally cover a sizable share
                # of the foreground; all gray Workbench data would score zero.
                self.assertGreater(result["view_color_saturation"]["front"]["mean"],.18)
                self.assertGreater(result["view_color_saturation"]["side"]["mean"],.05)

    def test_original_checkpoints_never_overwritten(self):
        for cr,original in (("dino","reader_hr_v1.pt"),
                            ("dragon","reader_hr.pt")):
            report=ROOT/"runs"/cr/"reader_fourview_p3_material_fullres_v1.validation.json"
            if not report.is_file():self.skipTest("P3 trained full-res checkpoint not ready")
            m=json.loads(report.read_text())
            self.assertEqual(m["training_appearance"],"source-material")
            self.assertEqual(m["validation_appearance"],"source-material")
            self.assertFalse(m["legacy_gray_appearance_explicitly_allowed"])
            self.assertEqual(set(m["train_pose_ids"])&set(m["validation_pose_ids"]),set())
            self.assertEqual(m["input_checkpoint_sha256"],
                sha256(ROOT/"runs"/cr/original))
            self.assertEqual(m["trained_checkpoint_sha256"],sha256(m["trained_checkpoint"]))
            self.assertNotEqual(Path(m["trained_checkpoint"]).resolve(),
                                (ROOT/"runs"/cr/original).resolve())

    def test_independent_3d_and_quality_gate(self):
        from tools.eval_p3_source_material import eval_material_run
        for cr in ("dino","dragon"):
            p=ROOT/"runs"/cr/"fourview_p3_material_fullres_acceptance_v1.json"
            if not p.is_file():self.skipTest("P3 3D independent GT report not ready")
            obj=json.loads(p.read_text(encoding="utf8"))
            self.assertEqual(obj,eval_material_run(cr,ROOT))
            self.assertEqual(obj["appearance"],"source-material")
            self.assertEqual(obj["train_pose_count"],24)
            self.assertEqual(obj["validation_pose_count"],4)
            self.assertTrue(obj["source_pose_gt_stored_separately"])
            self.assertEqual(obj["checkpoint_hash_integrity"],"PASS")
            self.assertIn("production_promotion_allowed",obj)


if __name__=="__main__":
    unittest.main()
