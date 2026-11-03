"""Independent P1 four-view sample audit regression and tamper detection."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from tools.audit_four_view_visibility import audit_one, VIEWS

ROOT=Path(__file__).resolve().parents[1]


class FourViewVisibilityArtifactTests(unittest.TestCase):
    def test_replay_and_raycast_for_available_creatures(self):
        for creature,data in (("dino","dataset_hr_v1"),("dragon","dataset_hr")):
            sample=ROOT/"runs"/creature/"four_view_visibility_p1_validation"
            rig=ROOT/"assets"/creature/"rigged/rig_info.json"
            src=ROOT/"runs"/creature/data
            if not (sample/"00000_visibility_meta.json").exists():
                self.skipTest("generated P1 fixture missing for "+creature)
            for i in (0,1):
                with self.subTest(creature=creature,sample=i):
                    r=audit_one(sample,i,rig,src/f"{i:05d}_meta.json",False)
                    self.assertEqual(len(r["views"]),4)
                    self.assertLess(r["replay_max_joint_world_error_m"],.001)
                    self.assertLess(r["camera_opposite_max_error_px"],.02)
                    for v in VIEWS:
                        self.assertGreater(r["views"][v]["render_mask_iou"],.88)
                        self.assertLess(r["views"][v]["ray_depth_p95_error_m"],.03)
                        self.assertLess(r["views"][v]["source_3d_gt_projection_max_error_px"],.1)
            self.assertEqual(len(list(sample.glob("*.mp4"))),0)

    def test_reject_tampered_part_id_not_a_valid_surface_gt(self):
        creature="dino"
        out=ROOT/"runs"/creature/"four_view_visibility_p1_validation"
        rig=ROOT/"assets"/creature/"rigged/rig_info.json"
        src=ROOT/"runs"/creature/"dataset_hr_v1/00000_meta.json"
        if not (out/"00000_visibility_meta.json").is_file():
            self.skipTest("local P1 fixture not available")
        with tempfile.TemporaryDirectory() as td:
            t=Path(td)
            shutil.copy2(out/"00000_visibility_meta.json",t)
            for v in VIEWS:
                shutil.copy2(out/f"00000_{v}.png",t)
                shutil.copy2(out/f"00000_{v}_labels.npz",t)
            f=t/"00000_front_labels.npz"
            with np.load(f) as z:
                payload={name:z[name].copy() for name in z.files}
            y,x=np.argwhere(payload["part_id"]>=0)[0]
            payload["part_id"][y,x]=-1
            np.savez_compressed(f,**payload)
            with self.assertRaisesRegex(ValueError,"depth-to-part"):
                audit_one(t,0,rig,src,False)

    def test_reject_source_metadata_not_matching_original(self):
        out=ROOT/"runs"/"dino"/"four_view_visibility_p1_validation"
        rig=ROOT/"assets/dino/rigged/rig_info.json"
        if not (out/"00000_visibility_meta.json").is_file():
            self.skipTest("local P1 fixture not available")
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"00000_meta.json"
            p.write_text(json.dumps({"joints_world":[]}))
            with self.assertRaisesRegex(ValueError,"provenance hash"):
                audit_one(out,0,rig,p,False)


if __name__=="__main__":
    unittest.main()
