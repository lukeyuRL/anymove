"""Pure Python unit contracts for 672px Human V3 Nose 315/45 dataset."""
import json
import math
import pathlib
import unittest

import numpy as np

from mocap.camera_geometry import project_orthographic
from tools import audit_human_v3_hr_v2_315_45 as audit
from tools import human_v3_fullbody_motion_v2 as motion

class HRV2DatasetContracts(unittest.TestCase):
    def test_projection_matrices_are_calibrated(self):
        rng=np.random.default_rng(20261104)
        xyz=rng.normal(size=(67,3))
        for slot,cam in audit.calibrated_cameras().items():
            matrix=np.asarray(cam["world_xyz1_to_pixel_uv_2x4"],dtype=np.float64)
            pos=np.column_stack((xyz,np.ones(len(xyz))))
            uv=pos@matrix.T
            target=project_orthographic(xyz,audit.ANGLE_MAP[slot],672,672,audit.M_PER_PX,audit.Z_CENTER)
            np.testing.assert_allclose(uv,target,rtol=0,atol=1e-10)
    def test_stereo_315_45_not_opposite(self):
        self.assertEqual(set(audit.ANGLE_MAP),{"front","side"})
        self.assertAlmostEqual(audit.ANGLE_MAP["front"],315)
        self.assertAlmostEqual(audit.ANGLE_MAP["side"],45)
    def test_expected_image_counts_and_splits(self):
        self.assertEqual(audit.EXPECTED_CLIPS,192)
        self.assertEqual(audit.EXPECTED_PAIRS,2688)
        self.assertEqual(audit.EXPECTED_PNG,5376)
        self.assertEqual(audit.SPLIT_PNG,{"train":4032,"validation":672,"test":672})
        self.assertEqual(audit.SLOT_COUNTS,{"front":2688,"side":2688})
        self.assertGreaterEqual(min(audit.SLOT_COUNTS.values()),2200)
        self.assertEqual(audit.PER_FAMILY_CLIPS,16)
        self.assertEqual(len(motion.FAMILIES),12)
    def test_oracle_crop_boxes_are_in_bounds(self):
        name_data=json.loads(pathlib.Path("runs/human/fullbody_finger_v2_hr_315_45/smoke/smoke_manifest.json").read_text(encoding="utf8"))
        names=name_data["joint_names"]
        row=name_data["rows"]["train"][2]
        crop=audit.oracle_training_crop_hints(row,names)
        self.assertEqual(set(crop),{"L_hand","R_hand","L_foot","R_foot"})
        for block in crop.values():
            x1,y1,x2,y2=block["xyxy_px"]
            self.assertGreaterEqual(x1,0)
            self.assertGreaterEqual(y1,0)
            self.assertLessEqual(x2,672)
            self.assertLessEqual(y2,672)
            self.assertEqual(x2-x1,y2-y1)
            self.assertGreaterEqual(x2-x1,112)
            self.assertEqual(block["source"],"TRAIN_GT_ONLY")
    def test_new_frame_schedule_never_includes_rest(self):
        frames=np.linspace(8,98,14).round().astype(int).tolist()
        self.assertEqual(len(frames),len(set(frames)))
        self.assertTrue(all(0<frame<106 for frame in frames))

if __name__=="__main__":
    unittest.main()
