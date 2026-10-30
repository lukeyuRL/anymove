"""Regression tests for reframe_existing_rig; no Blender needed."""
import json
from pathlib import Path
import tempfile
import unittest
from tools.reframe_existing_rig import reframe


class ReframeTests(unittest.TestCase):
    def test_updates_camera_without_resetting_authoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dino"
            rigged = root / "rigged"
            rigged.mkdir(parents=True)
            creature = {
                "name": "dino", "paint": {"FL": {"color": "#F53CD1", "bones": ["foo"]}},
                "feet": ["foot_left"], "custom_field": 99,
                "layout": {"W_FRONT": 640, "W_SIDE": 640,
                           "M_PER_PX": .03647943, "GROUND_V": 630},
            }
            info = {
                "bones": [{"name": "root", "limits_deg": [[-12, 24], [-5, 5], [-3, 3]]}],
                "keypoints": [{"name": "hip", "rest": [0,0,1]}],
                "layout": {"H": 720, "W_FRONT": 640, "W_SIDE": 640,
                           "M_PER_PX": .03647943, "GROUND_V": 630,
                           "Z_CENTER": 9.85, "Y_CENTER": 0, "X_CENTER": 0},
            }
            report = {
                "bounds_min": [-2.0444216728, -7.3383641243, -.0268925745],
                "bounds_max": [2.0446081161, 8.6281785965, 5.9318022728],
                "warnings": ["prior issue"],
            }
            names = ["creature.json", "rigged/rig_info.json", "rigged/rig_import_report.json"]
            for path, value in zip(names, (creature, info, report)):
                (root / path).write_text(json.dumps(value), encoding="utf-8")
            result = reframe(root)
            self.assertEqual(result["status"], "framing_updated")
            revised = json.loads((root / "creature.json").read_text())
            revised_info = json.loads((root / "rigged" / "rig_info.json").read_text())
            self.assertEqual((revised["layout"]["W_FRONT"], revised["layout"]["W_SIDE"]),
                             (400, 880))
            self.assertEqual(revised["dataset"]["yaw_max_deg"], 10.0)
            self.assertEqual(revised["paint"], creature["paint"])
            self.assertEqual(revised["feet"], ["foot_left"])
            self.assertEqual(revised["custom_field"], 99)
            self.assertEqual(revised_info["bones"], info["bones"])
            self.assertEqual(revised_info["keypoints"], info["keypoints"])
            self.assertLess(revised["layout"]["M_PER_PX"], .027)
            self.assertEqual(len(result["backups"]), 3)
            self.assertTrue(all(Path(p).is_file() for p in result["backups"]))
            again = reframe(root)
            self.assertEqual(again["status"], "unchanged")

    def test_reject_missing_metadata_without_editing(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                reframe(root)


if __name__ == "__main__":
    unittest.main()
