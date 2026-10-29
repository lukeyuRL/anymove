"""Fail closed when Dragon Solver silently inherits Wolf camera layout."""
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from mocap.layout_guard import verify_layout_for_rig


class CameraLayoutGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rig=pathlib.Path(self.temp.name)/"assets"/"dragon"/"rigged"/"rig_info.json"
        self.rig.parent.mkdir(parents=True)
        self.rig.write_text("{}",encoding="utf-8")
        config=self.rig.parent.parent/"creature.json"
        config.write_text(json.dumps({"name":"dragon","layout":{
            "W_FRONT":640,"W_SIDE":640,
            "M_PER_PX":.006875,"GROUND_V":630}}),encoding="utf-8")

    def test_correct_dragon_camera_layout_passes(self):
        current=SimpleNamespace(W_FRONT=640,W_SIDE=640,
                                M_PER_PX=.006875,GROUND_V=630)
        result=verify_layout_for_rig(self.rig,current)
        self.assertEqual(result["status"],"camera_layout_verified_for_rig")

    def test_wolf_fallback_is_explicitly_blocked(self):
        wrong=SimpleNamespace(W_FRONT=420,W_SIDE=860,
                              M_PER_PX=1.9/720,GROUND_V=630)
        with self.assertRaisesRegex(RuntimeError,"CAMERA_LAYOUT_MISMATCH"):
            verify_layout_for_rig(self.rig,wrong)
        with self.assertRaisesRegex(RuntimeError,"MOCAP_CREATURE"):
            verify_layout_for_rig(self.rig,wrong)

    def test_missing_camera_field_is_blocked(self):
        wrong=SimpleNamespace(W_FRONT=640,W_SIDE=640,M_PER_PX=.006875)
        with self.assertRaisesRegex(RuntimeError,"GROUND_V"):
            verify_layout_for_rig(self.rig,wrong)

    def test_unknown_legacy_rig_does_not_assume_dragon(self):
        path=pathlib.Path(self.temp.name)/"legacy.json"
        result=verify_layout_for_rig(path,SimpleNamespace())
        self.assertEqual(result["status"],"rig_config_not_inferred")


if __name__=="__main__":
    unittest.main()
