"""Video source integration UI headless helpers (no PySide2 installed)."""
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import test_iclone_dragon_preview as fixtures
import test_iclone_direct_import as direct_mock
import iclone_adapter.video_import_ui as video_ui
from iclone_adapter.schema import ContractError


class VideoImportUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DragonPreviewTests.setUpClass()
        cls.dragon=fixtures.DragonPreviewTests()

    def test_choose_external_python_never_uses_iclone_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp)
            self.assertIsNone(video_ui.find_external_python(p))
            exe=p/".venv"/"Scripts"/"python.exe"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"fake executable")
            self.assertEqual(video_ui.find_external_python(p),str(exe))

    def test_unique_capture_folder_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=pathlib.Path(tmp)
            first=video_ui.unique_capture_folder(parent,"F:\\videos\\dragon.mp4")
            self.assertFalse(first.exists())
            first.mkdir()
            self.assertTrue(first.exists())
            next_path=video_ui.unique_capture_folder(parent,"F:\\videos\\dragon.mp4")
            self.assertNotEqual(next_path,first)

    def test_worker_arguments_MP4_and_NPZ(self):
        video=video_ui.worker_arguments(
            "/x/demo.mp4","/y/output","/x/reader.pt","/x/ref.png")
        self.assertEqual(video[0:2],["-m","iclone_adapter.video_capture"])
        self.assertIn("--reader",video)
        self.assertIn("--reference",video)
        npz=video_ui.worker_arguments("/x/det.npz","/y/output")
        self.assertNotIn("--reader",npz)
        self.assertNotIn("--reference",npz)
        with self.assertRaises(ContractError):
            video_ui.worker_arguments("/x/demo.mp4","/y/output")
        with self.assertRaises(ContractError):
            video_ui.worker_arguments("/x/photo.jpg","/y/output")

    def test_clean_dragon_preflight_without_scene_mutation(self):
        sdk=direct_mock.SDK()
        snapshot=self.dragon.make_target()
        with mock.patch.object(video_ui,"snapshot",return_value=snapshot):
            with mock.patch.object(video_ui,"check_clean_clip",
                      return_value={"existing_clip_count":0}):
                result=video_ui.preflight_video_target(sdk)
        self.assertEqual(result["matched_bones"],42)
        self.assertAlmostEqual(result["fbx_import_scale"],100.,places=5)
        self.assertEqual(result["project_fps"],24.)
        self.assertEqual(sdk.RFileIO.saved,[])

    def test_preflight_rejects_other_rig_and_nonzero_frame(self):
        sdk=direct_mock.SDK()
        target=self.dragon.make_target()
        # Changing a helper display name alone does not invalidate the
        # same Skin-Bone hierarchy. Modify actual Head Rest position.
        next(b for b in target["bones"] if b["name"]=="head")[
            "snapshot_world_matrix"][3]+=25.0
        with mock.patch.object(video_ui,"snapshot",return_value=target):
            with self.assertRaises(ContractError):
                video_ui.preflight_video_target(sdk)
        sdk.RGlobal.t=10
        with self.assertRaisesRegex(ContractError,"Frame 0"):
            video_ui.preflight_video_target(sdk)

    def test_preflight_rejects_nonempty_scene(self):
        sdk=direct_mock.SDK()
        target=self.dragon.make_target()
        with mock.patch.object(video_ui,"snapshot",return_value=target):
            with mock.patch.object(
                video_ui,"check_clean_clip",
                side_effect=ContractError("dirty existing Clip")):
                with self.assertRaisesRegex(ContractError,"dirty"):
                    video_ui.preflight_video_target(sdk)


if __name__=="__main__":
    unittest.main()
