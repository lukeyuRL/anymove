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

    def test_prompt_worker_arguments_does_not_send_requests(self):
        command=video_ui.prompt_worker_arguments(
            "the Dragon roars","/tmp/new","/tmp/model.pt","/tmp/ref.png")
        self.assertEqual(command[:2],["-m","iclone_adapter.prompt_capture"])
        self.assertEqual(command[command.index("--model")+1],"h3max")
        self.assertIn("--action",command)
        self.assertNotIn("fal.ai key",str(command))
        with self.assertRaises(ContractError):
            video_ui.prompt_worker_arguments(
                "","/tmp/new","/tmp/model.pt","/tmp/ref.png")

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

    @staticmethod
    def _capture_widgets(root):
        widgets=mock.Mock()
        widgets.QMessageBox.Yes=1
        widgets.QMessageBox.No=2
        widgets.QMessageBox.question.return_value=1
        widgets.QInputDialog.getMultiLineText.return_value=(
            "the dragon takes off, roars, then lands.",True)
        widgets.QFileDialog.getExistingDirectory.return_value=str(root)
        core=mock.Mock()
        core.QProcess.NormalExit=0
        panel=mock.Mock()
        coordinator=video_ui.VideoCaptureCoordinator(
            panel,widgets,core,object())  # no selected RLPy actor
        return coordinator,widgets,panel

    @staticmethod
    def _prepare_capture_assets(root):
        reader=root/"runs"/"dragon"/"reader.pt"
        reference=root/"runs"/"dragon"/"ref"/"ref.png"
        python=root/".venv"/"Scripts"/"python.exe"
        for path in (reader,reference,python):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b"test fixture")
        return reader,reference,python

    def test_prompt_generation_ignores_selected_dragon_until_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            self._prepare_capture_assets(root)
            coordinator,widgets,_=self._capture_widgets(root)
            with mock.patch.object(video_ui,"REPO",root):
                with mock.patch.object(video_ui,"_credential_available",
                                       return_value=True):
                    with mock.patch.object(
                        video_ui,"preflight_video_target",
                        side_effect=AssertionError("early rig check")):
                        with mock.patch.object(
                            coordinator,"_launch_worker") as launch:
                            with mock.patch.object(
                                video_ui,"preflight_external_python",
                                return_value={"executable":"mock-python"}):
                                coordinator.start_prompt()
            launch.assert_called_once()
            args,kwargs=launch.call_args
            self.assertEqual(args[2]["mode"],"prompt")
            self.assertNotIn("target",args[2])
            self.assertIn("--action",args[1])
            self.assertTrue(widgets.QMessageBox.question.called)
            widgets.QMessageBox.warning.assert_not_called()

    def test_existing_video_capture_ignores_selected_dragon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            self._prepare_capture_assets(root)
            source=root/"input.mp4"
            source.write_bytes(b"fixture")
            coordinator,widgets,_=self._capture_widgets(root)
            widgets.QFileDialog.getOpenFileName.return_value=(str(source),"")
            with mock.patch.object(video_ui,"REPO",root):
                with mock.patch.object(video_ui,"_input_plan",return_value={}):
                    with mock.patch.object(
                        video_ui,"preflight_video_target",
                        side_effect=AssertionError("early rig check")):
                        with mock.patch.object(
                            coordinator,"_launch_worker") as launch:
                            with mock.patch.object(
                                video_ui,"preflight_external_python",
                                return_value={"executable":"mock-python"}):
                                coordinator.start()
            launch.assert_called_once()
            self.assertEqual(launch.call_args[0][2]["mode"],"existing")
            self.assertNotIn("target",launch.call_args[0][2])
            widgets.QMessageBox.warning.assert_not_called()

    def test_missing_fal_key_blocks_before_paid_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            self._prepare_capture_assets(root)
            coordinator,widgets,_=self._capture_widgets(root)
            with mock.patch.object(video_ui,"REPO",root):
                with mock.patch.object(video_ui,"_credential_available",
                                       return_value=False):
                    with mock.patch.object(
                        coordinator,"_launch_worker") as launch:
                        with mock.patch.object(
                            video_ui,"preflight_external_python",
                            return_value={"executable":"mock-python"}):
                            coordinator.start_prompt()
            launch.assert_not_called()
            widgets.QMessageBox.question.assert_not_called()
            self.assertIn("FAL_KEY",str(widgets.QMessageBox.warning.call_args))

    def test_prompt_requires_explicit_billable_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            self._prepare_capture_assets(root)
            coordinator,widgets,_=self._capture_widgets(root)
            widgets.QMessageBox.question.return_value=widgets.QMessageBox.No
            with mock.patch.object(video_ui,"REPO",root):
                with mock.patch.object(video_ui,"_credential_available",
                                       return_value=True):
                    with mock.patch.object(
                        coordinator,"_launch_worker") as launch:
                        with mock.patch.object(
                            video_ui,"preflight_external_python",
                            return_value={"executable":"mock-python"}):
                            coordinator.start_prompt()
            launch.assert_not_called()
            widgets.QMessageBox.warning.assert_not_called()

    def test_solved_motion_preserved_if_selected_target_is_not_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            capture=root/"capture"
            capture.mkdir()
            motion=capture/"motion.json"
            motion.write_text('{"test":"ready"}',encoding="utf-8")
            (capture/video_ui.RECEIPT).write_text(
                json.dumps({"status":"motion_json_ready",
                            "motion_file":str(motion)}),encoding="utf-8")
            coordinator,widgets,panel=self._capture_widgets(root)
            coordinator.job={"capture_path":str(capture),"mode":"prompt"}
            with mock.patch.object(
                video_ui,"prepare_direct_import",
                side_effect=ContractError("scene already has Motion Clips")):
                with mock.patch.object(
                    video_ui,"execute_direct_import") as write:
                    coordinator._finished(0,coordinator.core.QProcess.NormalExit)
            write.assert_not_called()
            self.assertTrue(motion.exists())
            self.assertIn("scene already has Motion Clips",
                          str(widgets.QMessageBox.warning.call_args))
            self.assertIn("import deferred",
                          str(widgets.QMessageBox.warning.call_args).lower())
            self.assertIn(str(motion),
                          str(panel.out.appendPlainText.call_args))


    def test_external_python_preflight_rejects_wrong_version(self):
        check=mock.Mock(returncode=0,stdout=json.dumps({
            "version":[3,12],"executable":"python312","missing":[]}))
        with mock.patch.object(video_ui.subprocess,"run",
                               return_value=check) as runner:
            with self.assertRaisesRegex(ContractError,"requires Python 3.11 or 3.14"):
                video_ui.preflight_external_python(
                    "fake-python",("torch","numpy"))
        self.assertEqual(runner.call_count,1)

    def test_external_python_preflight_accepts_python_314(self):
        info={"version":[3,14],"executable":"python314","missing":[]}
        check=mock.Mock(returncode=0,stdout=json.dumps(info))
        with mock.patch.object(video_ui.subprocess,"run",return_value=check):
            self.assertEqual(video_ui.preflight_external_python(
                "python314",("torch","numpy")),info)

    def test_external_python_preflight_rejects_missing_modules(self):
        check=mock.Mock(returncode=0,stdout=json.dumps({
            "version":[3,11],"executable":"python311",
            "missing":["torch","transformers"]}))
        with mock.patch.object(video_ui.subprocess,"run",return_value=check):
            with self.assertRaisesRegex(ContractError,"missing packages"):
                video_ui.preflight_external_python(
                    "fake-python",("torch","transformers"))

    def test_external_python_preflight_accepts_matching_environment(self):
        info={"version":[3,11],"executable":"python311","missing":[]}
        check=mock.Mock(returncode=0,stdout=json.dumps(info))
        with mock.patch.object(video_ui.subprocess,"run",return_value=check):
            self.assertEqual(video_ui.preflight_external_python(
                "fake-python",("numpy",)),info)

    def test_no_receipt_shows_external_worker_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            coordinator,widgets,panel=self._capture_widgets(root)
            coordinator.job={
                "capture_path":str(root/"not-created"),
                "python":"F:/anymove/.venv/Scripts/python.exe",
                "mode":"prompt"}
            coordinator._worker_log_tail=(
                "Traceback: AI prompt worker needs external Python 3.11")
            coordinator._finished(1,coordinator.core.QProcess.NormalExit)
            warning=str(widgets.QMessageBox.warning.call_args)
            self.assertIn("no receipt",warning)
            self.assertIn("Traceback",warning)
            self.assertIn("Exit code: 1",warning)
            self.assertIn("F:/anymove/.venv/Scripts/python.exe",warning)
            self.assertTrue(panel.out.appendPlainText.called)



    def test_quality_gate_blocked_preserves_motion_and_requires_manual_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            capture=root/"capture_h3max_001"
            capture.mkdir()
            motion=capture/"motion.json"
            motion.write_text('{"test":"completed"}',encoding="utf-8")
            receipt={
                "status":"quality_gate_blocked",
                "error":"side reprojection 15.2 px > 15 px",
                "motion_file":str(motion),
                "motion_summary":{
                    "frame_count":124,
                    "quality_status":"needs_review",
                    "quality_issues":["side reprojection 15.2 px > 15 px"],
                    "reprojection_px":{"front_x_only":7.0,"side":15.2}
                }
            }
            (capture/video_ui.RECEIPT).write_text(
                json.dumps(receipt),encoding="utf-8")
            coordinator,widgets,panel=self._capture_widgets(root)
            coordinator.job={
                "capture_path":str(capture),"mode":"existing",
                "python":"python.exe"}
            with mock.patch.object(video_ui,"prepare_direct_import") as prepare:
                with mock.patch.object(video_ui,"execute_direct_import") as write:
                    coordinator._finished(2,coordinator.core.QProcess.NormalExit)
            prepare.assert_not_called()
            write.assert_not_called()
            warning=str(widgets.QMessageBox.warning.call_args)
            self.assertIn("quality review",warning.lower())
            self.assertIn("15.2 px",warning)
            self.assertIn("motion.json",warning)
            self.assertIn(str(motion),warning)
            self.assertIn("NOT been modified",warning)
            self.assertIn("Manual import does NOT certify",warning)
            self.assertTrue(motion.is_file())
            self.assertTrue(panel.out.appendPlainText.called)

    def test_quality_blocked_without_motion_is_not_reported_as_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            capture=root/"capture_h3max_001"
            capture.mkdir()
            (capture/video_ui.RECEIPT).write_text(json.dumps({
                "status":"quality_gate_blocked",
                "error":"side reprojection 15.2 px > 15 px",
                "motion_file":str(capture/"missing.json")
            }),encoding="utf-8")
            coordinator,widgets,_=self._capture_widgets(root)
            coordinator.job={"capture_path":str(capture),"mode":"existing"}
            coordinator._finished(2,coordinator.core.QProcess.NormalExit)
            warning=str(widgets.QMessageBox.warning.call_args)
            self.assertIn("Dragon Capture not imported",warning)
            self.assertNotIn("Motion Solver FINISHED",warning)
            self.assertNotIn("Saved motion.json (NOT discarded)",warning)


if __name__=="__main__":
    unittest.main()
