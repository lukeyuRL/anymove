"""End-to-end Dragon direct import orchestration: source -> native iProject.

Uses the REAL pure-Python converter, manifest, file-safety and source rig
checks. Only RLPy SDK and the previously validated Motion Writer / Controller
readback are substituted with controllable offline fakes.
"""
import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import test_iclone_dragon_preview as fixtures

import iclone_adapter.direct_import as direct
from iclone_adapter.schema import ContractError


class FPS:
    def ToDouble(self):
        return 24.


class Global:
    def __init__(self):
        self.t=0
    def GetFps(self):
        return FPS()
    def GetTime(self):
        return self.t
    def SetTime(self,instant):
        self.t=instant
        return True


class Skeleton:
    def GetClipCount(self):
        return 0


class Dragon:
    def __init__(self):
        self.skel=Skeleton()
    def GetSkeletonComponent(self):
        return self.skel
    def GetName(self):
        return "rigged"


class Selection:
    def __init__(self,obj):
        self.obj=obj
    def GetSelectedObjects(self):
        return [self.obj]


class FileIO:
    def __init__(self,save_mode="ok"):
        self.save_mode=save_mode
        self.saved=[]
    def SaveProject(self,path):
        self.saved.append(path)
        if self.save_mode=="error":
            return False
        pathlib.Path(path).write_bytes(b"FAKE_NATIVE_IPROJECT_CONTENT")
        return True


class SDK:
    def __init__(self,save_mode="ok"):
        self.actor=Dragon()
        self.RScene=Selection(self.actor)
        self.RGlobal=Global()
        self.RFileIO=FileIO(save_mode)
    @staticmethod
    def GetFrameIndex(time,fps):
        return time
    @staticmethod
    def IndexedFrameTime(frame,fps):
        return int(frame)


class DirectImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DragonPreviewTests.setUpClass()
        cls.info=fixtures.DragonPreviewTests.source
        cls.dragon=fixtures.DragonPreviewTests()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=pathlib.Path(self.tmp.name)
        self.sdk=SDK()
        self.target=self.dragon.make_target()
        self.motion=self.dragon.make_motion()
        self.path=self.root/"motion.json"
        self.path.write_text(json.dumps(self.motion),encoding="utf-8")
        self.snap_patch=mock.patch.object(direct,"snapshot",return_value=self.target)
        self.guard_patch=mock.patch.object(
            direct,"preflight_clean_dragon",
            return_value={"result":"guarded_clean_pose_and_exposed_layer"})
        self.snap_patch.start()
        self.guard_patch.start()
        self.addCleanup(self.snap_patch.stop)
        self.addCleanup(self.guard_patch.stop)

    def prepare(self):
        return direct.prepare_direct_import(str(self.path),rlpy=self.sdk)

    def writer_success(self,motion,fps_confirmed,rlpy):
        self.assertTrue(fps_confirmed)
        self.assertEqual(rlpy,self.sdk)
        self.assertEqual(len(motion["tracks"]),42)
        self.assertEqual(len(motion["position_tracks"]),1)
        self.assertEqual(motion["frame_count"],4)
        return {"clip_count_after":1,
                "clip_strategy":"create_new",
                "status":"fk_keys_verified_pose_changed",
                "warnings":[]}

    def audit_success(self,motion,rlpy,clip_index):
        self.assertEqual(clip_index,0)
        self.assertEqual(rlpy,self.sdk)
        return {
            "status":"all_frame_controller_values_match",
            "verified_scalar_samples":motion["frame_count"]*(42+1)*3,
            "mismatched_scalar_samples":0,
            "channels_with_wrong_key_count":0,
            "maximum_abs_error_in_channel_units":4.5e-8,
            "all_integer_frame_controller_data_verified":True
        }

    def test_preflight_is_read_only_and_builds_full_fk(self):
        with mock.patch.object(direct,"write_selected") as writer:
            plan=self.prepare()
            self.assertEqual(plan["frames"],4)
            self.assertEqual(plan["rotation_bones"],42)
            self.assertEqual(plan["position_bones"],1)
            self.assertEqual(plan["expected_scalar_samples"],4*43*3)
            self.assertEqual(plan["source_profile"],direct.PROFILE_ID)
            self.assertEqual(plan["fk"]["root_motion"]["root_position_space"],
                             "armature")
            self.assertEqual(plan["fk"]["root_motion"]["root_position_scale"],1.)
            self.assertTrue(plan["fk"]["calibration_status"].startswith("unverified_"))
            self.assertFalse(any(self.root.glob("AnyMove_Dragon_*")))
            writer.assert_not_called()

    def test_one_click_saves_native_project_and_success_receipt(self):
        with mock.patch.object(direct,"write_selected",
                               side_effect=self.writer_success) as writer:
            with mock.patch.object(direct,"audit_full_controller_frames",
                                   side_effect=self.audit_success) as audit:
                result=direct.execute_direct_import(
                    self.prepare(),self.root,rlpy=self.sdk)
        writer.assert_called_once()
        audit.assert_called_once()
        self.assertEqual(result["status"],
                         "completed_full_frame_verified_project_saved")
        self.assertEqual(result["verified_scalar_samples"],516)
        self.assertTrue(result["iProject_saved"])
        self.assertTrue(pathlib.Path(result["iProject_path"]).is_file())
        folder=pathlib.Path(result["output_directory"])
        self.assertTrue((folder/"runtime_controller_audit.json").exists())
        self.assertTrue((folder/"offline_frame_audit.json").exists())
        self.assertTrue((folder/"iclone_fk_full.json").exists())
        self.assertTrue((folder/"import_manifest.json").exists())
        saved=json.loads((folder/"import_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"],result["status"])
        self.assertTrue(saved["playhead_restored_to_frame0"])
        self.assertEqual(len(self.sdk.RFileIO.saved),1)

    def test_optional_clean_template_saved_before_motion_keys(self):
        def verify_writer(motion,fps_confirmed,rlpy):
            folders=list(self.root.glob("AnyMove_Dragon_*"))
            self.assertEqual(len(folders),1)
            template=folders[0]/direct.CLEAN_TEMPLATE_NAME
            self.assertTrue(template.is_file())
            self.assertGreater(template.stat().st_size,0)
            self.assertTrue(self.sdk.RFileIO.saved)
            self.assertEqual(self.sdk.RFileIO.saved[0],str(template))
            return self.writer_success(motion,fps_confirmed,rlpy)
        with mock.patch.object(direct,"write_selected",
                               side_effect=verify_writer) as writer:
            with mock.patch.object(direct,"audit_full_controller_frames",
                                   side_effect=self.audit_success):
                result=direct.execute_direct_import(
                    self.prepare(),self.root,rlpy=self.sdk,
                    preserve_clean_template=True)
        writer.assert_called_once()
        self.assertEqual(result["status"],
                         "completed_full_frame_verified_project_saved")
        self.assertEqual(len(self.sdk.RFileIO.saved),2)
        self.assertEqual(result["clean_template_iProject"],
                         self.sdk.RFileIO.saved[0])
        self.assertNotEqual(self.sdk.RFileIO.saved[0],
                            result["iProject_path"])
        self.assertTrue(pathlib.Path(result["clean_template_iProject"]).is_file())

    def test_clean_template_save_failure_blocks_before_any_keys(self):
        self.sdk.RFileIO=FileIO("error")
        with mock.patch.object(direct,"write_selected") as writer:
            with self.assertRaisesRegex(
                    ContractError,"NO animation keys written"):
                direct.execute_direct_import(
                    self.prepare(),self.root,rlpy=self.sdk,
                    preserve_clean_template=True)
        writer.assert_not_called()
        folders=list(self.root.glob("AnyMove_Dragon_*"))
        self.assertEqual(len(folders),1)
        receipt=json.loads((folders[0]/"import_receipt.json").read_text())
        self.assertEqual(receipt["status"],
                         "clean_template_backup_failed_no_keys_written")
        self.assertFalse(receipt["scene_may_contain_new_keys"])
        self.assertEqual(len(self.sdk.RFileIO.saved),1)

    def test_reusable_template_requires_native_project_save(self):
        with self.assertRaisesRegex(ContractError,"native SaveProject"):
            direct.execute_direct_import(
                self.prepare(),self.root,rlpy=self.sdk,
                save_project=False,preserve_clean_template=True)
        self.assertFalse(any(self.root.glob("AnyMove_Dragon_*")))

    def test_fps_and_nonzero_time_rejected_before_package_creation(self):
        self.sdk.RGlobal.t=3
        with self.assertRaisesRegex(ContractError,"Frame 0"):
            self.prepare()
        self.sdk.RGlobal.t=0
        self.sdk.RGlobal.GetFps=lambda: type(
            "FPS30",(),{"ToDouble":lambda self:30.})()
        with self.assertRaisesRegex(ContractError,"24 FPS"):
            self.prepare()
        self.assertFalse(any(self.root.glob("AnyMove_Dragon_*")))

    def test_wrong_source_rig_rejected_before_scene_mutation(self):
        self.motion["bones"]=list(reversed(self.motion["bones"]))
        self.path.write_text(json.dumps(self.motion),encoding="utf-8")
        with mock.patch.object(direct,"write_selected") as writer:
            with self.assertRaises(ContractError):
                self.prepare()
            writer.assert_not_called()
        self.assertFalse(any(self.root.glob("AnyMove_Dragon_*")))

    def test_second_package_is_unique_and_does_not_overwrite(self):
        with mock.patch.object(direct,"write_selected",
                               side_effect=self.writer_success):
            with mock.patch.object(direct,"audit_full_controller_frames",
                                   side_effect=self.audit_success):
                prepared=self.prepare()
                first=direct.execute_direct_import(prepared,self.root,rlpy=self.sdk)
                second=direct.execute_direct_import(prepared,self.root,rlpy=self.sdk)
        self.assertNotEqual(first["output_directory"],second["output_directory"])
        self.assertTrue(first["iProject_saved"])
        self.assertTrue(second["iProject_saved"])
        self.assertTrue(pathlib.Path(first["iProject_path"]).exists())

    def test_source_scene_changed_after_confirmation_fails_no_write(self):
        plan=self.prepare()
        with mock.patch.object(direct,"preflight_clean_dragon",
                               side_effect=ContractError("not a clean scene")):
            with mock.patch.object(direct,"write_selected") as writer:
                with self.assertRaisesRegex(ContractError,"not a clean scene"):
                    direct.execute_direct_import(plan,self.root,rlpy=self.sdk)
                writer.assert_not_called()
        self.assertFalse(any(self.root.glob("AnyMove_Dragon_*")))

    def test_fullframe_controller_audit_failure_does_not_save_project(self):
        bad={
            "status":"controller_readback_failed",
            "verified_scalar_samples":515,
            "mismatched_scalar_samples":1,
            "channels_with_wrong_key_count":0,
            "maximum_abs_error_in_channel_units":0.3
        }
        with mock.patch.object(direct,"write_selected",
                               side_effect=self.writer_success):
            with mock.patch.object(direct,"audit_full_controller_frames",
                                   return_value=bad):
                outcome=direct.execute_direct_import(
                    self.prepare(),self.root,rlpy=self.sdk)
        self.assertEqual(outcome["status"],
                         "controller_audit_failed_project_not_saved")
        self.assertEqual(outcome["mismatched_scalar_samples"],1)
        self.assertFalse(outcome["iProject_saved"])
        self.assertEqual(self.sdk.RFileIO.saved,[])
        receipt=pathlib.Path(outcome["output_directory"])/"import_receipt.json"
        self.assertTrue(receipt.exists())

    def test_runtime_audit_sdk_exception_leaves_warning_and_receipt(self):
        with mock.patch.object(direct,"write_selected",
                               side_effect=self.writer_success):
            with mock.patch.object(direct,"audit_full_controller_frames",
                                   side_effect=RuntimeError("GetValue failed")):
                outcome=direct.execute_direct_import(
                    self.prepare(),self.root,rlpy=self.sdk)
        self.assertEqual(outcome["status"],
                         "controller_audit_exception_project_not_saved")
        self.assertEqual(outcome["controller_audit_status"],"exception")
        self.assertTrue(outcome["scene_may_contain_new_keys"])
        self.assertEqual(self.sdk.RFileIO.saved,[])

    def test_write_failure_records_possible_rollback_problem(self):
        with mock.patch.object(direct,"write_selected",
                               side_effect=ContractError("SDK SetValue failed")):
            with self.assertRaisesRegex(ContractError,
                                        "Dragon SDK write failed"):
                direct.execute_direct_import(self.prepare(),self.root,
                                             rlpy=self.sdk)
        receipts=list(self.root.glob(
            "AnyMove_Dragon_*/import_receipt.json"))
        self.assertEqual(len(receipts),1)
        receipt=json.loads(receipts[0].read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"],
                         "sdk_write_failed_rollback_status_must_be_checked")
        self.assertTrue(receipt["scene_may_contain_new_keys"])

    def test_SaveProject_missing_or_returns_error_not_complete(self):
        self.sdk.RFileIO=object()
        with mock.patch.object(direct,"write_selected",
                               side_effect=self.writer_success):
            with mock.patch.object(direct,"audit_full_controller_frames",
                                   side_effect=self.audit_success):
                outcome=direct.execute_direct_import(
                    self.prepare(),self.root,rlpy=self.sdk)
        self.assertEqual(outcome["status"],
                         "controller_verified_manual_project_save_required")
        self.assertFalse(outcome["iProject_saved"])
        self.sdk.RFileIO=FileIO("error")
        with mock.patch.object(direct,"write_selected",
                               side_effect=self.writer_success):
            with mock.patch.object(direct,"audit_full_controller_frames",
                                   side_effect=self.audit_success):
                outcome=direct.execute_direct_import(
                    self.prepare(),self.root,rlpy=self.sdk)
        self.assertEqual(outcome["status"],
                         "controller_verified_native_project_save_failed")
        self.assertFalse(outcome["iProject_saved"])

    def test_disable_native_project_save_is_not_a_full_save(self):
        with mock.patch.object(direct,"write_selected",
                               side_effect=self.writer_success):
            with mock.patch.object(direct,"audit_full_controller_frames",
                                   side_effect=self.audit_success):
                outcome=direct.execute_direct_import(
                    self.prepare(),self.root,rlpy=self.sdk,save_project=False)
        self.assertEqual(outcome["status"],
                         "completed_full_frame_verified_manual_project_save")
        self.assertEqual(self.sdk.RFileIO.saved,[])


if __name__=="__main__":
    unittest.main()
