"""Full-frame RLPy Controller audit, using offline iClone SDK mocks."""
import copy
import json
import pathlib
import tempfile
import unittest

from iclone_adapter.import_pipeline import (
    save_bundle,verify_fk_manifest_for_writer,OUTPUT_MOTION,
    OUTPUT_MANIFEST
)
from iclone_adapter.runtime_controller_audit import (
    audit_full_controller_frames,AUDIT_VERSION
)
from iclone_adapter.motion_writer import write_selected
from iclone_adapter.schema import ContractError

from test_iclone_root_writer import setup_root
from test_iclone_dragon_preview import DragonPreviewTests


class FullFrameRuntimeAuditTests(unittest.TestCase):
    def test_mock_7frames_rotation_and_position_all_samples_pass(self):
        avatar,rlpy,motion=setup_root()
        write_selected(motion,rlpy=rlpy,fps_confirmed=True)
        old_time=rlpy.RGlobal.GetTime()
        report=audit_full_controller_frames(motion,rlpy=rlpy)
        self.assertEqual(report["status"],"all_frame_controller_values_match")
        self.assertEqual(report["schema_version"],AUDIT_VERSION)
        self.assertEqual(report["expected_scalar_samples"],7*(2+1)*3)
        self.assertEqual(report["verified_scalar_samples"],63)
        self.assertEqual(report["mismatched_scalar_samples"],0)
        self.assertEqual(report["channels_with_wrong_key_count"],0)
        self.assertFalse(report["evaluated_pose_verified"])
        self.assertEqual(old_time,rlpy.RGlobal.GetTime())

    def test_single_mismatched_frame_is_detected(self):
        avatar,rlpy,motion=setup_root()
        write_selected(motion,rlpy=rlpy,fps_confirmed=True)
        clip=avatar.skeleton.GetClip(0)
        ctrl=clip.GetControl("Layer",avatar.skeleton.bones[0]).block.GetControl(
            "Position/PositionX")
        ctrl.values[3]=(ctrl.values[3][0],44.0)
        report=audit_full_controller_frames(motion,rlpy=rlpy)
        self.assertEqual(report["status"],"controller_readback_failed")
        self.assertEqual(report["mismatched_scalar_samples"],1)
        self.assertGreater(report["maximum_abs_error_in_channel_units"],1.0)
        self.assertIn("Frame 3",str(report["first_failures"]))

    def test_missing_key_and_channel_are_detected(self):
        avatar,rlpy,motion=setup_root()
        write_selected(motion,rlpy=rlpy,fps_confirmed=True)
        clip=avatar.skeleton.GetClip(0)
        a=clip.GetControl("Layer",avatar.skeleton.bones[1]).block
        a.GetControl("Rotation/RotationZ").RemoveKey(4)
        report=audit_full_controller_frames(motion,rlpy=rlpy)
        self.assertGreater(report["channels_with_wrong_key_count"],0)
        self.assertEqual(report["status"],"controller_readback_failed")
        b=clip.GetControl("Layer",avatar.skeleton.bones[0]).block
        b.ctrls.pop("Position/PositionY")
        report=audit_full_controller_frames(motion,rlpy=rlpy)
        self.assertEqual(report["status"],"controller_readback_failed")
        self.assertLess(report["verified_scalar_samples"],
                        report["expected_scalar_samples"])

    def test_stale_skeleton_wrong_clip_and_size_limit_fail(self):
        avatar,rlpy,motion=setup_root()
        write_selected(motion,rlpy=rlpy,fps_confirmed=True)
        with self.assertRaises(ContractError):
            audit_full_controller_frames(motion,clip_index=100,rlpy=rlpy)
        with self.assertRaisesRegex(ContractError,"too many scalar keys"):
            audit_full_controller_frames(
                motion,rlpy=rlpy,max_scalar_samples=2)
        bad=copy.deepcopy(motion)
        bad["skeleton_hash"]="other_skeleton"
        with self.assertRaises(ContractError):
            audit_full_controller_frames(bad,rlpy=rlpy)


class BundleWriterIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DragonPreviewTests.setUpClass()
        cls.fixture=DragonPreviewTests()
        cls.info=DragonPreviewTests.source

    def test_manifest_checked_before_iClone_write_and_mutation_blocked(self):
        m=self.fixture.make_motion()
        target=self.fixture.make_target()
        with tempfile.TemporaryDirectory() as folder:
            out=pathlib.Path(folder)/"import"
            save_bundle(out,m,target,self.info,self.info)
            path=out/OUTPUT_MOTION
            data=json.loads(path.read_text(encoding="utf-8"))
            good=verify_fk_manifest_for_writer(path,data)
            self.assertEqual(good["status"],
                             "fk_payload_matches_adjacent_manifest")
            changed=copy.deepcopy(data)
            bone_id=next(iter(changed["tracks"]))
            changed["tracks"][bone_id][1][0]+=.001
            with self.assertRaisesRegex(ContractError,"SHA-256"):
                verify_fk_manifest_for_writer(path,changed)
            (out/OUTPUT_MANIFEST).unlink()
            with self.assertRaisesRegex(ContractError,"adjacent"):
                verify_fk_manifest_for_writer(path,data)

    def test_legacy_preview_still_runs_without_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            file=pathlib.Path(folder)/"test_fk.json"
            file.write_text("{}",encoding="utf-8")
            self.assertEqual(
                verify_fk_manifest_for_writer(file,{} )["status"],
                "legacy_FK_json_no_bundle_manifest")


if __name__=="__main__":
    unittest.main()
