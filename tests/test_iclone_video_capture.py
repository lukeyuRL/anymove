"""External Dragon video/NPZ -> det.npz -> motion.json worker contract tests.

No GPU, NumPy, OpenCV or torch required for these orchestration tests.
"""
import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import test_iclone_dragon_preview as fixture
import iclone_adapter.video_capture as capture
from iclone_adapter.schema import ContractError


class CaptureWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DragonPreviewTests.setUpClass()
        cls.dragon=fixture.DragonPreviewTests()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=pathlib.Path(self.tmp.name)
        self.video=self.root/"test.mp4"
        self.video.write_bytes(b"MP4 test data")
        self.det=self.root/"test.npz"
        self.det.write_bytes(b"npz test data")
        self.reader=self.root/"reader.pt"
        self.reader.write_bytes(b"reader test data")
        self.ref=self.root/"ref.png"
        self.ref.write_bytes(b"png test data")
        self.output=self.root/"session"

    def motion(self):
        data=self.dragon.make_motion()
        data["fps"]=24.0
        data["reproj_px"]={"front_x_only":2.1,"side":4.8}
        data["sync"]={"paw_agreement":{"score":.93}}
        return data

    def test_preflight_video_and_detection_inputs(self):
        with mock.patch.object(capture.sys,"version_info",(3,11,0)):
            a=capture._input_plan(self.video,self.reader,self.ref,self.output)
            self.assertEqual(a["mode"],"video")
            b=capture._input_plan(self.det,output_dir=self.output)
            self.assertEqual(b["mode"],"detections")
            self.assertEqual(a["required_python"],"3.11")
            self.assertNotIn("reader.pt",str(b["input_path"]))

    def test_wrong_python_and_missing_reader_fail(self):
        with mock.patch.object(capture.sys,"version_info",(3,8,0)):
            with self.assertRaisesRegex(ContractError,"Python 3.11"):
                capture._input_plan(self.det,output_dir=self.output)
        with mock.patch.object(capture.sys,"version_info",(3,11,0)):
            with self.assertRaisesRegex(ContractError,"reader"):
                capture._input_plan(self.video,output_dir=self.output)

    def test_existing_output_directory_rejected_not_overwritten(self):
        self.output.mkdir()
        with mock.patch.object(capture.sys,"version_info",(3,11,0)):
            with self.assertRaisesRegex(ContractError,"already exists"):
                capture._input_plan(self.det,output_dir=self.output)

    def test_motion_quality_and_fps_gate(self):
        motion_path=self.root/"motion.json"
        motion_path.write_text(json.dumps(self.motion()),encoding="utf-8")
        good=capture._read_motion_stats(motion_path,4)
        self.assertTrue(good["accepted"])
        self.assertEqual(good["frame_count"],4)
        altered=self.motion()
        altered["fps"]=29.97
        motion_path.write_text(json.dumps(altered),encoding="utf-8")
        with self.assertRaisesRegex(ContractError,"24-FPS"):
            capture._read_motion_stats(motion_path,4)
        bad=self.motion()
        bad["sync"]["paw_agreement"]["score"]=.3
        motion_path.write_text(json.dumps(bad),encoding="utf-8")
        rejected=capture._read_motion_stats(motion_path,4)
        self.assertEqual(rejected["quality_status"],"needs_review")
        self.assertFalse(rejected["accepted"])
        self.assertTrue(capture._read_motion_stats(
            motion_path,4,allow_low_quality=True)["accepted"])

    def test_full_npz_capture_without_neural_import_in_test(self):
        motion=self.motion()
        recorded=[]
        def fake_runner(command,env):
            recorded.append(list(command))
            self.assertEqual(env["MOCAP_CREATURE"],"assets/dragon")
            self.assertEqual(command[2],"mocap.solve")
            pathlib.Path(command[5]).write_text(json.dumps(motion),
                                                  encoding="utf-8")
        with mock.patch.object(capture.sys,"version_info",(3,11,0)):
            with mock.patch.object(capture,"_read_detection_stats",
                return_value={"frame_count":4,"joint_count":50,"fps":24.0}):
                result=capture.run_capture(
                    self.det,self.output,runner=fake_runner)
        self.assertEqual(result["status"],"motion_json_ready")
        self.assertEqual(len(recorded),1)
        self.assertEqual(recorded[0][1:3],["-m","mocap.solve"])
        self.assertTrue((self.output/"motion.json").exists())
        self.assertTrue((self.output/"det.npz").exists())
        receipt=json.loads((self.output/"capture_receipt.json").read_text())
        self.assertEqual(receipt["source_motion_sha256"],result[
            "source_motion_sha256"])

    def test_video_calls_detect_then_solve_and_catches_missing_output(self):
        motion=self.motion()
        calls=[]
        def runner(command,env):
            calls.append(command[2])
            if command[2]=="mocap.detect":
                pathlib.Path(command[6]).write_bytes(b"fake det")
            elif command[2]=="mocap.solve":
                pathlib.Path(command[5]).write_text(
                    json.dumps(motion),encoding="utf-8")
        with mock.patch.object(capture.sys,"version_info",(3,11,0)):
            with mock.patch.object(capture,"_read_detection_stats",
                return_value={"frame_count":4,"joint_count":50,"fps":24.0}):
                report=capture.run_capture(
                    self.video,self.output,
                    reader_path=self.reader,ref_path=self.ref,runner=runner)
        self.assertEqual(calls,["mocap.detect","mocap.solve"])
        self.assertEqual(report["status"],"motion_json_ready")
        self.assertTrue((self.output/"overlay.mp4").exists() is False)

    def test_process_failure_receipt_kept(self):
        def broken(command,env):
            raise RuntimeError("CUDA unavailable")
        with mock.patch.object(capture.sys,"version_info",(3,11,0)):
            with self.assertRaisesRegex(RuntimeError,"CUDA unavailable"):
                capture.run_capture(
                    self.video,self.output,
                    reader_path=self.reader,ref_path=self.ref,runner=broken)
        payload=json.loads((self.output/"capture_receipt.json").read_text())
        self.assertEqual(payload["status"],"capture_failed")
        self.assertIn("CUDA unavailable",payload["error"])


if __name__=="__main__":
    unittest.main()
