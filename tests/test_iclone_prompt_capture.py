"""Prompt -> fal -> video -> motion.json orchestration, no remote calls."""
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import iclone_adapter.prompt_capture as prompt
from iclone_adapter.schema import ContractError


class DragonPromptCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=pathlib.Path(self.tmp.name)
        self.out=self.root/"prompt_capture"
        self.reader=self.root/"reader.pt"
        self.ref=self.root/"ref.png"
        self.reader.write_bytes(b"fake reader")
        self.ref.write_bytes(b"fake ref")
        self.action="the dragon raises both wings and roars, then settles."

    def plan(self,action=None,require_credentials=False):
        return prompt._prompt_plan(
            action or self.action,self.out,self.reader,self.ref,
            require_python311=False,require_credentials=require_credentials)

    def test_billable_requires_explicit_credentials(self):
        with mock.patch.object(prompt,"_credential_available",return_value=False):
            with self.assertRaisesRegex(ContractError,"FAL_KEY"):
                self.plan(require_credentials=True)
        with mock.patch.object(prompt,"_credential_available",return_value=True):
            result=self.plan(require_credentials=True)
        self.assertTrue(result["billable_external_video_request"])
        self.assertEqual(result["model"],"h3max")

    def test_invalid_action_model_or_output_block_before_charge(self):
        with self.assertRaisesRegex(ContractError,"action"):
            self.plan(action="jump")
        with self.assertRaisesRegex(ContractError,"unsupported"):
            prompt._prompt_plan(self.action,self.out,self.reader,self.ref,
                                model="bad",require_credentials=False,
                                require_python311=False)
        self.out.mkdir()
        with self.assertRaisesRegex(ContractError,"NEW directory"):
            self.plan()

    def test_prompt_to_video_to_motion_success_status(self):
        calls=[]
        def fake_run(cmd,env):
            calls.append(list(cmd))
            if len(cmd)>=2 and cmd[1]=="tools/make_prompt.py":
                pathlib.Path(cmd[-1]).write_text(
                    "Two-camera Dragon; action: "+self.action,
                    encoding="utf-8")
            elif len(cmd)>=3 and cmd[2]=="mocap.generate":
                pathlib.Path(cmd[6]).mkdir(exist_ok=True)
                (pathlib.Path(cmd[6])/"h3max.mp4").write_bytes(b"fake mp4")
        local={"status":"motion_json_ready",
               "motion_file":str(self.root/"motion.json"),
               "source_motion_sha256":"sha123"}
        with mock.patch.object(prompt.sys,"version_info",(3,11,0)):
            with mock.patch.object(prompt,"_credential_available",
                                   return_value=True):
                with mock.patch.object(prompt,"run_capture",
                        return_value=local) as capture:
                    result=prompt.run_prompt_capture(
                        self.action,self.out,self.reader,self.ref,
                        runner=fake_run)
        self.assertEqual(result["status"],"motion_json_ready")
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[1][2],"mocap.generate")
        capture.assert_called_once()
        receipt=json.loads((self.out/"capture_receipt.json").read_text())
        self.assertEqual(receipt["generated_video"],
                         str(self.out/"generated"/"h3max.mp4"))
        self.assertEqual(receipt["source_motion_sha256"],"sha123")

    def test_missing_generated_video_fails_without_attempting_solver(self):
        def fake_run(cmd,env):
            if len(cmd)>1 and cmd[1]=="tools/make_prompt.py":
                pathlib.Path(cmd[-1]).write_text("test prompt",encoding="utf-8")
        with mock.patch.object(prompt.sys,"version_info",(3,11,0)):
            with mock.patch.object(prompt,"_credential_available",
                                   return_value=True):
                with mock.patch.object(prompt,"run_capture") as capture:
                    with self.assertRaisesRegex(ContractError,
                                                "did not produce expected"):
                        prompt.run_prompt_capture(
                            self.action,self.out,self.reader,self.ref,
                            runner=fake_run)
                    capture.assert_not_called()
        result=json.loads((self.out/"capture_receipt.json").read_text())
        self.assertEqual(result["status"],"prompt_capture_failed")
        self.assertIn("did not produce expected",result["error"])


if __name__=="__main__":
    unittest.main()
