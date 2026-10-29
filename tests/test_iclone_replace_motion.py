"""Backed-up iClone motion replacement; mock runtime scene/clip only."""
import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import test_iclone_dragon_preview as fixtures
import iclone_adapter.replace_motion as repl
from iclone_adapter.schema import ContractError


class Status:
    def __init__(self,error=False):
        self.error=error
    def IsError(self):
        return self.error


class Fps:
    @staticmethod
    def ToDouble():
        return 24.0


class Skeleton:
    def __init__(self,events,clips=2):
        self.events=events
        self.clips=[object() for _ in range(clips)]
        self.previous=list(self.clips)
        self.failure=False
    def GetClipCount(self):
        return len(self.clips)
    def GetClip(self,index):
        return self.clips[index]
    def DeleteClip(self,clip):
        self.events.append(("delete",clip))
        if self.failure:
            return Status(error=True)
        self.clips.remove(clip)
        return Status()


class IO:
    def __init__(self,events,skel):
        self.events=events
        self.skel=skel
        self.fail_save=False
        self.fail_load=False
    def SaveProject(self,path):
        self.events.append(("save",str(path)))
        if self.fail_save:
            return Status(True)
        pathlib.Path(path).write_bytes(b"FULL ORIGINAL PROJECT")
        return Status()
    def LoadProject(self,path):
        self.events.append(("load",str(path)))
        if self.fail_load:
            return Status(True)
        self.skel.clips=list(self.skel.previous)
        return Status()


class Actor:
    @staticmethod
    def GetName():
        return "Dragon"
    def Update(self):
        pass


class Global:
    def __init__(self,events):
        self.events=events
    def SetTime(self,time):
        self.events.append(("time",time))
        return Status()


class SDK:
    def __init__(self,events,skel):
        self.RFileIO=IO(events,skel)
        self.RGlobal=Global(events)
    @staticmethod
    def IndexedFrameTime(frame,fps):
        return frame


class ReplaceMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DragonPreviewTests.setUpClass()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=pathlib.Path(self.tmp.name)
        self.dragon=fixtures.DragonPreviewTests()
        self.motion=self.dragon.make_motion()
        self.source=self.root/"motion.json"
        self.source.write_text(json.dumps(self.motion),encoding="utf-8")
        self.actor=Actor()
        self.events=[]
        self.skel=Skeleton(self.events,clips=2)
        self.sdk=SDK(self.events,self.skel)
        self.target=self.dragon.make_target()
        patches=[
            mock.patch.object(repl,"_selected_and_fps",
                              return_value=(self.actor,Fps())),
            mock.patch.object(repl,"_get_skeleton",return_value=self.skel),
            mock.patch.object(repl,"snapshot",return_value=self.target)
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_eligibility_checks_exact_bone_hierarchy_without_deleting(self):
        report=repl.inspect_replacement_eligibility(self.source,rlpy=self.sdk)
        self.assertEqual(report["existing_clip_count"],2)
        self.assertEqual(report["matched_dragon_bones"],42)
        self.assertFalse(report["rest_pose_verified"])
        self.assertEqual(self.events,[])

    def test_eligibility_rejects_incompatible_dragon_hierarchy(self):
        for entry in self.target["bones"]:
            if entry["name"]=="head":
                entry["parent_id"]="skin:/rigged/rig"
        with self.assertRaisesRegex(ContractError,"exact 42-Bone"):
            repl.inspect_replacement_eligibility(self.source,rlpy=self.sdk)
        self.assertEqual(self.events,[])

    def test_successful_backup_then_delete_then_strict_import(self):
        def prepare(motion_path,rlpy):
            self.assertEqual(self.skel.GetClipCount(),0)
            self.events.append(("strict_rest",str(motion_path)))
            return {"source":"ok"}
        def execute(prepared,folder,rlpy,save_project):
            self.assertEqual(self.skel.GetClipCount(),0)
            self.assertEqual(prepared,{"source":"ok"})
            self.assertTrue(save_project)
            self.events.append(("new_import",str(folder)))
            return {
                "status":"completed_full_frame_verified_project_saved",
                "iProject_path":str(self.root/"final.iProject"),
                "output_directory":str(self.root)
            }
        with mock.patch.object(repl,"prepare_direct_import",
                               side_effect=prepare):
            with mock.patch.object(repl,"execute_direct_import",
                                   side_effect=execute):
                result=repl.replace_dragon_motion_with_backup(
                    self.source,self.root,rlpy=self.sdk)
        self.assertEqual(result["removed_clip_count"],2)
        self.assertTrue(pathlib.Path(result["backup_iProject"]).is_file())
        self.assertTrue(pathlib.Path(result["receipt_path"]).is_file())
        steps=[e[0] for e in self.events]
        self.assertEqual(steps,["save","delete","delete","time",
                                "strict_rest","new_import"])
        self.assertEqual(self.skel.GetClipCount(),0)
        self.assertEqual(self.source.read_text(encoding="utf-8"),
                         json.dumps(self.motion))

    def test_full_scene_backup_failure_causes_no_clip_deletions(self):
        self.sdk.RFileIO.fail_save=True
        with self.assertRaisesRegex(ContractError,"save pre-replacement"):
            repl.replace_dragon_motion_with_backup(
                self.source,self.root,rlpy=self.sdk)
        self.assertEqual(self.skel.GetClipCount(),2)
        self.assertEqual([e[0] for e in self.events],["save"])

    def test_rest_mismatch_after_clip_deletion_restores_original(self):
        with mock.patch.object(repl,"prepare_direct_import",
                               side_effect=ContractError("bad Rest geometry")):
            with self.assertRaisesRegex(ContractError,"original project was restored"):
                repl.replace_dragon_motion_with_backup(
                    self.source,self.root,rlpy=self.sdk)
        self.assertEqual(self.skel.GetClipCount(),2)
        self.assertEqual([e[0] for e in self.events],
                         ["save","delete","delete","time","load"])

    def test_delete_clip_sdk_failure_reloads_original_scene(self):
        self.skel.failure=True
        with self.assertRaisesRegex(ContractError,"original project was restored"):
            repl.replace_dragon_motion_with_backup(
                self.source,self.root,rlpy=self.sdk)
        self.assertEqual(self.skel.GetClipCount(),2)
        self.assertEqual([e[0] for e in self.events],
                         ["save","delete","load"])

    def test_writer_failure_reloads_original_scene(self):
        with mock.patch.object(repl,"prepare_direct_import",
                               return_value={"source":"ok"}):
            with mock.patch.object(repl,"execute_direct_import",
                                   side_effect=ContractError("SDK write failed")):
                with self.assertRaisesRegex(ContractError,"original project was restored"):
                    repl.replace_dragon_motion_with_backup(
                        self.source,self.root,rlpy=self.sdk)
        self.assertEqual(self.skel.GetClipCount(),2)
        self.assertEqual(self.events[-1][0],"load")

    def test_load_backup_failure_retains_backup_file_path(self):
        self.sdk.RFileIO.fail_load=True
        self.skel.failure=True
        with self.assertRaisesRegex(ContractError,"Open the untouched backup manually"):
            repl.replace_dragon_motion_with_backup(
                self.source,self.root,rlpy=self.sdk)
        self.assertTrue(list(self.root.glob("AnyMove_BeforeMotionReplace*.iProject")))


if __name__=="__main__":
    unittest.main()
