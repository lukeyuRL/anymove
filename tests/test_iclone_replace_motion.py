"""Clean-template Dragon scene recovery; no iClone SDK needed for tests."""
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
    def __init__(self,clips=1):
        self.clips=[object() for _ in range(clips)]
    def GetClipCount(self):
        return len(self.clips)


class Actor:
    @staticmethod
    def GetName():
        return "Dragon"
    def GetSkeletonComponent(self):
        return self.skel


class Scene:
    def __init__(self,actor,events):
        self.actor=actor
        self.events=events
        self.selection=[actor]
    def GetAvatars(self,kind=None):
        return [self.actor]
    def GetProps(self):
        return []
    def GetSelectedObjects(self):
        return self.selection
    def ClearSelectObjects(self):
        self.events.append(("clear_selection",None))
        self.selection=[]
        return Status()
    def SelectObject(self,obj):
        self.events.append(("select",obj.GetName()))
        self.selection=[obj]
        return Status()


class IO:
    def __init__(self,events,actor):
        self.events=events
        self.actor=actor
        self.fail_save=False
        self.fail_load=False
        self.fail_template=False
        self.template_path=""
    def SaveProject(self,path):
        self.events.append(("save",str(path)))
        if self.fail_save:
            return Status(True)
        pathlib.Path(path).write_bytes(b"PREVIOUS CURRENT SCENE")
        return Status()
    def LoadProject(self,path):
        self.events.append(("load",str(path)))
        if self.fail_load:
            return Status(True)
        self.actor.skel=Skeleton(
            clips=1 if path!=self.template_path or self.fail_template else 0)
        return Status()


class Global:
    def __init__(self,events):
        self.events=events
    def GetFps(self):
        return Fps()
    def SetTime(self,time):
        self.events.append(("set_time",time))
        return Status()


class SDK:
    EAvatarType_All=1
    def __init__(self,events,actor):
        self.RFileIO=IO(events,actor)
        self.RScene=Scene(actor,events)
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
        self.clean=self.root/"OriginalCleanDragon.iProject"
        self.clean.write_bytes(b"CLEAN DRAGON PROJECT")
        self.events=[]
        self.actor=Actor()
        self.actor.skel=Skeleton(clips=1)
        self.sdk=SDK(self.events,self.actor)
        self.sdk.RFileIO.template_path=str(self.clean.resolve())
        self.target=self.dragon.make_target()
        patches=[
            mock.patch.object(repl,"_selected_and_fps",
                              side_effect=lambda rlpy,fps:(self.actor,Fps())),
            mock.patch.object(repl,"snapshot",return_value=self.target),
            mock.patch.object(repl,"check_clean_clip",
                              side_effect=self._check_clean)
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _check_clean(self,skel,obj,rlpy):
        if skel.GetClipCount()!=0:
            raise ContractError("template has existing animation")
        return {"exposed_motion_empty":True}

    def test_hierarchy_eligibility_allows_already_deleted_motion(self):
        self.actor.skel=Skeleton(clips=0)
        report=repl.inspect_replacement_eligibility(self.source,rlpy=self.sdk)
        self.assertEqual(report["existing_clip_count"],0)
        self.assertTrue(report["rest_pose_verified"])
        self.assertEqual(self.events,[])

    def test_eligibility_accepts_pose_mismatch_but_does_not_claim_clean(self):
        target=copy.deepcopy(self.target)
        for bone in target["bones"]:
            if bone["name"]=="head":
                bone["snapshot_world_matrix"][3]+=100
        with mock.patch.object(repl,"snapshot",return_value=target):
            report=repl.inspect_replacement_eligibility(
                self.source,rlpy=self.sdk)
        self.assertFalse(report["rest_pose_verified"])
        self.assertEqual(self.events,[])

    def test_no_template_is_hard_block_without_scene_changes(self):
        with self.assertRaisesRegex(ContractError,
                                   "Cannot repair Rest Pose by deleting Motion"):
            repl.replace_dragon_motion_with_backup(
                self.source,self.root,rlpy=self.sdk)
        self.assertEqual(self.events,[])
        self.assertFalse(list(self.root.glob("AnyMove_BeforeSceneReplace*")))

    def test_clean_template_selection_requires_zero_motion_and_exact_rest(self):
        self.actor.skel=Skeleton(clips=0)
        out=repl.select_verified_clean_dragon(self.sdk)
        self.assertEqual(out["rest_geometry"],"compatible_static_rest_snapshot")
        self.assertAlmostEqual(out["observed_fbx_scale"],100.)
        self.assertTrue(out["clean_layer"])
        self.assertEqual(self.events[-2:],
                         [("clear_selection",None),("select","Dragon")])
        target=copy.deepcopy(self.target)
        for bone in target["bones"]:
            if bone["name"]=="head":
                bone["snapshot_world_matrix"][3]+=100
        self.events.clear()
        with mock.patch.object(repl,"snapshot",return_value=target):
            with self.assertRaisesRegex(ContractError,"EXACTLY one"):
                repl.select_verified_clean_dragon(self.sdk)
        self.assertEqual(self.events,[])

    def test_success_backups_and_loads_clean_scene_without_deleting_clips(self):
        def prepare(motion_path,rlpy):
            self.assertEqual(self.actor.skel.GetClipCount(),0)
            self.events.append(("strict_rest",str(motion_path)))
            return {"source":"ok"}
        def execute(prepared,folder,rlpy,save_project,preserve_clean_template):
            self.assertEqual(self.actor.skel.GetClipCount(),0)
            self.assertTrue(save_project)
            self.assertTrue(preserve_clean_template)
            self.events.append(("new_import",str(folder)))
            return {
                "status":"completed_full_frame_verified_project_saved",
                "iProject_path":str(self.root/"final.iProject"),
                "output_directory":str(self.root)}
        with mock.patch.object(repl,"prepare_direct_import",side_effect=prepare):
            with mock.patch.object(repl,"execute_direct_import",
                                   side_effect=execute):
                result=repl.replace_dragon_motion_with_backup(
                    self.source,self.root,rlpy=self.sdk,
                    clean_project_path=self.clean)
        steps=[e[0] for e in self.events]
        self.assertEqual(steps,["save","load","set_time","clear_selection",
                                "select","strict_rest","new_import"])
        self.assertNotIn("delete",steps)
        self.assertTrue(pathlib.Path(result["original_scene_backup"]).is_file())
        self.assertTrue(pathlib.Path(result["receipt_path"]).is_file())
        self.assertEqual(result["status"],
                         "scene_recreated_from_clean_dragon_and_motion_imported")
        self.assertEqual(self.source.read_text(),json.dumps(self.motion))

    def test_failed_backup_does_not_load_template(self):
        self.sdk.RFileIO.fail_save=True
        with self.assertRaisesRegex(ContractError,"save entire scene"):
            repl.replace_dragon_motion_with_backup(
                self.source,self.root,rlpy=self.sdk,
                clean_project_path=self.clean)
        self.assertEqual([e[0] for e in self.events],["save"])

    def test_animated_template_rejected_and_original_restored(self):
        self.sdk.RFileIO.fail_template=True
        with self.assertRaisesRegex(ContractError,"Previous scene reloaded"):
            repl.replace_dragon_motion_with_backup(
                self.source,self.root,rlpy=self.sdk,
                clean_project_path=self.clean)
        self.assertEqual(self.actor.skel.GetClipCount(),1)
        self.assertEqual([e[0] for e in self.events],
                         ["save","load","set_time","load"])

    def test_failed_clean_project_import_restores_original_scene(self):
        with mock.patch.object(repl,"prepare_direct_import",
                               side_effect=ContractError("rig mismatch")):
            with self.assertRaisesRegex(ContractError,
                                        "Previous scene reloaded"):
                repl.replace_dragon_motion_with_backup(
                    self.source,self.root,rlpy=self.sdk,
                    clean_project_path=self.clean)
        self.assertEqual(self.actor.skel.GetClipCount(),1)
        self.assertEqual(self.events[-1][0],"load")

    def test_failed_writer_restores_original_scene(self):
        with mock.patch.object(repl,"prepare_direct_import",
                               return_value={"source":"ok"}):
            with mock.patch.object(repl,"execute_direct_import",
                                   side_effect=ContractError("FK write failed")):
                with self.assertRaisesRegex(ContractError,
                                            "Previous scene reloaded"):
                    repl.replace_dragon_motion_with_backup(
                        self.source,self.root,rlpy=self.sdk,
                        clean_project_path=self.clean)
        self.assertEqual(self.actor.skel.GetClipCount(),1)
        self.assertEqual(self.events[-1][0],"load")

    def test_failed_rollback_keeps_original_scene_backup_path(self):
        self.sdk.RFileIO.fail_template=True
        original=self.sdk.RFileIO.LoadProject
        def failing_restore(path):
            if path!=str(self.clean):
                return Status(True)
            return original(path)
        with mock.patch.object(self.sdk.RFileIO,"LoadProject",
                               side_effect=failing_restore):
            with self.assertRaisesRegex(ContractError,
                                        "Open the intact backup manually"):
                repl.replace_dragon_motion_with_backup(
                    self.source,self.root,rlpy=self.sdk,
                    clean_project_path=self.clean)
        self.assertTrue(list(self.root.glob("AnyMove_BeforeSceneReplace*.iProject")))


if __name__=="__main__":
    unittest.main()
