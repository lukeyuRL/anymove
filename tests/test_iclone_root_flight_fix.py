"""Backed-up experimental root flight rescale; no real RLPy required."""
import copy
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import test_iclone_dragon_preview as fixtures
import iclone_adapter.root_flight_fix as fix
from iclone_adapter.schema import ContractError


class Time:
    """iClone 8 RTime mock: ToInt only, deliberately NO GetValue."""
    def __init__(self,n):
        self.n=int(n)
    def ToInt(self):
        return self.n


class Fps:
    @staticmethod
    def ToDouble():
        return 24.0


class Status:
    def __init__(self,error=False):
        self.error=error
    def IsError(self):
        return self.error


class Axis:
    def __init__(self,values,events):
        self.values=list(values)
        self.events=events
        self.fail=False
    def GetKeyCount(self):
        return len(self.values)
    def GetValue(self,time,default):
        return Status(),self.values[time.ToInt()]
    def SetValue(self,time,value):
        self.events.append(("set",time.ToInt(),float(value)))
        if self.fail:
            return Status(True)
        self.values[time.ToInt()]=float(value)
        return Status()


class Clip:
    @staticmethod
    def ClipTimeToSceneTime(t):
        return t


class IO:
    def __init__(self,events,axes):
        self.events=events
        self.axes=axes
        self.original=[list(x.values) for x in axes]
        self.fail_save=False
        self.fail_load=False
    def SaveProject(self,path):
        self.events.append(("save",str(path)))
        if self.fail_save:
            return Status(True)
        pathlib.Path(path).write_bytes(b"mock iProject")
        return Status()
    def LoadProject(self,path):
        self.events.append(("restore",str(path)))
        if self.fail_load:
            return Status(True)
        for axis,values in zip(self.axes,self.original):
            axis.values[:]=values
        return Status()


class SDK:
    def __init__(self,events,axes):
        self.RFileIO=IO(events,axes)
        self.RGlobal=mock.Mock()
        self.RGlobal.GetFps.return_value=Fps()
    @staticmethod
    def IndexedFrameTime(frame,fps):
        return Time(frame)


class RootFlightFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DragonPreviewTests.setUpClass()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=pathlib.Path(self.tmp.name)
        self.motion=copy.deepcopy(fixtures.DragonPreviewTests().make_motion())
        self.motion["trans"]=[
            [0.0,0.0,0.0],[0.0,0.0,0.5],
            [0.0,0.0,1.2],[0.0,0.0,0.0]]
        self.path=self.root/"original_motion.json"
        self.path.write_text(json.dumps(self.motion),encoding="utf-8")
        _,_,expected=fix._load_source(self.path)
        self.events=[]
        self.axes=[Axis([row[i] for row in expected],self.events)
                   for i in range(3)]
        self.sdk=SDK(self.events,self.axes)
        self.stub=mock.patch.object(
            fix,"_controls",return_value=(
                object(),object(),Clip(),Fps(),self.axes))
        self.stub.start()
        self.addCleanup(self.stub.stop)

    def test_read_only_inspect_exposes_peak_height_and_no_writes(self):
        report=fix.inspect_root_scale(self.path,rlpy=self.sdk)
        self.assertEqual(report["frame_count"],4)
        self.assertEqual(report["peak_frame"],2)
        self.assertAlmostEqual(report["source_vertical_rise_m"],1.2)
        self.assertEqual(report["status"],"source_motion_position_matches_scale_1")
        self.assertEqual(report["key_counts_xyz"],[4,4,4])
        self.assertEqual(self.events,[])

    def test_backup_before_any_key_write_and_scaled_project_is_separate(self):
        result=fix.rescale_root_position(
            self.path,self.root,100.0,rlpy=self.sdk)
        self.assertEqual(result["status"],
                         "experimental_root_scale_keys_verified_project_saved")
        self.assertAlmostEqual(result["source_vertical_rise_m"],1.2)
        self.assertEqual(result["multiplier"],100.0)
        self.assertTrue(pathlib.Path(result["backup_iProject"]).is_file())
        self.assertTrue(pathlib.Path(result["adjusted_iProject"]).is_file())
        self.assertNotEqual(result["backup_iProject"],
                            result["adjusted_iProject"])
        self.assertEqual(self.events[0][0],"save")
        self.assertIn(("set",2,self.axes[0].values[2]),self.events)
        _,_,expected=fix._load_source(self.path)
        for i in range(3):
            for t in range(4):
                self.assertAlmostEqual(self.axes[i].values[t],
                                       expected[t][i]*100.0)
        self.assertEqual(self.motion["trans"][2][2],1.2)
        self.assertTrue(pathlib.Path(result["receipt_path"]).is_file())

    def test_wrong_clip_keys_must_not_be_rescaled_or_backed_up(self):
        self.axes[0].values[1]+=4.0
        with self.assertRaisesRegex(ContractError,"NOT the original pelvis-local"):
            fix.rescale_root_position(self.path,self.root,100.,rlpy=self.sdk)
        self.assertEqual(self.events,[])
        self.assertFalse(list(self.root.glob("*.iProject")))

    def test_backup_failure_must_not_write_any_position_keys(self):
        self.sdk.RFileIO.fail_save=True
        with self.assertRaisesRegex(ContractError,"backup original iProject"):
            fix.rescale_root_position(self.path,self.root,100.,rlpy=self.sdk)
        self.assertEqual([e[0] for e in self.events],["save"])

    def test_partial_setvalue_failure_restores_original_backup(self):
        old=[list(axis.values) for axis in self.axes]
        self.axes[1].fail=True
        with self.assertRaisesRegex(ContractError,"Original project reloaded"):
            fix.rescale_root_position(self.path,self.root,100.,rlpy=self.sdk)
        self.assertEqual(self.events[0][0],"save")
        self.assertEqual(self.events[-1][0],"restore")
        for axis,before in zip(self.axes,old):
            self.assertEqual(axis.values,before)
        self.assertTrue(list(self.root.glob("*.iProject")))

    def test_failure_to_restore_reports_existing_manual_backup(self):
        self.axes[0].fail=True
        self.sdk.RFileIO.fail_load=True
        with self.assertRaisesRegex(ContractError,"RESTORE also failed"):
            fix.rescale_root_position(self.path,self.root,100.,rlpy=self.sdk)
        self.assertTrue(list(self.root.glob("AnyMove_BeforeRootScale_*.iProject")))


    def test_pelvis_rest_basis_turns_source_up_into_negative_local_z(self):
        _,motion,local=fix._load_source(self.path)
        peak=2
        self.assertGreater(motion["trans"][peak][2]-motion["trans"][0][2],1.)
        self.assertLess(local[peak][2]-local[0][2],-1.)
        self.assertGreater(local[peak][1]-local[0][1],0.3)

    def test_already_scaled_local_clip_remains_identifiable_and_read_only(self):
        for axis in self.axes:
            axis.values[:]=[100.*v for v in axis.values]
        inspect=fix.inspect_root_scale(self.path,rlpy=self.sdk)
        self.assertAlmostEqual(inspect["detected_current_scale"],100.,places=5)
        self.assertEqual(inspect["status"],
                         "source_motion_position_matches_scaled_pelvis_local")
        self.assertLess(inspect["original_pelvis_local_peak_delta_xyz"][2],0)
        self.assertGreater(inspect["source_world_axis_peak_delta_xyz"][2],0)
        self.assertEqual(self.events,[])

    def test_replace_x100_downward_local_with_x100_source_xyz_up_without_double_scale(self):
        _,motion,local=fix._load_source(self.path)
        for axis in self.axes:
            axis.values[:]=[100.*v for v in axis.values]
        # Backup fake saves the state of this already x100 clip.
        self.sdk.RFileIO.original=[list(axis.values) for axis in self.axes]
        result=fix.rescale_root_position(
            self.path,self.root,100.0,rlpy=self.sdk,
            position_space="armature")
        self.assertEqual(self.events[0][0],"save")
        self.assertEqual(result["previous_local_position_scale"],100.)
        self.assertEqual(result["position_space"],"armature")
        self.assertEqual(result["multiplier"],100.)
        self.assertGreater(result["expected_stored_position_delta_xyz"][2],100.)
        self.assertAlmostEqual(result["expected_stored_position_delta_xyz"][2],
                               120.,places=5)
        self.assertAlmostEqual(self.axes[2].values[2],120.,places=4)
        self.assertAlmostEqual(self.axes[1].values[2],0.,places=4)
        self.assertAlmostEqual(self.axes[0].values[2],0.,places=4)
        self.assertTrue(pathlib.Path(result["backup_iProject"]).is_file())
        self.assertTrue(pathlib.Path(result["adjusted_iProject"]).is_file())
        self.assertTrue(pathlib.Path(result["receipt_path"]).is_file())

    def test_remap_rejects_previously_modified_arbitrary_axis_data(self):
        self.axes[0].values[2]+=0.7
        self.axes[1].values[2]-=0.2
        with self.assertRaisesRegex(ContractError,"NOT the original pelvis-local"):
            fix.rescale_root_position(
                self.path,self.root,100.,rlpy=self.sdk,
                position_space="armature")
        self.assertEqual(self.events,[])

    def test_reject_unknown_position_space_before_backup(self):
        with self.assertRaisesRegex(ContractError,"position_space"):
            fix.rescale_root_position(
                self.path,self.root,100.,rlpy=self.sdk,
                position_space="not_an_axis")
        self.assertEqual(self.events,[])

    def test_reremapping_armature_track_does_not_stack_a_second_100x(self):
        first=fix.rescale_root_position(
            self.path,self.root,100.,rlpy=self.sdk,
            position_space="armature")
        self.events.clear()
        with self.assertRaisesRegex(
                ContractError,"NOT the original pelvis-local"):
            fix.rescale_root_position(
                self.path,self.root,100.,rlpy=self.sdk,
                position_space="armature")
        self.assertEqual(self.events,[])

    def test_multiplier_restricted_to_opt_in_increase(self):
        for factor in (0,1.0,-1,201.,float("nan")):
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(ContractError,"multiplier"):
                    fix.rescale_root_position(
                        self.path,self.root,factor,rlpy=self.sdk)
        self.assertEqual(self.events,[])



class RootFlightClipTimeTests(unittest.TestCase):
    """Run the REAL _controls contract, not a patched-out shortcut."""

    def _probe(self,scene_origin_tick=0,conversion=True,valid_time=True):
        node=object()
        fake_axes=[mock.Mock() for _ in "XYZ"]
        block=mock.Mock()
        block.GetControl.side_effect=lambda path: fake_axes[
            "XYZ".index(path.rsplit("Position",1)[-1])]
        layer=mock.Mock()
        layer.GetDataBlock.return_value=block
        clip=mock.Mock()
        clip.GetControl.return_value=layer
        if conversion:
            clip.ClipTimeToSceneTime.side_effect=lambda zero: (
                Time(scene_origin_tick) if valid_time else object())
        else:
            clip.ClipTimeToSceneTime=None
        skel=mock.Mock()
        skel.GetClipCount.return_value=1
        skel.GetClip.return_value=clip
        skel.GetSkinBones.return_value=[node]
        sdk=mock.Mock()
        sdk.RGlobal.GetFps.return_value=Fps()
        sdk.IndexedFrameTime.side_effect=lambda frame,fps: Time(frame*250)
        # Critically, Time above has no GetValue. iClone 8 RTime.ToInt works.
        with mock.patch.object(fix,"selected_object",return_value=object()):
            with mock.patch.object(fix,"_get_skeleton",return_value=skel):
                with mock.patch.object(
                    fix,"build_bone_registry",return_value=(
                        [{"name":"pelvis","node":node}],{})):
                    return fix._controls(sdk),clip,fake_axes

    def test_iclone8_rtime_toint_only_zero_origin_is_accepted(self):
        outcome,clip,axes=self._probe(scene_origin_tick=0)
        self.assertEqual(outcome[-1],axes)
        self.assertEqual(clip.ClipTimeToSceneTime.call_count,1)

    def test_shifted_clip_start_is_still_rejected(self):
        with self.assertRaisesRegex(
                ContractError,"maps to scene tick 250"):
            self._probe(scene_origin_tick=250)

    def test_missing_clip_time_conversion_is_blocked(self):
        with self.assertRaisesRegex(
                ContractError,"ClipTimeToSceneTime unavailable"):
            self._probe(conversion=False)

    def test_unknown_time_type_is_blocked_without_writing(self):
        with self.assertRaisesRegex(
                ContractError,"RTime.ToInt ticks"):
            self._probe(valid_time=False)



if __name__=="__main__":
    unittest.main()
