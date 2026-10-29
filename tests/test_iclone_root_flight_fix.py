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
    def __init__(self,n):
        self.n=int(n)
    def GetValue(self):
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
        return Status(),self.values[time.GetValue()]
    def SetValue(self,time,value):
        self.events.append(("set",time.GetValue(),float(value)))
        if self.fail:
            return Status(True)
        self.values[time.GetValue()]=float(value)
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
        with self.assertRaisesRegex(ContractError,"NOT the original scale-1"):
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

    def test_multiplier_restricted_to_opt_in_increase(self):
        for factor in (0,1.0,-1,201.,float("nan")):
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(ContractError,"multiplier"):
                    fix.rescale_root_position(
                        self.path,self.root,factor,rlpy=self.sdk)
        self.assertEqual(self.events,[])


if __name__=="__main__":
    unittest.main()
