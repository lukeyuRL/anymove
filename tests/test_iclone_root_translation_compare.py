"""Root Translation capture and trajectory comparison using fake SDK."""
import json
import math
import os
import tempfile
import unittest

from iclone_adapter.root_translation_sampler import (
    VERSION,capture_root_sample,save_samples
)
from iclone_adapter.root_translation_compare import compare_root_trajectory
from iclone_adapter.schema import ContractError


class Vector:
    def __init__(self,*xyz):
        self.x,self.y,self.z=xyz


class Transform:
    def __init__(self,x,y,z):
        self.value=Vector(x,y,z)
    def T(self):
        return self.value


class Node:
    def __init__(self,name):
        self.name=name
    def GetName(self):
        return self.name
    def LocalTransform(self):
        return Transform(0,.32,.86)
    def WorldTransform(self):
        return Transform(0,32,86)


class Clip:
    def GetControl(self,key,node):
        return None


class Skel:
    def __init__(self,idx):
        self.idx=idx
        self.pelvis=Node("pelvis")
    def GetClipCount(self):
        return 0
    def ConvertToOriginalBoneAxis(self):
        return [(Node("rig"),Transform(0,0,0)),
                (self.pelvis,Transform(5*self.idx,32,86))]


class Object:
    def __init__(self,skel):
        self.skel=skel
    def GetName(self):
        return "rigged"
    def GetSkeletonComponent(self):
        return self.skel
    def WorldTransform(self):
        return Transform(0,0,0)


class Scene:
    def __init__(self,obj):
        self.obj=obj
    def GetSelectedObjects(self):
        return [self.obj]


class Fps:
    def ToDouble(self):
        return 24.0


class RLPy:
    def __init__(self,frame=0):
        self.frame=frame
        self.RScene=Scene(Object(Skel(frame)))
        self.RGlobal=self
    def GetFps(self):
        return Fps()
    def GetTime(self):
        return self.frame
    def GetFrameIndex(self,time,fps):
        return time


class RootSampleTests(unittest.TestCase):
    def test_manual_sample_does_not_set_time(self):
        fake=RLPy(3)
        report=capture_root_sample(fake)
        self.assertEqual(report["sample"]["frame"],3)
        self.assertEqual(report["sample"]["original_axis_local_xyz"],[15.,32.,86.])
        self.assertIsNone(report["sample"]["stored_pelvis_position_xyz"])

    def test_save_requires_frame0_and_two_more(self):
        records={i:capture_root_sample(RLPy(i)) for i in (0,31,62)}
        with tempfile.TemporaryDirectory() as folder:
            path=os.path.join(folder,"root.json")
            result=save_samples(path,records)
            self.assertEqual(len(result["samples"]),3)
            self.assertEqual(json.load(open(path,encoding="utf-8"))["schema_version"],
                             VERSION)
            with self.assertRaises(ContractError):
                save_samples(path,{31:records[31],62:records[62]})

    def test_compare_fits_correct_scale_and_direction(self):
        joints=[[ [f/100.,0.,0.] ] for f in range(124)]
        blender={"names":["hips"],"joints":joints,"fps":24.}
        samples={"schema_version":VERSION,"fps":24.,"samples":[
            {"frame":f,"original_axis_local_xyz":[float(f),32.,86.]}
            for f in (0,31,62,93,123)]}
        result=compare_root_trajectory(blender,samples,expected_scale=100.)
        self.assertEqual(result["result"],"direction_and_path_match_after_uniform_scaling")
        self.assertAlmostEqual(result["best_fit_native_units_per_blender_unit"],100.,places=8)
        self.assertAlmostEqual(result["suggested_position_scale_multiplier"],1.,places=7)
        self.assertAlmostEqual(result["path_shape_relative_rms_error"],0.,places=7)
        self.assertLess(result["max_direction_angle_deg"],1e-5)

    def test_detects_wrong_direction_and_static_positions(self):
        blender={"names":["hips"],"joints":[[[float(f),0,0]] for f in range(4)],"fps":24.}
        original={"schema_version":VERSION,"fps":24.,"samples":[
            {"frame":i,"original_axis_local_xyz":[0.,float(i),0.]}
            for i in (0,1,2,3)]}
        report=compare_root_trajectory(blender,original)
        self.assertEqual(report["result"],"opposite_or_incompatible_travel_direction")
        original["samples"]=[
            {"frame":i,"original_axis_local_xyz":[0.,0.,0.]}
            for i in (0,1,2,3)]
        self.assertEqual(compare_root_trajectory(blender,original)["result"],
                         "iclone_pelvis_translation_static")

    def test_fps_and_missing_baseline_fail(self):
        blender={"names":["hips"],"joints":[[[float(f),0,0]] for f in range(4)],"fps":30}
        original={"schema_version":VERSION,"fps":24.,"samples":[
            {"frame":i,"original_axis_local_xyz":[float(i),0,0]}
            for i in (0,1,2)]}
        with self.assertRaises(ContractError):
            compare_root_trajectory(blender,original)
        original["fps"]=30
        original["samples"]=original["samples"][1:]
        with self.assertRaises(ContractError):
            compare_root_trajectory(blender,original)


if __name__=="__main__":
    unittest.main()
