"""SDK mock tests for full FK/Position subframe interpolation survey."""
import unittest

from iclone_adapter.motion_writer import write_selected
from iclone_adapter.subframe_interpolation_audit import (
    survey_subframe_controllers,VERSION
)
from iclone_adapter.schema import ContractError
from test_iclone_root_writer import setup_root


class MockNative:
    def __init__(self,nonlinear=False,broken=False):
        self.actor,self.sdk,self.motion=setup_root()
        self.sdk.IndexedFrameTime=lambda frame,fps:int(frame)*100
        self.sdk.RTime=lambda ticks:int(ticks)
        write_selected(self.motion,rlpy=self.sdk,fps_confirmed=True)
        clip=self.actor.skeleton.GetClip(0)
        for layer in clip.layers.values():
            for name,ctrl in layer.block.ctrls.items():
                original=ctrl.GetValue
                def sample(time,out,control=ctrl,name=name,original=original):
                    if time%100==0:
                        return original(time,out)
                    entries=sorted(control.values)
                    lower=max((x for x in entries if x[0]<time),default=None)
                    upper=min((x for x in entries if x[0]>time),default=None)
                    if lower is None or upper is None:
                        return (False,0.0)
                    f=float(time-lower[0])/float(upper[0]-lower[0])
                    value=lower[1]+f*(upper[1]-lower[1])
                    if nonlinear and name=="Rotation/RotationX" and time==150:
                        value+=.2
                    if broken and name=="Position/PositionY" and time==150:
                        return (False,0.0)
                    return (True,value)
                ctrl.GetValue=sample


class MidpointAuditTests(unittest.TestCase):
    def test_all_midpoints_linear_and_timeline_unchanged(self):
        mock=MockNative()
        old=mock.sdk.RGlobal.GetTime()
        r=survey_subframe_controllers(mock.motion,rlpy=mock.sdk)
        self.assertEqual(r["schema_version"],VERSION)
        self.assertEqual(r["status"],
                         "native_subframes_match_linear_hypothesis")
        self.assertEqual(r["expected_scalar_samples"],54)
        self.assertEqual(r["read_scalar_samples"],54)
        self.assertEqual(r["nonlinear_vs_linear_scalar_samples"],0)
        self.assertEqual(r["overshooting_scalar_samples"],0)
        self.assertTrue(r["timeline_unchanged"])
        self.assertFalse(r["evaluated_pose_verified"])
        self.assertEqual(mock.sdk.RGlobal.GetTime(),old)

    def test_nonlinearity_is_diagnostic_not_failed_readback(self):
        mock=MockNative(nonlinear=True)
        r=survey_subframe_controllers(mock.motion,rlpy=mock.sdk)
        self.assertEqual(r["status"],
                         "native_subframes_deviate_from_linear_hypothesis")
        self.assertEqual(r["read_scalar_samples"],54)
        self.assertGreater(r["nonlinear_vs_linear_scalar_samples"],0)
        self.assertGreater(r["max_abs_deviation_from_linear_by_kind"][
            "Rotation"],.19)
        self.assertEqual(r["first_failures"],[])

    def test_failed_native_getvalue_blocks_complete_readback(self):
        mock=MockNative(broken=True)
        r=survey_subframe_controllers(mock.motion,rlpy=mock.sdk)
        self.assertEqual(r["status"],"subframe_readback_incomplete")
        self.assertLess(r["read_scalar_samples"],54)
        self.assertFalse(r["native_subframe_values_readable"])
        self.assertTrue(r["first_failures"])

    def test_sdk_time_unsupported_and_sample_limit(self):
        mock=MockNative()
        del mock.sdk.RTime
        with self.assertRaisesRegex(ContractError,"RLPy.RTime unavailable"):
            survey_subframe_controllers(mock.motion,rlpy=mock.sdk)
        mock=MockNative()
        mock.sdk.IndexedFrameTime=lambda frame,fps: int(frame)
        with self.assertRaisesRegex(ContractError,"no representable subframe"):
            survey_subframe_controllers(mock.motion,rlpy=mock.sdk)
        mock=MockNative()
        with self.assertRaisesRegex(ContractError,"limit exceeded"):
            survey_subframe_controllers(
                mock.motion,rlpy=mock.sdk,max_scalar_samples=10)

    def test_invalid_clip_rejected_without_time_change(self):
        mock=MockNative()
        with self.assertRaises(ContractError):
            survey_subframe_controllers(mock.motion,clip_index=20,
                                        rlpy=mock.sdk)


if __name__=="__main__":
    unittest.main()
