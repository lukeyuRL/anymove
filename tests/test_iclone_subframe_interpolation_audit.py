"""SDK mock tests for full FK/Position subframe interpolation survey."""
import unittest

from iclone_adapter.motion_writer import write_selected
from iclone_adapter.subframe_interpolation_audit import (
    survey_subframe_controllers, _midpoints, _ticks, VERSION
)
from iclone_adapter.schema import ContractError
from test_iclone_root_writer import setup_root


class IClone8RTime:
    """Replicate the real iClone 8 zero-argument constructor and factory.

    Calling IClone8RTime(125) deliberately raises:
    TypeError: __init__() takes 1 positional argument but 2 were given.
    """
    from_value_calls = []

    def __init__(self):
        self.native_ticks = 0

    @classmethod
    def FromValue(cls, tick):
        cls.from_value_calls.append(tick)
        obj = cls()
        obj.native_ticks = int(tick)
        return obj

    def ToInt(self):
        return self.native_ticks

    def GetValue(self):
        # Deliberately a different legacy unit convention: use ToInt.
        return self.native_ticks * 10


class MockNative:
    def __init__(self,nonlinear=False,broken=False):
        self.actor,self.sdk,self.motion=setup_root()
        self.sdk.IndexedFrameTime=lambda frame,fps:int(frame)*100
        self.sdk.RTime=IClone8RTime
        write_selected(self.motion,rlpy=self.sdk,fps_confirmed=True)
        clip=self.actor.skeleton.GetClip(0)
        for layer in clip.layers.values():
            for name,ctrl in layer.block.ctrls.items():
                original=ctrl.GetValue
                def sample(time,out,control=ctrl,name=name,original=original):
                    tick=_ticks(time)
                    if tick%100==0:
                        return original(tick,out)
                    entries=sorted(control.values)
                    lower=max((x for x in entries if x[0]<tick),default=None)
                    upper=min((x for x in entries if x[0]>tick),default=None)
                    if lower is None or upper is None:
                        return (False,0.0)
                    f=float(tick-lower[0])/float(upper[0]-lower[0])
                    value=lower[1]+f*(upper[1]-lower[1])
                    if nonlinear and name=="Rotation/RotationX" and tick==150:
                        value+=.2
                    if broken and name=="Position/PositionY" and tick==150:
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
        with self.assertRaisesRegex(ContractError,"RTime.FromValue unavailable"):
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


    def test_real_iclone8_constructor_is_not_called_with_ticks(self):
        mock=MockNative()
        with self.assertRaises(TypeError):
            IClone8RTime(125)
        IClone8RTime.from_value_calls[:]=[]
        midpoint=_midpoints(mock.sdk,mock.sdk.RGlobal.GetFps(),4)
        self.assertEqual([(i,_ticks(t),alpha) for i,t,alpha,tick
                          in midpoint],
                         [(0,50,.5),(1,150,.5),(2,250,.5)])
        self.assertEqual(IClone8RTime.from_value_calls,[50,150,250])
        self.assertTrue(all(isinstance(t,IClone8RTime)
                            for i,t,alpha,tick in midpoint))

    def test_24fps_6000ticks_per_second_uses_sdk_factory(self):
        mock=MockNative()
        mock.sdk.IndexedFrameTime=lambda frame,fps:frame*250
        midpoints=_midpoints(mock.sdk,mock.sdk.RGlobal.GetFps(),4)
        self.assertEqual([tick for _,_,_,tick in midpoints],[125,375,625])
        self.assertEqual([alpha for _,_,alpha,_ in midpoints],[.5,.5,.5])

    def test_missing_factory_and_wrong_factory_ticks_fail_closed(self):
        mock=MockNative()
        class NoFromValue:
            def __init__(self):
                pass
        mock.sdk.RTime=NoFromValue
        with self.assertRaisesRegex(ContractError,"FromValue unavailable"):
            survey_subframe_controllers(mock.motion,rlpy=mock.sdk)

        mock=MockNative()
        class BrokenFactory:
            @staticmethod
            def FromValue(tick):
                return IClone8RTime.FromValue(tick+1)
        mock.sdk.RTime=BrokenFactory
        with self.assertRaisesRegex(ContractError,"different tick"):
            survey_subframe_controllers(mock.motion,rlpy=mock.sdk)

if __name__=="__main__":
    unittest.main()
