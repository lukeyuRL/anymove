"""Unit contracts for TRAIN-only two-stage Human toe kinematic guard."""
import copy
import unittest

import numpy as np

from mocap.toe_tip_guard import (
    calibrate_from_train, calibrate_base_from_train, guard_foot_chain,
    ToeBaseGuardConfig, ToeTipGuardConfig, confidence_scales_from_diagnostics
)

def fixture():
    names=[
        "CC_Base_"+side+"_"+joint
        for side in ("L","R")
        for joint in ("Calf","Foot","ToeBase","ToeTip")
    ]+["OTHER_%02d"%i for i in range(59)]
    assert len(names)==67
    keys={name:i for i,name in enumerate(names)}
    angles=[315,45]
    rows=[]
    for slot,angle in zip(("front","side"),angles):
        for t in range(60):
            coords=np.zeros((67,2),dtype=float)
            for side_i,side in enumerate(("L","R")):
                foot=np.array((205 + t*.13 + side_i*18,
                               340 + 2*np.sin(t*.19+side_i)),dtype=float)
                bend=np.array((3*np.sin(t*.2+side_i),38+5*np.cos(t*.15+side_i)))
                matrix_base=np.array(((.12,.13),(.42,-.03)))
                matrix_tip=np.array(((.22,-.01),(.03,.15)))
                calf=foot-bend
                base=foot + bend@matrix_base
                tip=base+(base-foot)@matrix_tip
                for name,p in (("Calf",calf),("Foot",foot),("ToeBase",base),("ToeTip",tip)):
                    coords[keys["CC_Base_"+side+"_"+name]]=p
            rows.append({"split":"train","slot":slot,"camera_azimuth_deg":angle,
                         "frame":t+1,"uv":coords.tolist()})
    return {"joint_names":names,"angles_deg":angles,"samples":rows}

class FootChainGuardTests(unittest.TestCase):
    def setUp(self):
        self.data=fixture()
        self.base=calibrate_base_from_train(self.data)
        self.tip=calibrate_from_train(self.data)
        self.names=self.data["joint_names"]
        self.key={n:i for i,n in enumerate(self.names)}
        self.uv=np.asarray(self.data["samples"][17]["uv"],dtype=float)
    def test_calibration_is_train_only(self):
        altered=copy.deepcopy(self.data)
        huge=np.ones((67,2))*1e6
        altered["samples"] += [{
            "split":"test","slot":"front","camera_azimuth_deg":315,
            "frame":33,"uv":huge.tolist()
        }]
        a=calibrate_base_from_train(altered)
        b=calibrate_from_train(altered)
        for k in a:
            self.assertEqual(a[k].to_dict(), self.base[k].to_dict())
            self.assertEqual(b[k].to_dict(), self.tip[k].to_dict())
    def test_normal_view_not_modified(self):
        result,diag=guard_foot_chain(self.uv,self.names,"front",self.base,self.tip,(315,45))
        np.testing.assert_allclose(result,self.uv)
        self.assertEqual(diag["base_repaired"],0)
        self.assertEqual(diag["tip_repaired"],0)
    def test_catastrophic_tip_only_is_repaired(self):
        v=self.uv.copy()
        tip=self.key["CC_Base_L_ToeTip"]
        v[tip]+=[207,-130]
        result,diag=guard_foot_chain(v,self.names,"front",self.base,self.tip,(315,45))
        self.assertEqual(diag["base_repaired"],0)
        self.assertEqual(diag["tip_repaired"],1)
        np.testing.assert_allclose(result[tip],self.uv[tip],atol=1e-6)
        np.testing.assert_allclose(result[self.key["CC_Base_R_ToeTip"]],
                                   self.uv[self.key["CC_Base_R_ToeTip"]])
    def test_misplaced_base_and_tip_are_repaired(self):
        v=self.uv.copy()
        base=self.key["CC_Base_L_ToeBase"]
        tip=self.key["CC_Base_L_ToeTip"]
        v[[base,tip]] += [90,-70]
        result,diag=guard_foot_chain(v,self.names,"front",self.base,self.tip,(315,45))
        self.assertEqual(diag["base_repaired"],1)
        self.assertEqual(diag["tip_repaired"],1)
        np.testing.assert_allclose(result[[base,tip]],self.uv[[base,tip]],atol=1e-6)
    def test_bad_foot_anchor_is_flagged_untrusted(self):
        v=self.uv.copy()
        foot=self.key["CC_Base_L_Foot"]
        base=self.key["CC_Base_L_ToeBase"]
        tip=self.key["CC_Base_L_ToeTip"]
        v[[foot,base,tip]] += [200,-140]
        result,diag=guard_foot_chain(v,self.names,"front",self.base,self.tip,(315,45))
        self.assertEqual(diag["base_untrusted"],1)
        self.assertEqual(diag["tip_untrusted"],1)
        np.testing.assert_allclose(result[[base,tip]],v[[base,tip]])
    def test_confidence_scaling_flags_recovery_and_missing(self):
        v=self.uv.copy()
        foot=self.key['CC_Base_L_Foot']
        base=self.key['CC_Base_L_ToeBase']
        tip=self.key['CC_Base_L_ToeTip']
        v[[base,tip]] += [90,-70]
        _,diag=guard_foot_chain(v,self.names,'front',self.base,self.tip,(315,45))
        quality=confidence_scales_from_diagnostics(self.names,diag)
        self.assertEqual(quality[base],0.25)
        self.assertEqual(quality[tip],0.25)
        self.assertEqual(quality[foot],1.0)
        v=self.uv.copy()
        v[[foot,base,tip]] += [200,-140]
        _,diag=guard_foot_chain(v,self.names,'front',self.base,self.tip,(315,45))
        quality=confidence_scales_from_diagnostics(self.names,diag)
        self.assertEqual(quality[base],0)
        self.assertEqual(quality[tip],0)
        with self.assertRaisesRegex(ValueError,'inferred_scale'):
            confidence_scales_from_diagnostics(self.names,diag,1.5)

    def test_mismatched_camera_pair_rejected(self):
        with self.assertRaisesRegex(ValueError,"angle"):
            guard_foot_chain(self.uv,self.names,"front",self.base,self.tip,(0,90))
    def test_binary_metadata_serialization(self):
        for cfg in self.base.values():
            self.assertEqual(ToeBaseGuardConfig.from_dict(cfg.to_dict()),cfg)
        for cfg in self.tip.values():
            self.assertEqual(ToeTipGuardConfig.from_dict(cfg.to_dict()),cfg)
    def test_batched_inputs(self):
        pts=np.stack([self.uv,self.uv],axis=0)
        tip=self.key["CC_Base_L_ToeTip"]
        pts[1,tip]+=[200,-140]
        res,diag=guard_foot_chain(pts,self.names,"front",self.base,self.tip,(315,45))
        self.assertEqual(diag["tip_repaired"],1)
        self.assertEqual(diag["toe_tip_flags"]["CC_Base_L_ToeTip"]["replaced"],[False,True])
        np.testing.assert_allclose(res[1,tip],self.uv[tip],atol=1e-6)

if __name__=="__main__":
    unittest.main()
