"""Pure Python tests for the single-hierarchy Parent/Child A/B convention test."""
import copy
import math
import unittest

import test_iclone_dragon_preview as fixture
from iclone_adapter.anymove_bridge import from_euler_xyz, qmul, qinverse_unit
from iclone_adapter.parent_child_probe import (
    ANCHORS_DEG, _quat_from_rest_matrix, control_source_motion,
    make_probe, relative_rest_basis,
)
from iclone_adapter.schema import ContractError
from iclone_adapter.motion_contract import validate_motion


class ParentChildProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DragonPreviewTests.setUpClass()
        cls.info=fixture.DragonPreviewTests.source
        cls.target=fixture.DragonPreviewTests().make_target()

    def test_same_pair_two_controlled_modes(self):
        source,a,b,manifest=make_probe(self.info,self.target)
        self.assertEqual(source["fps"],24.)
        self.assertEqual(manifest["frames"],169)
        self.assertEqual(source["bones"][0],"pelvis")
        self.assertEqual(len(source["euler"]),169)
        self.assertEqual(len(a["tracks"]),2)
        self.assertEqual(len(b["tracks"]),2)
        self.assertEqual(a["calibration_status"],"unverified_identity_same_rig_preview")
        self.assertEqual(b["calibration_status"],"unverified_rest_basis_conjugation_preview")
        validate_motion(a,self.target)
        validate_motion(b,self.target)
        byname={x["name"]:x["id"] for x in self.target["bones"]}
        parent=byname["wing_arm_L"]
        child=byname["wing_fore_L"]
        self.assertAlmostEqual(a["tracks"][parent][12][0],math.radians(25),places=7)
        self.assertEqual(a["tracks"][child][12],[0.,0.,0.])
        self.assertAlmostEqual(a["tracks"][child][108][1],math.radians(25),places=7)
        self.assertEqual(a["tracks"][parent][108],[0.,0.,0.])
        self.assertEqual(a["tracks"][parent][0],[0.,0.,0.])
        self.assertEqual(b["tracks"][child][0],[0.,0.,0.])
        self.assertTrue(all(x==[0.,0.,0.] for x in source["trans"]))
        self.assertTrue(all(x==[1.,0.,0.,0.] for x in source["root_quat"]))
        # A genuinely tilted wing bone must produce two different hypotheses.
        err=max(abs(x-y) for x,y in zip(
            a["tracks"][child][108],b["tracks"][child][108]))
        self.assertGreater(err,0.1)
        self.assertEqual(manifest["sample_frames"][-2]["label"],"parent_x_plus_child_y")

    def test_conjugation_is_rotation_correct(self):
        source,a,b,_=make_probe(self.info,self.target)
        byname={x["name"]:x["id"] for x in self.target["bones"]}
        for bone in ("wing_arm_L","wing_fore_L"):
            bid=byname[bone]
            basis=relative_rest_basis(self.info,bone)
            for frame in (0,12,36,60,84,108,132,156,168):
                original=from_euler_xyz(a["tracks"][bid][frame])
                predicted=qmul(qmul(basis,original),qinverse_unit(basis))
                actual=from_euler_xyz(b["tracks"][bid][frame])
                # Quaternion sign equivalence, do not compare components alone.
                dot=abs(sum(x*y for x,y in zip(predicted,actual)))
                self.assertAlmostEqual(dot,1.,places=6)

    def test_rest_basis_quaternion_identity_and_quarter_turn(self):
        identity=[1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.]
        q=_quat_from_rest_matrix(identity)
        self.assertAlmostEqual(abs(q[0]),1.)
        # +90deg rotation about X, row-major column vector matrix.
        ninety=[1.,0.,0.,0.,0.,0.,-1.,0.,0.,1.,0.,0.,0.,0.,0.,1.]
        q=_quat_from_rest_matrix(ninety)
        self.assertAlmostEqual(abs(q[1]),math.sqrt(.5),places=6)
        self.assertAlmostEqual(abs(q[0]),math.sqrt(.5),places=6)

    def test_rejects_non_parent_pair_and_pose_drift(self):
        with self.assertRaises(ContractError):
            control_source_motion(self.info,"wing_fore_L","wing_arm_L")
        with self.assertRaises(ContractError):
            control_source_motion(self.info,"pelvis","tail1")
        with self.assertRaises(ContractError):
            control_source_motion(self.info,"wing_arm_L","wing_fore_L",fps=29.97)
        bad=copy.deepcopy(self.target)
        bone=next(x for x in bad["bones"] if x["name"]=="wing_fore_L")
        bone["snapshot_world_matrix"][3]+=100.
        with self.assertRaises(ContractError):
            make_probe(self.info,bad)

    def test_other_real_bone_chains(self):
        for parent,child in (("neck1","neck2"),("tail2","tail3")):
            _,a,b,m=make_probe(self.info,self.target,parent,child)
            self.assertEqual(m["parent"],parent)
            self.assertEqual(m["child"],child)
            self.assertEqual(len(a["tracks"]),2)
            self.assertEqual(len(b["tracks"]),2)


if __name__=="__main__":
    unittest.main()
