"""Root conversion math and strict contract tests; no live iClone dependency."""
import copy
import math
import unittest

import test_iclone_dragon_preview as fixture
from iclone_adapter.anymove_bridge import from_euler_xyz,qmul,qinverse_unit
from iclone_adapter.dragon_preview import make_preview
from iclone_adapter.rest_basis import root_rest_basis
from iclone_adapter.schema import ContractError
from iclone_adapter.motion_contract import validate_motion


class RootConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.DragonPreviewTests.setUpClass()
        cls.info=fixture.DragonPreviewTests.source
        cls.target=fixture.DragonPreviewTests().make_target()

    def source(self):
        data=fixture.DragonPreviewTests().make_motion()
        data["root_quat"][1]=[math.cos(.2),math.sin(.2),0.,0.]
        data["root_quat"][2]=[math.cos(.25),0.,math.sin(.25),0.]
        data["trans"][1]=[1.0,-.3,.5]
        return data

    def test_rotation_only_root(self):
        source=self.source()
        out=make_preview(source,self.target,self.info,bones="all",
                         basis="rest",root_mode="rotation")
        self.assertEqual(len(out["tracks"]),42)
        self.assertNotIn("position_tracks",out)
        self.assertEqual(out["root_motion"]["components"],"rotation_only")
        self.assertFalse(out["root_motion"]["calibrated"])
        rid=next(b["id"] for b in self.target["bones"] if b["name"]=="pelvis")
        base=root_rest_basis(self.info)
        expected=qmul(qmul(base,source["root_quat"][1]),qinverse_unit(base))
        actual=from_euler_xyz(out["tracks"][rid][1])
        self.assertAlmostEqual(abs(sum(x*y for x,y in zip(expected,actual))),
                               1.,places=6)
        validate_motion(out,self.target)

    def test_full_root_needs_explicit_scale_and_has_position_tracks(self):
        source=self.source()
        with self.assertRaises(ContractError):
            make_preview(source,self.target,self.info,bones="all",basis="rest",
                         root_mode="full")
        out=make_preview(source,self.target,self.info,bones="all",
                         basis="rest",root_mode="full",root_position_scale=2.)
        rid=out["root_motion"]["target_bone_id"]
        self.assertEqual(out["position_tracks"][rid][1],[2.,-.6,1.])
        self.assertEqual(out["root_motion"]["root_position_scale"],2.)
        self.assertEqual(out["root_motion"]["components"],"rotation_and_translation")
        validate_motion(out,self.target)

    def test_rejects_invalid_root_samples_and_metadata(self):
        source=self.source()
        source["root_quat"]=source["root_quat"][:2]
        with self.assertRaises(ContractError):
            make_preview(source,self.target,self.info,bones="all",basis="rest",
                         root_mode="rotation")
        source=self.source()
        source["trans"][1]=[0.,float("nan"),0.]
        with self.assertRaises(ContractError):
            make_preview(source,self.target,self.info,bones="all",basis="rest",
                         root_mode="full",root_position_scale=1.)
        source=self.source()
        out=make_preview(source,self.target,self.info,bones="all",
                         basis="rest",root_mode="full",root_position_scale=1.)
        bad=copy.deepcopy(out)
        bad["position_tracks"]["evil"]=[[0.,0.,0.]]*4
        with self.assertRaises(ContractError):
            validate_motion(bad,self.target)
        bad=copy.deepcopy(out)
        bad["root_motion"]["calibrated"]=True
        with self.assertRaises(ContractError):
            validate_motion(bad,self.target)

    def test_root_mode_does_not_change_existing_41_joint_samples(self):
        source=self.source()
        baseline=make_preview(source,self.target,self.info,bones="all",basis="rest")
        root=make_preview(source,self.target,self.info,bones="all",
                          basis="rest",root_mode="rotation")
        for key,track in baseline["tracks"].items():
            self.assertEqual(track,root["tracks"][key])

    def test_root_requires_rest_basis(self):
        with self.assertRaises(ContractError):
            make_preview(self.source(),self.target,self.info,
                         basis="raw",root_mode="rotation")


if __name__=="__main__":
    unittest.main()
