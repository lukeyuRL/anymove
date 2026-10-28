"""Root Position writer tests (mock SDK); no actual root retarget assertion."""
import unittest

from iclone_adapter.motion_contract import make_sine_test
from iclone_adapter.motion_writer import write_selected
from iclone_adapter.rlpy_reader import snapshot
from iclone_adapter.schema import ContractError
from test_iclone_adapter import Avatar,RLPyMock,DataBlock,Ctrl


def setup_root(available=True):
    avatar=Avatar()
    rlpy=RLPyMock(avatar)
    baseline=snapshot(avatar,rlpy=rlpy)
    roots={b["name"]:b["id"] for b in baseline["bones"]}
    motion=make_sine_test(baseline,roots["Bone2"],frames=7)
    motion["tracks"][roots["Bone1"]]=[[.1*f,0.,0.] for f in range(7)]
    motion["position_tracks"]={roots["Bone1"]:[[f*.12,0.,0.] for f in range(7)]}
    motion["root_motion"]={
        "mode":"pelvis_layer","components":"rotation_and_translation",
        "source_root":"pelvis","target_bone_id":roots["Bone1"],
        "root_position_scale":1.0,"calibrated":False}
    motion["calibration_status"]="unverified_rest_basis_conjugation_preview"
    motion["expected_unposed_world_matrices"]={
        b["id"]:b["snapshot_world_matrix"] for b in baseline["bones"]}
    original_add=avatar.skeleton.AddClip
    def patched_add(t):
        c=original_add(t)
        old_get=c.GetControl
        def get_control(key,bone):
            result=old_get(key,bone)
            if result is not None and bone.GetName()=="Bone1" and available:
                for axis in "XYZ":
                    name="Position/Position"+axis
                    result.block.ctrls.setdefault(name,Ctrl())
            return result
        c.GetControl=get_control
        return c
    avatar.skeleton.AddClip=patched_add
    return avatar,rlpy,motion


class RootWriterTests(unittest.TestCase):
    def test_full_root_writes_rot_and_pos_on_pelvis(self):
        avatar,rlpy,m=setup_root()
        result=write_selected(m,rlpy=rlpy,fps_confirmed=True)
        self.assertEqual(result["frame_count"],7)
        self.assertEqual(result["root_motion"]["components"],"rotation_and_translation")
        self.assertAlmostEqual(
            result["root_position_readback"][m["root_motion"]["target_bone_id"]][
                "max_abs_error"],0.,places=7)
        bone=avatar.skeleton.bones[0]
        ctrl=avatar.skeleton.GetClip(0).GetControl("Layer",bone).block.GetControl(
            "Position/PositionX")
        self.assertEqual(ctrl.GetKeyCount(),7)
        self.assertAlmostEqual(ctrl.GetValue(6,0.0)[1],.72)

    def test_missing_root_position_control_fails_and_rolls_back(self):
        avatar,rlpy,m=setup_root(available=False)
        with self.assertRaisesRegex(ContractError,"Position XYZ Layer controls unavailable"):
            write_selected(m,rlpy=rlpy,fps_confirmed=True)
        self.assertEqual(avatar.skeleton.GetClipCount(),0)
        self.assertTrue(avatar.skeleton.rollback)

    def test_position_write_failure_rolls_back(self):
        avatar,rlpy,m=setup_root()
        original_add=avatar.skeleton.AddClip
        def broken_add(t):
            clip=original_add(t)
            root=clip.GetControl("Layer",avatar.skeleton.bones[0])
            class Broken(Ctrl):
                def SetValue(self,time,value):
                    return False
            root.block.ctrls["Position/PositionX"]=Broken()
            return clip
        avatar.skeleton.AddClip=broken_add
        with self.assertRaises(ContractError):
            write_selected(m,rlpy=rlpy,fps_confirmed=True)
        self.assertEqual(avatar.skeleton.GetClipCount(),0)
        self.assertTrue(avatar.skeleton.rollback)


if __name__=="__main__":
    unittest.main()
