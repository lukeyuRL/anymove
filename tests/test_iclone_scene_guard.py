"""Fail-closed experimental Dragon guard tests, without iClone installed."""
import unittest
from iclone_adapter.motion_contract import make_sine_test
from iclone_adapter.motion_writer import write_selected
from iclone_adapter.rlpy_reader import snapshot
from iclone_adapter.schema import ContractError
from iclone_adapter.scene_guard import check_unposed_pose
from test_iclone_adapter import Avatar, RLPyMock, Ctrl, Transform


def fixture(avatar,rlpy):
    sk=snapshot(avatar,rlpy=rlpy)
    bid=next(b["id"] for b in sk["bones"] if b["name"]=="Bone2")
    motion=make_sine_test(sk,bid,frames=8)
    motion["calibration_status"]="unverified_rest_basis_conjugation_preview"
    motion["expected_unposed_world_matrices"]={b["id"]:b["snapshot_world_matrix"] for b in sk["bones"]}
    return motion


class SceneGuardTests(unittest.TestCase):
    def test_empty_scene_and_default_clip(self):
        for existing in (False,True):
            with self.subTest(existing=existing):
                a=Avatar()
                r=RLPyMock(a)
                if existing:
                    a.skeleton.AddClip(0)
                m=fixture(a,r)
                result=write_selected(m,rlpy=r,fps_confirmed=True)
                self.assertTrue(result["scene_guard"]["pose"]["pose_match"])
                self.assertEqual(result["clip_strategy"],"reuse_default" if existing else "create_new")
                self.assertEqual(a.skeleton.GetClipCount(),1)

    def test_reused_scene_is_refused_even_when_non_target_has_keys(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        clip=a.skeleton.AddClip(0)
        channel=clip.GetControl("Layer",a.skeleton.bones[0]).block.GetControl("Rotation/RotationX")
        channel.SetValue(0,.3)
        with self.assertRaisesRegex(ContractError,"existing FK/Position/Scale"):
            write_selected(m,rlpy=r,fps_confirmed=True)
        self.assertEqual(channel.GetKeyCount(),1)
        self.assertEqual(a.skeleton.GetClipCount(),1)

    def test_default_clip_length_is_not_assumed_empty(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        a.skeleton.AddClip(0).SetLength(24)
        with self.assertRaisesRegex(ContractError,"default Clip has nonzero"):
            write_selected(m,rlpy=r,fps_confirmed=True)

    def test_reject_existing_actor_animation_and_multiple_clips(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        a.skeleton.AddClip(0)
        a.skeleton.AddClip(32)
        with self.assertRaisesRegex(ContractError,"Clip count"):
            write_selected(m,rlpy=r,fps_confirmed=True)
        a.skeleton.clips.pop()
        control=Ctrl()
        control.SetValue(0,1)
        a.GetControl=lambda path: control if path=="Transform" else None
        with self.assertRaisesRegex(ContractError,"Transform keys"):
            write_selected(m,rlpy=r,fps_confirmed=True)

    def test_reject_pose_mismatch_and_missing_baseline(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        node=a.skeleton.bones[1]
        original=node.WorldTransform
        def changed():
            mat=original().raw[:]
            mat[12]+=20.0
            return Transform(mat)
        node.WorldTransform=changed
        with self.assertRaisesRegex(ContractError,"NOT in original unposed pose"):
            write_selected(m,rlpy=r,fps_confirmed=True)
        self.assertEqual(a.skeleton.GetClipCount(),0)
        m.pop("expected_unposed_world_matrices")
        with self.assertRaisesRegex(ContractError,"no unposed Bone baseline"):
            write_selected(m,rlpy=r,fps_confirmed=True)

    def test_cannot_resubmit_over_written_dragon_animation(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        write_selected(m,rlpy=r,fps_confirmed=True)
        with self.assertRaises(ContractError):
            write_selected(m,rlpy=r,fps_confirmed=True)
        self.assertEqual(a.skeleton.GetClipCount(),1)

    def test_reference_check_rejects_extra_bone(self):
        a=Avatar()
        r=RLPyMock(a)
        sk=snapshot(a,rlpy=r)
        ref={b["id"]:b["snapshot_world_matrix"] for b in sk["bones"]}
        ref["alien"]=[1.0]*16
        with self.assertRaisesRegex(ContractError,"IDs differ"):
            check_unposed_pose(ref,sk)


if __name__=="__main__":
    unittest.main()
