"""Fail-closed experimental Dragon guard tests, without iClone installed."""
import unittest
from iclone_adapter.motion_contract import make_sine_test
from iclone_adapter.motion_writer import write_selected
from iclone_adapter.rlpy_reader import snapshot
from iclone_adapter.schema import ContractError
from iclone_adapter.scene_guard import check_unposed_pose,inspect_actor_transform
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
        class AnimatedActor(Ctrl):
            def MaxControlTime(self):
                return 200
        control=AnimatedActor()
        control.SetValue(200,1)
        a.GetControl=lambda path: control if path=="Transform" else None
        with self.assertRaisesRegex(ContractError,"not at Frame 0"):
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


    def test_single_transform_key_at_zero_is_normal_import_state(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        class DefaultActor(Ctrl):
            def MaxControlTime(self):
                return 0
        ctrl=DefaultActor()
        ctrl.SetValue(0,17.2)  # nonidentity object initial transform is OK
        a.GetControl=lambda name: ctrl if name=="Transform" else None
        report=write_selected(m,rlpy=r,fps_confirmed=True)
        self.assertEqual(report["scene_guard"]["clip"]["actor_transform"]["status"],
                         "single_static_frame0_transform_key")
        self.assertEqual(ctrl.GetKeyCount(),1)  # cannot delete/modify baseline

    def test_transform_keys_at_multiple_times_with_same_values_are_static(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        class V3:
            def __init__(self,*args):
                self.x,self.y,self.z=args
        class Q4:
            def __init__(self,*args):
                self.x,self.y,self.z,self.w=args
        class Tr:
            def __init__(self,p):
                self.p=p
            def T(self):
                return V3(*self.p)
            def R(self):
                return Q4(0,0,0,1)
            def S(self):
                return V3(1,1,1)
        class Key:
            def __init__(self):
                self.frame=0
                self.pose=Tr((0,0,0))
            def GetTime(self):
                return self.frame
            def GetTransform(self):
                return self.pose
        class ConstantActor:
            def __init__(self):
                self.p=[(0,(2,3,4)),(100,(2,3,4))]
            def GetKeyCount(self):
                return len(self.p)
            def MaxControlTime(self):
                return 100
            def GetTransformKeyAt(self,index,key):
                time,pose=self.p[index]
                key.frame=time
                key.pose=Tr(pose)
                return True
        r.RTransformKey=Key
        ctrl=ConstantActor()
        a.GetControl=lambda name: ctrl if name=="Transform" else None
        report=write_selected(m,rlpy=r,fps_confirmed=True)
        self.assertEqual(report["scene_guard"]["clip"]["actor_transform"]["status"],
                         "multiple_constant_transform_keys")
        self.assertEqual(ctrl.GetKeyCount(),2)

    def test_varying_actor_transform_keys_block_before_clip_write(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        class V3:
            def __init__(self,*args):
                self.x,self.y,self.z=args
        class Q4:
            def __init__(self,*args):
                self.x,self.y,self.z,self.w=args
        class Tr:
            def __init__(self,p):
                self.p=p
            def T(self):
                return V3(*self.p)
            def R(self):
                return Q4(0,0,0,1)
            def S(self):
                return V3(1,1,1)
        class Key:
            def __init__(self):
                self.frame=0
                self.pose=None
            def GetTime(self):
                return self.frame
            def GetTransform(self):
                return self.pose
        class AnimatedActor:
            def GetKeyCount(self):
                return 2
            def MaxControlTime(self):
                return 100
            def GetTransformKeyAt(self,index,key):
                key.frame=0 if index==0 else 100
                key.pose=Tr((0,0,0) if index==0 else (2,0,0))
                return True
        r.RTransformKey=Key
        a.GetControl=lambda name: AnimatedActor() if name=="Transform" else None
        with self.assertRaisesRegex(ContractError,"actor Transform is ANIMATED"):
            write_selected(m,rlpy=r,fps_confirmed=True)
        self.assertEqual(a.skeleton.GetClipCount(),0)

    def test_multiple_actor_keys_without_sdk_readback_are_blocked(self):
        a=Avatar()
        r=RLPyMock(a)
        class Unknown:
            def GetKeyCount(self):
                return 2
            def MaxControlTime(self):
                return 100
        a.GetControl=lambda name: Unknown() if name=="Transform" else None
        with self.assertRaisesRegex(ContractError,"SDK key readback is unavailable"):
            inspect_actor_transform(a,r)

    def test_actor_scalar_channels_are_independently_audited(self):
        a=Avatar()
        r=RLPyMock(a)
        m=fixture(a,r)
        class Axis(Ctrl):
            def MaxControlTime(self):
                return 200
        class Data:
            def GetControl(self,key):
                return axis if key=="Position/PositionX" else None
        class Actor(Ctrl):
            def MaxControlTime(self):
                return 0
            def GetDataBlock(self):
                return Data()
        actor=Actor()
        actor.SetValue(0,1)
        axis=Axis()
        axis.SetValue(200,10)
        a.GetControl=lambda name: actor if name=="Transform" else None
        with self.assertRaisesRegex(ContractError,"Transform channel may animate"):
            write_selected(m,rlpy=r,fps_confirmed=True)
        self.assertEqual(a.skeleton.GetClipCount(),0)


if __name__=="__main__":
    unittest.main()
