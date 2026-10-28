"""Motion Bones vs Skin Bones audit must stay read-only."""
import unittest

from iclone_adapter.animation_bone_audit import audit_animation_bones
from iclone_adapter.motion_contract import make_sine_test
from iclone_adapter.motion_writer import write_selected
from iclone_adapter.rlpy_reader import snapshot
from test_iclone_adapter import Avatar, Bone, RLPyMock


class AuditTests(unittest.TestCase):
    def test_no_motion_api_reported_without_changing_keys_or_time(self):
        avatar=Avatar()
        rlpy=RLPyMock(avatar)
        rest=snapshot(avatar,rlpy=rlpy)
        bone=next(b["id"] for b in rest["bones"] if b["name"]=="Bone2")
        motion=make_sine_test(rest,bone,frames=8)
        write_selected(motion,rlpy=rlpy,fps_confirmed=True)
        ctrl=avatar.skeleton.GetClip(0).layers["Bone2"].block.ctrls[
            "Rotation/RotationX"]
        n_before=ctrl.GetKeyCount()
        t_before=rlpy.RGlobal.GetTime()
        result=audit_animation_bones(motion,rlpy=rlpy)
        self.assertEqual(result["node_sets"]["skin_bones"][
            "target_bones_with_any_fk_key"],1)
        self.assertEqual(result["node_sets"]["skin_bones"][
            "target_bones_readback_matches_1e3"],1)
        self.assertFalse(result["node_sets"]["motion_bones"]["available"])
        self.assertEqual(result["clip_index"],0)
        self.assertEqual(ctrl.GetKeyCount(),n_before)
        self.assertEqual(rlpy.RGlobal.GetTime(),t_before)

    def test_differently_identified_candidate_bone_is_visible(self):
        avatar=Avatar()
        rlpy=RLPyMock(avatar)
        rest=snapshot(avatar,rlpy=rlpy)
        target=next(b["id"] for b in rest["bones"] if b["name"]=="Bone2")
        motion=make_sine_test(rest,target,frames=8)
        write_selected(motion,rlpy=rlpy,fps_confirmed=True)
        class AliasBone(Bone):
            def GetID(self):
                return 2000+self.key
        avatar.skeleton.GetAllAnimationBone=lambda:[
            AliasBone(1),AliasBone(2)]
        result=audit_animation_bones(motion,rlpy=rlpy)
        anim=result["node_sets"]["all_animation_bones"]
        self.assertTrue(anim["available"])
        self.assertEqual(anim["matching_target_names"],1)
        self.assertEqual(len(result["same_name_identity_differences"][
            "all_animation_bones"]),1)
        self.assertEqual(anim["target_bones_with_any_fk_key"],1)


if __name__=="__main__":
    unittest.main()
