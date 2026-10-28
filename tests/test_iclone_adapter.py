"""Offline unit tests (no iClone installation needed)."""
import copy
import math
import unittest

from iclone_adapter.schema import (
    VERSION, IDENTITY, ContractError, inverse, mul, relative,
    skeleton_hash, validate_skeleton, world_matrices,
)
from iclone_adapter.motion_contract import make_sine_test, validate_motion
from iclone_adapter.rlpy_reader import snapshot, probe_matrix_layout
from iclone_adapter.motion_writer import write_selected
from iclone_adapter.p1_inspector import inspect_selected_animation, read_float_value
from iclone_adapter.fk_pose_validator import validate_live_fk_pose
from iclone_adapter.bone_registry import build_bone_registry, _lineage
from iclone_adapter.diagnostics import raw_skeleton_report, save_failure_report


def sample_skeleton():
    a = list(IDENTITY)
    b = list(IDENTITY)
    b[3] = 2.0
    return {
        "schema_version": VERSION,
        "reference": {"kind": "current_snapshot"},
        "bones": [
            {"id": "root", "path": "/root", "parent_id": None,
             "snapshot_local_matrix": a, "snapshot_world_matrix": a,
             "rest_local_matrix": None},
            {"id": "arm", "path": "/root/arm", "parent_id": "root",
             "snapshot_local_matrix": b, "snapshot_world_matrix": b,
             "rest_local_matrix": None},
        ],
    }


class MathTests(unittest.TestCase):
    def test_inverse_and_fk(self):
        data = sample_skeleton()
        b = data["bones"][1]["snapshot_local_matrix"]
        self.assertAlmostEqual(mul(b, inverse(b))[3], 0.0)
        f = world_matrices([
            {"id": "arm", "parent_id": "root", "rest_local_matrix": b},
            {"id": "root", "parent_id": None, "rest_local_matrix": IDENTITY},
        ])
        self.assertAlmostEqual(f["arm"][3], 2.0)
        self.assertEqual(relative(IDENTITY, b), b)

    def test_cycle_orphan_and_duplicate(self):
        bad = sample_skeleton()
        bad["bones"][0]["parent_id"] = "arm"
        with self.assertRaises(ContractError):
            validate_skeleton(bad)
        bad = sample_skeleton()
        bad["bones"][1]["parent_id"] = "missing"
        with self.assertRaises(ContractError):
            validate_skeleton(bad)
        bad = sample_skeleton()
        bad["bones"][1]["id"] = "root"
        with self.assertRaises(ContractError):
            validate_skeleton(bad)

    def test_references_and_motion(self):
        data = sample_skeleton()
        self.assertTrue(validate_skeleton(data))
        m = make_sine_test(data, "arm")
        self.assertTrue(validate_motion(m, data))
        d = copy.deepcopy(data)
        d["bones"][1]["snapshot_world_matrix"][3] = 33
        self.assertEqual(skeleton_hash(d), skeleton_hash(data))
        d["bones"][1]["id"] = "changed"
        with self.assertRaises(ContractError):
            validate_motion(m, d)
        m["space"] = "blender_euler_xyz"
        with self.assertRaises(ContractError):
            validate_motion(m, data)


class Matrix:
    def __init__(self, raw=None):
        self.raw = list(raw) if raw is not None else list(IDENTITY)

    def M(self, row, col):
        # Emulates the Legacy SWIG float& overload that failed in iClone 8.
        return object()

    def GetRow(self, row):
        # RVector4.__getitem__ returns a Python float.
        return tuple(self.raw[4*row:4*row+4])

    def MakeIdentity(self):
        self.raw = list(IDENTITY)

    def SetTranslate(self, vec):
        self.raw[12:15] = list(vec)


class ColumnMatrix(Matrix):
    def SetTranslate(self, vec):
        self.raw[3], self.raw[7], self.raw[11] = vec


class Transform:
    def __init__(self, raw):
        self.raw = raw

    def Matrix(self):
        return Matrix(self.raw)


class Bone:
    def __init__(self, key, parent=None):
        self.key = key
        self.parent = parent

    def GetID(self):
        return self.key

    def GetName(self):
        return "Bone" + str(self.key)

    def GetParent(self):
        return self.parent

    def LocalTransform(self):
        return Transform([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, self.key, 0, 0, 1])

    def WorldTransform(self):
        return self.LocalTransform()

    def Update(self):
        return None


class Ctrl:
    def __init__(self):
        self.values = []

    def SetValue(self, time, val):
        self.values.append((time, val))
        return True

    def GetKeyCount(self):
        return len(self.values)

    def GetValue(self, time, output):
        matches = [value for stamp, value in self.values if stamp == time]
        return (True, matches[-1] if matches else 0.0)

    def RemoveKey(self, time):
        self.values = [(t, v) for t, v in self.values if t != time]
        return True


class DataBlock:
    def __init__(self):
        self.ctrls = {name: Ctrl() for name in (
            "Rotation/RotationX", "Rotation/RotationY", "Rotation/RotationZ")}

    def GetControl(self, key):
        return self.ctrls.get(key)


class Layer:
    def __init__(self):
        self.block = DataBlock()

    def GetDataBlock(self):
        return self.block


class Clip:
    def __init__(self, start=0):
        self.layers = {}
        self.length = None
        self.start = start

    def SetLength(self, length):
        self.length = length
        return True

    def ClipTimeToSceneTime(self, local_time):
        return self.start + local_time

    def GetClipLength(self):
        return self.length or 0

    def GetLength(self):
        return self.length or 0

    def GetClipLength(self):
        return self.length or 0

    def GetLastKeyTime(self):
        return max(0, (self.length or 0) - 1)

    def GetStartOffset(self):
        return 0

    def GetControl(self, name, bone):
        if name != "Layer":
            return None
        return self.layers.setdefault(bone.GetName(), Layer())


class Skeleton:
    def __init__(self):
        a = Bone(1)
        self.bones = [a, Bone(2, a)]
        self.clips = []
        self.rollback = False

    def GetSkinBones(self):
        return self.bones

    def IsBoneAnimated(self, bone):
        return True

    def GetClipCount(self):
        return len(self.clips)

    def GetClip(self, index):
        return self.clips[index]

    def AddClip(self, time):
        clip = Clip(time)
        self.clips.append(clip)
        return clip

    def DeleteClip(self, clip):
        self.clips.remove(clip)
        self.rollback = True
        return True


class Avatar:
    def __init__(self):
        self.skeleton = Skeleton()

    def GetName(self):
        return "Demo"

    def GetType(self):
        return "Avatar"

    def GetSkeletonComponent(self):
        return self.skeleton

    def Update(self):
        return None


class Scene:
    def __init__(self, avatar):
        self.avatar = avatar

    def GetSelectedObjects(self):
        return [self.avatar]


class Fps:
    def ToDouble(self):
        return 24.0


class Global:
    def __init__(self):
        self.fps = Fps()

    def GetTime(self):
        return getattr(self, "time", 0)

    def SetTime(self, time):
        self.time = time
        return True

    def GetFps(self):
        return self.fps

    def GetEndTime(self):
        return getattr(self, "end_time", 48)

    def SetEndTime(self, end):
        self.end_time = end
        return True


class RLPyMock:
    class RStatus:
        Failure = False

    RMatrix4 = Matrix
    RVector3 = staticmethod(lambda x, y, z: (x, y, z))

    def __init__(self, avatar):
        self.RScene = Scene(avatar)
        self.RGlobal = Global()

    @staticmethod
    def GetFrameIndex(time, fps):
        return time

    @staticmethod
    def IndexedFrameTime(frame, fps):
        return frame


class RLPyTests(unittest.TestCase):
    def test_matrix_layout_probe_both_conventions(self):
        avatar = Avatar()
        rlpy = RLPyMock(avatar)
        self.assertEqual(probe_matrix_layout(rlpy), "transpose")
        rlpy.RMatrix4 = ColumnMatrix
        self.assertEqual(probe_matrix_layout(rlpy), "direct")

    def test_lineage_repeated_swig_wrapper_ids_not_reused(self):
        class DisposableWrapper:
            def __init__(self, depth):
                self.depth = depth
            def GetName(self):
                return "node_%s" % self.depth
            def GetParent(self):
                return DisposableWrapper(self.depth - 1) if self.depth else None
        self.assertEqual(len(_lineage(DisposableWrapper(15))), 16)

    def test_raw_diagnostic_can_run_with_duplicate_native_ids(self):
        avatar = Avatar()
        rlpy = RLPyMock(avatar)
        avatar.skeleton.bones[0].GetID = lambda: 0
        avatar.skeleton.bones[1].GetID = lambda: 0
        report = raw_skeleton_report(rlpy, failure="fixture")
        self.assertEqual(report["skin_bone_count"], 2)
        self.assertEqual(report["bones"][0]["native_scene_id"], "0")
        self.assertEqual(report["bones"][1]["parent_name"], "Bone1")
        self.assertEqual(report["failure"], "fixture")

    def test_repeated_root_node_is_coalesced_from_getskinbones(self):
        # The real 103-row CC test contains RL_BoneRoot twice: indices 0 and 4,
        # same GetID/name/parent. It represents 102 distinct skin records.
        avatar = Avatar()
        root, arm = avatar.skeleton.bones
        alias = Bone(root.key)
        avatar.skeleton.bones = [root, arm, alias]
        r = RLPyMock(avatar)
        data = snapshot(avatar, rlpy=r)
        self.assertEqual(len(data["bones"]), 2)
        stats = data["identity_diagnostics"]
        self.assertEqual(stats["raw_skin_bone_count"], 3)
        self.assertEqual(stats["canonical_bone_count"], 2)
        self.assertEqual(stats["coalesced_duplicate_count"], 1)
        self.assertEqual(stats["coalesced_duplicates"][0]["duplicate_raw_index"], 2)
        parent = next(b for b in data["bones"] if b["name"] == "Bone1")
        child = next(b for b in data["bones"] if b["name"] == "Bone2")
        self.assertEqual(child["parent_id"], parent["id"])
        # P1 must use the same canonical hierarchy and be able to resolve it.
        m = make_sine_test(data, child["id"], frames=4)
        out = write_selected(m, rlpy=r, fps_confirmed=True)
        self.assertEqual(out["frame_count"], 4)

    def test_same_id_name_parent_but_different_pose_keeps_distinct(self):
        # A shared ID alone must not hide real siblings with different poses.
        a, b = Bone(9), Bone(9)
        b.LocalTransform = lambda: Transform([
            1,0,0,0, 0,1,0,0, 0,0,1,0, 42,0,0,1
        ])
        nodes, stats = build_bone_registry([a, b])
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len({node["path"] for node in nodes}), 2)
        self.assertEqual(stats["coalesced_duplicate_count"], 0)

    def test_duplicate_scene_ids_preserve_hierarchy(self):
        avatar = Avatar()
        r = RLPyMock(avatar)
        for node in avatar.skeleton.bones:
            node.GetID = lambda: 0  # Simulate real iClone duplicate RINode IDs.
        data = snapshot(avatar, rlpy=r)
        self.assertEqual(len(data["bones"]), 2)
        self.assertEqual(len({b["id"] for b in data["bones"]}), 2)
        self.assertEqual(data["identity_diagnostics"]["native_id_duplicate_nodes"], 2)
        child = next(b for b in data["bones"] if b["name"] == "Bone2")
        parent = next(b for b in data["bones"] if b["name"] == "Bone1")
        self.assertEqual(child["parent_id"], parent["id"])
        motion = make_sine_test(data, child["id"], frames=4)
        result = write_selected(motion, rlpy=r, fps_confirmed=True)
        self.assertEqual(result["frame_count"], 4)
        self.assertIn("Bone2", avatar.skeleton.clips[0].layers)

    def test_same_name_different_parent_paths(self):
        class NamedBone(Bone):
            def __init__(self, key, name, parent=None):
                super(NamedBone, self).__init__(key, parent)
                self.name = name
            def GetName(self):
                return self.name
            def GetID(self):
                return 0
        root = NamedBone(1, "root")
        left = NamedBone(2, "left", root)
        right = NamedBone(3, "right", root)
        a = NamedBone(4, "joint", left)
        b = NamedBone(5, "joint", right)
        entries, diag = build_bone_registry([root, left, right, a, b])
        self.assertEqual(len({v["path"] for v in entries}), 5)
        self.assertNotEqual(entries[3]["parent_id"], entries[4]["parent_id"])
        self.assertEqual(diag["native_id_duplicate_nodes"], 5)

    def test_snapshot_and_writer(self):
        avatar = Avatar()
        r = RLPyMock(avatar)
        data = snapshot(avatar, rlpy=r)
        self.assertEqual(len(data["bones"]), 2)
        self.assertEqual(data["reference"]["kind"], "current_snapshot")
        # Reallusion row-vector translation is transposed into column vector.
        self.assertEqual(data["bones"][0]["snapshot_world_matrix"][3], 1.0)
        bone = next(b for b in data["bones"] if b["name"] == "Bone2")
        m = make_sine_test(data, bone["id"], frames=6)
        with self.assertRaises(ContractError):
            write_selected(m, rlpy=r)
        result = write_selected(m, rlpy=r, fps_confirmed=True)
        self.assertEqual(result["frame_count"], 6)
        self.assertEqual(result["status"], "fk_keys_verified_pose_unchanged")
        self.assertEqual(result["target_bone_name"], "Bone2")
        self.assertTrue(result["fk_readback"][bone["id"]]["xyz_key_counts"][0] >= 2)
        self.assertEqual(r.RGlobal.GetTime(), result["peak_frame"])
        self.assertEqual(avatar.skeleton.clips[0].length, 6)
        self.assertEqual(len(avatar.skeleton.clips), 1)
        self.assertEqual(len(avatar.skeleton.clips[0].layers["Bone2"].block.ctrls[
            "Rotation/RotationX"].values), 6)
        with self.assertRaises(ContractError):
            write_selected(m, rlpy=r, fps_confirmed=True)

    def test_one_default_clip_is_reused_not_rejected(self):
        avatar = Avatar()
        rlpy = RLPyMock(avatar)
        avatar.skeleton.AddClip(0)  # A normal initial clip, no FK layer keys.
        sk = snapshot(avatar, rlpy=rlpy)
        bone = next(b["id"] for b in sk["bones"] if b["name"] == "Bone2")
        motion = make_sine_test(sk, bone, frames=8)
        result = write_selected(motion, fps_confirmed=True, rlpy=rlpy)
        self.assertEqual(result["clip_strategy"], "reuse_default")
        self.assertEqual(result["clip_count_before"], 1)
        self.assertEqual(result["clip_count_after"], 1)
        self.assertEqual(avatar.skeleton.GetClipCount(), 1)
        self.assertEqual(avatar.skeleton.GetClip(0).start, 0)

    def test_prekeyed_default_clip_is_preserved_and_motion_appended(self):
        avatar = Avatar()
        rlpy = RLPyMock(avatar)
        old = avatar.skeleton.AddClip(0)
        old.SetLength(24)
        old_ctrl = old.GetControl("Layer", avatar.skeleton.bones[1]).GetDataBlock().GetControl("Rotation/RotationX")
        old_ctrl.SetValue(0, 0.23)
        sk = snapshot(avatar, rlpy=rlpy)
        bone = next(b["id"] for b in sk["bones"] if b["name"] == "Bone2")
        motion = make_sine_test(sk, bone, frames=8)
        result = write_selected(motion, fps_confirmed=True, rlpy=rlpy)
        self.assertEqual(result["clip_strategy"], "append_new")
        self.assertEqual(result["clip_count_before"], 1)
        self.assertEqual(result["clip_count_after"], 2)
        self.assertEqual(result["clip_start_ticks"], 25)
        self.assertEqual(old_ctrl.GetKeyCount(), 1)
        self.assertEqual(old_ctrl.GetValue(0, 0.)[1], 0.23)
        self.assertEqual(avatar.skeleton.GetClip(1).start, 25)
        self.assertEqual(rlpy.RGlobal.GetTime(), 25 + result["peak_frame"])

    def test_reused_clip_error_rolls_back_only_new_keys(self):
        avatar = Avatar()
        rlpy = RLPyMock(avatar)
        clip = avatar.skeleton.AddClip(0)
        sk = snapshot(avatar, rlpy=rlpy)
        bone = next(b["id"] for b in sk["bones"] if b["name"] == "Bone2")
        motion = make_sine_test(sk, bone, frames=6)
        ctrl = clip.GetControl("Layer", avatar.skeleton.bones[1]).GetDataBlock().GetControl("Rotation/RotationX")
        original_set = ctrl.SetValue
        def fails_after_two(time, value):
            if time == 2:
                return False
            return original_set(time, value)
        ctrl.SetValue = fails_after_two
        with self.assertRaises(ContractError):
            write_selected(motion, fps_confirmed=True, rlpy=rlpy)
        self.assertEqual(avatar.skeleton.GetClipCount(), 1)
        self.assertEqual(ctrl.GetKeyCount(), 0)
        self.assertEqual(clip.GetLength(), 0)

    def test_preview_validator_detects_static_bone_evaluation(self):
        avatar=Avatar()
        rlpy=RLPyMock(avatar)
        rest=snapshot(avatar,rlpy=rlpy)
        bone=next(b["id"] for b in rest["bones"] if b["name"]=="Bone2")
        motion=make_sine_test(rest,bone,frames=12,amplitude_deg=30.)
        motion["calibration_status"]="unverified_identity_same_rig_preview"
        written=write_selected(motion,rlpy=rlpy,fps_confirmed=True)
        time_before=rlpy.RGlobal.GetTime()
        keys_before=avatar.skeleton.GetClip(0).layers["Bone2"].block.ctrls[
            "Rotation/RotationX"].GetKeyCount()
        report=validate_live_fk_pose(rest,motion,rlpy=rlpy)
        self.assertEqual(report["result"],"bone_transform_static_axis_unverifiable")
        self.assertIn("Bone2",report["static_node_bones"])
        self.assertEqual(report["static_node_bone_count"],1)
        self.assertEqual(report["axis_testable_bone_count"],0)
        self.assertIsNone(report["consistent_bone_count"])
        self.assertEqual(report["over_10deg_bones"],[])
        self.assertEqual(rlpy.RGlobal.GetTime(),time_before)
        self.assertEqual(avatar.skeleton.GetClipCount(),1)
        self.assertEqual(avatar.skeleton.GetClip(0).layers["Bone2"].block.ctrls[
            "Rotation/RotationX"].GetKeyCount(),keys_before)

    def test_preview_validator_accepts_matching_local_and_world_poses(self):
        avatar=Avatar()
        rlpy=RLPyMock(avatar)
        rest=snapshot(avatar,rlpy=rlpy)
        child=avatar.skeleton.bones[1]
        bone=next(b["id"] for b in rest["bones"] if b["name"]=="Bone2")
        motion=make_sine_test(rest,bone,frames=12,amplitude_deg=30.)
        motion["calibration_status"]="unverified_identity_same_rig_preview"
        written=write_selected(motion,rlpy=rlpy,fps_confirmed=True)
        import math
        def animated_transform():
            index=min(int(rlpy.RGlobal.GetTime()),motion["frame_count"]-1)
            x=motion["tracks"][bone][index][0]
            c,si=math.cos(x),math.sin(x)
            # RLPy mock storage uses row vectors; canonical rotation transposes it.
            raw=[1,0,0,0, 0,c,si,0, 0,-si,c,0,
                 child.key,0,0,1]
            return Transform(raw)
        child.LocalTransform=animated_transform
        child.WorldTransform=animated_transform
        report=validate_live_fk_pose(rest,motion,rlpy=rlpy)
        self.assertEqual(report["result"],"consistent_in_sampled_fk_axes")
        self.assertEqual(report["consistent_bone_count"],1)
        self.assertGreater(report["by_bone"][0]["observed_temporal_rotation_deg"],20.)
        self.assertLess(report["by_bone"][0]["max_post_rest_error_deg"],0.01)
        self.assertGreater(report["by_bone"][0]["max_requested_angle_deg"],20.)

    def test_existing_clip_inspector_reports_key_counts(self):
        avatar = Avatar()
        rlpy = RLPyMock(avatar)
        sk = snapshot(avatar, rlpy=rlpy)
        b = next(entry["id"] for entry in sk["bones"]
                 if entry["name"] == "Bone2")
        motion = make_sine_test(sk, b, frames=6)
        result = write_selected(motion, rlpy=rlpy, fps_confirmed=True)
        report = inspect_selected_animation(rlpy)
        self.assertEqual(report["clip_count"], 1)
        self.assertTrue(report["clips"][0]["fk_keys_present"])
        self.assertEqual(report["clips"][0]["total_fk_keys"], 18)
        self.assertEqual(report["clips"][0]["keyed_bones"], 1)

    def test_nonconstant_track_without_keys_rolls_back(self):
        avatar = Avatar()
        rlpy = RLPyMock(avatar)
        sk = snapshot(avatar, rlpy=rlpy)
        b = next(entry["id"] for entry in sk["bones"]
                 if entry["name"] == "Bone2")
        motion = make_sine_test(sk, b, frames=6)
        orig_add = avatar.skeleton.AddClip
        def add_missing_keys(t):
            clip = orig_add(t)
            class NotStored(Ctrl):
                def GetKeyCount(self):
                    return 0
            bone_ctrl = clip.GetControl("Layer", avatar.skeleton.bones[1])
            bone_ctrl.block.ctrls["Rotation/RotationX"] = NotStored()
            return clip
        avatar.skeleton.AddClip = add_missing_keys
        with self.assertRaises(ContractError):
            write_selected(motion, rlpy=rlpy, fps_confirmed=True)
        self.assertEqual(avatar.skeleton.GetClipCount(), 0)
        self.assertTrue(avatar.skeleton.rollback)

    def test_failed_clip_preflight_rolls_back(self):
        avatar = Avatar()
        r = RLPyMock(avatar)
        m = make_sine_test(snapshot(avatar, rlpy=r), next(b["id"] for b in snapshot(avatar, rlpy=r)["bones"] if b["name"] == "Bone2"), frames=3)
        class BrokenClip:
            def SetLength(self, length):
                return True

            def GetControl(self, *args):
                return None
        def bad_add(time):
            clip = BrokenClip()
            avatar.skeleton.clips.append(clip)
            return clip
        avatar.skeleton.AddClip = bad_add
        with self.assertRaises(ContractError):
            write_selected(m, rlpy=r, fps_confirmed=True)
        self.assertTrue(avatar.skeleton.rollback)
        self.assertEqual(len(avatar.skeleton.clips), 0)

    def test_fps_mismatch_rejected_before_clip(self):
        avatar = Avatar()
        r = RLPyMock(avatar)
        m = make_sine_test(snapshot(avatar, rlpy=r), next(b["id"] for b in snapshot(avatar, rlpy=r)["bones"] if b["name"] == "Bone2"))
        r.RGlobal.fps.ToDouble = lambda: 30.0
        with self.assertRaises(ContractError):
            write_selected(m, rlpy=r, fps_confirmed=True)
        self.assertEqual(avatar.skeleton.GetClipCount(), 0)

    def test_sdk_status_error_triggers_rollback(self):
        avatar = Avatar()
        r = RLPyMock(avatar)
        m = make_sine_test(snapshot(avatar, rlpy=r), next(b["id"] for b in snapshot(avatar, rlpy=r)["bones"] if b["name"] == "Bone2"))
        class Status:
            def IsError(self):
                return True
        original_add = avatar.skeleton.AddClip
        def bad_add(t):
            clip = original_add(t)
            clip.SetLength = lambda value: Status()
            return clip
        avatar.skeleton.AddClip = bad_add
        with self.assertRaises(ContractError):
            write_selected(m, rlpy=r, fps_confirmed=True)
        self.assertTrue(avatar.skeleton.rollback)

    def test_writer_rejects_mismatched_skeleton(self):
        avatar = Avatar()
        r = RLPyMock(avatar)
        m = make_sine_test(snapshot(avatar, rlpy=r), next(b["id"] for b in snapshot(avatar, rlpy=r)["bones"] if b["name"] == "Bone2"))
        m["skeleton_hash"] = "invalid"
        with self.assertRaises(ContractError):
            write_selected(m, rlpy=r, fps_confirmed=True)
        self.assertEqual(avatar.skeleton.GetClipCount(), 0)


if __name__ == "__main__":
    unittest.main()
