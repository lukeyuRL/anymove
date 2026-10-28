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
from iclone_adapter.bone_registry import build_bone_registry


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


class Ctrl:
    def __init__(self):
        self.values = []

    def SetValue(self, time, val):
        self.values.append((time, val))
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
    def __init__(self):
        self.layers = {}
        self.length = None

    def SetLength(self, length):
        self.length = length
        return True

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

    def AddClip(self, time):
        clip = Clip()
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
        return 0

    def GetFps(self):
        return self.fps


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
        self.assertEqual(avatar.skeleton.clips[0].length, 6)
        self.assertEqual(len(avatar.skeleton.clips), 1)
        self.assertEqual(len(avatar.skeleton.clips[0].layers["Bone2"].block.ctrls[
            "Rotation/RotationX"].values), 6)
        with self.assertRaises(ContractError):
            write_selected(m, rlpy=r, fps_confirmed=True)

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
