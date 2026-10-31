"""CPU tests for generic same-rig Dragon/Dino/new-creature FK conversion."""
import copy
import json
import math
import pathlib
import tempfile
import unittest

from iclone_adapter.generic_creature import (
    make_creature_preview, prepare_package, verify_generic_fk,
    FK_FILENAME, MANIFEST_FILENAME,
)
from iclone_adapter.import_pipeline import verify_fk_manifest_for_writer
from iclone_adapter.motion_contract import validate_motion
from iclone_adapter.schema import ContractError, IDENTITY, VERSION


def matrix_at(x=0., y=0., z=0.):
    rows = [list(IDENTITY[i * 4:(i + 1) * 4]) for i in range(4)]
    for i, t in enumerate((x, y, z)):
        rows[i][3] = t
    return rows


def same_rig(name="DinoRoot", scale=125.):
    """Synthetic rig, not a claim about the actual user's 92-joint Dino."""
    rest = [
        {"name": name, "parent": -1, "matrix_local": matrix_at()},
        {"name": "Back", "parent": 0, "matrix_local": matrix_at(y=0.8)},
        {"name": "Tail", "parent": 1, "matrix_local": matrix_at(y=1.6)},
    ]
    rig = {"bones": rest,
           "keypoints": [{"name": "joint_%02d" % i} for i in range(5)]}
    target = {"schema_version": VERSION,
              "reference": {"kind": "current_snapshot"},
              "bones": []}
    helper_id = "skin:/CreatureArmature"
    target["bones"].append({
        "id": helper_id, "name": "CreatureArmature",
        "path": "/CreatureArmature", "parent_id": None,
        "snapshot_local_matrix": list(IDENTITY),
        "snapshot_world_matrix": list(IDENTITY),
    })
    for i, bone in enumerate(rest):
        path = "/CreatureArmature/" + "/".join(b["name"] for b in rest[:i + 1])
        bid = "skin:" + path
        parent_id = helper_id if i == 0 else target["bones"][-1]["id"]
        world = [float(v) for row in bone["matrix_local"] for v in row]
        for row in range(3):
            for column in range(4):
                world[row * 4 + column] *= scale
        target["bones"].append({
            "id": bid, "name": bone["name"], "path": path,
            "parent_id": parent_id, "snapshot_local_matrix": list(IDENTITY),
            "snapshot_world_matrix": world,
        })
    return rig, target


def sample_motion(rig, fps=30.):
    names = [b["name"] for b in rig["bones"]]
    frames = []
    for i in range(4):
        frames.append([
            [0., 0., 0.],
            [0.1 * i, 0., 0.],
            [0., 0., 0.03 * i],
        ])
    return {
        "bones": names,
        "fps": fps,
        "euler": frames,
        "trans": [[0., 0., 0.1 * i] for i in range(4)],
        "root_quat": [
            [math.cos(0.02 * i), math.sin(0.02 * i), 0., 0.]
            for i in range(4)
        ],
    }


class GenericCreatureTests(unittest.TestCase):
    def setUp(self):
        self.source, self.target = same_rig()
        self.motion = sample_motion(self.source)

    def test_dino_nonpelvis_root_and_dynamic_bone_count(self):
        fk, report = make_creature_preview(
            self.motion, self.source, self.target)
        self.assertEqual(report["result"], "compatible_static_rest_snapshot")
        self.assertAlmostEqual(report["source_to_target_uniform_scale"], 125.)
        self.assertEqual(fk["generic_source_bone_count"], 3)
        self.assertEqual(fk["generic_source_joint_count"], 5)
        self.assertEqual(len(fk["tracks"]), 2)
        self.assertNotIn("root_motion", fk)
        self.assertEqual(fk["fps"], 30)
        self.assertTrue(validate_motion(fk, self.target))

    def test_dino_full_root_and_position_tracks(self):
        fk, _ = make_creature_preview(
            self.motion, self.source, self.target,
            root_mode="full", root_position_scale=1.5,
            root_position_space="root_local",
        )
        root = fk["root_motion"]
        self.assertEqual(root["source_root"], "DinoRoot")
        self.assertEqual(root["mode"], "root_layer")
        self.assertEqual(root["root_position_space"], "root_local")
        self.assertAlmostEqual(
            fk["position_tracks"][root["target_bone_id"]][3][2], 0.45)
        self.assertEqual(len(fk["tracks"]), 3)
        self.assertTrue(validate_motion(fk, self.target))

    def test_rotation_only_works_with_unusual_root(self):
        fk, _ = make_creature_preview(
            self.motion, self.source, self.target, root_mode="rotation")
        self.assertEqual(fk["root_motion"]["components"], "rotation_only")
        self.assertEqual(len(fk["tracks"]), 3)
        self.assertNotIn("position_tracks", fk)

    def test_root_position_only_action_is_permitted(self):
        for frame in self.motion["euler"]:
            for xyz in frame:
                xyz[:] = [0., 0., 0.]
        self.motion["root_quat"] = [[1., 0., 0., 0.]] * 4
        fk, _ = make_creature_preview(self.motion, self.source, self.target,
                                      root_mode="full", root_position_scale=1.)
        self.assertIn("DinoRoot:position", fk["preview_bones_with_rotation_change"])
        self.assertTrue(validate_motion(fk, self.target))

    def test_wrong_bone_hierarchy_or_pose_fails(self):
        wrong = copy.deepcopy(self.target)
        wrong["bones"][-1]["parent_id"] = wrong["bones"][0]["id"]
        with self.assertRaisesRegex(ContractError, "preflight FAILED"):
            make_creature_preview(self.motion, self.source, wrong)
        wrong = copy.deepcopy(self.target)
        wrong["bones"][-1]["snapshot_world_matrix"][7] += 50.
        with self.assertRaisesRegex(ContractError, "preflight FAILED"):
            make_creature_preview(self.motion, self.source, wrong)

    def test_source_bone_order_root_and_duplicate_are_rejected(self):
        wrong = copy.deepcopy(self.motion)
        wrong["bones"] = list(reversed(wrong["bones"]))
        with self.assertRaisesRegex(ContractError, "Bone names/order"):
            make_creature_preview(wrong, self.source, self.target)
        bad_rig = copy.deepcopy(self.source)
        bad_rig["bones"][0]["parent"] = 1
        with self.assertRaisesRegex(ContractError, "one first root"):
            make_creature_preview(self.motion, bad_rig, self.target)
        bad_rig = copy.deepcopy(self.source)
        bad_rig["bones"][2]["name"] = "Back"
        with self.assertRaisesRegex(ContractError, "unique"):
            make_creature_preview(self.motion, bad_rig, self.target)

    def test_full_root_requires_explicit_scale(self):
        with self.assertRaisesRegex(ContractError, "root Translation requires explicit"):
            make_creature_preview(self.motion, self.source, self.target, root_mode="full")
        with self.assertRaisesRegex(ContractError, "only valid with full"):
            make_creature_preview(self.motion, self.source, self.target,
                                  root_mode="none", root_position_scale=1.)
        with self.assertRaisesRegex(ContractError, "root space"):
            make_creature_preview(self.motion, self.source, self.target,
                                  root_mode="full", root_position_scale=1.,
                                  root_position_space="pelvis_local")

    def test_hash_bundle_and_write_time_tamper_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            out = pathlib.Path(folder) / "creature"
            fkpath = prepare_package(out, self.motion, self.source, self.target,
                                     root_mode="full", root_position_scale=1.)
            self.assertTrue((out / MANIFEST_FILENAME).is_file())
            self.assertEqual(fkpath.name, FK_FILENAME)
            self.assertEqual(verify_generic_fk(fkpath)["status"],
                             "generic_fk_payload_matches_manifest")
            self.assertEqual(verify_fk_manifest_for_writer(fkpath)["status"],
                             "generic_fk_payload_matches_manifest")
            with self.assertRaisesRegex(ContractError, "already exists"):
                prepare_package(out, self.motion, self.source, self.target)
            data = json.loads(fkpath.read_text())
            root_id = data["root_motion"]["target_bone_id"]
            data["position_tracks"][root_id][1][2] += 10.
            fkpath.write_text(json.dumps(data))
            with self.assertRaisesRegex(ContractError, "verified manifest"):
                verify_fk_manifest_for_writer(fkpath)

    def test_original_dragon_still_converts_without_fortytwo_hardcode(self):
        from test_iclone_dragon_preview import DragonPreviewTests
        DragonPreviewTests.setUpClass()
        fixtures = DragonPreviewTests()
        target = fixtures.make_target()
        source = fixtures.make_motion()
        fk, report = make_creature_preview(
            source, DragonPreviewTests.source, target,
            root_mode="full", root_position_scale=1.,
        )
        self.assertEqual(report["matched_unique_names"], 42)
        self.assertEqual(len(fk["tracks"]), 42)
        self.assertEqual(fk["root_motion"]["mode"], "pelvis_layer")
        self.assertTrue(validate_motion(fk, target))


if __name__ == "__main__":
    unittest.main()
