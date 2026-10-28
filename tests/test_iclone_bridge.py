import copy
import math
import unittest

from iclone_adapter.anymove_bridge import (
    convert, from_euler_xyz, make_mapping_template, to_euler_xyz,
)
from iclone_adapter.schema import ContractError, IDENTITY, VERSION


def skel():
    return {
        "schema_version": VERSION,
        "reference": {"kind": "current_snapshot"},
        "bones": [
            {"id": "1", "path": "/root", "parent_id": None,
             "snapshot_local_matrix": list(IDENTITY),
             "snapshot_world_matrix": list(IDENTITY)},
            {"id": "2", "path": "/root/arm", "parent_id": "1",
             "snapshot_local_matrix": list(IDENTITY),
             "snapshot_world_matrix": list(IDENTITY)},
        ],
    }


def source():
    return {"bones": ["pelvis", "forearm_L"], "fps": 24,
            "euler": [[[0, 0, 0], [0, 0, 0]],
                      [[0, 0, 0], [math.pi/6, 0, 0]],
                      [[0, 0, 0], [0, 0, 0]]],
            "trans": [[0, 0, 0]] * 3}


class BridgeTests(unittest.TestCase):
    def test_euler_quaternion_roundtrip(self):
        for vec in ([0, 0, 0], [.4, -.2, .3], [-.7, .5, -1.]):
            out = to_euler_xyz(from_euler_xyz(vec))
            for a, b in zip(vec, out):
                self.assertAlmostEqual(a, b, places=7)

    def test_refuses_implicit_mapping(self):
        m = make_mapping_template(source(), skel())
        with self.assertRaises(ContractError):
            convert(source(), skel(), m)
        m["bone_map"]["forearm_L"]["target_id"] = "2"
        with self.assertRaises(ContractError):
            convert(source(), skel(), m)
        m["bone_map"]["forearm_L"]["calibrated"] = True
        out = convert(source(), skel(), m)
        self.assertEqual(out["frame_count"], 3)
        self.assertAlmostEqual(out["tracks"]["2"][1][0], math.pi/6)
        self.assertIn("root_translation_skipped", out["limitations"])

    def test_requires_target_hash(self):
        s = skel()
        m = make_mapping_template(source(), s)
        m["bone_map"]["forearm_L"].update(target_id="2", calibrated=True)
        bad = copy.deepcopy(s)
        bad["bones"][1]["id"] = "3"
        with self.assertRaises(ContractError):
            convert(source(), bad, m)


if __name__ == "__main__":
    unittest.main()
