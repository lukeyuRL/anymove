"""Pure-Python tests: no Blender dependency or mutation."""
import unittest
from tools.import_existing_rig import (
    build_rig_info, camera_layout, creature_draft, parent_first,
)


def bone(name, parent, head, tail):
    return {
        "name": name, "parent": parent,
        "matrix_local": [[1, 0, 0, head[0]], [0, 1, 0, head[1]],
                         [0, 0, 1, head[2]], [0, 0, 0, 1]],
        "length": 1.0, "head": head, "tail": tail,
        "limits_deg": [[-20, 20], [-20, 20], [-20, 20]],
    }


class ExistingRigImportTests(unittest.TestCase):
    def setUp(self):
        # Deliberately shuffled parent order; connected ends must be shared.
        self.records = [
            bone("leg_R", "hips", [-.5, 0, .8], [-.5, -.4, 0]),
            bone("head", "spine", [0, -.8, 1.5], [0, -1.2, 1.5]),
            bone("spine", "hips", [0, -.2, 1.0], [0, -.8, 1.5]),
            bone("hips", None, [0, 0, .8], [0, -.2, 1.0]),
            bone("leg_L", "hips", [.5, 0, .8], [.5, -.4, 0]),
        ]
        self.bounds = [[-.8, -1.5, 0], [.8, -1.5, 0],
                       [-.8, 1.1, 1.9], [.8, 1.1, 1.9]]

    def test_order_and_parent_ids(self):
        ordered = parent_first(self.records)
        self.assertEqual([b["name"] for b in ordered],
                         ["hips", "leg_L", "leg_R", "spine", "head"])
        info = build_rig_info(self.records, camera_layout(self.bounds))
        self.assertEqual([b["parent"] for b in info["bones"]], [-1, 0, 0, 0, 3])

    def test_connected_keypoints_are_shared(self):
        info = build_rig_info(self.records, camera_layout(self.bounds))
        self.assertEqual(info["bones"][0]["tail_joint"],
                         info["bones"][3]["head_joint"])
        self.assertEqual(info["bones"][3]["tail_joint"],
                         info["bones"][4]["head_joint"])
        names = [k["name"] for k in info["keypoints"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(b["head_joint"] in names and b["tail_joint"] in names
                            for b in info["bones"]))

    def test_high_res_two_camera_layout(self):
        layout = camera_layout(self.bounds)
        self.assertEqual((layout["W_FRONT"], layout["W_SIDE"], layout["H"]),
                         (640, 640, 720))
        self.assertGreater(layout["M_PER_PX"], 0)
        cfg = creature_draft("dinosaur", self.bounds, layout)
        self.assertEqual(cfg["name"], "dinosaur")
        self.assertEqual(cfg["paint"], {})
        self.assertEqual(cfg["feet"], [])
        self.assertEqual(cfg["layout"]["M_PER_PX"], layout["M_PER_PX"])

    def test_reject_multiple_roots(self):
        with self.assertRaisesRegex(ValueError, "one Skeleton root"):
            parent_first(self.records + [bone("other", None, [0, 0, 0], [0, 0, 1])])

    def test_reject_missing_parent(self):
        changed = [dict(b) for b in self.records]
        changed[0]["parent"] = "missing"
        with self.assertRaisesRegex(ValueError, "missing Bone parent"):
            parent_first(changed)

    def test_reject_nonfinite_joint(self):
        changed = [dict(b) for b in self.records]
        changed[0]["head"] = [float("nan"), 0, 0]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_rig_info(changed, camera_layout(self.bounds))

    def test_reject_zero_width_limit(self):
        changed = [dict(b) for b in self.records]
        changed[0]["limits_deg"] = [[0, 0], [-20, 20], [-20, 20]]
        with self.assertRaisesRegex(ValueError, "positive rotation limit"):
            build_rig_info(changed, camera_layout(self.bounds))


    def test_uniform_Armature_rotation_and_scale(self):
        from tools.import_existing_rig import ensure_similarity
        matrix = [[0, -2, 0, 5],
                  [2, 0, 0, 1],
                  [0, 0, 2, 7],
                  [0, 0, 0, 1]]
        self.assertAlmostEqual(ensure_similarity(matrix, "Armature"), 2.0)

    def test_reject_unsafe_nonuniform_or_reflected_Armature(self):
        from tools.import_existing_rig import ensure_similarity
        matrices = [
            [[2, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 1, 0], [0, 0, 0, 1]],
            [[-1, 0, 0, 0], [0, 1, 0, 0],
             [0, 0, 1, 0], [0, 0, 0, 1]],
            [[1, .1, 0, 0], [0, 1, 0, 0],
             [0, 0, 1, 0], [0, 0, 0, 1]],
        ]
        for matrix in matrices:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                ensure_similarity(matrix, "Armature")


    def test_near_identical_fbX_joint_endpoints_are_shared(self):
        changed = [dict(b) for b in self.records]
        changed[2]["tail"] = [0.0000025, -.800003, 1.500001]
        changed[1]["head"] = [-.000002, -.799999, 1.500000]
        info = build_rig_info(changed, camera_layout(self.bounds))
        self.assertEqual(info["bones"][3]["tail_joint"],
                         info["bones"][4]["head_joint"])
        self.assertEqual(len(info["keypoints"]),
                         len(set(k["name"] for k in info["keypoints"])))
    def test_elongated_dinosaur_uses_more_side_width(self):
        bounds = [
            [-2.0444216728, -7.3383641243, -.0268925745],
            [2.0446081161, 8.6281785965, 5.9318022728],
        ]
        layout = camera_layout(bounds)
        self.assertEqual(layout["W_FRONT"] + layout["W_SIDE"], 1280)
        self.assertEqual((layout["W_FRONT"], layout["W_SIDE"]), (400, 880))
        self.assertLess(layout["M_PER_PX"], .027)
        self.assertGreater(5.95 / layout["M_PER_PX"], 220)
        draft = creature_draft("dino", bounds, layout)
        self.assertEqual(draft["dataset"]["yaw_max_deg"], 10.0)
        # Check XY projections stay inside the permitted split at +/-10deg.
        import math
        corners = [(x, y) for x in (-2.0444216728, 2.0446081161)
                   for y in (-7.3383641243, 8.6281785965)]
        for angle in (-10, 0, 10):
            co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            for x, y in corners:
                xp, yp = x*co - y*si, x*si + y*co
                self.assertLess(abs(xp) + .08,
                                layout["M_PER_PX"] * layout["W_FRONT"] / 2)
                self.assertLess(abs(yp) + .25,
                                layout["M_PER_PX"] * layout["W_SIDE"] / 2)

    def test_wide_and_balanced_creatures_keep_legacy_layout(self):
        bounds = [[-3, -1, 0], [3, 1, 2]]
        layout = camera_layout(bounds)
        self.assertEqual((layout["W_FRONT"], layout["W_SIDE"]), (640, 640))
        self.assertNotIn("dataset", creature_draft("dragon", bounds, layout))


if __name__ == "__main__":
    unittest.main()
