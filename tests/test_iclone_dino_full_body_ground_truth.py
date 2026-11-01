"""CPU/NumPy regression for Dino full-body Blender-truth evaluation.

Synthetic positions here test metric correctness only; production truth
is produced independently by Blender pose-bone evaluation.
"""
import json
import pathlib
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from tools.evaluate_dino_full_body import evaluate, classify_joint, GROUPS


@unittest.skipUnless(np is not None, "NumPy optional for lightweight CI")
class DinoFullBodyGroundTruthTests(unittest.TestCase):
    def make_example(self, T=7):
        names = [
            "joint:CC_Dinosaur_Pelvis:head",
            "joint:CC_Dinosaur_Spine02:tail",
            "joint:CC_Dinosaur_Neck:tail",
            "joint:CC_Dinosaur_L_UpperArm:tail",
            "joint:CC_Dinosaur_R_Thigh:tail",
            "joint:CC_Dinosaur_R_LeftToe03:tail",
            "joint:CC_Dinosaur_Tail07:tail",
            "joint:CC_Dinosaur_MysteryHelper:tail",
        ]
        rig = {
            "bones": [{"name": "pelvis", "head_joint": names[0]},
                      {"name": "tail", "head_joint": names[6]}],
            "keypoints": [{"name": n, "bone": 0, "end": "head"}
                          for n in names],
        }
        truth = np.zeros((T, len(names), 3), dtype=np.float64)
        truth[:, :, 0] = np.arange(T)[:, None]*.04
        truth[:, :, 1] = np.arange(len(names))[None, :]*.12
        truth[:, :, 2] = 1.2 + np.arange(T)[:, None]*.002
        source = {"names": names, "joints": truth.tolist(), "fps": 24}
        motion = {"fps": 24, "bones": ["pelvis", "tail"],
                  "joint_names": names,
                  "joints": truth.tolist(), "trans": [[0, 0, 0]]*T,
                  "root_quat": [[1, 0, 0, 0]]*T}
        changed = json.loads(json.dumps(motion))
        for t in range(T):
            changed["joints"][t][5][0] += .10
            changed["joints"][t][6][1] += .20
        return source, rig, motion, changed

    def test_zero_error_and_two_anatomically_localized_errors(self):
        truth, rig, same, changed = self.make_example()
        report, rows, frames = evaluate(
            truth, rig, {"baseline": same, "p1": changed})
        self.assertEqual(report["frames"], 7)
        self.assertEqual(len(rows), 2*8)
        self.assertEqual(len(frames), 2*7)
        self.assertEqual(report["keypoints"], 8)
        self.assertEqual(report["variants"]["baseline"]["position_error_cm"]["max"], 0)
        self.assertEqual(report["variants"]["p1"]["per_group"]["tail"]["position_error_cm"]["mean"], 20)
        self.assertEqual(report["variants"]["p1"]["per_group"]["feet_toes"]["position_error_cm"]["mean"], 10)
        self.assertEqual(report["variants"]["p1"]["root_world_keypoint_position_error_cm"]["max"], 0)
        self.assertEqual(report["unclassified_joint_names"],
                         ["joint:CC_Dinosaur_MysteryHelper:tail"])
        self.assertEqual(sum(report["group_counts"].values()), 8)

    def test_temporal_velocity_error_and_acceleration_error_not_just_smoothness(self):
        truth, rig, base, altered = self.make_example()
        for t in range(7):
            altered["joints"][t][4][0] += .01*t*t
        report, _, _ = evaluate(truth, rig, {"noisy": altered})
        hind = report["variants"]["noisy"]["per_group"]["hindlimbs"]
        self.assertGreater(hind["velocity_error_cm_s"]["mean"], 0)
        self.assertGreater(hind["acceleration_error_cm_s2"]["mean"], 0)
        self.assertEqual(report["variants"]["noisy"]["root_world_keypoint_velocity_error_cm_s"]["mean"], 0)

    def test_rejects_fps_joint_order_and_truncated_sequence(self):
        truth, rig, base, altered = self.make_example()
        wrong = json.loads(json.dumps(base))
        wrong["fps"] = 30
        with self.assertRaisesRegex(ValueError, "FPS mismatch"):
            evaluate(truth, rig, {"wrong_fps": wrong})
        wrong = json.loads(json.dumps(base))
        wrong["joint_names"].reverse()
        with self.assertRaisesRegex(ValueError, "ordering"):
            evaluate(truth, rig, {"wrong_names": wrong})
        wrong = json.loads(json.dumps(base))
        wrong["joints"] = wrong["joints"][:-1]
        with self.assertRaisesRegex(ValueError, "expected finite"):
            evaluate(truth, rig, {"truncated": wrong})
        wrong = json.loads(json.dumps(base))
        wrong["joints"][2][0][1] = float("nan")
        with self.assertRaisesRegex(ValueError, "expected finite"):
            evaluate(truth, rig, {"nan": wrong})

    def test_groups_are_exhaustive_and_can_override_unclassified(self):
        truth, rig, base, _ = self.make_example()
        name = rig["keypoints"][-1]["name"]
        report, _, _ = evaluate(
            truth, rig, {"base": base},
            group_overrides={name: "torso_root"})
        self.assertFalse(report["unclassified_joint_names"])
        self.assertEqual(report["group_counts"]["torso_root"], 3)
        with self.assertRaisesRegex(ValueError, "overrides"):
            evaluate(truth, rig, {"base": base},
                     group_overrides={name: "missing_group"})
        for n in [x["name"] for x in rig["keypoints"]]:
            self.assertIn(classify_joint(n), GROUPS)

    def test_actual_dino_calvicle_index_mid_are_forelimbs(self):
        for name in (
            "joint:CC_Dinosaur_L_Calvicle:head",
            "joint:CC_Dinosaur_L_Calvicle:tail",
            "joint:CC_Dinosaur_R_Index01:head",
            "joint:CC_Dinosaur_L_Index02:tail",
            "joint:CC_Dinosaur_R_Mid01:head",
            "joint:CC_Dinosaur_L_Mid02:tail",
        ):
            self.assertEqual(classify_joint(name), "forelimbs")
        # A syntactic bone endpoint suffix MUST NOT make any limb a Tail.
        self.assertEqual(
            classify_joint("joint:CC_Dinosaur_R_LeftToe03:tail"),
            "feet_toes")
        self.assertEqual(
            classify_joint("joint:CC_Dinosaur_Tail07:tail"), "tail")

    def test_unknown_group_remains_visible_not_dropped(self):
        truth, rig, base, _ = self.make_example()
        report, rows, _ = evaluate(truth, rig, {"base": base})
        self.assertEqual(report["group_counts"]["unclassified"], 1)
        self.assertEqual(len({r["joint"] for r in rows}), len(rig["keypoints"]))


if __name__ == "__main__":
    unittest.main()
