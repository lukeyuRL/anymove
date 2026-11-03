"""Offline integrity/regression tests for Human H3 Ref2VA V6 fourcam snapshot.

No fal credentials, cloud calls, Blender, model weights, video or NumPy.
Runs against only checked-in source/evidence snapshots.
"""
import copy
import hashlib
import json
import pathlib
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
CODE = REPO / "tools" / "human_h3_ref2va_v6"
DATA = REPO / "docs" / "human_h3_ref2va_v6"


def load(filename):
    return json.loads((DATA / filename).read_text(encoding="utf8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def matches_original_snapshot(path, manifest_entry):
    """Compare the content to original source bytes across Git CRLF/LF checkouts."""
    raw = path.read_bytes()
    options = [raw]
    if path.suffix.lower() in (".py", ".md", ".json", ".txt"):
        lf = raw.replace(b"\r\n", b"\n")
        options.extend((lf, lf.replace(b"\n", b"\r\n")))
    return any(
        len(candidate) == manifest_entry["bytes"]
        and hashlib.sha256(candidate).hexdigest() == manifest_entry["sha256"]
        for candidate in options
    )


class HumanFourcamV6SnapshotTests(unittest.TestCase):

    def test_all_evidence_and_script_hashes(self):
        manifest = load("SNAPSHOT_MANIFEST.json")
        self.assertEqual(manifest["schema"], "anymove.human_ref2va_v6_git_snapshot/1")
        self.assertEqual(len(manifest["source_files"]), 27)
        self.assertGreaterEqual(len(manifest["evidence_files"]), 11)
        for item in manifest["source_files"]:
            with self.subTest(item=item["name"]):
                p = CODE / item["name"]
                self.assertTrue(p.is_file())
                self.assertTrue(matches_original_snapshot(p, item), msg=p.name)
                compile(p.read_text(encoding="utf8"), str(p), "exec")
        for item in manifest["evidence_files"]:
            with self.subTest(item=item["name"]):
                p = DATA / item["name"]
                self.assertTrue(p.is_file())
                self.assertTrue(matches_original_snapshot(p, item), msg=p.name)

    def test_training_dataset_acceptance(self):
        data = load("dataset_offline_acceptance.json")
        dataset = data["final_zip"]
        self.assertEqual(data["status"], "PASS_OFFLINE_DATASET")
        self.assertEqual(dataset["training_clips"], 28)
        self.assertEqual(dataset["heldout_clips"], 8)
        self.assertEqual(dataset["members"], 112)
        self.assertEqual(dataset["sha256"], "76faf527f9e5cd7f9521b8a42f306b7f07fda04f6c2d7cc2d4d233f57f102141")
        merged = load("dataset_merge_qa.json")
        self.assertEqual(merged["status"], "PASS_MERGED_OFFLINE_FILES_AND_INTEGRITY")
        self.assertEqual(merged["same_view_layout"], ["left_45", "right_45", "back", "top"])
        self.assertEqual(merged["aspect_ratio"], "1:1")

    def test_fal_debug_and_model_validation(self):
        dbg = load("fal_1step_debug_audit.json")
        self.assertEqual(dbg["status"], "PASS_PREPROCESSED_28_FOUR_CAMERA_SAMPLES")
        self.assertEqual(dbg["preprocessed_samples"], 28)
        self.assertEqual(dbg["captions_with_FOURCAM"], 28)
        self.assertTrue(dbg["reference_video_and_image_present_for_all_28"])
        self.assertEqual(dbg["reference1_sparse_video_frames"], 9)
        model = load("model_validation.json")
        self.assertEqual(model["status"], "PASS_400STEP_MODEL_ARTIFACTS")
        self.assertEqual(model["steps"], 400)
        self.assertEqual(model["training_source"], "H3_BASE")
        self.assertEqual(model["config"]["instance_prompt"], "FOURCAM")
        self.assertEqual(model["config"]["training_type"], "ref2va")
        self.assertEqual(model["weights"]["tensors"], 208)
        self.assertEqual(model["weights"]["bytes"], 65628056)
        self.assertEqual(model["weights"]["sha256"], "cc837f9122caa0619b5357a25563e319b8438102f07128ad057d2dbe70add6a2")

    def test_paid_ab_requests_only_change_lora_scale(self):
        controls = load("ab_request_controls_sanitized.json")
        self.assertEqual(set(controls["experiments"]), {"forward_left", "backward_right"})
        all_requests = set()
        for case, value in controls["experiments"].items():
            self.assertEqual(set(value["requests"]), {"scale0", "scale1"})
            a = copy.deepcopy(value["requests"]["scale0"])
            b = copy.deepcopy(value["requests"]["scale1"])
            self.assertEqual(a.pop("loras")[0]["scale"], 0.0)
            self.assertEqual(b.pop("loras")[0]["scale"], 1.0)
            self.assertEqual(a, b)
            self.assertEqual(a["seed"], 42)
            self.assertEqual(a["aspect_ratio"], "1:1")
            self.assertEqual(a["resolution"], "768P")
            self.assertTrue(a["prompt"].startswith("FOURCAM "))
            self.assertEqual(len(a["reference_image_urls"]), 1)
            self.assertEqual(len(a["reference_video_urls"]), 1)
            for rid in value["request_ids"].values():
                self.assertNotIn(rid, all_requests)
                all_requests.add(rid)
        self.assertEqual(len(all_requests), 4)

    def test_controlled_ab_summaries(self):
        record = load("ab_summary_sanitized.json")
        self.assertEqual(record["status"], "PASS_FOUR_PAID_VIDEOS_COMPLETED")
        self.assertEqual(set(record["full_videos"]), {"forward_left", "backward_right"})
        for results in record["full_videos"].values():
            self.assertEqual(set(results), {"scale0", "scale1"})
            for item in results.values():
                self.assertNotIn("source_url", item)
                self.assertEqual(item["local_video"], "EXTERNAL_RUNTIME_VIDEO_NOT_GIT_TRACKED")
                self.assertEqual(len(item["sha256"]), 64)
        results = record["statistics"]
        self.assertGreater(results["forward_left"]["scale1"]["eight_regions_mean_IoU_active"],
                           results["forward_left"]["scale0"]["eight_regions_mean_IoU_active"])
        self.assertGreater(results["backward_right"]["scale1"]["eight_regions_mean_IoU_active"],
                           results["backward_right"]["scale0"]["eight_regions_mean_IoU_active"])
        combined = results["combined"]
        self.assertEqual(combined["fal_paid_inference_videos_submitted"], 4)
        self.assertEqual(combined["sixteen_view_color_pairs_where_scale1_IoU_higher"], 13)
        self.assertEqual(combined["out_of_total_pairs"], 16)
        self.assertAlmostEqual(combined["all_color_regions_mean_active_IoU_scale0"], 0.6546, places=4)
        self.assertAlmostEqual(combined["all_color_regions_mean_active_IoU_scale1"], 0.68419, places=5)
        truth = record["blender_3d_gt"]["families"]
        self.assertEqual(truth["forward_left"]["R"]["hand_joint_max_3d_displacement_from_rest_m"], 0)
        self.assertEqual(truth["backward_right"]["L"]["hand_joint_max_3d_displacement_from_rest_m"], 0)

    def test_all_views_have_tracked_red_and_blue_regions(self):
        for case in ("forward_left", "backward_right"):
            data = load(case + "_color_iou.json")
            self.assertEqual(data["status"], "PASS")
            for scale in ("scale0", "scale1"):
                stats = data["metrics"][scale]
                self.assertEqual(len(stats), 8)
                for v in ("left_45", "right_45", "back", "top"):
                    for c in ("red", "blue"):
                        m = stats[f"{v}_{c}"]
                        self.assertGreater(m["frames_source_has_min20_color_px"], 80)
                        self.assertGreaterEqual(m["mean_mask_IoU_all_nonoccluded"], 0)
                        self.assertLessEqual(m["mean_mask_IoU_all_nonoccluded"], 1)


if __name__ == "__main__":
    unittest.main()
