"""Pure geometric and contract regression for the N-view P2 pipeline."""
import json
import unittest
from pathlib import Path
import numpy as np
import torch

from mocap.multiview import (
    check_packet,load_packet,project_np,linear_cameras,triangulate)
from mocap.solve_multiview import torch_project

ROOT=Path(__file__).resolve().parents[1]


def local_fixture(creature="dino",mode="synthetic_noisy"):
    base=ROOT/"runs"/creature/"four_view_visibility_p1_validation"
    file=base/f"p2_{mode}.npz"
    if not file.is_file():
        raise unittest.SkipTest(f"{creature} local P2 packet missing")
    return load_packet(file)


class NViewContractTests(unittest.TestCase):
    def test_real_camera_shapes_rig_names_and_gt_reference(self):
        for creature in ("dino","dragon"):
            p,meta=local_fixture(creature)
            rig=json.loads((ROOT/"assets"/creature/"rigged/rig_info.json").read_text())
            check_packet(p,rig_names=[k["name"] for k in rig["keypoints"]])
            self.assertEqual(meta["V"],4)
            self.assertEqual(meta["T"],2)
            self.assertTrue(meta["independent_stills"])
            self.assertEqual(meta["J"],len(rig["keypoints"]))
            self.assertNotIn("joints_world",p)
            self.assertNotIn("gt_world",p)

    def test_exact_gt_camera_projection_matches_saved_labels(self):
        for creature in ("dino","dragon"):
            p,meta=local_fixture(creature)
            path=ROOT/"runs"/creature/"four_view_visibility_p1_validation/p2_synthetic_noisy.ground_truth.npz"
            with np.load(path) as gt:
                world=gt["joints_world"]
                labels=gt["exact_uv"]
            result=project_np(world,p["world_to_camera"],
                              p["image_size_px"],p["meters_per_px"])
            self.assertLess(float(np.abs(result-labels).max()),.02)

    def test_numpy_and_torch_all_views_same_pixels(self):
        p,meta=local_fixture("dino")
        shape=(meta["T"],meta["J"],3)
        points=np.random.default_rng(16).normal(size=shape).astype(float)
        py=project_np(points,p["world_to_camera"],p["image_size_px"],p["meters_per_px"])
        th=torch_project(torch.tensor(points,dtype=torch.float64),
                  torch.tensor(p["world_to_camera"],dtype=torch.float64),
                  torch.tensor(p["image_size_px"],dtype=torch.float64),
                  torch.tensor(p["meters_per_px"],dtype=torch.float64)).numpy()
        np.testing.assert_allclose(th,py,atol=1e-9)

    def test_two_view_and_four_view_exact_triangulation(self):
        for creature in ("dino","dragon"):
            p,_=local_fixture(creature)
            with np.load(ROOT/"runs"/creature/"four_view_visibility_p1_validation/p2_synthetic_noisy.ground_truth.npz") as gt:
                world=gt["joints_world"]
                exact=gt["exact_uv"]
            # Deterministically recover geometric 3D positions without noise.
            clean={k:v.copy() for k,v in p.items()}
            clean["uv"]=exact
            clean["confidence"]=np.ones_like(p["confidence"])
            clean["valid"]=np.ones_like(p["valid"])
            for ids in ([0,1],[0,1,2],[0,1,2,3]):
                tri,status=triangulate(clean,index_views=ids)
                self.assertEqual(status["triangulable_count"],world.shape[0]*world.shape[1])
                np.testing.assert_allclose(tri,world,atol=1e-4)

    def test_parallel_cameras_have_no_false_3d(self):
        p,_=local_fixture("dino")
        x,status=triangulate(p,index_views=[0,2])
        self.assertEqual(status["triangulable_count"],0)
        self.assertFalse(status["triangulable"].any())

    def test_poison_unselected_cameras_cannot_change_two_view_result(self):
        p,_=local_fixture("dino")
        tri,stat=triangulate(p,index_views=[0,1])
        bad={k:v.copy() for k,v in p.items()}
        bad["uv"][2:]+=10000
        bad["confidence"][2:]=500
        # These back/opposite observations are out-of-sample for the 2-view run.
        tri2,s=triangulate(bad,index_views=[0,1])
        np.testing.assert_array_equal(tri,tri2)
        np.testing.assert_array_equal(stat["triangulable"],s["triangulable"])

    def test_fail_closed_bad_camera_names_3d_leak_and_invalid_shapes(self):
        p,meta=local_fixture("dino")
        broken={k:v.copy() for k,v in p.items()}
        broken["view_ids"][1]=broken["view_ids"][0]
        with self.assertRaisesRegex(ValueError,"duplicate View"):
            check_packet(broken)
        broken={k:v.copy() for k,v in p.items()}
        broken["world_to_camera"][0,0,0]=2
        with self.assertRaisesRegex(ValueError,"rigid rotation"):
            check_packet(broken)
        broken={k:v.copy() for k,v in p.items()}
        broken["gt_world"]=np.zeros((2,meta["J"],3))
        with self.assertRaisesRegex(ValueError,"GT must never"):
            check_packet(broken)
        broken={k:v.copy() for k,v in p.items()}
        broken["confidence"][:]=np.nan
        with self.assertRaisesRegex(ValueError,"invalid UV"):
            check_packet(broken)

    def test_quality_head_fields_are_explicitly_uncalibrated_not_gt(self):
        p,_=local_fixture('dino')
        q={k:v.copy() for k,v in p.items()}
        q['visibility_score']=np.full(q['confidence'].shape,.8,dtype=np.float32)
        q['expected_error_px']=np.full(q['confidence'].shape,4,dtype=np.float32)
        q['quality_semantics']=np.asarray('synthetic_cutout_only_uncalibrated')
        check_packet(q)
        q['quality_semantics']=np.asarray('true_visibility_probability')
        with self.assertRaisesRegex(ValueError,'uncalibrated'):
            check_packet(q)
        q['quality_semantics']=np.asarray('synthetic_cutout_only_uncalibrated')
        q['visibility_score'][0,0,0]=1.5
        with self.assertRaisesRegex(ValueError,'outside'):
            check_packet(q)
        q['visibility_score'][0,0,0]=.8
        q['joint_surface_part_visible']=np.zeros(q['confidence'].shape,dtype=np.int8)
        with self.assertRaisesRegex(ValueError,'GT mesh surface'):
            check_packet(q)

    def test_reader_ood_packet_contains_only_2d_information(self):
        base=ROOT/"runs"/"dino"/"four_view_visibility_p1_validation/p2_reader_ood.npz"
        if not base.exists():
            self.skipTest("Reader smoke input not yet generated")
        p,meta=load_packet(base)
        self.assertEqual(meta["source_kind"],"reader_inference")
        self.assertNotIn("joint_surface_part_visible",p)
        self.assertNotIn("joints_world",p)


if __name__=="__main__":
    unittest.main()
