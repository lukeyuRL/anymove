"""Tests for orthographic z-buffer and skinned surface-part visibility targets."""
import unittest
import numpy as np
from mocap.surface_visibility import (
    dominant_triangle_parts,rasterize_camera_mesh,part_visibility_targets,
    camera_axes_consistency,UNLABELED_SURFACE
)


class SurfaceVisibilityTests(unittest.TestCase):
    def overlapping_surfaces(self):
        # Near A overlays far B, same projected silhouette, normal +Z.
        verts=np.array([
            [-.2,-.2,-2],[.2,-.2,-2],[0,.3,-2],
            [-.2,-.2,-4],[.2,-.2,-4],[0,.3,-4]
        ],dtype=float)
        faces=np.array([[0,1,2],[3,4,5]])
        return verts,faces

    def test_depth_occludes_rear_part_and_amodal_preserved(self):
        v,f=self.overlapping_surfaces()
        r=rasterize_camera_mesh(v,f,np.array([0,1]),width=64,height=64,
                                 meters_per_px=.01,n_parts=2)
        self.assertGreater(int(r["amodal_pixel_count"][0]),100)
        self.assertEqual(int(r["amodal_pixel_count"][0]),
                         int(r["amodal_pixel_count"][1]))
        self.assertEqual(int(r["visible_pixel_count"][1]),0)
        self.assertAlmostEqual(float(r["surface_visible_fraction"][0]),1,places=6)
        self.assertAlmostEqual(float(r["surface_visible_fraction"][1]),0,places=6)
        finite=np.isfinite(r["depth_m"])
        np.testing.assert_allclose(r["depth_m"][finite],2.,atol=1e-5)
        self.assertTrue(np.all(r["part_id"][~finite]==-1))

    def test_triangle_order_does_not_change_closest_depth(self):
        v,f=self.overlapping_surfaces()
        a=rasterize_camera_mesh(v,f,np.array([0,1]),width=64,height=64,
                 meters_per_px=.01,n_parts=2)
        b=rasterize_camera_mesh(v,f[::-1],np.array([1,0]),width=64,height=64,
                 meters_per_px=.01,n_parts=2)
        np.testing.assert_array_equal(a["part_id"],b["part_id"])
        np.testing.assert_allclose(a["depth_m"],b["depth_m"],equal_nan=True)

    def test_ambiguous_surface_still_occludes(self):
        v,f=self.overlapping_surfaces()
        r=rasterize_camera_mesh(v,f,np.array([UNLABELED_SURFACE,1]),width=64,height=64,
                 meters_per_px=.01,n_parts=2)
        self.assertEqual(r["visible_pixel_count"][1],0)
        self.assertGreater(r["ambiguous_surface_pixels"],0)
        self.assertTrue(np.isnan(r["surface_visible_fraction"][0]))

    def test_joint_labels_are_surface_part_semantics_with_unknown(self):
        v,f=self.overlapping_surfaces()
        r=rasterize_camera_mesh(v,f,np.array([0,1]),width=64,height=64,
                 meters_per_px=.01,n_parts=3)
        labels=part_visibility_targets(r,np.array([0,1,2]),min_area_px=4)
        self.assertEqual(labels.tolist(),[1,0,-1])
        r2=rasterize_camera_mesh(v,f,np.array([0,1]),width=64,height=64,
                 meters_per_px=.01,n_parts=3)
        self.assertEqual(part_visibility_targets(r2,np.array([0]),min_area_px=99999)[0],-1)

    def test_skinning_mapping_detects_uncertain_faces(self):
        weights=np.array([
            [0.7,0.3],[.8,.2],[.7,.3],
            [.52,.48],[.5,.5],[.48,.52],
        ])
        f=np.array([[0,1,2],[3,4,5]])
        p,stats=dominant_triangle_parts(weights,f,min_weight=.45,min_margin=.08)
        self.assertEqual(p.tolist(),[0,-2])
        self.assertEqual(stats["assigned_triangles"],1)
        self.assertEqual(stats["ambiguous_triangles"],1)

    def test_invalid_inputs_fail_closed(self):
        v,f=self.overlapping_surfaces()
        with self.assertRaisesRegex(ValueError,"invalid camera mesh"):
            rasterize_camera_mesh(v,f,[0,50],width=64,height=64,
                                  meters_per_px=.01,n_parts=2)
        with self.assertRaisesRegex(ValueError,"invalid camera mesh"):
            rasterize_camera_mesh(v,f,[0,1],width=64,height=64,
                                  meters_per_px=-.01,n_parts=2)
        with self.assertRaisesRegex(ValueError,"invalid skinning"):
            dominant_triangle_parts(np.ones((2,2)),f)
        r=rasterize_camera_mesh(v,f,[0,1],width=64,height=64,
                                meters_per_px=.01,n_parts=2)
        with self.assertRaisesRegex(ValueError,"invalid part-to-keypoint"):
            part_visibility_targets(r,[1000])

    def test_flip_camera_pair(self):
        front=np.array([[8,20],[12,30]])
        back=np.array([[32,20],[28,30]])
        self.assertTrue(camera_axes_consistency(front,back,40))
        self.assertFalse(camera_axes_consistency(front,front,40))


if __name__=="__main__":
    unittest.main()
