"""Dino 41-class camera-aware argmax SkinWeight colour regression."""
import json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"runs/dino/skinmax_camera_aware_v1"
VIEWS=("front","side","back","opposite_side")

class ViewAwareDinoSkinWeightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regions=json.loads((P/"skinmax_regions.json").read_text(encoding="utf8"))
        cls.graph=json.loads((P/"camera_conflicts.json").read_text(encoding="utf8"))
        cls.palette=json.loads((P/"optimized_palette_constrained_v4.json").read_text(encoding="utf8"))
        cls.actual=json.loads((P/"rendered_camera_colour_audit_v4.json").read_text(encoding="utf8"))
        cls.applied=json.loads((P/"test_render_v4/viewaware_color_audit.json").read_text(encoding="utf8"))
    def test_categories_are_derived_from_weights_not_fixed_limit(self):
        reg=self.regions;pal=self.palette
        self.assertEqual(reg["classes"],41)
        self.assertEqual(len(pal["bone_colors"]),41)
        self.assertEqual(len(set(pal["bone_colors"].values())),41)
        self.assertEqual(set(pal["bone_colors"]),set(reg["selected_bone_names"]))
        self.assertEqual(sum(reg["dominant_vertex_count"].values()),4909)
        self.assertEqual(reg["total_mesh_vertices"],12232)
        self.assertEqual(reg["noncolorable_vertex_count"],7323)
        self.assertEqual(len(reg["unused_or_zero_dominant_weight_bones"]),13)
        self.assertFalse(set(reg["unused_or_zero_dominant_weight_bones"])&set(pal["bone_colors"]))
        self.assertEqual(list(reg["categories"].values()).count("hind_legs"),22)
        self.assertEqual(list(reg["categories"].values()).count("forearms_hands"),12)
        self.assertEqual(list(reg["categories"].values()).count("tail"),7)
        self.assertGreaterEqual(min(reg["dominant_vertex_count"].values()),30)
        self.assertEqual(pal["bone_colors"]["CC_Dinosaur_L_Thigh"],"#2979FF")
        self.assertEqual(pal["bone_colors"]["CC_Dinosaur_R_Thigh"],"#E53935")
    def test_projection_priority_and_rendered_collision_are_reported_not_hidden(self):
        g=self.graph;p=self.palette;a=self.actual
        self.assertEqual(g["sample_count"],32)
        self.assertTrue(set(g["poses"]).isdisjoint({9001,9002,9007}))
        self.assertEqual(g["view_weights"],{"front":2.,"side":2.,"back":1.,"opposite_side":1.})
        self.assertGreater(g["view_summary"]["front"]["close_pair_count"],200)
        self.assertGreater(g["view_summary"]["side"]["close_pair_count"],180)
        self.assertLessEqual(p["postrefinement_camera_near_conflicts"]["front"]["pairs_below_deltaE35"],1)
        self.assertEqual(p["postrefinement_camera_near_conflicts"]["side"]["pairs_below_deltaE35"],0)
        self.assertEqual(a["view_metrics"]["front"]["rendered_color_neighbor_pairs_deltaE76_below25"],0)
        self.assertEqual(a["view_metrics"]["side"]["rendered_color_neighbor_pairs_deltaE76_below25"],2)
        self.assertTrue(a["no_colour_can_restore_fully_occluded_mesh"])
        self.assertTrue(p["render_diagnostic_colours_used_for_refinement"])
    def test_vertex_argmax_and_materials_only_appearance_modified(self):
        report=self.applied
        self.assertEqual(report["colour_count"],41)
        self.assertEqual(report["coloured_vertex_count"],4909)
        self.assertEqual(report["uncoloured_vertex_count"],7323)
        self.assertEqual(report["vertex_color_domain"],"POINT")
        self.assertTrue(report["no_skin_weights_or_geometry_modified"])
        self.assertTrue(report["original_base_color_preserved_on_uncoloured_regions"])
        self.assertEqual(report["zero_dominant_bones_coloured"],[])
        self.assertTrue((P/"dino_viewaware_skinmax_v4.blend").exists())
    def test_all_3_poses_4_camera_gt_unchanged(self):
        ref=ROOT/"runs/dino/p4_blind_fourview_v1"
        dst=P/"test_render_v4"
        for pose in (9001,9002,9007):
            for view in VIEWS:
                name=f"{pose:05d}_{view}_labels.npz"
                with np.load(ref/name,allow_pickle=False) as original, \
                     np.load(dst/name,allow_pickle=False) as modified:
                    np.testing.assert_array_equal(original["part_id"],modified["part_id"])
                    np.testing.assert_allclose(original["depth_m"],modified["depth_m"],equal_nan=True)
                    np.testing.assert_allclose(original["joints_uv"],modified["joints_uv"],atol=1e-5)
                self.assertTrue((dst/f"{pose:05d}_{view}.png").exists())
    def test_p1_orthographic_camera_raycast_validation(self):
        q=json.loads((P/"test_render_v4/viewaware_render_geom_audit_v4.json").read_text(encoding="utf8"))
        self.assertEqual(q["status"],"PASS")
        self.assertEqual(q["num_samples"],3)
        self.assertEqual(q["num_views_per_sample"],4)


if __name__=="__main__":unittest.main()
