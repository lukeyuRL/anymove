"""QA the Dino 20 colour classes against existing untouched P4 GT renders."""
import json
import unittest
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"runs/dino/color20_candidate_v1"
ORIG=ROOT/"runs/dino/p4_blind_fourview_v1"
VIEWS=("front","side","back","opposite_side")


class DinoSkinMax20Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.palette=json.loads((BASE/"skinmax_palette_v2.json").read_text(encoding="utf8"))
        cls.geo=json.loads((BASE/"skinmax_geometry.json").read_text(encoding="utf8"))
        cls.audit=json.loads((BASE/"test_render_v1/color_assignment_audit.json").read_text(encoding="utf8"))

    def test_zero_dominant_weight_bones_have_no_colour(self):
        p=self.palette
        self.assertEqual(len(p["bone_colors"]),20)
        self.assertEqual(len(set(p["bone_colors"].values())),20)
        self.assertEqual(set(p["bone_colors"]),set(self.geo["selected_bone_names"]))
        self.assertEqual(sum(p["skinmax_vertex_counts"].values()),2484)
        self.assertEqual(self.geo["unweighted_vertex_count"],0)
        self.assertEqual(self.geo["total_mesh_vertices"],12232)
        self.assertEqual(p["color_class_count"],20)
        self.assertEqual(len(p["excluded_bones_without_dominant_skin_weight"]),8)
        self.assertFalse(set(p["excluded_bones_without_dominant_skin_weight"])&set(p["bone_colors"]))
        self.assertEqual(p["bone_colors"]["CC_Dinosaur_L_Calf"],"#2979FF")
        self.assertEqual(p["bone_colors"]["CC_Dinosaur_R_Calf"],"#E53935")
        self.assertTrue(min(p["skinmax_vertex_counts"].values())>0)

    def test_coloured_vertex_count_and_original_texture_audit(self):
        a=self.audit
        self.assertEqual(a["selected_vertex_count"],2484)
        self.assertEqual(a["unselected_vertex_count"],9748)
        self.assertEqual(a["total_mesh_vertices"],12232)
        self.assertEqual(a["vertex_color_domain"],"POINT")
        self.assertTrue(a["original_materials_preserved"])
        self.assertTrue(a["no_geometry_or_skin_weight_changed"])
        self.assertEqual(len(a["counts_by_bone"]),20)

    def test_source_geometry_and_depth_unchanged_on_8_views(self):
        for f in (9001,9002):
            for v in VIEWS:
                with self.subTest(sample=f,view=v):
                    path=f"{f:05d}_{v}_labels.npz"
                    with np.load(BASE/"test_render_v1"/path,allow_pickle=False) as altered, \
                         np.load(ORIG/path,allow_pickle=False) as untouched:
                        np.testing.assert_array_equal(altered["part_id"],untouched["part_id"])
                        np.testing.assert_allclose(altered["depth_m"],untouched["depth_m"],equal_nan=True)
                        np.testing.assert_allclose(altered["joints_uv"],untouched["joints_uv"],atol=1e-4)
                    img=cv2.imread(str(BASE/"test_render_v1"/f"{f:05d}_{v}.png"),cv2.IMREAD_UNCHANGED)
                    old=cv2.imread(str(ORIG/f"{f:05d}_{v}.png"),cv2.IMREAD_UNCHANGED)
                    self.assertEqual(img.shape,old.shape)
                    # Source and colored renders use different Cycles samples.
                    self.assertGreater(int((np.max(np.abs(img[:,:,:3].astype(np.int16)-
                                            old[:,:,:3].astype(np.int16)),axis=2)>35).sum()),150)

    def test_blender_preview_copy_is_separate(self):
        path=BASE/"dino_skinmax20_preview.blend"
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size,1_000_000)
        self.assertTrue((BASE/"test_render_v1/skinmax20_render_audit.json").is_file())


if __name__=="__main__":unittest.main()
