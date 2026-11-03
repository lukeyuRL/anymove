"""Dino P2 visual overlay/report should preserve detection dimensions."""
import unittest

try:
    import numpy as np
    import cv2
except ImportError:
    np=None
    cv2=None

if np is not None and cv2 is not None:
    from tools.review_dino_p2 import quality_report, draw_one
    from mocap import layout


@unittest.skipUnless(np is not None and cv2 is not None,
                     "NumPy/OpenCV covered in visibility-numpy CI")
class DinoP2VisualReviewTests(unittest.TestCase):
    def make_detection(self):
        det={
            "names":np.array(["joint:CC_Dinosaur_Tail07:tail",
                              "joint:CC_Dinosaur_L_LeftToe03:tail"]),
            "fps":np.array(24.0),
            "calib":np.eye(2,3,dtype=np.float32),
        }
        for view in ("front","side"):
            det[view+"_uv"]=np.tile(np.array([[50.,75.],[100.,140.]],np.float32),(4,1,1))
            det[view+"_expected_error_px"]=np.array(
                [[2.,18.],[2.,9.],[15.,2.],[2.,2.]],np.float32)
            det[view+"_synthetic_visibility_score"]=np.array(
                [[.8,.2],[.9,.5],[.1,.9],[.8,.9]],np.float32)
        return det

    def test_report_flags_high_errors_and_ranks_joint_by_mean(self):
        det=self.make_detection()
        summary=quality_report(det)
        self.assertEqual(summary["frames"],4)
        self.assertEqual(summary["keypoints"],2)
        self.assertEqual(summary["views"]["side"]["joint_frames_predicted_error_over_12px"],2)
        self.assertEqual(summary["views"]["side"]["joint_frames_synthetic_cutout_score_below_0_35"],2)
        self.assertEqual(summary["views"]["side"]["top_mean_predicted_error_joints"][0]["name"],
                         "joint:CC_Dinosaur_L_LeftToe03:tail")

    def test_overlay_dimensions_and_original_inputs_untouched(self):
        det=self.make_detection()
        frame=np.zeros((layout.H,layout.W,3),np.uint8)
        before=det["front_uv"].copy()
        canvas=draw_one(frame,det,0)
        self.assertEqual(canvas.shape,(layout.H,layout.W+540,3))
        self.assertTrue(np.any(canvas))
        np.testing.assert_array_equal(det["front_uv"],before)

    def test_missing_or_invalid_quality_fails_closed(self):
        det=self.make_detection()
        det["front_expected_error_px"][1,0]=np.nan
        with self.assertRaisesRegex(ValueError,"invalid"):
            quality_report(det)
        det["front_expected_error_px"][1,0]=2.
        det.pop("side_synthetic_visibility_score")
        with self.assertRaises(KeyError):
            quality_report(det)


if __name__=="__main__":
    unittest.main()
