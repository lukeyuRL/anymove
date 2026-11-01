"""No-network regressions for H3Max Dino three-view reference and sync audit."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
    from tools.build_dino_three_view_ref import compose
    from tools.audit_fal_three_view import segment_magenta,correlate_shift,summary
except ImportError:
    np=None


@unittest.skipUnless(np is not None,"NumPy, OpenCV and Pillow optional outside visibility CI")
class DinoFalThreeViewSmokeTests(unittest.TestCase):
    def test_compose_preserves_three_source_aspects_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            sources={}
            for name,size,color in (
                    ("front",(400,720),(255,80,20,255)),
                    ("side",(880,720),(25,180,100,255)),
                    ("top",(880,720),(200,50,200,255))):
                path=root/(name+".png")
                Image.new("RGBA",size,color).save(path)
                sources[name]=path
            target=root/"reference"
            target.mkdir()
            file,manifest=compose(sources["front"],sources["side"],sources["top"],target)
            with Image.open(file) as image:
                self.assertEqual(image.size,(1280,720))
            info=json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(info["canvas"],[1280,720])
            self.assertEqual(info["divider_x"],[320,800])
            self.assertEqual(info["panels"]["front"]["render_height"],576)
            self.assertEqual(info["panels"]["side"]["render_height"],393)
            self.assertEqual(info["panels"]["top"]["render_height"],393)
            self.assertEqual(info["panels"]["front"]["panel_left"],0)
            self.assertEqual(info["panels"]["side"]["panel_left"],320)
            self.assertEqual(info["panels"]["top"]["panel_left"],800)
            for name in sources:
                self.assertEqual(info["panels"][name]["source_sha256"],__import__("hashlib").sha256(sources[name].read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                compose(sources["front"],sources["side"],sources["top"],target)

    def test_reject_wrong_camera_view_pixel_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            p=[]
            for name,size in (("front",(400,720)),("side",(500,720)),("top",(880,720))):
                path=root/(name+".png")
                Image.new("RGBA",size,(20,60,90,255)).save(path)
                p.append(path)
            out=root/"reference";out.mkdir()
            with self.assertRaisesRegex(ValueError,"resolution"):
                compose(*p,out)

    def test_magenta_tail_tip_isolated_from_gray_and_colored_limbs(self):
        import cv2
        im=np.zeros((100,120,3),np.uint8)
        im[:]=(134,130,128)
        cv2.rectangle(im,(12,40),(24,60),(255,0,255),-1) # magenta BGR
        cv2.rectangle(im,(80,10),(94,24),(255,20,5),-1) # bright blue
        found=segment_magenta(im,0,0,120,100)
        self.assertGreater(found["pixels"],200)
        self.assertAlmostEqual(found["center"][0],18,delta=2)
        self.assertAlmostEqual(found["center"][1],50,delta=2)
        empty=segment_magenta(im,40,0,80,100)
        self.assertIsNone(empty["center"])

    def test_motion_lag_finds_known_shift_but_flags_static(self):
        t=np.arange(180,dtype=float)
        wave=30*np.sin(t*.14)+10*np.sin(t*.31)
        shifted=np.r_[np.zeros(5),wave[:-5]]
        out=correlate_shift(wave,shifted,12)
        self.assertEqual(out["best_lag_frames"],5)
        self.assertGreater(out["best_correlation"],.999)
        static=correlate_shift(np.ones(180),np.ones(180),12)
        self.assertIsNone(static["best_lag_frames"])

    def test_summary_preserves_missing_observations(self):
        report=summary([1,2,float("nan"),3])
        self.assertEqual(report["n"],3)
        self.assertEqual(report["median"],2)


if __name__=="__main__":
    unittest.main()
