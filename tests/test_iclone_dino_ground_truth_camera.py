"""Independent Dino Blender camera JSON versus fixed 2-view layout."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from tools.audit_dino_gt_camera import audit_camera


@unittest.skipUnless(np is not None, "NumPy optional in lightweight CI")
class DinoGroundTruthCameraAuditTests(unittest.TestCase):
    def make_case(self):
        H,Wf,Ws,mp,Z=720,640,640,.006875,1.85625
        geometry=dict(H=H,W_FRONT=Wf,W_SIDE=Ws,M_PER_PX=mp,Z_CENTER=Z)
        Rfront=np.array([[1,0,0],[0,0,1],[0,-1,0]],float)
        Rside=np.array([[0,-1,0],[0,0,1],[-1,0,0]],float)
        def cam(R, position, width):
            t=-(R@np.array(position))
            return {
                "type":"ORTHO", "R":R.tolist(),"t":t.tolist(),
                "f":1/mp,"cx":width/2,"cy":H/2,"res":[width,H]}
        front=cam(Rfront,[0,-20,Z],Wf)
        side=cam(Rside,[-20,0,Z],Ws)
        points=np.zeros((4,3,3),float)
        points[:,:,0]=np.linspace(-.2,.3,4)[:,None]
        points[:,:,1]=np.array([-1.,0,1.])[None,:]
        points[:,:,2]=np.linspace(.2,1.8,4)[:,None]
        return {"joints":points.tolist()},front,side,geometry

    def test_same_blender_camera_matches_solver_layout(self):
        truth, front, side, layout=self.make_case()
        report=audit_camera(truth,front,side,layout)
        self.assertEqual(report["status"],
                         "blender_cameras_match_anymove_orthographic_layout")
        self.assertLess(report["views"]["front"]["max_pixel_geometry_difference"],.001)
        self.assertLess(report["views"]["side"]["max_pixel_geometry_difference"],.001)

    def test_camera_rotation_and_scale_mismatch_fails_closed(self):
        truth, front, side, layout=self.make_case()
        side["t"][0]+=0.05
        with self.assertRaisesRegex(ValueError,"differs"):
            audit_camera(truth,front,side,layout)
        truth, front, side, layout=self.make_case()
        front["res"]=[420,720]
        with self.assertRaisesRegex(ValueError,"resolution"):
            audit_camera(truth,front,side,layout)
        truth, front, side, layout=self.make_case()
        side["type"]="PERSP"
        with self.assertRaisesRegex(ValueError,"orthographic"):
            audit_camera(truth,front,side,layout)


if __name__ == "__main__":
    unittest.main()
