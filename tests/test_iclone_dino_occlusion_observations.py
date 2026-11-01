"""Per-view observation omission must not delete any frame/key or use occluded UV."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from mocap.occlusion_observations import prepare_masked_solver_input
    from mocap import layout
    from tools.make_dino_occlusion_ablations import build


@unittest.skipUnless(np is not None, "NumPy optional in lightweight CI")
class DinoOccludedObservationTests(unittest.TestCase):
    def setUp(self):
        self.T = 8
        self.names=(
            ["joint:CC_Dinosaur_Tail%02d:tail" % i for i in range(9)] +
            ["joint:CC_Dinosaur_L_LeftToe%02d:tail" % i for i in range(34)] +
            ["joint:CC_Dinosaur_Hip%02d:head" % i for i in range(49)])
        self.rig={"keypoints":[{"name":n,"rest":[0.,0.,float(i)*.03]}
                               for i,n in enumerate(self.names)]}
        self.raw={
            "names":np.asarray(self.names),
            "fps":np.asarray(24.),
            "calib":np.asarray([[1.,0.,0.],[0.,1.,0.]]),
            "front_uv":np.ones((self.T,92,2),np.float32)*100,
            "side_uv":np.ones((self.T,92,2),np.float32)*220,
            "front_conf":np.ones((self.T,92),np.float32)*.8,
            "side_conf":np.ones((self.T,92),np.float32)*.7,
        }

    def test_mesh_proxy_masks_supported_intersections_only(self):
        geom={"names":np.asarray(self.names),"frame_count":np.asarray(self.T)}
        for v in ("front","side"):
            geom[v+"_mesh_blocked_proxy"]=np.zeros((self.T,92),bool)
            geom[v+"_mesh_support"]=np.zeros((self.T,92),bool)
            geom[v+"_surface_crossings"]=np.zeros((self.T,92),np.uint8)
        geom["front_mesh_blocked_proxy"][2,0]=True
        geom["front_mesh_support"][2,0]=True
        geom["front_surface_crossings"][2,0]=2
        # Unsupported e.g. virtual tip is unknown: never silently mask.
        geom["front_surface_crossings"][1,8]=3
        out,audit=build(self.raw,self.rig,"mesh",geometry=geom)
        self.assertFalse(out["front_visible_mask"][2,0])
        self.assertTrue(out["front_visible_mask"][1,8])
        self.assertTrue(out["side_visible_mask"].all())
        self.assertEqual(audit["views"]["front"]["masked"],1)
        for key in self.raw:
            np.testing.assert_array_equal(out[key],self.raw[key])

    def test_p2_error_and_oracle_are_separate_from_visibility(self):
        q={k:v.copy() for k,v in self.raw.items()}
        q["front_expected_error_px"]=np.zeros((self.T,92),np.float32)
        q["side_expected_error_px"]=np.zeros((self.T,92),np.float32)
        q["front_expected_error_px"][4,8]=29.
        result,audit=build(self.raw,self.rig,"p2_error",quality=q,threshold_px=10)
        self.assertFalse(result["front_visible_mask"][4,8])
        self.assertTrue(result["side_visible_mask"][4,8])
        self.assertEqual(audit["mode"],"p2_error")
        with self.assertRaisesRegex(ValueError,"required"):
            build(self.raw,self.rig,"oracle_gt")
        with self.assertRaisesRegex(ValueError,"unused"):
            build(self.raw,self.rig,"p2_error",quality=q,
                  truth={"names":self.names})

    def test_masked_pixel_outliers_cannot_reach_solver_observations(self):
        masks={k:v.copy() for k,v in self.raw.items()}
        for v in ("front","side"):
            masks[v+"_visible_mask"]=np.ones((self.T,92),bool)
        masks["front_visible_mask"][3:6,8]=False
        masks["side_visible_mask"][4,8]=False
        a,ms,audit=prepare_masked_solver_input(
            {k:v.copy() for k,v in self.raw.items()
             if k in ("front_uv","side_uv","front_conf","side_conf")},
            masks,self.rig)
        # A perturbed raw UV at a masked image location MUST be completely
        # ignored for both initialization and downstream fit weighting.
        braw={k:v.copy() for k,v in masks.items()}
        braw["front_uv"][3:6,8]=123456.
        braw["side_uv"][4,8]=-345678.
        b,_,_=prepare_masked_solver_input(
            {k:braw[k] for k in ("front_uv","side_uv","front_conf","side_conf")},
            braw,self.rig)
        for view in ("front","side"):
            np.testing.assert_allclose(a[view+"_uv"],b[view+"_uv"])
            np.testing.assert_allclose(a[view+"_conf"],b[view+"_conf"])
        self.assertTrue(np.all(a["front_conf"][3:6,8]==0))
        self.assertEqual(audit["joint_frames_missing_in_both_views"],1)
        self.assertEqual(len(a["front_uv"]),self.T)
        self.assertTrue(audit["no_animation_keys_deleted"])

    def test_no_reference_in_view_falls_back_to_rest_not_gt(self):
        masks={k:v.copy() for k,v in self.raw.items()}
        masks["front_visible_mask"]=np.ones((self.T,92),bool)
        masks["side_visible_mask"]=np.ones((self.T,92),bool)
        masks["front_visible_mask"][:,8]=False
        inp={k:masks[k] for k in ("front_uv","side_uv","front_conf","side_conf")}
        out,_,audit=prepare_masked_solver_input(inp,masks,self.rig)
        point=np.asarray([self.rig["keypoints"][8]["rest"]])
        rest_front,_=layout.project(point)
        np.testing.assert_allclose(out["front_uv"][:,8,:],rest_front[0])
        self.assertTrue(np.all(out["front_conf"][:,8]==0))
        self.assertEqual(audit["views"]["front"]["largest_missing_run_frames"],self.T)
        self.assertEqual(len(audit["views"]["front"]["unanchored_joints"]),1)

    def test_reject_shape_dtype_stack_or_wrong_rig(self):
        masked={k:v.copy() for k,v in self.raw.items()}
        masked["front_visible_mask"]=np.ones((self.T,92),bool)
        masked["side_visible_mask"]=np.ones((self.T,92),bool)
        obs={k:masked[k] for k in ("front_uv","side_uv","front_conf","side_conf")}
        masked["side_visible_mask"]=np.ones((self.T,91),bool)
        with self.assertRaisesRegex(ValueError,"boolean"):
            prepare_masked_solver_input(obs,masked,self.rig)
        masked["side_visible_mask"]=np.ones((self.T,92),np.uint8)
        with self.assertRaisesRegex(ValueError,"boolean"):
            prepare_masked_solver_input(obs,masked,self.rig)
        masked["side_visible_mask"]=np.ones((self.T,92),bool)
        masked["front_track_weight"]=np.ones((self.T,92))
        with self.assertRaisesRegex(ValueError,"stack"):
            prepare_masked_solver_input(obs,masked,self.rig)
        masked.pop("front_track_weight")
        self.rig["keypoints"][0]["name"]="wrong"
        with self.assertRaisesRegex(ValueError,"order"):
            prepare_masked_solver_input(obs,masked,self.rig)


if __name__=="__main__":
    unittest.main()
