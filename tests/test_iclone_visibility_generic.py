"""P0 creature-generic two-view reliability regression tests.

These validate abstention, character-independent rig topology,
camera-aligned geometric checks, side/front weighting, and immutability.
They do NOT claim true mesh occlusion or 3D motion accuracy.
"""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from mocap.visibility_generic import PROFILE, generic_view_weights
from mocap.solve import triangulate


def arbitrary_rig():
    # The names intentionally match neither Dragon nor Dino.
    base = [
        {"name":"core","parent":-1,"head_joint":"origin","tail_joint":"origin"},
        {"name":"appendage_A","parent":0,"head_joint":"a","tail_joint":"a_tip"},
        {"name":"appendage_B","parent":0,"head_joint":"b","tail_joint":"b_tip"},
    ]
    points=[
        {"name":"origin","bone":0,"end":"head","rest":[0,0,0]},
        {"name":"a","bone":1,"end":"head","rest":[10,0,0]},
        {"name":"a_tip","bone":1,"end":"tail","rest":[10,0,10]},
        {"name":"b","bone":2,"end":"head","rest":[-10,0,0]},
        {"name":"b_tip","bone":2,"end":"tail","rest":[-10,0,10]},
    ]
    return {
        "bones":base,"keypoints":points,
        "layout":{"H":720,"W_FRONT":400,"W_SIDE":880,
                  "M_PER_PX":.01,"Z_CENTER":3.2},
    }


def observations(T=20):
    names=[k["name"] for k in arbitrary_rig()["keypoints"]]
    uv=np.array([[200,400],[100,310],[100,270],[300,310],[300,270]],float)
    d={"front_uv":np.broadcast_to(uv,(T,5,2)).copy(),
       "side_uv":np.broadcast_to(uv,(T,5,2)).copy(),
       "front_conf":np.full((T,5),.8),
       "side_conf":np.full((T,5),.8)}
    return d,names


class GenericVisibilityTests(unittest.TestCase):
    def test_no_evidence_abstains_both_views(self):
        d,n=observations()
        before={k:v.copy() for k,v in d.items()}
        w,r=generic_view_weights(d,n,arbitrary_rig())
        np.testing.assert_array_equal(w["front"],1.)
        np.testing.assert_array_equal(w["side"],1.)
        self.assertEqual(r["profile"],PROFILE)
        self.assertTrue(r["not_true_mesh_occlusion"])
        self.assertTrue(r["not_single_view_method"])
        self.assertEqual(r["views"]["front"]["joint_frames_downweighted"],0)
        for k in d:np.testing.assert_array_equal(d[k],before[k])

    def test_view_specific_overlap_downweights_only_that_view(self):
        d,n=observations()
        ia=n.index("a_tip")
        # A tip is cloned on top of remote B tip in Front, and fails shared Z;
        # side contains separated A tip. All segments still satisfy bone length.
        d["front_uv"][5:16,ia]=d["front_uv"][5:16,n.index("b_tip")]
        d["side_uv"][5:16,ia,1]+=42.
        w,r=generic_view_weights(d,n,arbitrary_rig())
        self.assertLess(float(w["front"][10,ia]),.92)
        self.assertEqual(float(w["side"][10,ia]),1.0)
        self.assertTrue((w["front"]>=.45).all() and (w["side"]>=.45).all())
        self.assertEqual(float(w["front"][10,n.index("origin")]),1.)

    def test_2d_overlong_bone_is_geometrically_unreliable(self):
        d,n=observations()
        j=n.index("b_tip")
        d["front_uv"][5:17,j,0]+=1500.
        w,r=generic_view_weights(d,n,arbitrary_rig())
        self.assertLess(float(w["front"][10,j]),.5)
        self.assertEqual(float(w["side"][10,j]),1.)
        self.assertGreater(r["views"]["front"]["max_projected_excess_m"],1)

    def test_bad_shapes_unknown_names_and_nonfinite_fail_closed(self):
        d,n=observations()
        with self.assertRaisesRegex(ValueError,"exact rig keypoint order"):
            generic_view_weights(d,n[::-1],arbitrary_rig())
        d2={k:v.copy() for k,v in d.items()}
        d2["side_uv"][2,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            generic_view_weights(d2,n,arbitrary_rig())
        wrong=arbitrary_rig()
        wrong["layout"]["M_PER_PX"]=-.5
        with self.assertRaisesRegex(ValueError,"positive"):
            generic_view_weights(d,n,wrong)
        with self.assertRaisesRegex(ValueError,"odd positive"):
            generic_view_weights(d,n,arbitrary_rig(),median_window=2)

    def test_temporary_custom_names_are_not_required(self):
        rig=arbitrary_rig()
        d,n=observations()
        mapping={x:"unrelated_%d"%i for i,x in enumerate(n)}
        for p in rig["keypoints"]:p["name"]=mapping[p["name"]]
        for b in rig["bones"]:
            b["head_joint"]=mapping[b["head_joint"]]
            b["tail_joint"]=mapping[b["tail_joint"]]
        w,r=generic_view_weights(d,[mapping[x] for x in n],rig)
        self.assertEqual(w["front"].shape,(20,5))
        self.assertGreaterEqual(r["anatomy"]["eligible_distant_keypoint_pairs"],2)

    def test_solver_side_weights_change_triangulation_as_expected(self):
        import os
        # Keep original layout constants, this only checks relative Z mixture.
        d,n=observations()
        d["front_uv"][:,0,1]=300
        d["side_uv"][:,0,1]=340
        base=triangulate(d)
        ones=np.ones((len(d["front_uv"]),len(n)))
        np.testing.assert_allclose(base,triangulate(d,side_visibility=ones))
        s=np.ones_like(ones);s[:,0]=.5
        adjusted=triangulate(d,side_visibility=s)
        # Front v smaller -> higher world Z, reducing Side means higher Z.
        self.assertGreater(float(adjusted[4,0,2]),float(base[4,0,2]))
        with self.assertRaisesRegex(ValueError,"side visibility"):
            triangulate(d,side_visibility=np.full_like(s,np.nan))

    def test_both_real_creatures_use_the_same_api(self):
        base=Path(__file__).resolve().parents[1]
        for creature,clip in (("dragon","hr_roar"),("dino","test_walk")):
            rig=base/"assets"/creature/"rigged"/"rig_info.json"
            det=base/"runs"/creature/"clips"/clip/"det.npz"
            if not rig.is_file() or not det.is_file():
                self.skipTest("requires local "+creature+" rig/Reader")
            info=json.loads(rig.read_text(encoding="utf8"))
            with np.load(det) as raw:
                d={k:raw[k].copy() for k in
                   ("front_uv","side_uv","front_conf","side_conf")}
                names=list(raw["names"])
            weights,audit=generic_view_weights(d,names,info)
            with self.subTest(creature=creature):
                self.assertEqual(weights["front"].shape,(124,len(info["keypoints"])))
                self.assertEqual(weights["side"].shape,(124,len(info["keypoints"])))
                self.assertLess(float(weights["front"].mean()),1.)
                self.assertLess(float(weights["side"].mean()),1.)
                self.assertEqual(audit["num_bones"],len(info["bones"]))
                self.assertEqual(audit["not_single_view_method"],True)
                json.dumps(audit)


if __name__=="__main__":
    unittest.main()
