"""The same-rig Dragon Preview is deliberately NOT an approved mapping."""
import copy
import json
import unittest

from iclone_adapter.dragon_preview import make_preview
from iclone_adapter.rest_basis import relative_rest_basis
from iclone_adapter.anymove_bridge import (
    from_euler_xyz, qmul, qinverse_unit
)
from iclone_adapter.schema import ContractError, IDENTITY, VERSION
from iclone_adapter.verify_import import DEFAULT_SOURCE


class DragonPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DEFAULT_SOURCE,encoding="utf-8") as f:
            cls.source=json.load(f)

    def make_target(self):
        bones=[
            {"id":"skin:/rigged","path":"/rigged","name":"rigged",
             "parent_id":None, "snapshot_local_matrix":list(IDENTITY),
             "snapshot_world_matrix":list(IDENTITY)},
            {"id":"skin:/rigged/rig","path":"/rigged/rig","name":"rig",
             "parent_id":"skin:/rigged","snapshot_local_matrix":list(IDENTITY),
             "snapshot_world_matrix":list(IDENTITY)}
        ]
        names={}
        for s in self.source["bones"]:
            parent_name=self.source["bones"][s["parent"]]["name"] if s["parent"]>=0 else "rig"
            parent_path=names[parent_name] if s["parent"]>=0 else "/rigged/rig"
            path=parent_path+"/"+s["name"]
            names[s["name"]]=path
            mat=[float(v) for row in s["matrix_local"] for v in row]
            scene=list(mat)
            for i in range(3):
                for j in range(4):
                    scene[4*i+j]*=100.
            bones.append({"id":"skin:"+path,"path":path,"name":s["name"],
                          "parent_id":"skin:"+parent_path,
                          "snapshot_local_matrix":list(IDENTITY),
                          "snapshot_world_matrix":scene})
        return {"schema_version":VERSION,"reference":{"kind":"current_snapshot"},
                "bones":bones}

    def make_motion(self):
        names=[b["name"] for b in self.source["bones"]]
        samples=[[[0.,0.,0.] for _ in names] for _ in range(4)]
        samples[1][names.index("head")]=[0.2,0.,0.]
        samples[2][names.index("tail2")]=[-0.15,0.,0.06]
        return {"bones":names,"fps":24.0,"euler":samples,
                "trans":[[0.,0.,0.]]*4,
                "root_quat":[[1.,0.,0.,0.]]*4}

    def test_preview_keeps_calibration_unverified(self):
        target=self.make_target()
        m=make_preview(self.make_motion(),target,self.source)
        self.assertEqual(m["calibration_status"],
                         "unverified_identity_same_rig_preview")
        self.assertEqual(m["frame_count"],4)
        self.assertEqual(m["fps"],24)
        self.assertEqual(len(m["tracks"]),4)
        self.assertEqual(m["import_validation"]["matched_bones"],42)
        self.assertEqual(m["preview_bones_with_rotation_change"],["head","tail2"])
        name_map={b["name"]:b["id"] for b in target["bones"]}
        self.assertEqual(m["tracks"][name_map["head"]][1],[0.2,0.,0.])
        self.assertNotIn(name_map["pelvis"],m["tracks"])
        self.assertIn("root_translation_skipped",m["limitations"])

    def test_all_mode_includes_nonroot_41(self):
        m=make_preview(self.make_motion(),self.make_target(),self.source,bones="all")
        self.assertEqual(len(m["tracks"]),41)

    def test_all_mode_rest_conversion_uses_same_formula_as_wing_probe(self):
        target=self.make_target()
        m=self.make_motion()
        wing_idx=m["bones"].index("wing_fore_L")
        m["euler"][1][wing_idx]=[0.13,-0.25,0.35]
        raw=make_preview(m,target,self.source,bones="all")
        converted=make_preview(m,target,self.source,bones="all",basis="rest")
        self.assertEqual(len(converted["tracks"]),41)
        self.assertEqual(converted["rotation_basis_mode"],"rest")
        self.assertEqual(converted["calibration_status"],
                         "unverified_rest_basis_conjugation_preview")
        self.assertEqual(len(converted["experimental_rest_quaternion_wxyz"]),41)
        self.assertIn("root_translation_skipped",converted["limitations"])
        self.assertEqual(raw["preview_source_bones"],converted["preview_source_bones"])
        bid=next(b["id"] for b in target["bones"] if b["name"]=="wing_fore_L")
        self.assertNotEqual(raw["tracks"][bid][1],converted["tracks"][bid][1])
        b=relative_rest_basis(self.source,"wing_fore_L")
        q=from_euler_xyz(raw["tracks"][bid][1])
        expected=qmul(qmul(b,q),qinverse_unit(b))
        actual=from_euler_xyz(converted["tracks"][bid][1])
        self.assertAlmostEqual(abs(sum(x*y for x,y in zip(expected,actual))),
                               1.,places=6)
        self.assertEqual(converted["tracks"][bid][0],[0.,0.,0.])
        self.assertEqual(converted["tracks"][bid][3],[0.,0.,0.])

    def test_refuses_unknown_basis(self):
        with self.assertRaises(ContractError):
            make_preview(self.make_motion(),self.make_target(),self.source,
                         basis="made_up")

    def test_refuses_unverified_rest_change(self):
        target=self.make_target()
        for b in target["bones"]:
            if b["name"]=="head":
                b["snapshot_world_matrix"][3]+=50.
        with self.assertRaises(ContractError):
            make_preview(self.make_motion(),target,self.source)

    def test_refuses_wrong_order_wrong_fps_and_static_pose(self):
        target=self.make_target()
        m=self.make_motion()
        m["bones"]=list(reversed(m["bones"]))
        with self.assertRaises(ContractError):
            make_preview(m,target,self.source)
        m=self.make_motion()
        m["fps"]=29.97
        with self.assertRaises(ContractError):
            make_preview(m,target,self.source)
        m=self.make_motion()
        for frame in m["euler"]:
            for row in frame:
                row[:]=[0.,0.,0.]
        with self.assertRaises(ContractError):
            make_preview(m,target,self.source)

    def test_rejects_root_unknown_duplicates(self):
        for val in ("pelvis","unknown","head,head","","head,unknown"):
            with self.subTest(val=val):
                with self.assertRaises(ContractError):
                    make_preview(self.make_motion(),self.make_target(),
                                 self.source,bones=val)


if __name__=="__main__":
    unittest.main()
