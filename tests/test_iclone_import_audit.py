"""Verify importer audit with source's real Dragon rig_info and synthetic iClone."""
import json
import pathlib
import unittest

from iclone_adapter.schema import IDENTITY, VERSION
from iclone_adapter.verify_import import DEFAULT_SOURCE, audit_rig, create_mapping


class ImportAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DEFAULT_SOURCE, encoding="utf-8") as f:
            cls.source = json.load(f)

    def _fixture(self):
        # Simulate FBX pipeline preserving all rest bones but adding 100x rig.
        bones = [
            {"id":"skin:/rigged","name":"rigged","path":"/rigged",
             "parent_id":None,"snapshot_local_matrix":list(IDENTITY),
             "snapshot_world_matrix":list(IDENTITY)},
            {"id":"skin:/rigged/rig","name":"rig","path":"/rigged/rig",
             "parent_id":"skin:/rigged","snapshot_local_matrix":list(IDENTITY),
             "snapshot_world_matrix":list(IDENTITY)}
        ]
        paths={}
        for b in self.source["bones"]:
            parent_index=b["parent"]
            parent_name=self.source["bones"][parent_index]["name"] if parent_index>=0 else "rig"
            path=(paths[parent_name] if parent_index>=0 else "/rigged/rig")+"/"+b["name"]
            paths[b["name"]]=path
            mat=[float(v) for row in b["matrix_local"] for v in row]
            scene_mat=list(mat)
            for i in range(3):
                for j in range(4):
                    scene_mat[i*4+j]*=100
            bones.append({"id":"skin:"+path, "name":b["name"],"path":path,
                          "parent_id":"skin:"+(paths[parent_name] if parent_index>=0 else "/rigged/rig"),
                          "snapshot_world_matrix":scene_mat,
                          "snapshot_local_matrix":list(IDENTITY)})
        return {"schema_version":VERSION,"reference":{"kind":"current_snapshot"},
                "bones":bones}

    def test_original_42_bones_aligned_under_100x_scale(self):
        target=self._fixture()
        report=audit_rig(self.source,target)
        self.assertEqual(report["result"],"compatible_static_rest_snapshot")
        self.assertEqual(report["matched_unique_names"],42)
        self.assertEqual(report["matched_parent_relations"],42)
        self.assertAlmostEqual(report["source_to_target_uniform_scale"],100.,places=6)
        self.assertAlmostEqual(report["max_position_error_source_units"],0.,places=7)
        mapping=create_mapping(self.source,target,report)
        self.assertEqual(len(mapping["bone_map"]),41)
        self.assertFalse(any(x["calibrated"] for x in mapping["bone_map"].values()))
        self.assertEqual(mapping["root_policy"],"skip")

    def test_reject_parent_or_pose_deviation(self):
        target=self._fixture()
        for bone in target["bones"]:
            if bone["name"]=="tail2":
                bone["parent_id"]="skin:/rigged/rig"
        report=audit_rig(self.source,target)
        self.assertEqual(report["result"],"needs_mapping_review")
        with self.assertRaises(Exception):
            create_mapping(self.source,target,report)
        target=self._fixture()
        for bone in target["bones"]:
            if bone["name"]=="tail2":
                bone["snapshot_world_matrix"][3]+=100.0
        report=audit_rig(self.source,target)
        self.assertEqual(report["result"],"needs_mapping_review")


if __name__ == "__main__":
    unittest.main()
