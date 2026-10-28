"""Full source-to-FK Dragon import package with reference-only profile."""
import copy
import json
import pathlib
import tempfile
import unittest

import test_iclone_dragon_preview as fixtures
from iclone_adapter.import_pipeline import (
    PROFILE_ID, OUTPUT_MOTION, OUTPUT_AUDIT, OUTPUT_MANIFEST,
    build_bundle, save_bundle, verify_bundle
)
from iclone_adapter.schema import ContractError
from iclone_adapter.verify_import import DEFAULT_SOURCE


class SameRigBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DragonPreviewTests.setUpClass()
        cls.info=fixtures.DragonPreviewTests.source
        cls.dragon=fixtures.DragonPreviewTests()

    def target(self):
        return self.dragon.make_target()

    def moving_motion(self,frames=124):
        original=self.dragon.make_motion()
        names=original["bones"]
        original["fps"]=24.
        original["euler"]=[]
        original["root_quat"]=[]
        original["trans"]=[]
        for t in range(frames):
            original["euler"].append([
                [(.004*t) if j==names.index("head") else 0.,
                 (.001*t) if j==names.index("wing_fore_L") else 0.,0.]
                for j in range(len(names))])
            original["root_quat"].append([1.,0.,0.,0.])
            original["trans"].append([.001*t,.005*t,-.003*t])
        return original

    def test_entire_124_frame_clip_transforms_42_bones(self):
        source=self.moving_motion()
        motion,audit,manifest=build_bundle(source,self.target(),self.info,
                                           self.info)
        self.assertEqual(manifest["profile_id"],PROFILE_ID)
        self.assertEqual(audit["result"],"all_serialized_frames_match_source_math")
        self.assertEqual(audit["frames_checked"],124)
        self.assertEqual(audit["rotation_samples_checked"],5208)
        self.assertEqual(audit["root_position_samples_checked"],124)
        self.assertEqual(len(motion["tracks"]),42)
        self.assertEqual(motion["root_motion"]["root_position_space"],"pelvis_local")
        self.assertEqual(motion["root_motion"]["root_position_scale"],1.0)
        self.assertTrue(motion["calibration_status"].startswith("unverified_"))
        self.assertLess(audit["maximum_quaternion_error_deg"],.001)
        self.assertLess(audit["maximum_root_position_error_source_units"],1e-8)
        self.assertEqual(manifest["observed_evidence"]["other_avatars_or_rigs"],
                         "NOT_VERIFIED")

    def test_save_verify_and_integrity_fail_closed(self):
        source=self.moving_motion(14)
        with tempfile.TemporaryDirectory() as temp:
            output=pathlib.Path(temp)/"dragon_import"
            save_bundle(output,source,self.target(),self.info,self.info)
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {OUTPUT_MOTION,OUTPUT_AUDIT,OUTPUT_MANIFEST})
            result=verify_bundle(output,source,self.target(),self.info,self.info)
            self.assertEqual(result["frames"],14)
            with self.assertRaisesRegex(ContractError,"directory exists"):
                save_bundle(output,source,self.target(),self.info,self.info)
            altered=json.loads((output/OUTPUT_MOTION).read_text(encoding="utf-8"))
            k=next(iter(altered["tracks"]))
            altered["tracks"][k][3][0]+=.4
            (output/OUTPUT_MOTION).write_text(json.dumps(altered),encoding="utf-8")
            with self.assertRaisesRegex(ContractError,"payload hash mismatch"):
                verify_bundle(output,source,self.target(),self.info,self.info)

    def test_overwrite_safe_and_reject_unrelated_files(self):
        source=self.moving_motion(8)
        with tempfile.TemporaryDirectory() as temp:
            folder=pathlib.Path(temp)/"bundle"
            save_bundle(folder,source,self.target(),self.info,self.info)
            save_bundle(folder,source,self.target(),self.info,self.info,
                        overwrite=True)
            self.assertTrue((folder/OUTPUT_MANIFEST).exists())
            (folder/"important_user_file.txt").write_text("private")
            with self.assertRaisesRegex(ContractError,"unrelated files"):
                save_bundle(folder,source,self.target(),self.info,self.info,
                            overwrite=True)

    def test_reject_incompatible_source_or_wrong_scale(self):
        source=self.moving_motion(4)
        target=self.target()
        original=copy.deepcopy(self.info)
        original["bones"][2]["matrix_local"][0][3]+=.003
        with self.assertRaisesRegex(ContractError,"exact checked-in"):
            build_bundle(source,target,original,self.info)
        target=self.target()
        for entry in target["bones"]:
            if entry["name"]=="head":
                entry["snapshot_world_matrix"][3]+=10.
        with self.assertRaises(ContractError):
            build_bundle(source,target,self.info,self.info)
        target=self.target()
        for entry in target["bones"]:
            for row in (0,1,2):
                for column in range(4):
                    entry["snapshot_world_matrix"][row*4+column]*=.5
        with self.assertRaisesRegex(ContractError,"100x"):
            build_bundle(source,target,self.info,self.info)

    def test_reject_other_rig_fps_or_root_missing(self):
        source=self.moving_motion(4)
        bad=copy.deepcopy(source)
        bad["fps"]=30.
        with self.assertRaisesRegex(ContractError,"24 FPS"):
            build_bundle(bad,self.target(),self.info,self.info)
        bad=copy.deepcopy(source)
        bad["bones"]=list(reversed(bad["bones"]))
        with self.assertRaisesRegex(ContractError,"Bone names/order"):
            build_bundle(bad,self.target(),self.info,self.info)
        bad=copy.deepcopy(source)
        bad["root_quat"]=bad["root_quat"][:2]
        with self.assertRaises(ContractError):
            build_bundle(bad,self.target(),self.info,self.info)

    def test_verify_blocks_mismatched_source_even_if_bundle_untouched(self):
        source=self.moving_motion(9)
        with tempfile.TemporaryDirectory() as temp:
            folder=pathlib.Path(temp)/"bundle"
            save_bundle(folder,source,self.target(),self.info,self.info)
            modified=copy.deepcopy(source)
            modified["euler"][3][7][0]+=.01
            with self.assertRaisesRegex(ContractError,"does not match current input"):
                verify_bundle(folder,modified,self.target(),self.info,self.info)


if __name__=="__main__":
    unittest.main()
