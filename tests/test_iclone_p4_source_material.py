"""P4-A/B plan, specialized observation selection and independent test gates."""
import json,unittest
from pathlib import Path
import numpy as np
from mocap.fourview_data import FourViewDataset
from tools.plan_p4_source_material import select,sha

ROOT=Path(__file__).resolve().parents[1]
class P4PlanTests(unittest.TestCase):
    def test_all_plans_are_reproducible_and_exclude_old_p3(self):
        for creature,source in (('dino','dataset_hr_v1'),('dragon','dataset_hr')):
            path=ROOT/'runs'/creature
            p=json.loads((path/'p4_source_material_split_v1.json').read_text())
            again=select(path/source,creature,seed=41111)
            self.assertEqual(p['train_ids'],again['train_ids'])
            self.assertEqual(p['val_ids'],again['val_ids'])
            self.assertEqual(p['pilot_ids_render'],again['pilot_ids_render'])
            self.assertEqual(len(p['train_ids']),240)
            self.assertEqual(len(p['val_ids']),36)
            self.assertEqual(len(p['pilot_train_ids']),64)
            self.assertEqual(len(p['pilot_val_ids']),12)
            self.assertFalse(set(p['train_ids'])&set(p['val_ids']))
            self.assertTrue(set(p['pilot_train_ids'])<=set(p['train_ids']))
            self.assertTrue(set(p['pilot_val_ids'])<=set(p['val_ids']))
            self.assertTrue(all(x>100 and x%25!=0 for x in p['train_ids']))
            self.assertTrue(all(x>=125 and x%25==0 for x in p['val_ids']))
            self.assertTrue(set(p['test_ids']).isdisjoint(p['train_ids']+p['val_ids']))
            self.assertTrue(all(x>=9001 for x in p['test_ids']))
            for i in p['pilot_ids_render']:
                m=json.loads((path/source/f'{i:05d}_meta.json').read_text())
                self.assertFalse(m['blur'])
                self.assertEqual(m['scale'],1.)

    def test_specialized_loader_only_trains_two_extra_views(self):
        for creature,source in (('dino','dataset_fourview_p3_material_fullres_v1'),
                                ('dragon','dataset_fourview_p3_material_fullres_v1')):
            base=ROOT/'runs'/creature/source
            rig=ROOT/'assets'/creature/'rigged/rig_info.json'
            ds=FourViewDataset(base,rig,[1],train=True,views=('back','opposite_side'))
            self.assertEqual(ds.views,('back','opposite_side'))
            self.assertEqual(len(ds),2)
            self.assertEqual([int(ds[i]['view_index']) for i in range(2)],[2,3])
            self.assertEqual(ds.appearance_mode,'source-material')
            with self.assertRaisesRegex(ValueError,'invalid selected'):
                FourViewDataset(base,rig,[1],train=True,views=('back','back'))

    def test_original_source_reused_readonly_without_generating_new_front(self):
        for creature,i,source in [('dino',184,'dataset_hr_v1'),
                                   ('dragon',269,'dataset_hr')]:
            root=ROOT/'runs'/creature
            pilot=root/'p4_reuse_sanity_v1'
            if not (pilot/f'{i:05d}_visibility_meta.json').is_file():
                self.skipTest('P4 source image reuse smoke not available')
            m=json.loads((pilot/f'{i:05d}_visibility_meta.json').read_text())
            self.assertTrue(m['source_front_side_reused'])
            self.assertTrue(m['source_material_colors_preserved'])
            self.assertEqual(m['render_scale'],1.)
            for view in ('front','side'):
                self.assertEqual(sha(root/source/f'{i:05d}_{view}.png'),
                                 sha(pilot/f'{i:05d}_{view}.png'))
            for view in ('back','opposite_side'):
                self.assertTrue((pilot/f'{i:05d}_{view}.png').exists())
                self.assertTrue((pilot/f'{i:05d}_{view}_labels.npz').exists())

    def test_novel_test_gt_metadata_is_seed_disjoint_and_not_training(self):
        for creature,source in [('dino','dataset_hr_v1'),
                                ('dragon','dataset_hr')]:
            root=ROOT/'runs'/creature
            plan=json.loads((root/'p4_source_material_split_v1.json').read_text())
            for i in plan['test_ids']:
                file=root/'p4_blind_source_pose_v1'/f'{i:05d}_meta.json'
                if not file.exists():
                    self.skipTest('new blind source poses not yet generated')
                m=json.loads(file.read_text())
                self.assertEqual(m['schema'],'anymove.p4_independent_3d_blender_pose/1')
                self.assertGreater(m['sample_seed'],9000)
                self.assertTrue(m['not_model_training'])
                self.assertTrue(m['not_natural_motion'])
                self.assertFalse(m['blur'])
                self.assertEqual(m['scale'],1.)
                self.assertEqual(len(m['joints_world']),
                     92 if creature=='dino' else 50)

    def test_pilot_checkpoint_preserves_original_hash(self):
        root=ROOT/'runs/dino'
        path=root/'reader_backopp_p4_specialized_smoke_v1.validation.json'
        if not path.exists():self.skipTest('P4 pilot model not yet trained')
        x=json.loads(path.read_text())
        self.assertEqual(x['train_views'],['back','opposite_side'])
        self.assertEqual(x['checkpoint_selection_metric'],'back_opposite')
        self.assertEqual(x['input_checkpoint_sha256'],sha(root/'reader_hr_v1.pt'))
        self.assertEqual(x['validation_pose_ids'],[25,50,75,100])


if __name__=='__main__':unittest.main()
