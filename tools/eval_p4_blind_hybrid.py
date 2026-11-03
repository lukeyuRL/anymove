"""P4 blind-test 3D evaluation for original two-view vs hybrid four-view.

The blind test comprises 12 newly generated static 3D poses (IDs 9001..9012)
from independent seeds NEVER in the 2,200 training/validation poses.
No GT is supplied to either Reader or the N-camera FK Solver.
The paired bootstrap resamples POSES, not joint measurements.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()


def stats(err):
    a=np.asarray(err,float)
    return {'mean_mm':round(float(np.mean(a)),4),
            'median_mm':round(float(np.median(a)),4),
            'p95_mm':round(float(np.percentile(a,95)),4),
            'max_mm':round(float(np.max(a)),4)}


def compare(repo,creature,*,boot_count=4000,seed=41111):
    root=Path(repo).resolve()
    work=root/'runs'/creature
    blind=work/'p4_blind_fourview_v1'
    audit=json.loads((blind/'blind_audit_v1.json').read_text(encoding='utf8'))
    plan=json.loads((work/'p4_source_material_split_v1.json').read_text(encoding='utf8'))
    if not audit['new_synthetic_blind_poses'] or audit['status']!='PASS':
        raise ValueError('P4 independent blind source metadata audit not PASS')
    if set(plan['test_ids'])&(set(plan['train_ids'])|set(plan['val_ids'])):
        raise ValueError('test ID crossed P4 train/validation')
    original=blind/'p4_original_blind_reader.npz'
    hybrid=blind/'p4_hybrid_blind_reader.npz'
    obs_orig,obs_new={},{}
    for filename,target in ((original,obs_orig),(hybrid,obs_new)):
        with np.load(filename,allow_pickle=False) as z:
            for key in z.files:
                target[key]=z[key].copy()
    for key in ('frame_ids','names','world_to_camera','image_size_px','meters_per_px'):
        if not np.array_equal(obs_orig[key],obs_new[key]):
            raise ValueError('original and hybrid camera/pose mismatched')
    if not np.array_equal(obs_orig['frame_ids'],plan['test_ids']):
        raise ValueError('source GT/test ID leaked or changed')
    np.testing.assert_array_equal(obs_orig['uv'][:2],obs_new['uv'][:2])
    np.testing.assert_array_equal(obs_orig['confidence'][:2],obs_new['confidence'][:2])
    manifest=json.loads((hybrid.with_suffix('.manifest.json')).read_text())
    if manifest['original_input_sha256']!=sha(original):
        raise ValueError('original 2D packet changed')
    eval_gt=blind/'p4_original_blind_reader.ground_truth.npz'
    with np.load(eval_gt,allow_pickle=False) as z:
        world=z['joints_world'].copy()
        exact=z['exact_uv'].copy()
        view_label=z['part_visibility'].copy()
        names=z['names'].tolist()
        ids=z['frame_ids'].copy()
    if world.shape[:2]!=(12,92 if creature=='dino' else 50):
        raise ValueError('unexpected 12-pose blind GT shape')
    if not np.array_equal(ids,obs_orig['frame_ids']):
        raise ValueError('GT/test ID order mismatch')
    if names!=obs_orig['names'].tolist():
        raise ValueError('GT/keypoint ordering mismatch')
    outputs={
        'original_two':blind/'p4_original_fk_2v.npz',
        'original_four':blind/'p4_original_fk_4v.npz',
        'hybrid_two':blind/'p4_hybrid_fk_2v.npz',
        'hybrid_four':blind/'p4_hybrid_fk_4v.npz',
    }
    results={}
    by_pose={}
    for tag,p in outputs.items():
        with np.load(p,allow_pickle=False) as z:
            values={key:z[key].copy() for key in z.files}
        if values['world_joints'].shape!=world.shape or not np.array_equal(values['frame_ids'],ids):
            raise ValueError('P4 FK output/3D GT frames differ: '+tag)
        if values['names'].tolist()!=names:
            raise ValueError('FK saved keypoint names mismatch')
        expected=['front','side'] if tag.endswith('two') else ['front','side','back','opposite_side']
        if values['view_ids'].tolist()!=expected:
            raise ValueError('selected cameras mismatch: '+tag)
        error=np.linalg.norm(values['world_joints']-world,axis=-1)*1000
        results[tag]=stats(error)
        by_pose[tag]=error.mean(axis=1)
        results[tag]['per_pose_mean_mm']=np.round(error.mean(axis=1),4).tolist()
        results[tag]['frames']=len(ids)
    # Exact check: hybrid has identical original selected-view observations.
    with np.load(outputs['hybrid_two'],allow_pickle=False) as z:
        ht=z['world_joints'].copy()
    with np.load(outputs['original_two'],allow_pickle=False) as z:
        ot=z['world_joints'].copy()
    identical_2view=float(np.max(np.linalg.norm(ht-ot,axis=-1))*1000)
    if identical_2view>1e-5:
        raise ValueError('P4 hybrid changed original Front/Side reference')
    delta=by_pose['hybrid_four']-by_pose['original_two']
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(ids),size=(boot_count,len(ids)))
    bootstrap=np.mean(delta[indices],axis=1)
    lo,hi=np.percentile(bootstrap,[2.5,97.5])
    # Correction to view-aware risk: the quality score is intentionally not
    # used in the solver yet; evaluate whether it is trustworthy first.
    p2d=np.linalg.norm(obs_new['uv']-exact,axis=-1)
    input_stats={}
    for vi,v in enumerate(('front','side','back','opposite_side')):
        have=obs_new['valid'][vi]
        input_stats[v]=stats((p2d[vi][have])*1) # pixel values, renamed below
        input_stats[v]={'mean_px':input_stats[v]['mean_mm'],
                        'p95_px':input_stats[v]['p95_mm'],
                        'num_observed':int(have.sum()),
                        'mesh_unknown_fraction':round(float((view_label[vi]==-1).mean()),4)}
    report={
        'schema':'anymove.p4_independent_blind_hybrid_3d/1',
        'creature':creature,
        'source_pose_ids':ids.tolist(),
        'source_3d_gt_only_in_evaluator':True,
        'source_pose_gen_seed_base':820240,
        'independent_new_procedural_pose_not_motion':True,
        'hybrid_original_front_side_2d_unchanged':True,
        'original_two_vs_hybrid_two_max_joint_difference_mm':round(identical_2view,8),
        'reader_input_2d_quality_px':input_stats,
        '3d_fk_error':results,
        'hybrid_four_minus_original_two_paired_mean_mm':round(float(delta.mean()),4),
        'paired_95pct_pose_bootstrap_CI_mm':[round(float(lo),4),round(float(hi),4)],
        'paired_improved_pose_count':int((delta<0).sum()),
        'blind_synthetic_static_pose_count':len(ids),
        'suggested_gate_3d_mean_and_ci_both_improve':bool(delta.mean()<0 and hi<0),
        'not_enough_for_natural_motion_promotion':True,
        'temporal_motion_evaluation_performed':False,
        'no_new_h3_video_generated':True,
    }
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',default='.')
    ap.add_argument('--creature',choices=('dino','dragon'),required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    dest=Path(args.out)
    if dest.exists():raise FileExistsError('P4 blind 3D report exists')
    result=compare(args.repo,args.creature)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='3d_fk_error'},indent=2))
    print('3D_ERROR',{k:{a:b for a,b in v.items() if a!='per_pose_mean_mm'}
                       for k,v in result['3d_fk_error'].items()})


if __name__=='__main__':main()
