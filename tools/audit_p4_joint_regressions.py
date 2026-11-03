"""P4 independent synthetic blind-test 3D per-joint and region regression audit.

Reports both improvements and worst regressions. Do not use this blind
audit for tuning parameters, since doing so converts it into validation.
"""
import argparse,json
from pathlib import Path
import numpy as np


def group(name,ancestors=()):
    """Categorize by Rig ancestry; wing fingers are not body/root bones."""
    s=' '.join([name,*ancestors]).lower()
    if 'wing' in s:return 'wings'
    if 'tail' in s:return 'tail'
    if any(k in s for k in ('foot','toe','calf','thigh','shin','metatarsus','hpaw','hock','knee','hipleg')):return 'hind_legs_feet'
    if any(k in s for k in ('upperarm','forearm','clavicle','hand','index','mid','metacarpus','scapula','fpaw')):return 'forelimbs'
    if any(k in s for k in ('head','neck','jaw','mouth','tongue','eye')):return 'head_neck'
    return 'body_root'


def stats(x):
    x=np.asarray(x,float)
    return {'mean_mm':round(float(x.mean()),4),
            'p95_mm':round(float(np.percentile(x,95)),4),
            'max_mm':round(float(x.max()),4)}


def run(repo,cr):
    root=Path(repo).resolve()
    blind=root/'runs'/cr/'p4_blind_fourview_v1'
    orig=blind/'p4_original_fk_2v.npz'
    hybrid=blind/'p4_hybrid_fk_4v.npz'
    gt=blind/'p4_original_blind_reader.ground_truth.npz'
    rig=root/'assets'/cr/'rigged/rig_info.json'
    info=json.loads(rig.read_text(encoding='utf8'))
    with np.load(gt,allow_pickle=False) as z:
        truth=z['joints_world'].copy()
        frame_ids=z['frame_ids'].copy()
        names=z['names'].tolist()
    with np.load(orig,allow_pickle=False) as z:
        o=z['world_joints'].copy()
        assert np.array_equal(z['frame_ids'],frame_ids)
    with np.load(hybrid,allow_pickle=False) as z:
        h=z['world_joints'].copy()
        assert np.array_equal(z['frame_ids'],frame_ids)
    error_o=np.linalg.norm(o-truth,axis=-1)*1000
    error_h=np.linalg.norm(h-truth,axis=-1)*1000
    if names!=[x['name'] for x in info['keypoints']]:
        raise ValueError('3D test and Rig keypoint names mismatch')
    # Classify by complete ancestry, not just localized bone-name tokens.
    # Dragon fingerN_L belongs to wing_fore_L; fpaw belongs to forelimb.
    ancestry=[]
    for b in info['bones']:
        lineage=[]
        parent=b['parent']
        while parent>=0:
            lineage.append(info['bones'][parent]['name'])
            parent=info['bones'][parent]['parent']
        ancestry.append(lineage)
    region={}
    perbone={}
    for idx,kp in enumerate(info['keypoints']):
        bone_index=kp['bone']
        bone=info['bones'][bone_index]['name']
        cat=group(bone,ancestry[bone_index])
        region.setdefault(cat,[]).append(idx)
        perbone.setdefault(bone,[]).append(idx)
    sections={}
    for k,ids in sorted(region.items()):
        a=error_o[:,ids]
        b=error_h[:,ids]
        sections[k]={
            'keypoint_count':len(ids),
            'original':stats(a),'hybrid':stats(b),
            'delta_mean_mm':round(float(b.mean()-a.mean()),4),
            'pose_improved_count':int(np.sum(b.mean(axis=1)<a.mean(axis=1))),
        }
    joint_errors=[]
    for j,(name,boneref) in enumerate(zip(names,[info['bones'][p['bone']]['name'] for p in info['keypoints']])):
        a=float(error_o[:,j].mean());b=float(error_h[:,j].mean())
        joint_errors.append({'joint':name,'bone':boneref,
             'original_mean_mm':round(a,3),
             'hybrid_mean_mm':round(b,3),
             'delta_mean_mm':round(b-a,3),
             'hybrid_max_mm':round(float(error_h[:,j].max()),2)})
    worst=np.unravel_index(np.argmax(error_h),error_h.shape)
    worst_o=np.unravel_index(np.argmax(error_o),error_o.shape)
    return {
        'schema':'anymove.p4_independent_gt_3d_region_audit/1',
        'creature':cr,'poses':len(frame_ids),'keypoints':len(names),
        'heldout_pose_ids':frame_ids.tolist(),
        'region_error':sections,
        'largest_mean_regressions':sorted(joint_errors,key=lambda x:x['delta_mean_mm'],reverse=True)[:15],
        'largest_mean_improvements':sorted(joint_errors,key=lambda x:x['delta_mean_mm'])[:15],
        'worst_original_3d_error':{
            'pose':int(frame_ids[worst_o[0]]),'joint':names[worst_o[1]],
            'mm':float(error_o[worst_o])},
        'worst_hybrid_3d_error':{
            'pose':int(frame_ids[worst[0]]),'joint':names[worst[1]],
            'mm':float(error_h[worst])},
        'not_independent_if_used_to_tune':True,
    }


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--repo',default='.')
    a.add_argument('--creature',choices=('dino','dragon'),required=True)
    a.add_argument('--out',required=True)
    args=a.parse_args()
    dst=Path(args.out)
    if dst.exists():raise FileExistsError('region audit already present')
    result=run(args.repo,args.creature)
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'creature':result['creature'],'region_error':result['region_error'],
                      'worst_original_3d_error':result['worst_original_3d_error'],
                      'worst_hybrid_3d_error':result['worst_hybrid_3d_error'],
                      'largest_regressions':result['largest_mean_regressions'][:6]},indent=2))


if __name__=='__main__':main()
