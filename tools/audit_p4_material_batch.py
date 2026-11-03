"""Independent P4-A batch audit: pose split, source sha and materials.

- A pilot dataset uses existing *unblurred* original Front/Side images and
  new material-coloured Back/Opposite camera renders.
- A genuinely untouched test dataset uses new synthetic Blender 3D pose
  metadata with independent seeds/IDs and four new render images.
- All 3D positions remain outside model input; no learned model in this audit.
"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mocap.fourview_data import sha256

VIEWS=('front','side','back','opposite_side')


def batch_audit(repo,creature,purpose,*,expect):
    root=Path(repo).resolve()
    data=root/'runs'/creature
    plan=json.loads((data/'p4_source_material_split_v1.json').read_text())
    if purpose=='pilot':
        ids=plan['pilot_ids_render']
        src=data/('dataset_hr_v1' if creature=='dino' else 'dataset_hr')
        dst=data/'p4_source_material_pilot_v1'
    elif purpose=='blind':
        ids=plan['test_ids']
        src=data/'p4_blind_source_pose_v1'
        dst=data/'p4_blind_fourview_v1'
    else:
        raise ValueError('P4 expected pilot or blind')
    if len(ids)!=expect or len(ids)!=len(set(ids)) or not dst.exists():
        raise ValueError('P4 unexpectedly incomplete batch selection')
    bones=92 if creature=='dino' else 50
    views=0
    min_satur=1.
    max_reproj=0.
    max_raycast=0.
    vis_unknown=[]
    reused_count=0
    for i in ids:
        meta_path=dst/f'{i:05d}_visibility_meta.json'
        if not meta_path.is_file():raise ValueError('missing P4 pose '+str(meta_path))
        m=json.loads(meta_path.read_text(encoding='utf8'))
        if (m['appearance_mode']!='source-material' or
                not m['source_material_colors_preserved'] or
                m['render_scale']!=1 or
                m['source_meta_sha256']!=sha256(src/f'{i:05d}_meta.json') or
                m['source_front_side_reused']!=(purpose=='pilot')):
            raise ValueError('P4 source material provenance mismatch: '+str(i))
        if m['replay_max_joint_world_error_m']>.001:
            raise ValueError('3D source FK replay mismatch')
        original=json.loads((src/f'{i:05d}_meta.json').read_text())
        if purpose=='pilot' and (original['blur'] or i<=100):
            raise ValueError('pilot dataset accidentally includes blur/P3')
        if purpose=='blind' and (original.get('schema')!='anymove.p4_independent_3d_blender_pose/1' or
                                 original['sample_seed']<100000 or
                                 not original['not_model_training']):
            raise ValueError('not an independent 3D test pose')
        max_reproj=max(max_reproj,m['replay_max_joint_world_error_m'])
        for v in VIEWS:
            d=m['views'][v]
            png=dst/f'{i:05d}_{v}.png'
            npz=dst/f'{i:05d}_{v}_labels.npz'
            if not png.is_file() or not npz.is_file():
                raise ValueError('P4 missing image/labels '+str(i)+' '+v)
            raw=cv2.imread(str(png),cv2.IMREAD_UNCHANGED)
            w,h=d['image_size_px']
            if raw is None or raw.shape!=(h,w,4):
                raise ValueError('P4 image size or RGBA mismatch')
            if w not in (int(m['views']['front']['image_size_px'][0]),
                          int(m['views']['side']['image_size_px'][0])) or h!=720:
                raise ValueError('P4 resolution is not full original size')
            if purpose=='pilot' and v in ('front','side'):
                if sha256(png)!=sha256(src/f'{i:05d}_{v}.png'):
                    raise ValueError('P4 reused original image was modified')
                reused_count+=1
            with np.load(npz,allow_pickle=False) as item:
                labs=item['joint_surface_part_visible']
                if labs.shape!=(bones,) or not np.isin(labs,[-1,0,1]).all():
                    raise ValueError('P4 surface-part visibility labels corrupt')
                vis_unknown.append(float((labs<0).mean()))
                if item['depth_m'].shape!=(h,w) or item['part_id'].shape!=(h,w):
                    raise ValueError('P4 surface raster dimensions mismatch')
            hsv=cv2.cvtColor(raw[:,:,:3],cv2.COLOR_BGR2HSV)
            foreground=raw[:,:,3]>64
            sat=foreground&(hsv[:,:,1]>110)
            min_satur=min(min_satur,float(sat.sum()/max(foreground.sum(),1)))
            ray=d['raycast_validation']['depth_error_m_p95']
            max_raycast=max(max_raycast,float(ray))
            if ray>.03 or not np.isfinite(ray):
                raise ValueError('P4 mesh depth not verified')
            views+=1
    return {
        'schema':'anymove.p4_material_batch_audit/1',
        'creature':creature,'purpose':purpose,
        'status':'PASS',
        'pose_ids':ids,'source_pose_count':len(ids),'view_count':views,
        'all_rgba_images_full_resolution':True,
        'source_front_side_reused_image_count':reused_count,
        'new_synthetic_blind_poses':purpose=='blind',
        'max_replay_3d_gt_error_m':max_reproj,
        'max_raycast_p95_error_m':max_raycast,
        'min_saturated_foreground_fraction':round(min_satur,6),
        'mean_unknown_surface_part_fraction':round(float(np.mean(vis_unknown)),6),
        'not_a_natural_motion_video':True,
        'no_h3_generated':True
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--creature',choices=('dino','dragon'),required=True)
    p.add_argument('--purpose',choices=('pilot','blind'),required=True)
    p.add_argument('--repo',default='.')
    p.add_argument('--expected',type=int,required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('P4 audit exists already')
    d=batch_audit(a.repo,a.creature,a.purpose,expect=a.expected)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:v if k!='pose_ids' else len(v) for k,v in d.items()},indent=2))


if __name__=='__main__':main()
