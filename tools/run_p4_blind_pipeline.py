"""P4 Pilot: run independent unseen-pose Reader/FK ablation, fail closed.

This script is a one-time local orchestration in the current invocation,
not a background automation. Requires already completed new 12-pose
Blender GT, original reader (already predicted) and a trained P4-B
Back/Opposite specialized checkpoint from the fresh 64/12 pose split.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE_VIEWS='front,side'
FOUR_VIEWS='front,side,back,opposite_side'
IDS=','.join(str(i) for i in range(9001,9013))


def check_complete(path):
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))


def execute(cmd,log):
    print('RUN', ' '.join(str(s) for s in cmd),flush=True)
    with log.open('a',encoding='utf8') as f:
        f.write('RUN '+ ' '.join(map(str,cmd))+'\n')
        f.flush()
        result=subprocess.run([sys.executable,*map(str,cmd)],cwd=ROOT,
                              stdout=f,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            tail=log.read_text(encoding='utf8',errors='replace').splitlines()[-18:]
            raise RuntimeError(f'P4 subprocess exited {result.returncode}:\n'+'\n'.join(tail))
    print('PASS',cmd[0],flush=True)


def run(cr):
    ck=ROOT/'runs'/cr/'reader_backopp_p4_pilot_v1.pt'
    model_report=ck.with_suffix('.validation.json')
    check_complete(ck);check_complete(model_report)
    m=json.loads(model_report.read_text(encoding='utf8'))
    if (m.get('train_views')!=['back','opposite_side'] or
            m.get('checkpoint_selection_metric')!='back_opposite' or
            m.get('train_poses')!=64 or m.get('val_poses')!=12 or
            m.get('training_appearance')!='source-material' or
            m.get('unfreeze_last_blocks',0) not in (0,1)):
        raise ValueError('unexpected P4 specialized training contract')
    blind=ROOT/'runs'/cr/'p4_blind_fourview_v1'
    data=blind/'p4_original_blind_reader.npz'
    check_complete(data);check_complete(blind/'blind_audit_v1.json')
    if json.loads((blind/'blind_audit_v1.json').read_text())['source_pose_count']!=12:
        raise ValueError('blind test not 12 independent poses')
    log=ROOT/'runs'/cr/'p4_blind_pipeline_v1.log'
    rig=ROOT/'assets'/cr/'rigged/rig_info.json'
    trained=blind/'p4_adapted_blind_reader.npz'
    if not trained.exists():
        execute(['tools/build_four_view_observations.py',
                 '--source',blind,'--rig',rig,'--out',trained,'--indices',IDS,
                 '--mode','reader','--checkpoint',ck],log)
    check_complete(trained)
    hybrid=blind/'p4_hybrid_blind_reader.npz'
    if not hybrid.exists():
        execute(['tools/blend_fourview_reader_p3.py',
                 '--original',data,'--finetuned',trained,'--out',hybrid],log)
    check_complete(hybrid)
    for reader,src in (('original',data),('hybrid',hybrid)):
        for mode,views in (('2v',BASE_VIEWS),('4v',FOUR_VIEWS)):
            dest=blind/f'p4_{reader}_fk_{mode}.npz'
            if dest.exists():
                continue
            execute(['-m','mocap.solve_multiview',src,rig,'--views',views,
                     '--iterations','240',
                     '--out-prefix',dest.with_suffix('')],log)
            check_complete(dest)
    # Separate evaluator reads 3D truth; the solver above did NOT.
    result=ROOT/'runs'/cr/'p4_blind_acceptance_v1.json'
    if not result.exists():
        execute(['tools/eval_p4_blind_hybrid.py',
                 '--repo',ROOT,'--creature',cr,'--out',result],log)
    check_complete(result)
    report=json.loads(result.read_text(encoding='utf8'))
    print('P4_BLIND_RESULTS',cr,json.dumps({
         'original_two_mm':report['3d_fk_error']['original_two']['mean_mm'],
         'hybrid_four_mm':report['3d_fk_error']['hybrid_four']['mean_mm'],
         'delta_mean_mm':report['hybrid_four_minus_original_two_paired_mean_mm'],
         'paired_95pct_pose_bootstrap_CI_mm':report['paired_95pct_pose_bootstrap_CI_mm'],
         'improved_pose_count':report['paired_improved_pose_count'],
         'n':report['blind_synthetic_static_pose_count'],
         'gate':report['suggested_gate_3d_mean_and_ci_both_improve']
         },indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--creature',choices=('dino','dragon'),required=True)
    args=p.parse_args()
    run(args.creature)
