#!/usr/bin/env python3
"""Finite real-pilot runner regression: independent replay and epoch-boundary resume."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def equal_state(left,right):
    import torch
    if isinstance(left,torch.Tensor): return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if type(left)!=type(right): return False
    if isinstance(left,dict): return left.keys()==right.keys() and all(equal_state(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)): return len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','stats','class-map','reference','outdir'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    from lunarrisk.benchmarks.classifier import stream_sha256
    import h5py
    import torch
    reference=json.loads((args.reference/'report.json').read_text())
    if reference['config']!={**reference['config'],'planned_epochs':2,'batch_size':3,
                            'purpose':'runner_validation','seed':42,'workers':0,'precision':'fp32'}:
        raise ValueError('expected fixed two-epoch, batch-three runner-validation reference')
    if reference['splits']['train']['rows']!=5 or reference['splits']['val']['rows']!=5:
        raise ValueError('only existing five-per-split pilot is authorized by this regression driver')
    args.outdir.mkdir(parents=True,exist_ok=False)
    # Keep downloaded-data readers isolated while explicitly reusing this trusted runtime.
    bootstrap=("import runpy,sys; runtime=sys.argv.pop(1); script=sys.argv.pop(1); "
               "sys.path.insert(0,runtime); sys.argv[0]=script; runpy.run_path(script,run_name='__main__')")
    common=[sys.executable,'-I','-c',bootstrap,str(Path(torch.__file__).resolve().parents[1]),
            str(ROOT/'scripts/p31_train_classifier.py'),'--data',str(args.data.resolve()),
            '--stats',str(args.stats.resolve()),'--class-map',str(args.class_map.resolve()),
            '--epochs','2','--batch-size','3','--expected-data-sha256',reference['inputs']['data_sha256']]
    def run(name,extra):
        command=common+['--outdir',str(args.outdir/name)]+extra
        with (args.outdir/(name+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        return json.loads((args.outdir/name/'report.json').read_text())
    interrupted=run('interrupted',['--stop-after-epoch','1'])
    resumed=run('resumed',['--resume',str(args.outdir/'interrupted/checkpoint.pt')])
    replay=run('replay',['--mode','replay','--checkpoint',str(args.reference/'checkpoint.pt')])
    reference_checkpoint=torch.load(args.reference/'checkpoint.pt',map_location='cpu',weights_only=True)
    resumed_checkpoint=torch.load(args.outdir/'resumed/checkpoint.pt',map_location='cpu',weights_only=True)
    checks={
        'interrupted_after_one_of_two_planned_epochs':interrupted['completed_epochs']==1,
        'resume_last_weights_exact':reference['last_weights_sha256']==resumed['last_weights_sha256'],
        'resume_best_weights_exact':reference['best_weights_sha256']==resumed['best_weights_sha256'],
        'resume_train_val_prediction_files_exact':reference['prediction_sha256']==resumed['prediction_sha256'],
        'fresh_process_replay_prediction_files_exact':reference['prediction_sha256']==replay['prediction_sha256'],
        'optimizer_state_exact':equal_state(reference_checkpoint['optimizer_state_dict'],resumed_checkpoint['optimizer_state_dict']),
        'rng_state_exact':equal_state(reference_checkpoint['rng'],resumed_checkpoint['rng']),
        'loader_generator_states_exact':equal_state(reference_checkpoint['loader_rng'],resumed_checkpoint['loader_rng']),
        'epoch_training_loss_and_val_metrics_exact':[(r['train_loss'],r['val']) for r in reference_checkpoint['history']]==[(r['train_loss'],r['val']) for r in resumed_checkpoint['history']],
        'no_test_scoring':all(not r['test_evaluated'] for r in (reference,interrupted,resumed,replay))}
    with h5py.File(args.reference/'features.h5','r') as cache:
        checks['feature_cache_provenance_and_last_partial_batch']=all(
            cache[name+'/features'].shape==(5,256) and cache[name+'/features'].chunks==(3,256)
            and cache[name].attrs['ordered_ids_sha256']==reference['splits'][name]['ordered_ids_sha256']
            for name in ('train','val')) and bool(json.loads(cache.attrs['provenance']))
    report={'status':'PASS' if all(checks.values()) else 'FAIL','scope':'runner validation only; not new scientific performance evidence',
            'source_commit':os.environ.get('LUNARRISK_GIT_SHA','UNKNOWN'),
            'validation_script_sha256':stream_sha256(__file__),'checks':checks,
            'reference_prediction_sha256':reference['prediction_sha256'],
            'checkpoint_sha256':{'uninterrupted':stream_sha256(args.reference/'checkpoint.pt'),
                                 'resumed':stream_sha256(args.outdir/'resumed/checkpoint.pt')},
            'runs':{name:{k:r[k] for k in ('completed_epochs','best_epoch','last_weights_sha256','best_weights_sha256',
                     'peak_process_rss_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes',
                     'train_window_seconds_this_process','train_samples_per_second_this_process')}
                    for name,r in [('uninterrupted',reference),('interrupted',interrupted),('resumed',resumed),('replay',replay)]}}
    (args.outdir/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if all(checks.values()) else 2


if __name__=='__main__': raise SystemExit(main())
