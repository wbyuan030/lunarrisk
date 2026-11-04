#!/usr/bin/env python3
"""Run the locked P32 four-head matrix, then the fixed descriptor control."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import p31_run_baselines as base
import p32_train_heads as train


def sequence(plan):
    primary=[(seed,method,False) for seed in plan['seeds'] for method in plan['methods']]
    control=[(seed,method,True) for seed in plan['seeds'] for method in plan['description_control']['methods']]
    return primary+control


def run_name(seed,method,permuted):
    return f'{method}_seed{seed}'+('_permuted' if permuted else '')


def validate_run(directory,plan,manifest_hash):
    import numpy as np
    import torch
    from lunarrisk.benchmarks.classifier import state_fingerprint
    report=base.read_json(directory/'report.json')
    identity=report['identity']
    base.require(report['status']=='PASS' and report['mode']=='train' and not report['test_scored']
                 and report['config']==plan['config'] and identity['manifest_sha256']==manifest_hash
                 and identity['source_sha256']==plan['source_sha256'], 'run identity mismatch')
    base.require(identity['seed'] in plan['seeds'] and identity['method'] in plan['methods']
                 and directory.name==run_name(identity['seed'],identity['method'],identity['descriptor_permuted']), 'unexpected run')
    expected_text=plan['description_control']['text_identity'] if identity['descriptor_permuted'] else plan['text_identity']
    base.require(identity['cache_sha256']==plan['encoders'][str(identity['seed'])]['sha256']
                 and identity['text_identity']==expected_text, 'cache/text provenance differs')
    if identity['descriptor_permuted']:
        base.require(identity['method'] in plan['description_control']['methods'], 'unexpected control')
    base.require(report['parameters']==plan['parameter_counts'][identity['method']], 'parameter count differs')
    history=report['history']; base.require(1<=len(history)<=plan['config']['epochs'],'invalid history length')
    selected=max(range(len(history)),key=lambda i:history[i]['val']['geology']['macro_f1_full'])
    base.require(report['best_epoch']==selected+1 and report['best_metric']==history[selected]['val']['geology']['macro_f1_full'], 'selection differs')
    stale=0; best=-1.
    for index,row in enumerate(history):
        metric=row['val']['geology']['macro_f1_full']
        base.require(row['epoch']==index+1 and row['train_rows']==plan['splits']['train']['rows'], 'dropped/repeated training rows')
        base.finite(row['train_loss'],'train loss'); base.finite(row['train_seconds'],'train seconds')
        stale=0 if metric>best else stale+1; best=max(best,metric)
        base.require(index==len(history)-1 or stale<plan['config']['patience'], 'continued after stopping')
    base.require(len(history)==plan['config']['epochs'] or stale==plan['config']['patience'], 'run stopped before budget/patience')
    checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
    base.require(checkpoint['identity']==identity and checkpoint['history']==history
                 and checkpoint['best_epoch']==report['best_epoch'] and checkpoint['best_metric']==report['best_metric']
                 and state_fingerprint(checkpoint['best_state_dict'])==report['selected_weights_sha256'], 'checkpoint mismatch')
    hashes={}
    for split in ('train','val'):
        loaded={}
        for task,classes in (('geology',49),('age',5),('age_from_geology',5)):
            path=directory/f'{split}_{task}_predictions.csv'
            hashes[path.name]=base.sha256(path)
            base.require(hashes[path.name]==report['prediction_sha256'][split][task], 'prediction hash mismatch')
            counts=plan['splits'][split][('age' if task=='age_from_geology' else task)+'_class_counts']
            info={'rows':plan['splits'][split]['rows'],'ordered_ids_sha256':plan['splits'][split]['ordered_ids_sha256'],'class_counts':counts}
            metrics=report['metrics'][split][task]
            base.check_metrics(metrics,counts,classes,split+task)
            base.check_predictions(path,split,info,metrics,classes)
            with path.open(newline='') as f:
                rows=list(csv.DictReader(f))
            p=np.asarray([[float(row[f'p_{c}']) for c in range(classes)] for row in rows])
            y=np.asarray([int(row['target']) for row in rows])
            recalculated=train.probability_metrics(p,y)
            for key in ('accuracy','macro_f1_full','macro_f1_supported','nll','multiclass_brier_sum'):
                base.require(np.isclose(recalculated[key],metrics[key],atol=1e-12,rtol=0),'recomputed metric differs: '+key)
            loaded[task]=p
        base.require(np.array_equal(train.age_aggregation(loaded['geology'],plan['geology_to_age']),loaded['age_from_geology']), 'Age aggregation differs')
    for name in ('checkpoint.pt','report.json'):
        hashes[name]=base.sha256(directory/name)
    return report,hashes


def verify_replay(original,replay):
    first,last=base.read_json(original/'report.json'),base.read_json(replay/'report.json')
    base.require(last['mode']=='replay' and not last['test_scored'],'invalid replay')
    for key in ('identity','config','metrics','prediction_sha256','selected_weights_sha256','best_epoch','best_metric','history'):
        base.require(first[key]==last[key],'replay differs: '+key)
    for split in ('train','val'):
        for task in ('geology','age','age_from_geology'):
            name=f'{split}_{task}_predictions.csv'
            base.require(base.sha256(original/name)==base.sha256(replay/name),'CSV replay differs: '+name)
    return {'status':'PASS','fresh_process':True,'prediction_files_exact':True,'csv_count':6,
            'report_sha256':base.sha256(replay/'report.json')}


def aggregate(reports,plan):
    result={'protocol':plan['protocol'],'completed_runs':len(reports),'expected_runs':len(sequence(plan)),
            'status':'COMPLETE' if len(reports)==len(sequence(plan)) else 'PARTIAL','methods':{},'test_scored':False}
    for method in plan['methods']:
        for permuted in (False,True):
            if permuted and method not in plan['description_control']['methods']: continue
            rows=[r for r in reports if r['identity']['method']==method and r['identity']['descriptor_permuted']==permuted]
            label=method+('_permuted' if permuted else '')
            group={'seeds':[r['identity']['seed'] for r in rows],'n':len(rows),'metrics':{}}
            for split in ('train','val'):
                group['metrics'][split]={}
                for task in ('geology','age','age_from_geology'):
                    group['metrics'][split][task]={}
                    for metric in ('accuracy','macro_f1_full','macro_f1_supported','nll','multiclass_brier_sum'):
                        values=[r['metrics'][split][task][metric] for r in rows]
                        group['metrics'][split][task][metric]={'values':values,'mean':statistics.mean(values) if values else None,
                                                            'sample_std_ddof1':statistics.stdev(values) if len(values)>1 else None}
            result['methods'][label]=group
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-root','manifest','outdir','runtime-directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--catalog',type=Path,default=ROOT/'protocols/p32_candidate_catalog.json')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args(argv)
    plan=base.read_json(args.manifest)
    manifest_hash=base.sha256(args.manifest)
    base.require(plan['source_sha256']==train.sources(),'source changed after lock')
    runtime=base.runtime_directory(args.runtime_directory)
    commands=[]
    for seed,method,permuted in sequence(plan):
        name=run_name(seed,method,permuted)
        command=[sys.executable,'-I','-c',base.BOOTSTRAP,str(runtime),str(ROOT/'scripts/p32_train_heads.py'),
                 '--mode','train','--cache-root',str(args.cache_root.resolve()),'--catalog',str(args.catalog.resolve()),
                 '--manifest',str(args.manifest.resolve()),'--outdir',str((args.outdir/name).resolve()),
                 '--seed',str(seed),'--method',method]
        if permuted: command.append('--permuted')
        commands.append((name,command))
    if not args.apply:
        print(json.dumps({'mode':'PREVIEW','runs':[name for name,cmd in commands],'test_scored':False},indent=2)); return 0
    args.outdir.mkdir(parents=True,exist_ok=True)
    with (args.outdir/'.launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reports=[]
        for name,command in commands:
            base.require(train.sources()==plan['source_sha256'] and base.sha256(args.manifest)==manifest_hash,'manifest/source drift')
            directory=args.outdir/name; replay=args.outdir/(name+'_replay')
            log=args.outdir/(name+'.stdout.log')
            if directory.exists() or log.exists():
                base.require((directory/'completion.json').is_file(),'incomplete output retained; choose new namespace')
                report,hashes=validate_run(directory,plan,manifest_hash)
                receipt=base.read_json(directory/'completion.json')
                base.require(receipt['artifact_sha256']==hashes and receipt['stdout_sha256']==base.sha256(log)
                             and receipt['replay']==verify_replay(directory,replay),'completed artifacts changed')
                print('REUSED '+name,flush=True)
            else:
                print('START '+name,flush=True)
                with log.open('x') as output: subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                report,hashes=validate_run(directory,plan,manifest_hash)
                replay_command=command.copy(); replay_command[replay_command.index('--mode')+1]='replay'
                replay_command[replay_command.index('--outdir')+1]=str(replay.resolve())
                replay_command+=['--checkpoint',str((directory/'checkpoint.pt').resolve())]
                with (args.outdir/(name+'.replay.log')).open('x') as output:
                    subprocess.run(replay_command,stdout=output,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                receipt={'status':'PASS','identity':report['identity'],'artifact_sha256':hashes,
                         'stdout_sha256':base.sha256(log),'replay':verify_replay(directory,replay),'test_scored':False}
                base.write_new(directory/'completion.json',receipt)
                print('VERIFIED '+name,flush=True)
            reports.append(report)
            aggregate_path=args.outdir/f'aggregate_{len(reports):02d}.json'
            summary=aggregate(reports,plan)
            if aggregate_path.exists():
                base.require(base.read_json(aggregate_path)==summary,'aggregate changed')
            else:
                base.write_new(aggregate_path,summary)
        print(json.dumps({'status':'COMPLETE','runs':len(reports),'test_scored':False}),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
