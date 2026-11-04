#!/usr/bin/env python3
"""Run the separately locked local-prefix baselines offline; preview by default."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import p31_run_baselines as base


def validate_plan(plan):
    reference=base.read_json(ROOT/'protocols/p31_baseline_eval_plan.json')
    base.require(plan['protocol']=='P31_LOCAL_PREFIX_STORED_DERIVED_V1' and
                 plan['data']['track']=='partial_stored_prefix' and
                 plan['data']['upstream_full_integrity_verified'] is False,'expected separate partial-data plan')
    for key in ('training','metrics','tasks','test'):
        base.require(plan[key]==reference[key],'subset must retain locked baseline '+key)
    for key in ('stats_sha256','class_map_sha256','revision'):
        base.require(plan['data'][key]==reference['data'][key],'subset source differs: '+key)
    base.require(plan['data']['upstream_bytes']==reference['data']['bytes'] and
                 plan['data']['upstream_sha256']==reference['data']['sha256'],'upstream provenance mismatch')
    for split in ('train','val'):
        base.require(plan['data'][split+'_rows']==plan['splits'][split]['rows'] and
                     plan['data'][split+'_ordered_ids_sha256']==plan['splits'][split]['ordered_ids_sha256'],
                     'inconsistent frozen subset inventory')


def identity_for(args,plan,task,seed):
    identity=base.identity_for(args,plan,task,seed)
    identity['config']['purpose']='subset_baseline'
    for key in ('metadata_sha256','range_report_sha256'):
        identity['inputs'][key]=plan['data'][key]
    identity['source_sha256']['src/lunarrisk/benchmarks/prefix.py']=base.sha256(ROOT/'src/lunarrisk/benchmarks/prefix.py')
    identity['launcher_sha256']=base.sha256(__file__)
    identity['verifier_sha256']=base.sha256(ROOT/'scripts/p31_run_baselines.py')
    return identity


def command_for(args,plan,runtime,task,seed):
    command=base.command_for(args,plan,runtime,task,seed)
    command[command.index('--purpose')+1]='subset_baseline'
    return command+['--metadata',str(args.metadata),'--range-report',str(args.range_report)]


def verify_replay(directory,replay,identity):
    report=base.read_json(replay/'report.json')
    original=base.read_json(directory/'report.json')
    base.require(report['mode']=='replay' and report['protocol']=='P31_LOCAL_PREFIX_STORED_DERIVED_V1'
                 and report['test_evaluated'] is False,'invalid partial-data replay')
    for key in ('config','inputs','source_sha256'):
        base.require(report[key]==identity[key],'replay identity mismatch: '+key)
    for key in ('splits','metrics','train_prior_metrics','prediction_sha256','best_weights_sha256','last_weights_sha256'):
        base.require(report[key]==original[key],'fresh process replay differs: '+key)
    for split in ('train','val'):
        for suffix in ('','_prior'):
            name=split+suffix+'_predictions.csv'
            base.require(base.sha256(directory/name)==base.sha256(replay/name),'replayed CSV differs: '+name)
    return {'status':'PASS','fresh_process':True,'prediction_files_exact':True,
            'report_sha256':base.sha256(replay/'report.json')}


def verify(directory,log,identity,plan,runtime):
    report,receipt=base.verify_run(directory,log,identity,plan,runtime,subset=True)
    receipt['replay']=verify_replay(directory,directory.parent/(directory.name+'_replay'),identity)
    return report,receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('data','metadata','range-report','plan','outdir'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--stats',type=Path,default=ROOT/'reports/evidence/p30_20261103/hf_channel_stats.json')
    parser.add_argument('--class-map',type=Path,default=ROOT/'reports/evidence/p30_20261103/geologic_units.json')
    parser.add_argument('--runtime-directory',type=Path)
    parser.add_argument('--max-new-runs',type=int)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args(argv)
    if args.max_new_runs is not None and args.max_new_runs<1: parser.error('max-new-runs must be positive')
    for key in ('data','metadata','range_report','plan','outdir','stats','class_map'):
        setattr(args,key,getattr(args,key).resolve())
    plan=base.read_json(args.plan)
    validate_plan(plan)
    runtime=base.runtime_directory(args.runtime_directory)
    commands=[command_for(args,plan,runtime,task,seed) for task,seed in base.SEQUENCE]
    if not args.apply:
        print(json.dumps({'mode':'DRY_RUN','protocol':plan['protocol'],'commands':commands,
                          'network_requests':0,'test_evaluated':False},indent=2))
        return 0
    # Children stream-hash the prefix once each; preflight checks the remaining small inputs.
    base.require(args.data.stat().st_size==plan['data']['bytes'],'local prefix size changed')
    for path,key in ((args.stats,'stats_sha256'),(args.class_map,'class_map_sha256'),
                     (args.metadata,'metadata_sha256'),(args.range_report,'range_report_sha256')):
        base.require(base.sha256(path)==plan['data'][key],'input hash changed: '+key)
    frozen=identity_for(args,plan,'age',42)
    args.outdir.mkdir(parents=True,exist_ok=True)
    with (args.outdir/'.launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=[]; new_runs=0
        try:
            for (task,seed),command in zip(base.SEQUENCE,commands):
                identity=identity_for(args,plan,task,seed)
                expected={**frozen,'config':{**frozen['config'],'task':task,'seed':seed}}
                base.require(identity==expected,'source/plan changed during prefix sequence')
                name=base.run_name(task,seed)
                directory,log=args.outdir/name,args.outdir/(name+'.stdout.log')
                receipt=directory/'completion.json'
                if directory.exists() or log.exists():
                    base.require(receipt.is_file() and log.is_file(),'incomplete output; preserve and choose new namespace: '+name)
                    report,verified=verify(directory,log,identity,plan,runtime)
                    base.require(base.read_json(receipt)==verified,'completed prefix artifacts changed')
                    print('REUSED '+name,flush=True)
                else:
                    if args.max_new_runs is not None and new_runs>=args.max_new_runs: break
                    print('START '+name,flush=True)
                    with log.open('x') as output:
                        subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                    base.verify_run(directory,log,identity,plan,runtime,subset=True)
                    replay=command.copy()
                    replay[replay.index('--mode')+1]='replay'
                    replay[replay.index('--outdir')+1]=str(args.outdir/(name+'_replay'))
                    replay+=['--checkpoint',str(directory/'checkpoint.pt')]
                    with (args.outdir/(name+'.replay.log')).open('x') as output:
                        subprocess.run(replay,stdout=output,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                    report,verified=verify(directory,log,identity,plan,runtime)
                    base.require(identity_for(args,plan,task,seed)==identity,'source/plan changed during run')
                    base.write_new(receipt,verified)
                    new_runs+=1
                    print('VERIFIED '+name,flush=True)
                completed.append((task,seed,report))
                base.save_aggregate(args.outdir,completed)
        except Exception:
            base.save_aggregate(args.outdir,completed)
            raise
        summary=base.save_aggregate(args.outdir,completed)
        print(json.dumps({'status':summary['status'],'protocol':plan['protocol'],'completed':len(completed),
                          'new_runs':new_runs,'test_evaluated':False}),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
