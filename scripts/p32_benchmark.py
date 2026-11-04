#!/usr/bin/env python3
"""Resident-pixel/shared-context timing for trained P32 heads; Train probe only."""
from __future__ import annotations
import argparse
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
import p31_run_baselines as base


def measure(operation,device):
    import torch
    for _ in range(5): operation()
    torch.cuda.synchronize(device)
    times=[]
    for _ in range(20):
        start=time.perf_counter(); operation(); torch.cuda.synchronize(device)
        times.append((time.perf_counter()-start)*1000)
    return {'median_ms':statistics.median(times),'mean_ms':statistics.mean(times),'repeats':20,'warmup':5}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','metadata','range-report','cache-root','p31-runs','head-runs','manifest','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    import numpy as np
    import torch
    from lunarrisk.benchmarks.classifier import ScratchClassifier,seed_all,make_loader,state_fingerprint,cpu_state
    from lunarrisk.benchmarks.prefix import PrefixClassifierDataset
    from lunarrisk.lunarjev.context import FrozenContext
    from lunarrisk.lunarjev.catalog import load_catalog,encode_catalog
    from lunarrisk.lunarjev.heads import build_head
    seed_all(42); device=torch.device('cuda:0')
    plan=base.read_json(ROOT/'protocols/p31_local_prefix_eval_plan.json')
    manifest=base.read_json(args.manifest)
    base.require(base.sha256(args.data)==plan['data']['sha256'],'prefix checksum changed')
    paths={'metadata_sha256':args.metadata,'range_report_sha256':args.range_report,
           'stats_sha256':ROOT/'reports/evidence/p30_20261103/hf_channel_stats.json',
           'class_map_sha256':ROOT/'reports/evidence/p30_20261103/geologic_units.json'}
    for key,path in paths.items(): base.require(base.sha256(path)==plan['data'][key],'probe input changed')
    ds=PrefixClassifierDataset(args.data,base.read_json(paths['stats_sha256']),base.read_json(paths['class_map_sha256']),
                               args.metadata,args.range_report,plan,'geology',0)
    batch=next(iter(make_loader(ds,16,False,torch.Generator().manual_seed(0))))
    pixels=batch['pixels'].to(device)
    encoded=encode_catalog(load_catalog(ROOT/'protocols/p32_candidate_catalog.json'))
    text={k:torch.from_numpy(v) for k,v in encoded['candidate_features'].items()}
    tasks={k:torch.from_numpy(v) for k,v in encoded['task_features'].items()}
    result={'status':'PASS','scope':'FP32 resident Train pixels; no HDF5 I/O or host/device transfer in timed regions',
            'probe_ordered_ids':list(batch['sample_id']),'batch_size':16,'device':torch.cuda.get_device_name(device),
            'script_sha256':base.sha256(__file__),'manifest_sha256':base.sha256(args.manifest),
            'test_scored':False,'network_requests':0,'encoders':{}}
    try:
        with torch.no_grad():
            for seed in (42,43,44):
                path=args.p31_runs/f'geology_seed{seed}'/'checkpoint.pt'
                base.require(base.sha256(path)==manifest['encoders'][str(seed)]['checkpoint_sha256'],'original checkpoint changed')
                original=torch.load(path,map_location='cpu',weights_only=True)
                model=ScratchClassifier(49); model.load_state_dict(original['best_state_dict'])
                before=state_fingerprint(cpu_state(model))
                context=FrozenContext(model).to(device)
                tokens=context(pixels)
                measured={'shared_visual_context':measure(lambda:context(pixels),device),'methods':{}}
                for method in ('B0','B1','B2','J'):
                    checkpoint_path=args.head_runs/f'{method}_seed{seed}'/'checkpoint.pt'
                    receipt=base.read_json(checkpoint_path.parent/'completion.json')
                    base.require(base.sha256(checkpoint_path)==receipt['artifact_sha256']['checkpoint.pt'],'head checkpoint changed')
                    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
                    head=build_head(method,text,tasks).to(device); head.load_state_dict(checkpoint['best_state_dict']); head.eval()
                    cached=head.forward_all(tokens)
                    direct=head.forward_all(context(pixels))
                    base.require(all(torch.equal(cached[k],direct[k]) for k in cached),'shared context versus direct predictions differ')
                    measured['methods'][method]={'head_two_tasks':measure(lambda:head.forward_all(tokens),device),
                                                 'encoder_plus_two_tasks':measure(lambda:head.forward_all(context(pixels)),device),
                                                 'checkpoint_sha256':base.sha256(checkpoint_path),
                                                 'context_reuse_predictions_exact':True}
                    del head,checkpoint
                base.require(before==state_fingerprint(cpu_state(model)),'timing changed encoder/BN')
                result['encoders'][str(seed)]=measured
                del context,model,original
        base.write_new(args.out,result)
    finally: ds.close()
    return 0


if __name__=='__main__': raise SystemExit(main())
