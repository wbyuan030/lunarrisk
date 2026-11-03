#!/usr/bin/env python3
"""Measure known-ID control latency on fixed public subsets, without model scoring."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','stats','class-map','outdir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--head-repeats',type=int,default=30)
    parser.add_argument('--end-to-end-repeats',type=int,default=10)
    args=parser.parse_args()
    if min(args.batch_size,args.head_repeats,args.end_to_end_repeats)<1:
        raise ValueError('positive batch/repeat counts required')
    import torch
    from lunarrisk.benchmarks.classifier import ClassifierDataset,ScratchClassifier,seed_all,stream_sha256
    from lunarrisk.benchmarks.candidate_heads import OrdinaryHead,CandidateIDHead,masked_softmax
    seed_all(42)
    device=torch.device(args.device)
    dataset=ClassifierDataset(args.data,json.loads(args.stats.read_text()),json.loads(args.class_map.read_text()),'geology')
    if len(dataset)<args.batch_size: raise ValueError('requested batch exceeds available Train rows')
    args.outdir.mkdir(parents=True,exist_ok=False)
    frontend=ScratchClassifier(49).features.to(device).eval()
    ordinary=OrdinaryHead().to(device).eval()
    candidate=CandidateIDHead().to(device).eval()
    def synchronize():
        if device.type=='cuda': torch.cuda.synchronize(device)
    def pixels():
        # Fixed first Train rows, identical for every condition; no labels or IDs enter the network.
        return torch.from_numpy(np.stack([dataset[i]['pixels'] for i in range(args.batch_size)])).to(device)
    def timed(function,repeats):
        for _ in range(3): function()
        synchronize()
        values=[]
        for _ in range(repeats):
            synchronize()
            start=time.perf_counter()
            function()
            synchronize()
            values.append((time.perf_counter()-start)*1000)
        return {'median_ms':float(np.median(values)),'p90_ms':float(np.percentile(values,90)),
                'min_ms':min(values),'max_ms':max(values),'repeats':repeats}
    rows=[]
    with torch.inference_mode():
        features=frontend(pixels())
        full=candidate(features,torch.arange(49,device=device))
        for count in (1,5,10,25,49):
            ids=torch.arange(count,device=device)
            mask=torch.zeros((args.batch_size,49),dtype=torch.bool,device=device)
            mask[:,ids]=True
            def ordinary_call(x): return masked_softmax(ordinary(x),mask)[:,ids]
            def candidate_call(x): return candidate(x,ids).softmax(-1)
            delta=float((candidate_call(features)-masked_softmax(full,mask)[:,ids]).abs().max())
            rows.append({'candidate_count':count,'subset':'first K public IDs, same for every sample',
                'own_full_mask_identity_max_abs_error':delta,
                'ordinary_head_including_mask':timed(lambda:ordinary_call(features),args.head_repeats),
                'candidate_id_control_head_including_validation':timed(lambda:candidate_call(features),args.head_repeats),
                'ordinary_read_normalize_transfer_frontend_head':timed(lambda:ordinary_call(frontend(pixels())),args.end_to_end_repeats),
                'candidate_read_normalize_transfer_frontend_head':timed(lambda:candidate_call(frontend(pixels())),args.end_to_end_repeats),
                'ordinary_linear_macs_per_sample':256*40+40*49,
                'candidate_linear_macs_per_sample':count*((256+16)*42+42)})
        frontend_timing=timed(lambda:frontend(pixels()),args.end_to_end_repeats)
    sources=['scripts/p31_candidate_control.py','src/lunarrisk/benchmarks/classifier.py',
             'src/lunarrisk/benchmarks/candidate_heads.py','src/lunarrisk/benchmarks/moonstone.py']
    report={'status':'P31_CANDIDATE_ID_STRUCTURAL_AND_LATENCY_CONTROL',
        'weights':'seed42 random initialization; no training, accuracy or semantic claim',
        'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        'torch':str(torch.__version__),'hip':torch.version.hip,'precision':'fp32',
        'batch_size':args.batch_size,'train_sample_ids':[str(x) for x in dataset.ids[dataset.indices[:args.batch_size]]],
        'test_evaluated':False,'source_sha256':{s:stream_sha256(ROOT/s) for s in sources},
        'inputs':{k:stream_sha256(getattr(args,k)) for k in ('data','stats','class_map')},
        'parameters':{'ordinary':sum(p.numel() for p in ordinary.parameters()),'candidate_id_control':sum(p.numel() for p in candidate.parameters())},
        'timing_scope':'synchronized sequential warm runs; head includes softmax, candidate input validation, and ordinary masking; end-to-end includes selected HDF5 pixel read, normalization, stack, host-to-device, encoder, head; filesystem caches not cleared; no backward/optimizer',
        'mac_scope':'linear multiply-accumulates only; excludes embedding, validation, concat, nonlinearities, softmax, masking and frontend; latency measured separately',
        'read_normalize_transfer_frontend':frontend_timing,'rows':rows}
    (args.outdir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    dataset.close()
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
