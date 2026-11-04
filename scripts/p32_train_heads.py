#!/usr/bin/env python3
"""Lock and train paired frozen-context P32 heads; existing Train/Val only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import p31_run_baselines as base

SOURCES = ('scripts/p32_train_heads.py','scripts/p32_run_heads.py',
           'src/lunarrisk/lunarjev/heads.py','src/lunarrisk/lunarjev/catalog.py',
           'src/lunarrisk/lunarjev/context.py','src/lunarrisk/benchmarks/classifier.py',
           'scripts/p31_run_baselines.py')
CONFIG = {'epochs':50,'patience':8,'batch_size':128,'lr':0.001,'weight_decay':0.01,
          'precision':'fp32','optimizer':'AdamW','schedule':'constant',
          'loss':'0.5*(mean Geology CE + mean Age CE)','selection':'strict Val full-49 Geology Macro-F1; earliest tie',
          'grad_clip':None,'configuration_search':False,'test_scored':False}


def text_identity(encoded):
    h = hashlib.sha256(json.dumps(encoded['metadata'],sort_keys=True,allow_nan=False).encode())
    for kind in ('candidate_features','task_features'):
        for task in ('geology','age'):
            a = encoded[kind][task]
            h.update((kind+task+str(a.dtype)+str(a.shape)).encode()); h.update(a.tobytes())
    h.update(json.dumps(encoded['geology_to_age']).encode())
    return h.hexdigest()


def sources():
    return {path:base.sha256(ROOT/path) for path in SOURCES}


def lock(args):
    from lunarrisk.lunarjev.catalog import load_catalog, encode_catalog, permute_encoded
    from lunarrisk.lunarjev.heads import build_head
    import torch
    from lunarrisk.benchmarks.classifier import seed_all
    seed_all(42)
    cache = base.read_json(args.cache_root/'manifest.json')
    subset = base.read_json(ROOT/'protocols/p31_local_prefix_eval_plan.json')
    base.require(cache['status']=='PASS' and not cache['test_accessed'], 'cache not verified')
    encoded = encode_catalog(load_catalog(args.catalog))
    permuted = permute_encoded(encoded)
    for seed,info in cache['encoders'].items():
        base.require(base.sha256(args.cache_root/info['file'])==info['sha256'], 'cache changed')
    text = {k:torch.from_numpy(v.copy()) for k,v in encoded['candidate_features'].items()}
    task = {k:torch.from_numpy(v.copy()) for k,v in encoded['task_features'].items()}
    params = {method:sum(p.numel() for p in build_head(method,text,task).parameters()) for method in ('B0','B1','B2','J')}
    plan = {'protocol':'P32_TAXONOMY_LEXICAL_HEADS_V1','status':'LOCKED_BEFORE_HEAD_SCORES',
            'design_sha256':base.sha256(ROOT/'protocols/p32_lunarjev_design.json'),
            'subset_plan_sha256':base.sha256(ROOT/'protocols/p31_local_prefix_eval_plan.json'),
            'cache_manifest_sha256':base.sha256(args.cache_root/'manifest.json'),'encoders':cache['encoders'],
            'catalog_sha256':base.sha256(args.catalog),'text_identity':text_identity(encoded),
            'text_metadata':encoded['metadata'],'geology_to_age':encoded['geology_to_age'],
            'source_sha256':sources(),'config':CONFIG,'seeds':[42,43,44], 'methods':['B0','B1','B2','J'],
            'parameter_counts':params,'splits':{s:{k:v for k,v in subset['splits'][s].items() if k!='indices'} for s in ('train','val')},
            'description_control':{'seed':20261104,'methods':['B2','J'],'seeds':[42,43,44],
                                   'after_primary_matrix':True,'text_identity':text_identity(permuted),
                                   'metadata':permuted['metadata']},
            'minority_classes_geology_train_support_lte':50,
            'tier':'TAXONOMY_ONLY_LOCAL; deterministic lexical TF-IDF; not a pretrained text model',
            'evaluation_role':'exploratory reused northern validation set; no unseen-class or independent test claim',
            'test_scored':False,'new_downloads':0}
    base.write_new(args.manifest,plan)
    print(json.dumps({'status':'LOCKED','parameters':params,'text_identity':plan['text_identity']}),flush=True)


def load_assets(args, plan):
    import h5py
    import numpy as np
    import torch
    from lunarrisk.lunarjev.catalog import load_catalog, encode_catalog, permute_encoded
    from lunarrisk.lunarjev.context import validate_arrays
    base.require(plan['source_sha256']==sources() and plan['config']==CONFIG, 'source/config drift')
    base.require(plan['catalog_sha256']==base.sha256(args.catalog)
                 and plan['cache_manifest_sha256']==base.sha256(args.cache_root/'manifest.json'), 'asset manifest drift')
    encoded = encode_catalog(load_catalog(args.catalog))
    base.require(text_identity(encoded)==plan['text_identity'], 'text representation drift')
    if args.permuted:
        base.require(args.method in plan['description_control']['methods'], 'no descriptor control for this method')
        encoded = permute_encoded(encoded)
        base.require(text_identity(encoded)==plan['description_control']['text_identity'], 'permutation drift')
    info = plan['encoders'][str(args.seed)]
    path = args.cache_root/info['file']
    base.require(base.sha256(path)==info['sha256'], 'context cache changed')
    arrays = {}
    with h5py.File(path,'r') as handle:
        base.require(set(handle)=={'train','val'} and handle.attrs['encoder_seed']==args.seed
                     and handle.attrs['best_weights_sha256']==info['best_weights_sha256'], 'wrong cache provenance')
        for split in ('train','val'):
            group = handle[split]
            row = {k:group[k][:] for k in ('tokens','pooled','geology','age')}
            row['ids'] = group['sample_ids'].asstr()[:].tolist()
            validate_arrays(row['tokens'],row['pooled'],row['ids'],row['geology'],row['age'],split,plan['splits'][split])
            mapping = np.asarray(plan['geology_to_age'])
            base.require(np.array_equal(mapping[row['geology']],row['age']), 'Age targets differ from public mapping')
            arrays[split] = {k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in row.items()}
    base.require(not set(arrays['train']['ids'])&set(arrays['val']['ids']), 'Train/Val overlap')
    return encoded, arrays


def probability_metrics(probability, targets):
    import numpy as np
    from lunarrisk.benchmarks.classifier import classification_metrics
    p = np.asarray(probability,dtype=np.float64)
    y = np.asarray(targets,dtype=np.int64)
    base.require(p.ndim==2 and len(p)==len(y) and np.isfinite(p).all() and (p>=0).all()
                 and np.allclose(p.sum(1),1,atol=1e-12,rtol=0), 'invalid probabilities')
    matrix = np.zeros((p.shape[1],p.shape[1]),dtype=np.int64)
    np.add.at(matrix,(y,p.argmax(1)),1)
    metric = classification_metrics(matrix)
    selected = p[np.arange(len(y)),y]
    nll = float(-np.log(selected).mean()) if (selected>0).all() else None
    metric.update(rows=len(y),nll=nll,nll_is_infinite=nll is None,
                  multiclass_brier_sum=float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean()))
    return metric


def age_aggregation(probability, mapping):
    import numpy as np
    p = np.asarray(probability,dtype=np.float64)
    result = np.zeros((len(p),5),dtype=np.float64)
    for c,age in enumerate(mapping):
        if not 0<=age<5:
            raise ValueError('all Geology outputs need audited Age mapping')
        result[:,age] += p[:,c]
    return result


def write_predictions(path, ids, targets, probability, split):
    with path.open('x',newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['sample_id','split','target']+[f'p_{c}' for c in range(probability.shape[1])])
        for sample,target,row in zip(ids,targets,probability):
            writer.writerow([sample,split,int(target),*row.tolist()])


def sync(device):
    import torch
    if device.type=='cuda': torch.cuda.synchronize(device)


def evaluate(model, arrays, device, mapping, outdir=None):
    import numpy as np
    import torch
    result, hashes = {}, {}
    model.eval()
    with torch.no_grad():
        for split,row in arrays.items():
            chunks = {task:[] for task in ('geology','age')}
            for begin in range(0,len(row['tokens']),CONFIG['batch_size']):
                logits = model.forward_all(row['tokens'][begin:begin+CONFIG['batch_size']].to(device))
                for task in chunks:
                    chunks[task].append(torch.softmax(logits[task].double(),dim=-1).cpu().numpy())
            probability = {task:np.concatenate(parts) for task,parts in chunks.items()}
            probability['age_from_geology'] = age_aggregation(probability['geology'],mapping)
            result[split],hashes[split] = {},{}
            for task,p in probability.items():
                y = row['age' if task=='age_from_geology' else task].numpy()
                result[split][task] = probability_metrics(p,y)
                if outdir is not None:
                    path = outdir/f'{split}_{task}_predictions.csv'
                    write_predictions(path,row['ids'],y,p,split)
                    hashes[split][task] = base.sha256(path)
    return result,hashes


def compact(metrics):
    return {task:{k:values[k] for k in ('accuracy','macro_f1_full','macro_f1_supported','nll','multiclass_brier_sum')}
            for task,values in metrics.items()}


def latency(model, tokens, device):
    import torch
    model.eval(); result = {}
    with torch.no_grad():
        for batch in (1,128):
            x = tokens[:batch].to(device)
            for _ in range(5): model.forward_all(x)
            sync(device); timings=[]
            for _ in range(20):
                began=time.perf_counter(); model.forward_all(x); sync(device)
                timings.append((time.perf_counter()-began)*1000)
            result[str(batch)] = {'batch':len(x),'repeats':20,'median_ms':statistics.median(timings),
                                  'mean_ms':statistics.mean(timings),'scope':'resident context, both tasks, head only, no pixel encoding/transfer'}
    return result


def prepare_device(name):
    import torch
    device = torch.device(name)
    if device.type=='cuda':
        base.require(torch.cuda.is_available(), 'existing GPU unavailable')
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    return device


def run(args):
    import torch
    from lunarrisk.benchmarks.classifier import seed_all,cpu_state,state_fingerprint,atomic_checkpoint,rng_state,peak_rss_bytes
    from lunarrisk.lunarjev.heads import build_head
    plan = base.read_json(args.manifest)
    base.require(args.seed in plan['seeds'] and args.method in plan['methods'], 'unsupported run')
    seed_all(args.seed)
    encoded,arrays = load_assets(args,plan)
    device = prepare_device(args.device)
    text = {k:torch.from_numpy(v.copy()) for k,v in encoded['candidate_features'].items()}
    tasks = {k:torch.from_numpy(v.copy()) for k,v in encoded['task_features'].items()}
    model = build_head(args.method,text,tasks).to(device)
    args.outdir.mkdir(parents=True,exist_ok=False)
    identity = {'manifest_sha256':base.sha256(args.manifest),'method':args.method,'seed':args.seed,
                'descriptor_permuted':args.permuted,'source_sha256':plan['source_sha256'],
                'cache_sha256':plan['encoders'][str(args.seed)]['sha256'],'text_identity':text_identity(encoded)}
    began=time.monotonic(); history=[]; best=-1.; best_epoch=0; stale=0; selected=None
    optimizer = torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    generator = torch.Generator().manual_seed(args.seed)
    train_seconds=0.; train_examples=0
    if args.mode=='train':
        for epoch in range(1,CONFIG['epochs']+1):
            model.train(); sync(device); start=time.monotonic(); total=0.; seen=0
            order=torch.randperm(len(arrays['train']['tokens']),generator=generator)
            for indices in order.split(CONFIG['batch_size']):
                row=arrays['train']
                logits=model.forward_all(row['tokens'][indices].to(device))
                loss=sum(torch.nn.functional.cross_entropy(logits[task],row[task][indices].to(device)) for task in ('geology','age'))/2
                base.require(bool(torch.isfinite(loss)), 'nonfinite training loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                base.require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'nonfinite gradient')
                optimizer.step(); total+=float(loss.detach())*len(indices); seen+=len(indices)
            sync(device); elapsed=time.monotonic()-start
            train_seconds+=elapsed; train_examples+=seen
            val,_=evaluate(model,{'val':arrays['val']},device,plan['geology_to_age'])
            metric=val['val']['geology']['macro_f1_full']
            if metric>best:
                best,best_epoch,selected,stale=metric,epoch,cpu_state(model),0
            else: stale+=1
            history.append({'epoch':epoch,'train_loss':total/seen,'train_rows':seen,'train_seconds':elapsed,
                            'val':compact(val['val'])})
            print(json.dumps({'method':args.method,'seed':args.seed,'permuted':args.permuted,**history[-1]}),flush=True)
            if stale>=CONFIG['patience']: break
        payload={'protocol':plan['protocol'],'identity':identity,'config':CONFIG,'history':history,
                 'best_epoch':best_epoch,'best_metric':best,'best_state_dict':selected,'last_state_dict':cpu_state(model),
                 'optimizer_state_dict':optimizer.state_dict(),'rng':rng_state(),'loader_rng':generator.get_state(),
                 'test_scored':False}
        atomic_checkpoint(args.outdir/'checkpoint.pt',payload)
    else:
        base.require(args.checkpoint is not None, 'replay requires checkpoint')
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        base.require(payload['identity']==identity and payload['config']==CONFIG and not payload['test_scored'], 'checkpoint identity mismatch')
        history=payload['history']; selected=payload['best_state_dict']; best_epoch=payload['best_epoch']; best=payload['best_metric']
    model.load_state_dict(selected)
    metrics,predictions=evaluate(model,arrays,device,plan['geology_to_age'],args.outdir)
    base.require(metrics['val']['geology']['macro_f1_full']==best, 'selected checkpoint metrics differ')
    report={'status':'PASS','protocol':plan['protocol'],'mode':args.mode,'identity':identity,'config':CONFIG,
            'history':history,'completed_epochs':len(history),'best_epoch':best_epoch,'best_metric':best,
            'selected_weights_sha256':state_fingerprint(selected),'metrics':metrics,'prediction_sha256':predictions,
            'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'train_window_seconds':train_seconds,'train_examples':train_examples,'wall_seconds':time.monotonic()-began,
            'parent_peak_rss_bytes':peak_rss_bytes(),
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
            'head_two_task_latency':latency(model,arrays['val']['tokens'],device),
            'test_scored':False,'network_requests':0,'frozen_encoder_used_from_cache':True,
            'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,'hip':torch.version.hip,
                       'device':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu','deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}}
    base.require(sources()==plan['source_sha256'],'source drift during run')
    base.write_new(args.outdir/'report.json',report)
    print(json.dumps({'status':'COMPLETE','method':args.method,'seed':args.seed,'permuted':args.permuted,'best_epoch':best_epoch}),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('lock','train','replay'),required=True)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,default=ROOT/'protocols/p32_candidate_catalog.json')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--method',choices=('B0','B1','B2','J'))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--permuted',action='store_true')
    parser.add_argument('--outdir',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv)
    if args.mode=='lock': lock(args)
    else:
        if args.method is None or args.outdir is None: parser.error('method and outdir required')
        run(args)
    return 0


if __name__=='__main__': raise SystemExit(main())
