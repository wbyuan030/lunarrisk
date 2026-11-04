#!/usr/bin/env python3
"""Export frozen best-Geology spatial tokens for the existing Train/Val subset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import p31_run_baselines as base

SOURCES = ('scripts/p32_cache_context.py', 'src/lunarrisk/lunarjev/context.py',
           'src/lunarrisk/benchmarks/classifier.py', 'src/lunarrisk/benchmarks/prefix.py',
           'src/lunarrisk/benchmarks/moonstone.py')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'metadata', 'range-report', 'runs-root', 'outdir'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=ROOT/'protocols/p31_local_prefix_eval_plan.json')
    parser.add_argument('--stats', type=Path, default=ROOT/'reports/evidence/p30_20261103/hf_channel_stats.json')
    parser.add_argument('--class-map', type=Path, default=ROOT/'reports/evidence/p30_20261103/geologic_units.json')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('positive batch size required')
    import h5py
    import numpy as np
    import torch
    from lunarrisk.benchmarks.classifier import ScratchClassifier, cpu_state, state_fingerprint, make_loader, seed_all, peak_rss_bytes
    from lunarrisk.benchmarks.prefix import PrefixClassifierDataset
    from lunarrisk.lunarjev.context import FrozenContext, validate_arrays
    seed_all(42)
    plan = base.read_json(args.plan)
    base.require(plan['protocol'] == 'P31_LOCAL_PREFIX_STORED_DERIVED_V1' and not plan['test']['runner_allows_test'], 'invalid subset plan')
    args.outdir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    base.require(args.data.stat().st_size == plan['data']['bytes'], 'prefix size changed')
    base.require(base.sha256(args.data) == plan['data']['sha256'], 'prefix SHA changed')
    for path, key in ((args.metadata, 'metadata_sha256'), (args.range_report, 'range_report_sha256'),
                      (args.stats, 'stats_sha256'), (args.class_map, 'class_map_sha256')):
        base.require(base.sha256(path) == plan['data'][key], 'input changed: '+key)
    hash_seconds = time.monotonic()-start
    stats, mapping = base.read_json(args.stats), base.read_json(args.class_map)
    datasets = {split: {task: PrefixClassifierDataset(args.data, stats, mapping, args.metadata,
                 args.range_report, plan, task, code) for task in ('geology', 'age')}
                for code, split in enumerate(('train', 'val'))}
    sources = {p: base.sha256(ROOT/p) for p in SOURCES}
    summary = {'protocol': 'P32_FROZEN_SPATIAL_CONTEXT_V1', 'sources': sources,
               'subset_plan_sha256': base.sha256(args.plan), 'inputs': plan['data'],
               'prefix_hash_verified': True, 'hash_seconds': hash_seconds,
               'test_accessed': False, 'network_requests': 0, 'encoders': {}}
    try:
        for seed in (42, 43, 44):
            run = args.runs_root/f'geology_seed{seed}'
            receipt, report = base.read_json(run/'completion.json'), base.read_json(run/'report.json')
            base.require(receipt['verification'] == 'PASS' and receipt['replay']['prediction_files_exact'], 'original receipt not verified')
            for name, expected in receipt['artifact_sha256'].items():
                path = run.parent/(run.name+'.stdout.log') if name == 'stdout' else run/name
                base.require(base.sha256(path) == expected, 'original run artifact changed: '+name)
            checkpoint = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=True)
            for key in ('inputs', 'splits', 'config', 'source_sha256'):
                base.require(checkpoint[key] == report[key], 'checkpoint/report mismatch: '+key)
            base.require(checkpoint['config']['task'] == 'geology' and checkpoint['config']['seed'] == seed
                         and not checkpoint['test_evaluated'] and checkpoint['plan_sha256'] == base.sha256(args.plan), 'wrong encoder provenance')
            expected_inputs = {'data_sha256': plan['data']['sha256'], 'data_bytes': plan['data']['bytes'],
                               **{k:plan['data'][k] for k in ('stats_sha256','class_map_sha256','metadata_sha256','range_report_sha256')}}
            base.require(checkpoint['inputs'] == expected_inputs, 'encoder input identity mismatch')
            base.require(all(base.sha256(ROOT/p) == value for p,value in checkpoint['source_sha256'].items()), 'P31 source changed')
            base.require(state_fingerprint(checkpoint['best_state_dict']) == report['best_weights_sha256'], 'best weights fingerprint mismatch')
            model = ScratchClassifier(49)
            model.load_state_dict(checkpoint['best_state_dict'])
            model.requires_grad_(False).eval()
            before = state_fingerprint(cpu_state(model))
            context = FrozenContext(model).to(args.device)
            if str(args.device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
            cache = args.outdir/f'context_seed{seed}.h5'
            began = time.monotonic()
            probe = {}
            with h5py.File(cache, 'x') as output:
                output.attrs['protocol'] = summary['protocol']
                output.attrs['encoder_seed'] = seed
                output.attrs['best_weights_sha256'] = before
                for split in ('train', 'val'):
                    ds = datasets[split]['geology']
                    age_ds = datasets[split]['age']
                    base.require(np.array_equal(ds.indices, age_ds.indices), 'task indices differ')
                    group = output.create_group(split)
                    n = len(ds)
                    group.create_dataset('tokens', (n,16,256), dtype='float32', chunks=(min(n,args.batch_size),16,256))
                    group.create_dataset('pooled', (n,256), dtype='float32')
                    group.create_dataset('geology', data=ds.targets)
                    group.create_dataset('age', data=age_ds.targets)
                    group.create_dataset('sample_ids', data=np.asarray(ds.ids[ds.indices], dtype=str).astype(object), dtype=h5py.string_dtype())
                    offset = 0
                    for batch in make_loader(ds, args.batch_size, False, torch.Generator().manual_seed(0)):
                        pixels = batch['pixels'].to(args.device)
                        values = context(pixels).cpu().numpy()
                        base.require(np.isfinite(values).all(), 'nonfinite encoder tokens')
                        end = offset+len(values)
                        group['tokens'][offset:end] = values
                        group['pooled'][offset:end] = values.mean(axis=1)
                        if offset == 0:
                            again = context(pixels).cpu().numpy()
                            probe[split] = {'batch_rows': len(values), 'repeat_exact': bool(np.array_equal(values,again))}
                            base.require(probe[split]['repeat_exact'], 'context repeat differs')
                        offset = end
                    base.require(offset == n, 'cache dropped final batch')
                    group.attrs['ordered_ids_sha256'] = ds.ordered_ids_hash
                    print(json.dumps({'seed':seed,'split':split,'rows':offset,'status':'EXPORTED'}),flush=True)
            after = state_fingerprint(cpu_state(model))
            base.require(before == after and all(p.grad is None and not p.requires_grad for p in context.parameters())
                         and all(not layer.training for layer in context.modules()), 'encoder/BN not frozen')
            with h5py.File(cache, 'r') as output:
                all_ids = []
                for split in ('train', 'val'):
                    group = output[split]
                    ids = group['sample_ids'].asstr()[:].tolist()
                    validate_arrays(group['tokens'][:], group['pooled'][:], ids, group['geology'][:], group['age'][:], split, plan['splits'][split])
                    base.require(np.array_equal(group['geology'][:], datasets[split]['geology'].targets)
                                 and np.array_equal(group['age'][:], datasets[split]['age'].targets), 'cache target order changed')
                    all_ids.append(set(ids))
                base.require(not all_ids[0]&all_ids[1], 'Train/Val overlap')
            info = {'seed':seed, 'file':cache.name, 'sha256':base.sha256(cache), 'bytes':cache.stat().st_size,
                    'checkpoint_sha256':base.sha256(run/'checkpoint.pt'), 'receipt_sha256':base.sha256(run/'completion.json'),
                    'best_weights_sha256':before,'selected_epoch':checkpoint['best_epoch'], 'frozen_parameters_and_bn':True,
                    'actual_tokens_shape_per_sample':[16,256], 'pooled_from_same_tokens':True, 'split_arrays_ids_targets_verified':True,
                    'repeat_probe':probe, 'export_wall_seconds':time.monotonic()-began, 'batch_size':args.batch_size,
                    'parent_peak_rss_bytes':peak_rss_bytes(),
                    'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else 0,
                    'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(args.device) if str(args.device).startswith('cuda') else 0}
            base.write_new(args.outdir/f'context_seed{seed}.json', info)
            summary['encoders'][str(seed)] = info
            print(json.dumps({'seed':seed,'status':'CACHE_VERIFIED','seconds':info['export_wall_seconds']}),flush=True)
            del model, context, checkpoint
        base.require(sources == {p:base.sha256(ROOT/p) for p in SOURCES}, 'source changed during export')
        summary['status'] = 'PASS'
        base.write_new(args.outdir/'manifest.json',summary)
        print('ALL_CONTEXT_CACHES_VERIFIED',flush=True)
    finally:
        for pair in datasets.values():
            for ds in pair.values():
                ds.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
