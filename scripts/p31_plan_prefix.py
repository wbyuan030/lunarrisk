#!/usr/bin/env python3
"""Lock an offline, metadata-only Train/Val subset of the retained HDF5 prefix."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from lunarrisk.benchmarks.moonstone import CHANNELS,age_target,geology_targets
from lunarrisk.benchmarks.prefix import PrefixFile,cached_blocks,MissingLocalBytes

PREFIX_BYTES=33964443553
UPSTREAM_BYTES=74449119347
UPSTREAM_SHA256='e3e96bccc044d856cd55883c9cda7628f02afbdc1b2d1818b401bbcc590df83f'
METADATA_SHA256='3b3ed8cade200f79f720beeba6a666365d66383dbb623337cd96ebec31a0a98d'
ORIGINAL_PLAN=ROOT/'protocols/p31_baseline_eval_plan.json'
SOURCES=('scripts/p31_plan_prefix.py','src/lunarrisk/benchmarks/prefix.py',
         'src/lunarrisk/benchmarks/moonstone.py')


def require(condition,message):
    if not condition: raise ValueError(message)


def ids_hash(ids):
    return hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest()


def stamp(path):
    state=Path(path).stat()
    return state.st_dev,state.st_ino,state.st_size,state.st_mtime_ns,state.st_ctime_ns


def stream_sha256(path,expected_bytes):
    """Exactly one bounded full-prefix pass; reject a concurrent write or replacement."""
    before=stamp(path)
    require(before[2]==expected_bytes,'unexpected local prefix size')
    digest=hashlib.sha256()
    size=0
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):
            size+=len(block)
            digest.update(block)
    require(size==expected_bytes and stamp(path)==before,'prefix mutated during hashing')
    return digest.hexdigest()


def read_json(path,expected=None):
    body=Path(path).read_bytes()
    digest=hashlib.sha256(body).hexdigest()
    require(expected is None or digest==expected,'audited SHA256 mismatch: '+str(path))
    return json.loads(body),digest


def metadata_targets(raw,mapping,original):
    require(tuple(raw['attributes']['channel_names'])==CHANNELS,'channel order mismatch')
    meta=raw['metadata']
    fields=('metadata/patch_id','metadata/split','metadata/center_lat',
            'metadata/geology/dominant_unit_id','metadata/geology/dominant_unit_name')
    arrays=[np.asarray(meta[key]) for key in fields]
    require(all(a.ndim==1 and len(a)==len(arrays[0]) for a in arrays),
            'metadata length/shape mismatch')
    ids,stored,latitude,units,names=arrays
    ids=np.asarray([str(value) for value in ids])
    require(len(ids)>0 and len(set(ids))==len(ids) and all(value and '\n' not in value for value in ids),
            'duplicate or invalid sample IDs')
    require(np.issubdtype(stored.dtype,np.integer) and np.isin(stored,[0,1,2]).all(),
            'invalid stored split')
    require(np.isfinite(latitude).all() and (np.abs(latitude)<=90).all(),'invalid latitude')
    geology=geology_targets(units,names,mapping)
    ages=np.asarray([age_target(name) for name in names])
    for code,key in ((0,'train'),(1,'val')):
        selected=ids[stored==code]
        require(len(selected)==original['data'][key+'_rows'] and
                ids_hash(selected)==original['data'][key+'_ordered_ids_sha256'],
                'original '+key+' split provenance mismatch')
    return ids,stored,latitude,ages,geology


def inventory(patches,targets,prefix_bytes,logical_bytes):
    """Only query chunk descriptors, including unavailable descriptors for every source row."""
    ids,stored,latitude,ages,geology=targets
    require(patches.shape==(len(ids),28,256,256) and patches.chunks==(1,28,256,256)
            and patches.dtype==np.dtype('float32'),'unexpected pixel schema')
    complete=[]
    outside=[]
    unavailable=[]
    missing={}
    counts={key:{name:0 for name in ('complete','outside_prefix','metadata_unavailable')}
            for key in ('train','val','test')}
    for index in range(len(ids)):
        key=('train','val','test')[int(stored[index])]
        try:
            info=patches.id.get_chunk_info_by_coord((index,0,0,0))
        except MissingLocalBytes as error:
            require(error.start>=0 and error.length>0 and error.start+error.length<=logical_bytes,
                    'invalid missing metadata range')
            unavailable.append(index)
            missing[(error.start,error.length)]=missing.get((error.start,error.length),0)+1
            counts[key]['metadata_unavailable']+=1
            continue
        offset,size=info.byte_offset,int(info.size)
        require(offset is None or (type(offset) is int and offset>=0 and offset+size<=logical_bytes),
                'invalid chunk byte range')
        require(size>=0 and (offset is not None or size==0),'invalid chunk allocation')
        record={'index':index,'sample_id':str(ids[index]),'byte_offset':offset,'compressed_bytes':size}
        if offset is not None and size>0 and offset+size<=prefix_bytes:
            complete.append(record)
            counts[key]['complete']+=1
        else:
            outside.append(record)
            counts[key]['outside_prefix']+=1
    selected=[row for row in complete if stored[row['index']] in (0,1)]
    splits={}
    for code,key in ((0,'train'),(1,'val')):
        indices=[row['index'] for row in selected if stored[row['index']]==code]
        require(bool(indices),'empty available '+key+' split')
        require((ages[indices]>=0).all(),'unknown Age target in shared frozen subset')
        splits[key]={'indices':indices,'rows':len(indices),'ordered_ids_sha256':ids_hash(ids[indices]),
                     'age_class_counts':np.bincount(ages[indices],minlength=5).tolist(),
                     'geology_class_counts':np.bincount(geology[indices],minlength=49).tolist(),
                     'latitude_min':float(min(latitude[indices])),
                     'latitude_max':float(max(latitude[indices]))}
    return {'splits':splits,'complete_chunks':selected,
            'availability':{'source_rows':len(ids),'prefix_bytes':prefix_bytes,
                            'complete_chunk_count':len(complete),'outside_count':len(outside),
                            'metadata_unavailable_count':len(unavailable),'by_original_split':counts,
                            'missing_metadata_ranges':[{'start':start,'length':length,'affected_indices':count}
                                for (start,length),count in sorted(missing.items())],
                            'test_chunks_complete_but_excluded':counts['test']['complete'],
                            'excluded_test_rows':int(sum(stored==2))},
            'exclusions':{'metadata_unavailable_indices':unavailable,'outside_prefix_chunks':outside,
                          'test_indices':np.flatnonzero(stored==2).tolist()}}


def scan(data,range_report,targets,prefix_bytes,logical_bytes):
    import h5py
    with PrefixFile(data,logical_bytes,cached_blocks(range_report),prefix_bytes) as adapter:
        with h5py.File(adapter,'r') as source:
            return inventory(source['patches'],targets,prefix_bytes,logical_bytes)


def make_plan(original,audit,hashes,original_sha256,source_sha256):
    """Scientific task, metric and training settings are copied without edits."""
    plan=copy.deepcopy(original)
    plan.update(protocol='P31_LOCAL_PREFIX_STORED_DERIVED_V1',status='LOCKED_BEFORE_MODEL_SCORING',
                **audit)
    plan['data'].update(bytes=PREFIX_BYTES,sha256=hashes['data'],upstream_bytes=UPSTREAM_BYTES,
                        upstream_sha256=UPSTREAM_SHA256,upstream_full_integrity_verified=False,
                        upstream_integrity_status='UNVERIFIED',track='partial_stored_prefix',
                        metadata_sha256=hashes['metadata'],range_report_sha256=hashes['range_report'],
                        stats_sha256=hashes['stats'],class_map_sha256=hashes['class_map'])
    for key in ('train','val'):
        plan['data'][key+'_rows']=audit['splits'][key]['rows']
        plan['data'][key+'_ordered_ids_sha256']=audit['splits'][key]['ordered_ids_sha256']
    plan['provenance']={'original_plan_path':'protocols/p31_baseline_eval_plan.json',
                        'original_plan_sha256':original_sha256,'original_plan':copy.deepcopy(original),
                        'source_sha256':source_sha256,
                        'selection':'Original stored split 0/1; complete physical chunks within the local prefix only; original indices/order retained.',
                        'user_direction':'先用这些数据吧，vpn流量快顶不住了'}
    plan['limitations']={'full_official_benchmark':False,'unseen_class_claim':False,
                        'semantic_claim':False,'sampling_bias':'Contiguous downloaded prefix; availability-selected and geographically biased, not a representative official benchmark.',
                        'loader_claim':original['data']['loader_claim'],
                        'upstream_full_integrity':'UNVERIFIED: pinned full-file size/SHA256 are upstream claims, not verification of the retained prefix.'}
    plan['resource_boundary'].update(network_allowed=False,bulk_download_authorized=False)
    plan.update(pixel_reads=0,network_requests=0,test_evaluated=False)
    return plan


def write_new(path,value):
    """Publish complete JSON with an exclusive hard link; no partial target or overwrite."""
    path=Path(path)
    # Use the job scratch directory when provided, including all offline tests.
    scratch=Path(os.environ['CLAUDE_JOB_DIR'])/'tmp' if os.environ.get('CLAUDE_JOB_DIR') else path.parent
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=scratch,
                                         prefix='.p31_plan_',suffix='.json',delete=False) as output:
            temporary=Path(output.name)
            json.dump(value,output,indent=2,allow_nan=False,ensure_ascii=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary,path)  # Atomic and fails if a target (including a symlink) already exists.
    finally:
        if temporary is not None: temporary.unlink()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','metadata','range-report','stats','class-map','outplan'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    if args.outplan.exists() or args.outplan.is_symlink(): raise FileExistsError(args.outplan)
    original,original_hash=read_json(ORIGINAL_PLAN)
    require(original['protocol']=='P31_03_STORED_DERIVED_SCRATCH_BASELINES_V1' and
            original['data']['bytes']==UPSTREAM_BYTES and original['data']['sha256']==UPSTREAM_SHA256
            and original['test']['runner_allows_test'] is False,'original baseline provenance mismatch')
    hashes={}
    raw,hashes['metadata']=read_json(args.metadata,METADATA_SHA256)
    _,hashes['stats']=read_json(args.stats,original['data']['stats_sha256'])
    mapping,hashes['class_map']=read_json(args.class_map,original['data']['class_map_sha256'])
    report,hashes['range_report']=read_json(args.range_report)
    require(all(report['input_sha256'][key]==hashes[name] for key,name in
                (('metadata','metadata'),('hf_stats','stats'),('class_map','class_map'))),
            'range report audited input provenance mismatch')
    targets=metadata_targets(raw,mapping,original)
    before=stamp(args.data)
    require(before[2]==PREFIX_BYTES,'unexpected local prefix size')
    # No per-row/per-epoch hashing and no full-file hash claim for this partial asset.
    hashes['data']=stream_sha256(args.data,PREFIX_BYTES)
    audit=scan(args.data,args.range_report,targets,PREFIX_BYTES,UPSTREAM_BYTES)
    require(stamp(args.data)==before,'prefix mutated during planning')
    require(read_json(args.range_report)[1]==hashes['range_report'],'range report mutated during planning')
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    plan=make_plan(original,audit,hashes,original_hash,sources)
    write_new(args.outplan,plan)
    print(json.dumps({'protocol':plan['protocol'],'prefix_sha256':hashes['data'],
                      'rows':{key:value['rows'] for key,value in plan['splits'].items()},
                      'network_requests':0,'pixel_reads':0,'test_evaluated':False}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
