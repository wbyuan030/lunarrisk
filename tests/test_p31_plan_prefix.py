"""Offline prefix locking: synthetic metadata/descriptors, no real asset hashing or scoring."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('p31_plan_prefix',ROOT/'scripts/p31_plan_prefix.py')
planner=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def sha(body): return hashlib.sha256(body).hexdigest()


def fixtures():
    original=json.loads((ROOT/'protocols/p31_baseline_eval_plan.json').read_text())
    names=['pN0','I1','N2','E3','C4']+[f'pN{i}' for i in range(5,49)]
    mapping={name:i+1 for i,name in enumerate(names)}
    raw={'attributes':{'channel_names':list(planner.CHANNELS)},
         'metadata':{'metadata/patch_id':[f'id{i}' for i in range(8)],
                     'metadata/split':[0,1,2,0,1,2,0,1],
                     'metadata/center_lat':[89.,87.,85.,7.,9.,-1.,-3.,-5.],
                     'metadata/geology/dominant_unit_id':list(range(1,9)),
                     'metadata/geology/dominant_unit_name':names[:8]}}
    for code,key in ((0,'train'),(1,'val')):
        ids=[value for value,split in zip(raw['metadata']['metadata/patch_id'],
                                        raw['metadata']['metadata/split']) if split==code]
        original['data'].update({key+'_rows':len(ids),key+'_ordered_ids_sha256':planner.ids_hash(ids)})
    return raw,mapping,original


class Descriptors:
    """An HDF5-like dataset which fails if any pixel read is attempted."""
    def __init__(self):
        self.shape=(8,28,256,256)
        self.chunks=(1,28,256,256)
        self.dtype=np.dtype('float32')
        self.id=self
        self.queries=[]
        self.ranges=[(10,10),(30,10),(50,10),(90,10),(98,3),(101,2),None,None]
    def __getitem__(self,key): raise AssertionError('pixel read forbidden')
    def get_chunk_info_by_coord(self,coord):
        self.queries.append(coord)
        value=self.ranges[coord[0]]
        if value is None: raise planner.MissingLocalBytes(120,8)
        return SimpleNamespace(byte_offset=value[0],size=value[1])


class TestPrefixInventory(unittest.TestCase):
    def setUp(self):
        self.raw,self.mapping,self.original=fixtures()
        self.targets=planner.metadata_targets(self.raw,self.mapping,self.original)
        self.patches=Descriptors()

    def inventory(self): return planner.inventory(self.patches,self.targets,100,200)

    def test_every_row_descriptor_only_original_indices_boundary_and_test_exclusion(self):
        audit=self.inventory()
        self.assertEqual(self.patches.queries,[(i,0,0,0) for i in range(8)])
        self.assertEqual(audit['splits']['train']['indices'],[0,3])
        self.assertEqual(audit['splits']['val']['indices'],[1])
        self.assertEqual([r['index'] for r in audit['complete_chunks']],[0,1,3])
        train=audit['splits']['train']
        self.assertEqual(train['ordered_ids_sha256'],sha(b'id0\nid3'))
        self.assertEqual((train['latitude_min'],train['latitude_max']),(7.,89.))
        self.assertEqual(train['age_class_counts'],[1,0,0,1,0])
        self.assertEqual(train['geology_class_counts'],[1,0,0,1]+[0]*45)
        self.assertEqual(audit['availability']['complete_chunk_count'],4)
        self.assertEqual(audit['availability']['outside_count'],2)
        self.assertEqual(audit['availability']['metadata_unavailable_count'],2)
        self.assertEqual(audit['availability']['test_chunks_complete_but_excluded'],1)
        self.assertEqual(audit['availability']['excluded_test_rows'],2)
        self.assertEqual(audit['exclusions']['test_indices'],[2,5])
        self.assertEqual(audit['exclusions']['metadata_unavailable_indices'],[6,7])
        self.assertEqual(audit['availability']['missing_metadata_ranges'],
                         [{'start':120,'length':8,'affected_indices':2}])

    def test_metadata_validation_rejects_misalignment_duplicates_split_and_label_drift(self):
        for mutation in ('length','duplicate','split','latitude','mapping','channel','provenance'):
            with self.subTest(mutation=mutation):
                raw,mapping,original=fixtures()
                meta=raw['metadata']
                if mutation=='length': meta['metadata/center_lat'].pop()
                if mutation=='duplicate': meta['metadata/patch_id'][1]='id0'
                if mutation=='split': meta['metadata/split'][0]=3
                if mutation=='latitude': meta['metadata/center_lat'][0]=float('nan')
                if mutation=='mapping': meta['metadata/geology/dominant_unit_id'][0]=49
                if mutation=='channel': raw['attributes']['channel_names'].reverse()
                if mutation=='provenance': original['data']['train_ordered_ids_sha256']='0'*64
                with self.assertRaises(ValueError): planner.metadata_targets(raw,mapping,original)

    def test_invalid_schema_and_ranges_do_not_silently_fill(self):
        for mutation in ('shape','chunks','dtype','negative','overflow','unallocated'):
            with self.subTest(mutation=mutation):
                self.patches=Descriptors()
                if mutation=='shape': self.patches.shape=(7,28,256,256)
                if mutation=='chunks': self.patches.chunks=(2,28,256,256)
                if mutation=='dtype': self.patches.dtype=np.dtype('float64')
                if mutation=='negative': self.patches.ranges[0]=(-1,10)
                if mutation=='overflow': self.patches.ranges[0]=(195,10)
                if mutation=='unallocated': self.patches.ranges[0]=(None,10)
                with self.assertRaises(ValueError): self.inventory()

    def test_no_usable_validation_rows_is_rejected(self):
        self.patches.ranges[1]=(101,3)
        with self.assertRaisesRegex(ValueError,'empty available val'): self.inventory()

    def test_unknown_age_refuses_ambiguous_shared_task_rows(self):
        self.raw['metadata']['metadata/geology/dominant_unit_name'][0]='unknown'
        self.mapping['unknown']=self.mapping.pop('pN0')
        self.targets=planner.metadata_targets(self.raw,self.mapping,self.original)
        with self.assertRaisesRegex(ValueError,'unknown Age'): self.inventory()

    def test_original_scientific_settings_copied_and_upstream_claim_explicit(self):
        original=copy.deepcopy(self.original)
        audit=self.inventory()
        hashes={key:character*64 for key,character in zip(
            ('data','metadata','range_report','stats','class_map'),'abcde')}
        plan=planner.make_plan(original,audit,hashes,'f'*64,{'source':'0'*64})
        self.assertEqual(plan['protocol'],'P31_LOCAL_PREFIX_STORED_DERIVED_V1')
        for key in ('training','metrics','tasks','baselines','test','candidate_controls'):
            self.assertEqual(plan[key],original[key])
        self.assertEqual(plan['provenance']['original_plan'],original)
        self.assertEqual(plan['data']['bytes'],33964443553)
        self.assertEqual(plan['data']['upstream_bytes'],74449119347)
        self.assertEqual(plan['data']['upstream_sha256'],planner.UPSTREAM_SHA256)
        self.assertFalse(plan['data']['upstream_full_integrity_verified'])
        self.assertEqual(plan['data']['upstream_integrity_status'],'UNVERIFIED')
        self.assertEqual(plan['data']['track'],'partial_stored_prefix')
        self.assertEqual(plan['data']['sha256'],hashes['data'])
        self.assertEqual(plan['data']['metadata_sha256'],hashes['metadata'])
        self.assertEqual(plan['data']['range_report_sha256'],hashes['range_report'])
        self.assertFalse(plan['test']['runner_allows_test'])
        self.assertIn('HOLD',plan['test']['model_scoring'])
        for key in ('full_official_benchmark','unseen_class_claim','semantic_claim'):
            self.assertFalse(plan['limitations'][key])
        self.assertEqual((plan['pixel_reads'],plan['network_requests'],plan['test_evaluated']),(0,0,False))
        plan['training']['seed'].append(99)
        self.assertEqual(original,self.original)


class TestPrefixPlannerIO(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.raw,self.mapping,self.original=fixtures()
        self.data=self.root/'prefix.partial'
        self.data.write_bytes(b'x'*100)
        self.metadata=self.root/'metadata.json'
        self.stats=self.root/'stats.json'
        self.class_map=self.root/'map.json'
        self.report=self.root/'report.json'
        self.original_path=self.root/'original.json'
        self.outplan=self.root/'lock.json'
        self.metadata.write_text(json.dumps(self.raw))
        self.stats.write_text('{}')
        self.class_map.write_text(json.dumps(self.mapping))
        self.original['data'].update(stats_sha256=sha(self.stats.read_bytes()),
                                     class_map_sha256=sha(self.class_map.read_bytes()))
        self.original_path.write_text(json.dumps(self.original))
        report={'input_sha256':{'metadata':sha(self.metadata.read_bytes()),
                               'hf_stats':sha(self.stats.read_bytes()),
                               'class_map':sha(self.class_map.read_bytes())},'hdf5_ranges':[]}
        self.report.write_text(json.dumps(report))
        self.argv=[item for flag,path in (('data',self.data),('metadata',self.metadata),
                    ('range-report',self.report),('stats',self.stats),('class-map',self.class_map),
                    ('outplan',self.outplan)) for item in ('--'+flag,str(path))]
        self.audit=planner.inventory(Descriptors(),planner.metadata_targets(self.raw,self.mapping,self.original),100,200)

    def invoke(self):
        with patch.object(planner,'PREFIX_BYTES',100),patch.object(planner,'ORIGINAL_PLAN',self.original_path), \
             patch.object(planner,'METADATA_SHA256',sha(self.metadata.read_bytes())), \
             contextlib.redirect_stdout(io.StringIO()):
            return planner.main(self.argv)

    def test_cli_hashes_prefix_once_and_publishes_deterministic_offline_lock(self):
        import socket
        import subprocess
        import urllib.request
        real_read=Path.read_bytes
        def guarded_read(path):
            if path==self.data: raise AssertionError('whole-prefix read_bytes forbidden')
            return real_read(path)
        with patch.object(planner,'scan',return_value=self.audit) as scan, \
             patch.object(planner,'stream_sha256',wraps=planner.stream_sha256) as stream, \
             patch.object(Path,'read_bytes',guarded_read), \
             patch.object(socket,'socket',side_effect=AssertionError('network forbidden')), \
             patch.object(urllib.request,'urlopen',side_effect=AssertionError('network forbidden')), \
             patch.object(subprocess,'run',side_effect=AssertionError('child execution forbidden')):
            self.assertEqual(self.invoke(),0)
            stream.assert_called_once_with(self.data,100)
            scan.assert_called_once()
            first=self.outplan.read_bytes()
            plan=json.loads(first)
            self.assertEqual(plan['data']['sha256'],sha(b'x'*100))
            self.assertEqual(plan['data']['bytes'],100)
            self.assertEqual(plan['splits']['train']['indices'],[0,3])
            self.outplan.unlink()
            self.invoke()
            self.assertEqual(first,self.outplan.read_bytes())

    def test_cli_refuses_existing_or_dangling_symlink_before_hashing(self):
        with patch.object(planner,'stream_sha256') as stream:
            for symlink in (False,True):
                if symlink: self.outplan.symlink_to(self.root/'absent')
                else: self.outplan.write_text('preserve')
                with self.assertRaises(FileExistsError): self.invoke()
                if not symlink: self.assertEqual(self.outplan.read_text(),'preserve')
                self.outplan.unlink()
            stream.assert_not_called()

    def test_cli_rejects_audited_hash_or_report_mismatch_without_prefix_pass(self):
        for path in (self.stats,self.class_map,self.report):
            with self.subTest(path=path),patch.object(planner,'stream_sha256') as stream:
                body=path.read_bytes()
                path.write_text('{}' if path!=self.stats else '{"changed":true}')
                with self.assertRaises((ValueError,KeyError)): self.invoke()
                self.assertFalse(self.outplan.exists())
                stream.assert_not_called()
                path.write_bytes(body)
        with patch.object(planner,'METADATA_SHA256','0'*64), \
             patch.object(planner,'ORIGINAL_PLAN',self.original_path), \
             patch.object(planner,'stream_sha256') as stream:
            with self.assertRaisesRegex(ValueError,'audited SHA256 mismatch'): planner.main(self.argv)
            stream.assert_not_called()

    def test_cli_wrong_size_and_concurrent_mutation_never_publish(self):
        self.data.write_bytes(b'short')
        with patch.object(planner,'stream_sha256') as stream:
            with self.assertRaisesRegex(ValueError,'unexpected local prefix size'): self.invoke()
            stream.assert_not_called()
        self.data.write_bytes(b'x'*100)
        def mutate(*args):
            before=self.data.stat()
            self.data.write_bytes(b'y'*100)
            # Same-size writes may share a filesystem clock tick. Make this
            # observable-metadata mutation fixture independent of wall-clock speed.
            os.utime(self.data,ns=(before.st_atime_ns,before.st_mtime_ns+1_000_000_000))
            return self.audit
        with patch.object(planner,'scan',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'prefix mutated during planning'): self.invoke()
        self.assertFalse(self.outplan.exists())

    def test_atomic_exclusive_write_and_serialization_failure(self):
        self.outplan.write_text('preserve')
        with self.assertRaises(FileExistsError): planner.write_new(self.outplan,{'value':1})
        self.assertEqual(self.outplan.read_text(),'preserve')
        self.outplan.unlink()
        with self.assertRaises(ValueError): planner.write_new(self.outplan,{'value':float('nan')})
        self.assertFalse(self.outplan.exists())
        with patch.object(planner.os,'link',side_effect=OSError('publication failed')):
            with self.assertRaises(OSError): planner.write_new(self.outplan,{'value':1})
        self.assertFalse(self.outplan.exists())
        planner.write_new(self.outplan,{'value':1})
        self.assertEqual(json.loads(self.outplan.read_text()),{'value':1})
        scratch=Path(os.environ['CLAUDE_JOB_DIR'])/'tmp' if os.environ.get('CLAUDE_JOB_DIR') else self.root
        self.assertFalse(list(scratch.glob('.p31_plan_*')))

    def test_streamed_hash_is_bounded_and_rejects_mutation(self):
        body=b'a'*(4*1024*1024+7)
        self.data.write_bytes(body)
        real_open=Path.open
        reads=[]
        class Tracked:
            def __init__(self,source): self.source=source
            def __enter__(self): return self
            def __exit__(self,*args): self.source.close()
            def read(self,size):
                reads.append(size)
                return self.source.read(size)
        def tracked_open(path,*args,**kwargs):
            source=real_open(path,*args,**kwargs)
            return Tracked(source) if path==self.data else source
        with patch.object(Path,'open',tracked_open):
            self.assertEqual(planner.stream_sha256(self.data,len(body)),sha(body))
        self.assertEqual(reads,[4*1024*1024]*3)
        with patch.object(planner,'stamp',side_effect=[(1,2,len(body),3,4),(1,2,len(body),4,5)]):
            with self.assertRaisesRegex(ValueError,'mutated during hashing'):
                planner.stream_sha256(self.data,len(body))

    def test_real_h5py_descriptor_scan_reads_no_pixel_chunks(self):
        import h5py
        local=io.BytesIO()
        with h5py.File(local,'w') as handle:
            dataset=handle.create_dataset('patches',(8,28,256,256),dtype='float32',
                                          chunks=(1,28,256,256),compression='gzip')
            for index in range(8): dataset[index]=index+1
        blob=local.getvalue()
        with h5py.File(io.BytesIO(blob),'r') as handle:
            chunks=[handle['patches'].id.get_chunk_info_by_coord((i,0,0,0)) for i in range(8)]
        # Cache the suffix, including otherwise available pixel bytes: cache coverage
        # must never make an out-of-prefix chunk eligible for the frozen subset.
        prefix_bytes=chunks[3].byte_offset+chunks[3].size
        self.data.write_bytes(blob[:prefix_bytes])
        cache=self.root/'suffix.bin'
        cache.write_bytes(blob[prefix_bytes:])
        self.report.write_text(json.dumps({'hdf5_ranges':[{'start':prefix_bytes,
            'length':len(blob)-prefix_bytes,'sha256':sha(blob[prefix_bytes:]),'cache':cache.name}]}))
        seen=[]
        base=planner.PrefixFile
        class TrackedPrefix(base):
            def read(self,size=-1):
                seen.append((self.tell(),self.tell()+size))
                return super().read(size)
        with patch.object(planner,'PrefixFile',TrackedPrefix):
            audit=planner.scan(self.data,self.report,
                               planner.metadata_targets(self.raw,self.mapping,self.original),prefix_bytes,len(blob))
        self.assertEqual(audit['availability']['complete_chunk_count'],4)
        self.assertEqual(audit['availability']['outside_count'],4)
        self.assertEqual(audit['splits']['train']['indices'],[0,3])
        self.assertEqual(audit['splits']['val']['indices'],[1])
        self.assertTrue(seen)
        for chunk in chunks:
            self.assertFalse(any(start<chunk.byte_offset+chunk.size and end>chunk.byte_offset
                                 for start,end in seen),'descriptor scan read a pixel chunk')


if __name__=='__main__': unittest.main()
