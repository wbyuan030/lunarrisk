"""Offline verification of P31 full-crop evidence; reads only existing local cache files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zlib


def sha(data):
    return hashlib.sha256(data).hexdigest()


class CachedFile(io.RawIOBase):
    def __init__(self,blocks,size):
        self.blocks,self.size,self.position=blocks,size,0
        self.reads=[]
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        position={0:0,1:self.position,2:self.size}[whence]+offset
        if not 0<=position<=self.size: raise ValueError('seek outside asset')
        self.position=position
        return position
    def read(self,size=-1):
        if size<0: raise ValueError('unbounded cache read')
        start_position=self.position
        end=min(self.position+size,self.size)
        output=[]
        while self.position<end:
            covering=[(start,body) for start,body in self.blocks if start<=self.position<start+len(body)]
            if not covering: raise ValueError('requested bytes absent from offline cache; network forbidden')
            start,body=max(covering,key=lambda pair:pair[0]+len(pair[1]))
            count=min(end-self.position,start+len(body)-self.position)
            output.append(body[self.position-start:self.position-start+count])
            self.position+=count
        self.reads.append({'start':start_position,'length':self.position-start_position})
        return b''.join(output)
    def readinto(self,target):
        body=self.read(len(target))
        target[:len(body)]=body
        return len(body)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo,run=args.repo.resolve(),args.run.resolve()
    sys.path.insert(0,str(repo/'src'))
    import h5py
    import numpy as np
    from lunarrisk.benchmarks.p31_pair import check_lock,ranges,compare_crop
    from lunarrisk.benchmarks.moonstone import CHANNELS
    report_path=run/'report.json'
    report=json.loads(report_path.read_text())
    plan=json.loads((repo/'protocols/p31_loader_pair_plan.json').read_text())
    stats=json.loads((repo/'reports/evidence/p30_20261103/hf_channel_stats.json').read_text())
    check_lock(plan)
    assert report['plan_lock_sha256']==plan['lock_sha256']
    for source,digest in report['source_sha256'].items(): assert sha((repo/source).read_bytes())==digest,source
    blocks=[]
    def cached(descriptor):
        path=(run/descriptor['cache']).resolve()
        if not path.is_relative_to(run): raise ValueError('cache path escapes run directory')
        body=path.read_bytes()
        assert len(body)==descriptor['length'] and sha(body)==descriptor['sha256'],descriptor['cache']
        return body
    for row in report['hdf5_ranges']: blocks.append((row['start'],cached(row)))
    for i,(start,body) in enumerate(blocks):
        for other_start,other in blocks[:i]:
            left,right=max(start,other_start),min(start+len(body),other_start+len(other))
            if left<right: assert body[left-start:right-start]==other[left-other_start:right-other_start]
    keys=('sample_id','channel','y','start','length')
    expected={tuple(row[k] for k in keys) for row in ranges(plan['records'])}
    actual=[tuple(row[k] for k in keys) for row in report['mmap_ranges']]
    assert len(actual)==len(set(actual)) and set(actual)==expected
    arrays={record['sample_id']:np.empty((28,256,256),dtype=np.float32) for record in plan['records']}
    for row in report['mmap_ranges']:
        arrays[row['sample_id']][CHANNELS.index(row['channel']),row['y']]=np.frombuffer(cached(row),dtype='<f4')
    hdf5_asset=next(asset for asset in plan['assets'] if asset['path']=='benchmark/lunar_patches_v4.h5')
    comparisons=[]
    chunk_info=[]
    with CachedFile(blocks,hdf5_asset['bytes']) as adapter:
        with h5py.File(adapter,'r') as source:
            for record,previous in zip(plan['records'],report['comparisons']):
                sample,index=record['sample_id'],record['index']
                assert previous['record']==record and record['stored_split'] in (0,1)
                for field,key in [('patch_id','sample_id'),('row_idx','row_idx'),('col_idx','col_idx'),
                                  ('split','stored_split'),('geology/dominant_unit_id','unit_id')]:
                    assert int(source['metadata/'+field][index])==int(record[key])
                name=source['metadata/geology/dominant_unit_name'][index]
                assert (name.decode() if isinstance(name,bytes) else str(name))==record['unit_name']
                np.testing.assert_array_equal(source['metadata/has_channel'][index],record['metadata_has_channel'])
                read_start=len(adapter.reads)
                raw=source['patches'][index]
                pixel_read=max(adapter.reads[read_start:],key=lambda row:row['length'])
                matching=[row for row in report['hdf5_ranges'] if row['start']==pixel_read['start'] and row['length']==pixel_read['length']]
                assert len(matching)==1
                assert not source['patches'].shuffle and not source['patches'].fletcher32
                assert zlib.decompress(cached(matching[0]))==raw.tobytes()
                saved=run/('raw_'+sample+'.npy')
                assert sha(saved.read_bytes())==previous['raw_crop_sha256']
                stored=np.load(saved,allow_pickle=False)
                assert stored.tobytes()==raw.tobytes()
                assert sha(raw.tobytes())==previous['raw_pixel_sha256']
                assert sha(arrays[sample].tobytes())==previous['mmap_pixel_sha256']
                recalculated=compare_crop(raw,arrays[sample],stats)
                assert recalculated=={key:previous[key] for key in recalculated}
                assert recalculated['numerical_and_masks_pass']
                comparisons.append(recalculated)
                chunk_info.append({'sample_id':sample,'chunk_offset':[index,0,0,0],
                                   'byte_offset':pixel_read['start'],'compressed_bytes':pixel_read['length'],
                                   'exact_hdf5_body_requests':len(matching),'zlib_payload_equals_selected_raw_pixels':True})
    assert len(comparisons)==len(plan['records'])==2
    assert all(chunk['exact_hdf5_body_requests']==1 for chunk in chunk_info)
    total=sum(ch['compared_pixels'] for c in comparisons for ch in c['channels'])
    valid=sum(ch['mmap_valid_pixels'] for c in comparisons for ch in c['channels'])
    missing=total-valid
    zero=sum(ch['raw_finite_zero_pixels'] for c in comparisons for ch in c['channels'])
    assert all(c['mmap_group_presence']==[True]*7 for c in comparisons)
    assert zero>0 and any(0<ch['raw_missing_pixels']<65536 for c in comparisons for ch in c['channels'])
    hbytes=sum(len(body) for _,body in blocks)
    mbytes=sum(row['length'] for row in report['mmap_ranges'])
    transfer=report['transfer']
    assert hbytes==transfer['hdf5_body_bytes'] and mbytes==sum(transfer['worker_body_bytes'])
    assert hbytes+mbytes==transfer['new_returned_body_bytes']
    assert transfer['cumulative_conservative_body_bytes']==165257071
    result={'status':'PASS_OFFLINE_CACHE_REPLAY','network_requests':0,
            'source_execution_report_sha256':sha(report_path.read_bytes()),'plan_lock_sha256':plan['lock_sha256'],
            'audit_script_sha256':sha(Path(__file__).read_bytes()),
            'cache_hashes_verified':len(report['hdf5_ranges'])+len(report['mmap_ranges']),
            'expected_mmap_range_set_exact':True,'hdf5_overlapping_cached_bytes_consistent':True,
            'metadata_and_raw_crops_reloaded_from_cached_hdf5':True,
            'complete_numeric_masks_and_presence_recomputed':True,'selected_hdf5_chunks':chunk_info,
            'compared_pixels':total,'valid_pixels':valid,'missing_pixels':missing,'finite_raw_zero_pixels':zero,
            'max_abs_error':max(ch['max_abs_error'] for c in comparisons for ch in c['channels']),
            'not_close_pixels':sum(ch['not_close_pixels'] for c in comparisons for ch in c['channels']),
            'mask_disagreements':sum(ch['mask_disagreements'] for c in comparisons for ch in c['channels']),
            'new_returned_body_bytes':hbytes+mbytes,'cumulative_returned_body_bytes':78502175+hbytes+mbytes,
            'cumulative_conservative_body_bytes':165257071,
            'scope':'two prelocked Train/Val crops; roughness has no valid pixels; no global chunk enumeration or whole-asset checksum claim'}
    with args.output.open('x') as output: output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
