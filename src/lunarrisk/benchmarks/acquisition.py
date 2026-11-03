"""Single-copy streaming acquisition with strict resume headers and verified atomic rename."""
from __future__ import annotations
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request


def file_hash(path,chunk=4*1024*1024):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(chunk),b''): digest.update(block)
    return digest.hexdigest()


def response_contract(response,offset,size):
    if response.headers.get('Content-Encoding','identity').lower()!='identity':
        raise ValueError('encoded body forbidden')
    status=response.status
    if offset or status==206:
        if status!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{size-1}/{size}':
            raise ValueError('resume requires exact HTTP206 and Content-Range before body')
    elif status!=200:
        raise ValueError('full download requires HTTP200 or exact HTTP206')
    length=response.headers.get('Content-Length')
    if length is not None and int(length)!=size-offset:
        raise ValueError('response length differs from fixed asset')


def acquire(asset,outdir,opener,*,resume=False,chunk=4*1024*1024,attempts=3,
            workspace_bytes=80*1024**3,progress=None):
    if chunk<1 or attempts<1: raise ValueError('positive chunk and attempts required')
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    final=outdir/Path(asset['path']).name
    partial=final.with_suffix(final.suffix+'.partial')
    manifest_path=outdir/'download_manifest.json'
    with (outdir/'download.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if previous and previous['asset']!=asset: raise ValueError('existing download belongs to another asset')
        if partial.exists() and not previous: raise ValueError('unowned partial file; refusing adoption')
        if final.exists() and partial.exists(): raise ValueError('both final and partial exist; audit required')
        if partial.exists() and not resume: raise ValueError('owned partial requires explicit resume')
        size=asset['bytes']
        if partial.exists() and partial.stat().st_size>size: raise ValueError('partial exceeds fixed asset size')
        manifest=previous or {'asset':asset,'returned_body_bytes':0,'attempts':[],
                             'policy':'single partial/final copy; no snapshot; no automatic restart from zero'}
        def save(status,**fields):
            manifest.update(status=status,**fields)
            temporary=manifest_path.with_suffix('.json.partial')
            temporary.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
            temporary.replace(manifest_path)
        if final.exists():
            if final.stat().st_size!=size or file_hash(final,chunk)!=asset['sha256']:
                raise ValueError('existing final asset failed full verification')
            save('VERIFIED_EXISTING',verified_bytes=size,verified_sha256=asset['sha256'])
            return manifest
        offset=partial.stat().st_size if partial.exists() else 0
        if shutil.disk_usage(outdir).free<max(0,workspace_bytes-offset):
            raise ValueError('insufficient free workspace for single-copy acquisition')
        save('DOWNLOADING',partial_bytes=offset,chunk_bytes=chunk)
        begin=time.monotonic()
        try:
            for attempt in range(attempts):
                offset=partial.stat().st_size if partial.exists() else 0
                if offset==size: break
                headers={'Accept-Encoding':'identity'}
                if offset: headers['Range']=f'bytes={offset}-'
                row={'start_offset':offset,'returned_body_bytes':0,'status':'STARTED'}
                manifest['attempts'].append(row)
                save('DOWNLOADING')
                try:
                    with opener(Request(asset['url'],headers=headers),timeout=90) as response:
                        response_contract(response,offset,size)
                        with partial.open('ab') as target:
                            last=time.monotonic()
                            while offset<size:
                                body=response.read(min(chunk,size-offset))
                                if not body: raise EOFError('truncated fixed asset')
                                if len(body)>min(chunk,size-offset): raise ValueError('body exceeds requested read')
                                target.write(body)
                                offset+=len(body)
                                row['returned_body_bytes']+=len(body)
                                manifest['returned_body_bytes']+=len(body)
                                if time.monotonic()-last>=30:
                                    target.flush()
                                    save('DOWNLOADING',partial_bytes=offset)
                                    if progress: progress({'status':'DOWNLOADING','bytes':offset,'total_bytes':size,
                                                           'elapsed_seconds':time.monotonic()-begin})
                                    last=time.monotonic()
                            if response.read(1): raise ValueError('body exceeds fixed size')
                    row['status']='COMPLETE'
                    break
                except (OSError,EOFError) as error:
                    row.update(status='INTERRUPTED',error_type=type(error).__name__)
                    save('INTERRUPTED',partial_bytes=partial.stat().st_size if partial.exists() else 0)
                    if attempt+1==attempts: raise
            if not partial.exists() or partial.stat().st_size!=size: raise ValueError('asset size incomplete')
            save('VERIFYING',partial_bytes=size,download_seconds_this_process=time.monotonic()-begin)
            hash_begin=time.monotonic()
            actual=file_hash(partial,chunk)
            if actual!=asset['sha256']:
                save('HASH_MISMATCH',actual_sha256=actual)
                raise ValueError('full asset SHA256 mismatch; partial retained without rename')
            partial.replace(final)
            save('VERIFIED',verified_bytes=size,verified_sha256=actual,
                 hash_seconds=time.monotonic()-hash_begin,
                 transport_note='returned body bytes exclude redirect/HTTP/TLS overhead and bytes consumed internally before a transport exception')
            return manifest
        except Exception as error:
            if manifest.get('status')!='HASH_MISMATCH': save('FAILED',error_type=type(error).__name__,partial_bytes=partial.stat().st_size if partial.exists() else 0)
            raise
