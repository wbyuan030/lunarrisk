#!/usr/bin/env python3
"""Acquire the explicitly authorized single fixed HDF5; default offline dry-run."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    from lunarrisk.benchmarks.acquisition import acquire,file_hash
    authorization=json.loads((ROOT/'protocols/p31_acquisition_authorization.json').read_text())
    asset=authorization['asset']
    expected=('ayushprd/Moonstone','53099efd0f2db81eee96b28eae87e1efdd281310',
              'benchmark/lunar_patches_v4.h5',74449119347,
              'e3e96bccc044d856cd55883c9cda7628f02afbdc1b2d1818b401bbcc590df83f')
    if tuple(asset[k] for k in ('repository','revision','path','bytes','sha256'))!=expected:
        raise ValueError('asset is outside the single-file acquisition scope')
    if authorization['status']!='EXPLICIT_USER_AUTHORIZATION': raise ValueError('missing acquisition authorization')
    print(json.dumps({'mode':'apply' if args.apply else 'dry_run','asset':asset,
                      'single_copy':True,'test_scoring':False},indent=2),flush=True)
    if not args.apply: return 0
    from lunarrisk.benchmarks.p31_pair import PersistentOpener
    opener=PersistentOpener()
    try:
        report=acquire(asset,args.outdir,opener,resume=args.resume,
                       progress=lambda row:print(json.dumps(row),flush=True))
    finally:
        opener.close()
    print(json.dumps({'status':report['status'],'verified_bytes':report.get('verified_bytes'),
                      'verified_sha256':report.get('verified_sha256'),
                      'authorization_sha256':file_hash(ROOT/'protocols/p31_acquisition_authorization.json')},indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
