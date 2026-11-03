"""Offline acquisition boundary tests; tiny in-memory fake HTTP bodies only."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lunarrisk.benchmarks.acquisition import acquire


class Response(io.BytesIO):
    def __init__(self,body,status=200,headers=None):
        super().__init__(body)
        self.status=status
        self.headers=headers if headers is not None else {'Content-Length':str(len(body))}
        self.read_sizes=[]
    def read(self,size=-1):
        self.read_sizes.append(size)
        return super().read(size)


class TestAcquisition(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.body=b'0123456789abcdef'
        self.asset={'path':'benchmark/fixture.h5','bytes':len(self.body),'sha256':hashlib.sha256(self.body).hexdigest(),
                    'url':'https://example.invalid/fixed-fixture'}
    def tearDown(self): self.tmp.cleanup()
    def invoke(self,opener,**kwargs):
        return acquire(self.asset,self.root,opener,chunk=4,workspace_bytes=0,**kwargs)

    def test_streaming_size_hash_and_atomic_rename(self):
        response=Response(self.body)
        with patch.object(Path,'read_bytes',side_effect=AssertionError('whole file read')):
            result=self.invoke(lambda *a,**k:response)
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual((self.root/'fixture.h5').read_bytes(),self.body)
        self.assertFalse((self.root/'fixture.h5.partial').exists())
        self.assertLessEqual(max(response.read_sizes),4)
        self.assertEqual(result['returned_body_bytes'],len(self.body))

    def test_truncated_response_resumes_only_remaining_range(self):
        responses=[Response(self.body[:8],headers={'Content-Length':'16'}),
                   Response(self.body[8:],206,{'Content-Length':'8','Content-Range':'bytes 8-15/16'})]
        offsets=[]
        def opener(request,**kwargs):
            offsets.append(request.get_header('Range'))
            return responses.pop(0)
        result=self.invoke(opener)
        self.assertEqual(offsets,[None,'bytes=8-'])
        self.assertEqual(result['returned_body_bytes'],16)
        self.assertEqual((self.root/'fixture.h5').read_bytes(),self.body)

    def make_interrupted(self):
        with self.assertRaises(EOFError):
            self.invoke(lambda *a,**k:Response(self.body[:8],headers={'Content-Length':'16'}),attempts=1)

    def test_resume_rejects_http200_before_reading_or_appending(self):
        self.make_interrupted()
        response=Response(self.body)
        with self.assertRaises(ValueError): self.invoke(lambda *a,**k:response,resume=True)
        self.assertEqual(response.read_sizes,[])
        self.assertEqual((self.root/'fixture.h5.partial').read_bytes(),self.body[:8])
        self.assertFalse((self.root/'fixture.h5').exists())

    def test_resume_rejects_wrong_range_and_encoding_before_body(self):
        self.make_interrupted()
        for headers in ({'Content-Length':'8','Content-Range':'bytes 0-7/16'},
                        {'Content-Length':'8','Content-Range':'bytes 8-15/16','Content-Encoding':'gzip'}):
            response=Response(self.body[8:],206,headers)
            with self.assertRaises(ValueError): self.invoke(lambda *a,**k:response,resume=True)
            self.assertEqual(response.read_sizes,[])

    def test_hash_mismatch_never_becomes_final_asset(self):
        with self.assertRaises(ValueError): self.invoke(lambda *a,**k:Response(b'x'*16))
        self.assertTrue((self.root/'fixture.h5.partial').exists())
        self.assertFalse((self.root/'fixture.h5').exists())
        self.assertEqual(json.loads((self.root/'download_manifest.json').read_text())['status'],'HASH_MISMATCH')

    def test_existing_verified_asset_needs_no_network_and_tampering_fails(self):
        self.invoke(lambda *a,**k:Response(self.body))
        def forbidden(*args,**kwargs): raise AssertionError('redundant network request')
        self.assertEqual(self.invoke(forbidden)['status'],'VERIFIED_EXISTING')
        (self.root/'fixture.h5').write_bytes(b'X'*16)
        with self.assertRaises(ValueError): self.invoke(forbidden)

    def test_unowned_partial_and_insufficient_disk_fail_before_network(self):
        def forbidden(*args,**kwargs): raise AssertionError('network before preflight')
        partial=self.root/'fixture.h5.partial'
        partial.write_bytes(b'unknown')
        with self.assertRaises(ValueError): self.invoke(forbidden,resume=True)
        partial.unlink()
        with patch('lunarrisk.benchmarks.acquisition.shutil.disk_usage') as usage:
            usage.return_value.free=0
            with self.assertRaises(ValueError): acquire(self.asset,self.root,forbidden,workspace_bytes=100)


if __name__=='__main__': unittest.main()
