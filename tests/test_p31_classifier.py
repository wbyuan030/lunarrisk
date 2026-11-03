"""P3.1 streaming, split, mini-batch and state contracts; synthetic data only."""
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
try:
    import h5py
    import torch
    from lunarrisk.benchmarks.classifier import (
        ClassifierDataset,MetadataTargets,make_loader,stream_sha256,classification_metrics,
        seed_all,rng_state,restore_rng,atomic_checkpoint,state_fingerprint,cpu_state,export_features)
    from lunarrisk.benchmarks.moonstone import CHANNELS
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"torch/h5py needed for offline classifier fixtures")
class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.path=self.root/"fixture.h5"
        self.stats={name:[0,1] for name in CHANNELS}
        self.mapping={f"C{i}":i for i in range(1,50)}
        n=21
        with h5py.File(self.path,"w") as f:
            f.attrs["channel_names"]=json.dumps(CHANNELS)
            f.create_dataset("patches",shape=(n,28,256,256),dtype="float32",chunks=(1,28,256,256),fillvalue=1)
            meta=f.create_group("metadata")
            for key,values in (("patch_id",np.arange(n)),("row_idx",np.zeros(n,dtype=int)),
                    ("col_idx",np.arange(n)),("split",[0]*11+[1]*9+[2]),("center_lat",[89.]*n),
                    ("geology/dominant_unit_id",np.ones(n,dtype=int))):
                meta.create_dataset(key,data=values)
            meta.create_dataset("geology/dominant_unit_name",data=np.asarray(["C1"]*n,dtype=h5py.string_dtype()))

    def tearDown(self):
        self.tmp.cleanup()

    def test_stream_hash_uses_bounded_reads_not_path_read_bytes(self):
        path=self.root/"known.bin"
        raw=b"bounded-hash-fixture"*503
        path.write_bytes(raw)
        with patch.object(Path,"read_bytes",side_effect=AssertionError("whole file read")):
            self.assertEqual(stream_sha256(path,13),hashlib.sha256(raw).hexdigest())
        with self.assertRaises(ValueError): stream_sha256(path,0)

    def test_more_than_smoke_limit_last_batch_and_no_test_pixel_access(self):
        dataset=ClassifierDataset(self.path,self.stats,self.mapping)
        read=h5py.Dataset.__getitem__
        def guarded(source,index):
            if source.name=="/patches":
                self.assertIsInstance(index,int)
                self.assertLess(index,20)
            return read(source,index)
        with patch.object(h5py.Dataset,"__getitem__",guarded):
            batches=list(make_loader(dataset,4,False,torch.Generator().manual_seed(2)))
        self.assertEqual([len(b["target"]) for b in batches],[4,4,3])
        self.assertEqual([x for b in batches for x in b["sample_id"]],list(map(str,range(11))))
        dataset.close()
        with self.assertRaises(ValueError): ClassifierDataset(self.path,self.stats,self.mapping,split=2)

    def test_invalid_ids_split_target_and_metadata_lengths_fail(self):
        for kind in ("id","split","target","length"):
            with self.subTest(kind=kind):
                with h5py.File(self.path,"r+") as f:
                    if kind=="id": f["metadata/patch_id"][1]=0
                    if kind=="split": f["metadata/split"][0]=7
                    if kind=="target": f["metadata/geology/dominant_unit_id"][0]=0
                    if kind=="length":
                        del f["metadata/center_lat"]
                        f.create_dataset("metadata/center_lat",data=[89.]*20)
                with self.assertRaises(ValueError): ClassifierDataset(self.path,self.stats,self.mapping)
                # Restore independent invalid-fixture variants without rewriting any real asset.
                with h5py.File(self.path,"r+") as f:
                    if kind=="id": f["metadata/patch_id"][1]=1
                    if kind=="split": f["metadata/split"][0]=0
                    if kind=="target": f["metadata/geology/dominant_unit_id"][0]=1
                    if kind=="length":
                        del f["metadata/center_lat"]
                        f.create_dataset("metadata/center_lat",data=[89.]*21)

    def test_worker_pickle_drops_handle_and_prior_never_reads_pixels(self):
        dataset=ClassifierDataset(self.path,self.stats,self.mapping)
        dataset._file()
        cloned=pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(cloned._handle)
        self.assertEqual(cloned[0]["sample_id"],"0")
        with patch.object(ClassifierDataset,"__getitem__",side_effect=AssertionError("prior read pixels")):
            rows=list(make_loader(MetadataTargets(dataset),4,False,torch.Generator()))
        self.assertEqual(sum(len(r["target"]) for r in rows),11)
        dataset.close()
        cloned.close()

    def test_two_workers_keep_selected_ids_and_parent_handle(self):
        dataset=ClassifierDataset(self.path,self.stats,self.mapping)
        parent=dataset._file()
        batches=list(make_loader(dataset,4,False,torch.Generator().manual_seed(2),workers=2))
        self.assertEqual([x for b in batches for x in b["sample_id"]],list(map(str,range(11))))
        self.assertTrue(parent.id.valid)
        dataset.close()

    def test_macro_full_matches_upstream_zero_division_supported_is_secondary(self):
        matrix=np.zeros((49,49),dtype=int)
        matrix[4,4]=3
        result=classification_metrics(matrix)
        self.assertEqual(result["accuracy"],1)
        self.assertAlmostEqual(result["macro_f1_full"],1/49)
        self.assertEqual(result["macro_f1_supported"],1)
        self.assertEqual(result["per_class_recall"][0],0)

    def test_checkpoint_replay_and_rng_restoration(self):
        seed_all(42)
        model=torch.nn.Linear(4,3)
        x=torch.randn(2,4)
        reference=model(x).detach().clone()
        state=cpu_state(model)
        loader_rng=torch.Generator().manual_seed(8)
        loader_state=loader_rng.get_state()
        payload={"state":state,"rng":rng_state(),"loader":loader_state}
        path=self.root/"checkpoint.pt"
        atomic_checkpoint(path,payload)
        expected=(random.random(),np.random.random(),torch.rand(2),torch.randperm(11,generator=loader_rng))
        other=torch.nn.Linear(4,3)
        restored=torch.load(path,weights_only=True,map_location="cpu")
        other.load_state_dict(restored["state"])
        torch.testing.assert_close(other(x),reference,rtol=0,atol=0)
        self.assertEqual(state_fingerprint(cpu_state(other)),state_fingerprint(state))
        restore_rng(restored["rng"])
        loader_rng.set_state(restored["loader"])
        actual=(random.random(),np.random.random(),torch.rand(2),torch.randperm(11,generator=loader_rng))
        self.assertEqual(actual[:2],expected[:2])
        torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
        torch.testing.assert_close(actual[3],expected[3],rtol=0,atol=0)

    def test_feature_export_is_chunked_and_handles_final_partial_batch(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(28,256))
        dataset=ClassifierDataset(self.path,self.stats,self.mapping)
        path=self.root/"features.h5"
        export_features(Tiny(),{"train":dataset},path,torch.device("cpu"),4)
        with h5py.File(path) as output:
            self.assertEqual(output["train/features"].shape,(11,256))
            self.assertEqual(output["train/features"].chunks,(4,256))
            self.assertEqual(output["train/sample_ids"].asstr()[:].tolist(),list(map(str,range(11))))
        dataset.close()


if __name__=="__main__": unittest.main()
