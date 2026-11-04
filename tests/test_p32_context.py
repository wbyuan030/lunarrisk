import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from lunarrisk.benchmarks.classifier import ScratchClassifier, ids_sha256, state_fingerprint, cpu_state, seed_all
from lunarrisk.lunarjev.context import FrozenContext, validate_arrays


class ContextTests(unittest.TestCase):
    def test_frozen_spatial_shape_pool_and_bn(self):
        seed_all(42)
        model = ScratchClassifier(49)
        context = FrozenContext(model)
        before = state_fingerprint(cpu_state(model))
        context.train(True)
        self.assertFalse(any(module.training for module in context.modules()))
        pixels = torch.randn(2,28,256,256)
        tokens = context(pixels)
        self.assertEqual(tuple(tokens.shape), (2,16,256))
        self.assertFalse(tokens.requires_grad)
        self.assertTrue(torch.equal(tokens,context(pixels)))
        self.assertTrue(torch.allclose(tokens.mean(1),model.features(pixels),atol=1e-6,rtol=1e-5))
        self.assertEqual(before,state_fingerprint(cpu_state(model)))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in context.parameters()))

    def arrays(self):
        tokens = np.arange(2*16*256,dtype=np.float32).reshape(2,16,256)
        geology, age = np.array([0,1]), np.array([0,1])
        frozen = {'rows':2,'ordered_ids_sha256':ids_sha256(['a','b']),
                  'geology_class_counts':np.bincount(geology,minlength=49).tolist(),
                  'age_class_counts':np.bincount(age,minlength=5).tolist()}
        return [tokens,tokens.mean(1),['a','b'],geology,age,'train',frozen]

    def test_finite_arrays_and_ids(self):
        validate_arrays(*self.arrays())

    def test_test_split_rejected(self):
        args = self.arrays(); args[5]='test'
        with self.assertRaises(ValueError): validate_arrays(*args)

    def test_wrong_ids_rejected(self):
        args = self.arrays(); args[2]=['b','a']
        with self.assertRaises(ValueError): validate_arrays(*args)

    def test_nonfinite_rejected(self):
        args = self.arrays(); args[0][0,0,0]=np.nan
        with self.assertRaises(ValueError): validate_arrays(*args)

    def test_different_pooled_control_rejected(self):
        args = self.arrays(); args[1][0,0]+=1
        with self.assertRaises(ValueError): validate_arrays(*args)

    def test_wrong_support_rejected(self):
        args = self.arrays(); args[3][0]=2
        with self.assertRaises(ValueError): validate_arrays(*args)

    def test_invalid_spatial_input_rejected(self):
        with self.assertRaises(ValueError): FrozenContext(ScratchClassifier(49))(torch.zeros(1,28,128,128))
