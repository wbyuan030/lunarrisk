import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import p32_train_heads as train
import p32_run_heads as runner
from lunarrisk.lunarjev.catalog import load_catalog, encode_catalog, permute_encoded


class RunnerTests(unittest.TestCase):
    def test_gpu_initialized_before_peak_counter_reset(self):
        from unittest.mock import patch
        calls=[]
        with patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.init',side_effect=lambda:calls.append('init')), \
             patch('torch.cuda.reset_peak_memory_stats',side_effect=lambda device:calls.append('reset')):
            self.assertEqual(str(train.prepare_device('cuda:0')),'cuda:0')
        self.assertEqual(calls,['init','reset'])

    def test_unavailable_gpu_fails_before_counter_reset(self):
        from unittest.mock import patch
        with patch('torch.cuda.is_available',return_value=False), patch('torch.cuda.reset_peak_memory_stats') as reset:
            with self.assertRaisesRegex(ValueError,'GPU unavailable'): train.prepare_device('cuda:0')
            reset.assert_not_called()

    def test_probability_metrics_match_known_confusion(self):
        p=np.asarray([[.8,.2],[.3,.7],[.4,.6]])
        m=train.probability_metrics(p,[0,1,0])
        self.assertEqual(m['confusion'],[[1,1],[0,1]])
        self.assertAlmostEqual(m['accuracy'],2/3)
        self.assertAlmostEqual(m['macro_f1_full'],2/3)
        self.assertAlmostEqual(m['nll'],-np.log([.8,.7,.4]).mean())
        self.assertAlmostEqual(m['multiclass_brier_sum'],(.08+.18+.72)/3)

    def test_nonfinite_or_unnormalized_probabilities_rejected(self):
        for p in ([[np.nan,0.]],[[.2,.2]],[[-1.,2.]]):
            with self.assertRaises(ValueError): train.probability_metrics(p,[0])

    def test_age_aggregation_preserves_probability_mass(self):
        mapping=[i%5 for i in range(49)]
        p=np.ones((3,49))/49
        result=train.age_aggregation(p,mapping)
        self.assertTrue(np.allclose(result.sum(1),1))
        self.assertTrue(np.allclose(result[0],np.bincount(mapping,minlength=5)/49))
        with self.assertRaises(ValueError): train.age_aggregation(p,[-1]*49)

    def test_all_primary_seeds_before_descriptor_control(self):
        plan={'seeds':[42,43,44],'methods':['B0','B1','B2','J'],
              'description_control':{'methods':['B2','J']}}
        sequence=runner.sequence(plan)
        self.assertEqual(len(sequence),18)
        self.assertEqual(sequence[:4],[(42,m,False) for m in plan['methods']])
        self.assertFalse(any(permuted for seed,method,permuted in sequence[:12]))
        self.assertTrue(all(permuted for seed,method,permuted in sequence[12:]))

    def test_text_identity_changes_for_fixed_permutation(self):
        encoded=encode_catalog(load_catalog(Path(__file__).resolve().parents[1]/'protocols/p32_candidate_catalog.json'))
        self.assertEqual(train.text_identity(encoded),train.text_identity(encoded))
        self.assertNotEqual(train.text_identity(encoded),train.text_identity(permute_encoded(encoded)))

    def test_sample_std_is_explicit_and_incomplete_not_complete(self):
        plan={'protocol':'test','seeds':[42,43,44],'methods':['B0','B1','B2','J'],
              'description_control':{'methods':['B2','J']}}
        report={'identity':{'seed':42,'method':'B0','descriptor_permuted':False},
                'metrics':{s:{t:{k:.5 for k in ('accuracy','macro_f1_full','macro_f1_supported','nll','multiclass_brier_sum')}
                              for t in ('geology','age','age_from_geology')} for s in ('train','val')}}
        agg=runner.aggregate([report],plan)
        self.assertEqual(agg['status'],'PARTIAL')
        self.assertIsNone(agg['methods']['B0']['metrics']['val']['geology']['accuracy']['sample_std_ddof1'])
        self.assertEqual(agg['methods']['J']['n'],0)
