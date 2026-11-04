"""Separate local-prefix configuration gate; complete-data checks remain independent."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('p31_subset_trainer',ROOT/'scripts/p31_train_classifier.py')
trainer=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)


class TestSubsetRunner(unittest.TestCase):
    def setUp(self):
        original=json.loads((ROOT/'protocols/p31_baseline_eval_plan.json').read_text())
        self.plan={'protocol':'P31_LOCAL_PREFIX_STORED_DERIVED_V1',
                   'data':{'track':'partial_stored_prefix','upstream_full_integrity_verified':False,
                           'bytes':100,'sha256':'a'*64,
                           **{key:'b'*64 for key in ('stats_sha256','class_map_sha256','metadata_sha256','range_report_sha256')}},
                   'training':original['training'],'test':original['test']}
        self.inputs={'data_bytes':100,'data_sha256':'a'*64,
                     **{key:'b'*64 for key in ('stats_sha256','class_map_sha256','metadata_sha256','range_report_sha256')}}
        self.args=SimpleNamespace(track='partial_stored_prefix',epochs=50,batch_size=16,seed=42,
                                  lr=0.001,patience=5,precision='fp32',workers=0)

    def test_fixed_partial_plan_passes_without_claiming_complete_integrity(self):
        trainer.validate_subset_plan(self.args,self.plan,self.inputs)

    def test_full_protocol_or_false_complete_integrity_claim_is_rejected(self):
        for key,value in (('protocol','P31_03_STORED_DERIVED_SCRATCH_BASELINES_V1'),
                          ('integrity',True),('track','official_stored')):
            plan=copy.deepcopy(self.plan)
            if key=='protocol': plan[key]=value
            elif key=='integrity': plan['data']['upstream_full_integrity_verified']=value
            else: plan['data']['track']=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                trainer.validate_subset_plan(self.args,plan,self.inputs)

    def test_every_data_or_metadata_hash_and_size_must_match(self):
        for key in self.inputs:
            inputs=copy.deepcopy(self.inputs)
            inputs[key]=101 if key=='data_bytes' else 'c'*64
            with self.subTest(key=key),self.assertRaises(ValueError):
                trainer.validate_subset_plan(self.args,self.plan,inputs)

    def test_budget_changes_and_test_permission_are_rejected(self):
        for key,value in (('epochs',2),('batch_size',3),('seed',99),('lr',0.1),
                          ('patience',0),('precision','bf16'),('workers',2)):
            args=copy.copy(self.args); setattr(args,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):
                trainer.validate_subset_plan(args,self.plan,self.inputs)
        self.plan['test']['runner_allows_test']=True
        with self.assertRaises(ValueError): trainer.validate_subset_plan(self.args,self.plan,self.inputs)

    def test_prefix_options_cannot_reach_complete_data_or_pilot_loader(self):
        common=['--data','unused','--stats','unused','--class-map','unused','--outdir','unused']
        for extra in (['--purpose','baseline','--track','partial_stored_prefix'],
                      ['--metadata','unused'],['--range-report','unused'],
                      ['--purpose','subset_baseline'],
                      ['--purpose','subset_baseline','--track','partial_stored_prefix',
                       '--plan','unused','--metadata','unused','--range-report','unused','--stop-after-epoch','1']):
            with self.subTest(extra=extra),self.assertRaises(ValueError): trainer.main(common+extra)


if __name__=='__main__': unittest.main()
