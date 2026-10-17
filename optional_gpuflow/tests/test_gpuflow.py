import datetime as dt
import json
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpuflow import read_config, is_allowed, choose, check_spec

CFG = read_config(Path(__file__).resolve().parents[1] / 'config.json')

class TestPolicy(unittest.TestCase):
    def test_weekend_rule(self):
        node = CFG['nodes'][1]
        friday = dt.datetime(2026, 10, 16, 8, tzinfo=dt.timezone.utc)  # Friday afternoon China
        saturday = dt.datetime(2026, 10, 17, 8, tzinfo=dt.timezone.utc)
        self.assertFalse(is_allowed(node, friday))
        self.assertTrue(is_allowed(node, saturday))
    def test_rocm_excluded_for_cuda(self):
        job = next(j for j in CFG['jobs'] if j['id']=='joint_seed42')
        self.assertFalse(check_spec(job, CFG['nodes'][0])[0])
        self.assertTrue(check_spec(job, CFG['nodes'][1])[0])
    def test_four_gpus_only_on_weekend_node(self):
        job = next(j for j in CFG['jobs'] if j['id']=='joint_4gpu_ddp')
        self.assertFalse(check_spec(job, CFG['nodes'][0])[0])
        self.assertTrue(check_spec(job, CFG['nodes'][1])[0])
    def test_priorities(self):
        job = next(j for j in CFG['jobs'] if j['id']=='smoke_test')
        now = dt.datetime(2026, 10, 17, 8, tzinfo=dt.timezone.utc)
        rows = choose(CFG, job, now, live=False)
        self.assertEqual(rows[0][0]['name'], 'home_7900xt')
        self.assertTrue(rows[0][1])

if __name__ == '__main__': unittest.main()
