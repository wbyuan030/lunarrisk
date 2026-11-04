"""Offline serial/replay acceptance with synthetic result artifacts, no model execution."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import test_p31_run_baselines as fixtures

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('prefix_launcher',ROOT/'scripts/p31_run_prefix_baselines.py')
runner=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TestPrefixBaselines(unittest.TestCase):
    def setUp(self):
        fixtures.TestSerialBaselines.setUp(self)
        self.plan['protocol']='P31_LOCAL_PREFIX_STORED_DERIVED_V1'
        self.plan['data'].update(track='partial_stored_prefix',upstream_full_integrity_verified=False,
                                 upstream_bytes=74449119347,
                                 upstream_sha256='e3e96bccc044d856cd55883c9cda7628f02afbdc1b2d1818b401bbcc590df83f')
        self.stats=ROOT/'reports/evidence/p30_20261103/hf_channel_stats.json'
        self.mapping=ROOT/'reports/evidence/p30_20261103/geologic_units.json'
        self.metadata=self.root/'metadata.json'; self.metadata.write_text('{}')
        self.ranges=self.root/'ranges.json'; self.ranges.write_text('{}')
        for path,key in ((self.stats,'stats_sha256'),(self.mapping,'class_map_sha256'),
                         (self.metadata,'metadata_sha256'),(self.ranges,'range_report_sha256')):
            self.plan['data'][key]=runner.base.sha256(path)
        self.plan['splits']={}
        for split,n,counts in (('train',3,[2,1]),('val',2,[1,1])):
            self.plan['splits'][split]={'rows':n,'ordered_ids_sha256':self.plan['data'][split+'_ordered_ids_sha256'],
                                       'age_class_counts':counts+[0]*3,'geology_class_counts':counts+[0]*47}
        self.plan_path.write_text(json.dumps(self.plan))
        self.argv=['--data',str(self.data),'--outdir',str(self.outdir),'--stats',str(self.stats),
                   '--class-map',str(self.mapping),'--plan',str(self.plan_path),
                   '--metadata',str(self.metadata),'--range-report',str(self.ranges),
                   '--runtime-directory',str(self.runtime),'--max-new-runs','1']
        mocked=patch.object(runner.base,'inspect_checkpoint',side_effect=lambda path,runtime:runner.base.read_json(path))
        mocked.start(); self.addCleanup(mocked.stop)
        self.bad_replay=False; self.bad_integrity=False

    def invoke(self,apply=True):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.main(self.argv+(['--apply'] if apply else []))

    def fake_run(self,command,**kwargs):
        result=fixtures.TestSerialBaselines.fake_run(self,command,**kwargs)
        directory=Path(command[command.index('--outdir')+1])
        replay=command[command.index('--mode')+1]=='replay'
        for name in ('report.json','checkpoint.pt','asset_manifest.json'):
            value=runner.base.read_json(directory/name)
            for key in ('metadata_sha256','range_report_sha256'):
                if name=='asset_manifest.json': value[key]=self.plan['data'][key]
                else: value['inputs'][key]=self.plan['data'][key]
            if name=='asset_manifest.json':
                value.update(full_content_hash_verified_this_invocation=False,
                             local_prefix_hash_verified_this_invocation=True,
                             upstream_full_integrity_verified=self.bad_integrity,
                             upstream_bytes=self.plan['data']['upstream_bytes'],
                             upstream_sha256=self.plan['data']['upstream_sha256'])
            else:
                value['config']['purpose']='subset_baseline'
                value['source_sha256']['src/lunarrisk/benchmarks/prefix.py']=runner.base.sha256(ROOT/'src/lunarrisk/benchmarks/prefix.py')
                value['protocol']=self.plan['protocol']
            if name=='report.json':
                value.update(status='P31_PARTIAL_DATA_TRAINVAL',purpose='subset_baseline',mode='replay' if replay else 'train')
                if replay and self.bad_replay: value['prediction_sha256']['val']='0'*64
            (directory/name).write_text(json.dumps(value))
        return result

    def test_preview_no_data_hash_or_outputs(self):
        self.data.unlink()
        self.assertEqual(self.invoke(False),0)
        self.process.assert_not_called()
        self.assertFalse(self.outdir.exists())

    def test_first_seed_replay_then_restart_to_geology(self):
        self.assertEqual(self.invoke(),0)
        self.assertEqual(self.calls,[('age',42),('age',42)])
        receipt=runner.base.read_json(self.outdir/'age_seed42/completion.json')
        self.assertEqual(receipt['replay']['status'],'PASS')
        self.assertIn('metadata_sha256',receipt['identity']['inputs'])
        self.assertEqual(receipt['identity']['config']['purpose'],'subset_baseline')
        self.calls.clear()
        self.assertEqual(self.invoke(),0)
        self.assertEqual(self.calls,[('geology',42),('geology',42)])
        self.assertEqual(runner.base.read_json(self.outdir/'aggregate_02.json')['status'],'PARTIAL')
        for call in self.process.call_args_list:
            command=call.args[0]
            self.assertEqual(command[1:3],['-I','-c'])
            self.assertEqual(command[command.index('--track')+1],'partial_stored_prefix')
            self.assertNotIn('--stop-after-epoch',command)

    def test_failed_replay_blocks_next_seed_and_receipt(self):
        self.bad_replay=True
        with self.assertRaisesRegex(ValueError,'replay differs'): self.invoke()
        self.assertFalse((self.outdir/'age_seed42/completion.json').exists())
        self.assertEqual(self.calls,[('age',42),('age',42)])
        self.assertEqual(runner.base.read_json(self.outdir/'aggregate_00.json')['tasks']['age']['completed_seeds'],[])

    def test_false_full_integrity_claim_blocks_before_replay(self):
        self.bad_integrity=True
        with self.assertRaisesRegex(ValueError,'integrity scope'): self.invoke()
        self.assertEqual(self.calls,[('age',42)])
        self.assertFalse((self.outdir/'age_seed42/completion.json').exists())


if __name__=='__main__': unittest.main()
