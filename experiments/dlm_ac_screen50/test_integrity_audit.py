import tempfile
import unittest
from pathlib import Path
from experiments.dlm_multiscale_ac50.artifacts import read,write
from .integrity import freeze_checked,read_checked,ReadoutStore

class IntegrityTests(unittest.TestCase):
    def test_finite_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'artifact.json';freeze_checked(p,dict(fingerprint={'a':1},metrics={'AC':1.}))
            r=read(p);r['metrics']['AC']=2.;write(p,r)
            with self.assertRaisesRegex(RuntimeError,'payload SHA256'):read_checked(p)
    def test_readout_resume_identity_and_values(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'readouts.json';ReadoutStore(p,{'teacher':'A'},2,1).append([1.])
            s=ReadoutStore(p,{'teacher':'A'},2,1);s.append([2.]);self.assertEqual(s.complete(),[[1.],[2.]])
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):ReadoutStore(p,{'teacher':'B'},2,1)
            r=read(p);r['values'][0][0]=3.;write(p,r)
            with self.assertRaisesRegex(RuntimeError,'payload SHA256'):ReadoutStore(p,{'teacher':'A'},2,1)
    def test_incomplete_pair_not_committed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pair.json'
            # A temporary partial write never becomes the committed pair.
            p.with_suffix('.tmp').write_text('{"half":')
            self.assertFalse(p.exists())
            freeze_checked(p,dict(pair=0,metrics={'A':1.,'C':2.,'AC':3.}))
            self.assertEqual(read_checked(p)['metrics']['AC'],3.)


class ManifestTests(unittest.TestCase):
    def test_manifest_mutation_fails_before_dependencies(self):
        import json
        from unittest.mock import patch
        from . import prepare
        from experiments.dlm_multiscale_ac50.artifacts import sha
        original=read(prepare.ROOT/'audits/prelaunch_revision_1/manifest.json')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write(root/'manifest.json',original)
            write(root/'manifest_identity.json',dict(manifest_sha256=sha(root/'manifest.json')))
            original['probe_rates']=[.3,.7];write(root/'manifest.json',original)
            with patch.object(prepare,'ROOT',root):
                with self.assertRaisesRegex(RuntimeError,'SHA256 changed'):prepare.validate()
    def test_dry_run_does_not_poll_spawn_or_load(self):
        from unittest.mock import patch
        from . import run
        import contextlib,io
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write(root/'preparation_receipt.json',{})
            with patch.object(run,'ROOT',root),patch('experiments.dlm_ac_screen50.prepare.validate',return_value={'costs':{}}),patch.object(run.subprocess,'check_output',side_effect=AssertionError('poll')),patch.object(run.subprocess,'Popen',side_effect=AssertionError('spawn')),patch.object(run.subprocess,'run',side_effect=AssertionError('tmux')),contextlib.redirect_stdout(io.StringIO()) as output:
                run.launch(['3'],dry=True)
            self.assertIn('"gpu_queried": false',output.getvalue())


class ExchangeJournalTests(unittest.TestCase):
    def test_committed_round_recovers_without_replay(self):
        from .exchange_state import recover
        from .integrity import write_checked
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);before=dict(config_sha256='same',round=0,evaluations=0,done=False,counts=[3],loss=1.)
            after=dict(before,round=1,evaluations=8,counts=[2],loss=.9)
            write_checked(root/'state.json',before)
            # Crash after immutable decision commit, before incumbent pointer update.
            freeze_checked(root/'round0.json',dict(before=before,offers=[],after=after))
            self.assertEqual(recover(root,before),after)
            self.assertEqual(read_checked(root/'state.json'),after)
            self.assertEqual(recover(root,after),after)
    def test_wrong_incumbent_rejected(self):
        from .exchange_state import recover
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);before=dict(config_sha256='same',round=0,evaluations=0,done=False)
            freeze_checked(root/'round0.json',dict(before=dict(before,loss=2),after={},offers=[]))
            with self.assertRaisesRegex(RuntimeError,'incumbent'):recover(root,before)

class ShapeStorageTests(unittest.TestCase):
    def test_square_rejects_broadcast_and_nonfinite(self):
        import numpy as np
        from .core import square_metrics
        for teacher in [np.zeros((1,4,8)),np.full((2,4,8),np.nan)]:
            with self.assertRaises(ValueError):square_metrics(np.zeros((2,4,8)),teacher)
        self.assertEqual(square_metrics(np.zeros((2,4,8)),np.zeros((2,4,8)))['mean']['AC'],0.)
    def test_storage_fails_before_teacher_work(self):
        from .preflight import storage_check
        memory=dict(total_teacher_bytes=20,estimated_pair_ram_bytes=10)
        with self.assertRaisesRegex(RuntimeError,'disk'):storage_check(Path('/tmp'),memory,free_disk=19,free_ram=2**40)
        with self.assertRaisesRegex(RuntimeError,'RAM'):storage_check(Path('/tmp'),memory,free_disk=2**40,free_ram=9)
        self.assertEqual(storage_check(Path('/tmp'),memory,free_disk=2**40,free_ram=2**40)['fallback'],'none')
    def test_nonuniform_queries_are_equal_pair_weighted(self):
        import numpy as np
        from .core import vector_pair,mean_rows
        # Pair losses 1 and 4, with one and three queries: correct average is2.5, not3.25.
        rows=[]
        for q,scale in [(1,1),(3,2)]:
            p=np.tile(np.array([scale,-scale],dtype=np.float32),(q,1));d=np.zeros_like(p)
            rows.append(vector_pair([p,p],[d,d]))
        self.assertEqual(mean_rows(rows)['A'],2.5)



class SchemaTests(unittest.TestCase):
    def test_missing_controls_extra_arms_and_constants_fail(self):
        import copy
        from .prepare import validate_schema
        from .core import ARMS
        arm=dict(bank={},seed={},edge_weights={},reduction_denominators={},source_hashes={},matched_control=['control'],hypothesis='h',defaults_provenance='design')
        manifest=dict(arms={a:dict(arm) for a in ARMS},references={x:{} for x in ['legacy_Uniform','legacy_A','legacy_AC']},target=3489660928,total_prunable=6979321856,probe_rates=[.48,.52],exchange=dict(d=41,rounds=3,proposals=8,max_new=24),model={},evaluation={})
        legacy=dict(model={},evaluation={});validate_schema(manifest,legacy)
        for defect in ['control','extra','budget','probe']:
            m=copy.deepcopy(manifest)
            if defect=='control':del m['arms']['Square-AC']['matched_control']
            if defect=='extra':m['arms']['invented']={}
            if defect=='budget':m['target']+=1
            if defect=='probe':m['probe_rates']=[.4,.6]
            with self.assertRaises(ValueError):validate_schema(m,legacy)

if __name__=='__main__':unittest.main()
