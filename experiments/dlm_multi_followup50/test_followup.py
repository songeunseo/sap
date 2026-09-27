import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .core import ARMS, allocation_rates, DRAWS
from .prepare import derive, OLD
from . import run
from experiments.dlm_multiscale_ac50.artifacts import read, write, sha, mask_identity
from experiments.dlm_multiscale_ac50.core import metrics, rank_rates
from experiments.dlm_multiscale_ac50.evaluation import evaluate_requests

class AllocationTests(unittest.TestCase):
    def test_ties(self):
        rates,_=allocation_rates(np.full((32,8),-3.))
        for v in rates.values():np.testing.assert_allclose(v,.5,atol=1e-14)

    def test_fixed_radii(self):
        costs=np.repeat(np.arange(-16,16)[:,None],8,axis=1)
        rates,draws=allocation_rates(costs);base=rank_rates(costs.mean(1))
        np.testing.assert_allclose(rates[ARMS[0]],base,atol=1e-14)
        for arm,factor in zip(ARMS[1:],(.4,.7)):
            np.testing.assert_allclose(rates[arm],.5+factor*(base-.5),atol=1e-14)
        self.assertEqual(draws.shape,(DRAWS,8))
        self.assertLess(rates[ARMS[0]][31],rates[ARMS[0]][0])

    def test_shared_draws_no_rerank(self):
        costs=np.random.default_rng(10).normal(size=(32,8))
        rates,draws=allocation_rates(costs);expected=[]
        for draw in draws:
            g=costs[:,draw].mean(1)
            q=np.array([(np.sum(g<x)+.5*(np.sum(g==x)-1))/31 for x in g])
            expected.append(.5-.1*(q-q.mean()))
        np.testing.assert_allclose(rates[ARMS[0]],np.mean(expected,0),atol=1e-14)
        self.assertGreater(rates[ARMS[0]].min(),.45)
        self.assertLess(rates[ARMS[0]].max(),.55)

    def test_bad_input(self):
        for c in (np.zeros((8,32)),np.full((32,8),np.nan)):
            with self.assertRaises(ValueError):allocation_rates(c)

    def test_real_budget(self):
        c=read(OLD/'config.json');refs=read(c['legacy_manifests']['uniform']['path'])['entries'];costs=[]
        for b in range(32):
            x=read(OLD/'probes'/f'block{b:02d}.json')['conditions'];lo,hi=x['0.48'],x['0.52']
            costs.append([(h['metrics']['Multi']-l['metrics']['Multi'])/(hi['pruned']-lo['pruned']) for l,h in zip(lo['metrics']['sequences'],hi['metrics']['sequences'])])
        derived,_=derive(costs,refs,c['pruning']['pruned'])
        for a in derived.values():
            self.assertEqual(sum(k*r['shape'][0] for k,r in zip(a['row_counts'],refs)),3489660928)
            for b in range(32):
                for width in {r['shape'][1] for r in refs[b*7:b*7+7]}:
                    self.assertEqual(len({a['row_counts'][i] for i in range(b*7,b*7+7) if refs[i]['shape'][1]==width}),1)

class ReceiptTests(unittest.TestCase):
    def test_generation_count_and_incomplete_attempt_costs(self):
        calls=[0]
        def gen(_):calls[0]+=256;return 'ok'
        self.assertEqual(run.checked_generate(gen,calls,256)({}),'ok')
        with self.assertRaises(RuntimeError):run.checked_generate(gen,calls,255)({})
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)):
            r=Path(d);write(r/'attempts/a.json',dict(arm='Multi-Bag'))
            self.assertFalse(run.cost_accounting()['complete_accounting'])
            write(r/'costs/a.json',dict(stages=[dict(stage='gsm8k',forward_calls=77000)]))
            a=run.cost_accounting();self.assertTrue(a['complete_accounting']);self.assertEqual(a['generation_excess_over_nominal'],200)

    def fixture(self,root,arm):
        requests=[dict(example_id=i,doc_hash=f'd{i}',prompt_hash=f'p{i}',target_hash=f't{i}',reference_answer='#### 2',prompt='test') for i in range(100)]
        write(root/'requests.json',dict(development=requests,confirmation=[]))
        bank=dict(states=8,chains=[dict(query=[0],sequence_index=0,chain_index=0)]);write(root/'bank.json',bank)
        c=dict(protocol_hash='protocol',pruning=dict(pruned=896),banks={s:dict(path=str(root/'bank.json'),sha256=sha(root/'bank.json')) for s in ('calibration','diagnostic')})
        write(root/'config.json',c);ch=sha(root/'config.json')
        write(root/'allocation.json',dict(allocations={arm:dict(row_counts=[2]*224)}));entries=[]
        for i in range(224):
            p=root/'candidates'/arm/'masks'/f'{i}.bin';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'\x03')
            entries.append(dict(name=f'block_{i//7:02d}.p{i%7}',shape=[2,4],selected_mask=dict(path=str(p),file_sha256=sha(p),mask_sha256=f'm{i}',prune_per_row=2)))
        manifest=dict(entries=entries,pruned=896,config_sha256=ch,allocation_sha256=sha(root/'allocation.json'))
        write(root/'candidates'/arm/'mask_manifest.json',manifest);identity=mask_identity(manifest)
        write(root/'candidates'/arm/'model_identity.json',dict(config_sha256=ch,mask_identity=identity,sparse_model_sha256='sparse'))
        fp=dict(config_sha256=ch,mask_identity=identity,sparse_model_sha256='sparse',split='development',requests_sha256=sha(root/'requests.json'),protocol_hash='protocol')
        evaluate_requests(root/'gsm8k/development'/arm,requests,fp,'protocol',arm,lambda r:'answer 2',lambda r,t:dict(extracted_answer='2',correct=True),lambda *a,**k:None)
        for split in ('calibration','diagnostic'):
            p=root/'readouts'/arm/f'{split}.json';t=root/'readouts/dense'/f'{split}.json';values=[[0.]]*8
            write(t,dict(values=values));write(p,dict(fingerprint=dict(config_sha256=ch,mask_identity=identity,bank_sha256=sha(root/'bank.json')),values=values,complete=True))
            write(root/'diagnostics'/arm/f'{split}.json',dict(config_sha256=ch,mask_identity=identity,readout_sha256=sha(p),teacher_sha256=sha(t),**metrics(values,values,bank)))

    def test_completion_and_tamper(self):
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)):
            r=Path(d);arm=ARMS[0];self.fixture(r,arm)
            run.completion(arm,save=True);run.completion(arm)
            p=r/'gsm8k/development'/arm/'examples/0000.json';x=read(p);x['row']['generated_text']='changed';write(p,x)
            with self.assertRaises((ValueError,RuntimeError)):run.completion(arm)

    def test_resume_only_missing(self):
        from experiments.dlm_multiscale_ac50.test_cpu import request
        with tempfile.TemporaryDirectory() as d:
            n=[0]
            def failing(req):
                n[0]+=1
                if n[0]==3:raise RuntimeError('interruption')
                return 'answer 2'
            req=[request(i) for i in range(4)];grade=lambda r,t:dict(extracted_answer='2',correct=True);progress=lambda *a,**k:None
            with self.assertRaises(RuntimeError):evaluate_requests(Path(d),req,{},'protocol','test',failing,grade,progress)
            seen=[]
            result=evaluate_requests(Path(d),req,{},'protocol','test',lambda r:seen.append(r['example_id']) or 'answer 2',grade,progress)
            self.assertEqual(seen,[2,3]);self.assertEqual(result['correct'],4)

    def test_direct_worker_requires_audit_before_runtime_import(self):
        with patch.object(run,'launch_gates',side_effect=RuntimeError('audit required')) as gate:
            with self.assertRaisesRegex(RuntimeError,'audit required'):run.worker(ARMS[0])
            gate.assert_called_once()

    def test_launch_requires_current_cpu_receipt(self):
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)),patch.object(run,'validate'):
            r=Path(d);write(r/'config.json',{});test=r/'test.py';test.write_text('pass')
            cpu=dict(status='passed',returncode=0,config_sha256=sha(r/'config.json'),tests_run=8,test_sources={str(test):sha(test)})
            write(r/'cpu_validation.json',cpu)
            write(r/'audit_pass.json',dict(status='passed',config_sha256=sha(r/'config.json'),cpu_validation_sha256=sha(r/'cpu_validation.json')))
            run.launch_gates()
            test.write_text('changed')
            with self.assertRaises(RuntimeError):run.launch_gates()
            test.write_text('pass');cpu['returncode']=1;write(r/'cpu_validation.json',cpu)
            write(r/'audit_pass.json',dict(status='passed',config_sha256=sha(r/'config.json'),cpu_validation_sha256=sha(r/'cpu_validation.json')))
            with self.assertRaises(RuntimeError):run.launch_gates()

    def test_controller_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)),patch.dict(os.environ,TMUX='test'):
            write(Path(d)/'config.json',{});spawned=[];real=subprocess.Popen
            def popen(*a,**kw):
                p=real(*a,**kw);spawned.append(p);return p
            def command(*args):
                return [sys.executable,'-c','import time,sys; time.sleep(0.1);sys.exit(2)' if args[-1]==ARMS[0] else 'import time;time.sleep(60)']
            with patch.object(run,'launch_gates'),patch.object(run,'idle_devices'),patch.object(run,'command',command),patch.object(run.subprocess,'Popen',popen):
                with self.assertRaises(RuntimeError):run.pipeline(['1','2','3'])
            self.assertEqual(read(Path(d)/'execution.json')['status'],'failed')
            self.assertTrue(all(p.poll() is not None for p in spawned))

if __name__=='__main__':unittest.main()
