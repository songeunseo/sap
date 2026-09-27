import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from . import run
from experiments.dlm_multiscale_ac50.artifacts import read,write,sha,mask_identity
from experiments.dlm_multiscale_ac50.evaluation import evaluate_requests

class ValidationTests(unittest.TestCase):
    def test_existing_sample_and_prompt_integrity(self):
        old=read(run.OLD/'requests.json');new=copy.deepcopy(old);new['development']=old['confirmation'];new['confirmation']=[]
        run.validate_requests(new,old)
        self.assertEqual(len(run.expected_ids()),100);self.assertEqual(run.expected_ids()[0],209);self.assertEqual(run.expected_ids()[-1],1305)
        new['development']=copy.deepcopy(new['development']);new['development'][0]['prompt']+=' changed'
        with self.assertRaises(ValueError):run.validate_requests(new,old)

    def test_duplicate_and_overlap_rejected(self):
        old=read(run.OLD/'requests.json');new=copy.deepcopy(old);new['development']=copy.deepcopy(old['confirmation']);new['confirmation']=[]
        new['development'][1]=copy.deepcopy(new['development'][0])
        with self.assertRaises(ValueError):run.validate_requests(new,old)
        new['development']=copy.deepcopy(old['confirmation']);old['development'][0]=copy.deepcopy(new['development'][0])
        with self.assertRaises(ValueError):run.validate_requests(new,old)

    def test_direct_worker_gate(self):
        with patch.object(run,'gates',side_effect=RuntimeError('cpu required')) as gate:
            with self.assertRaisesRegex(RuntimeError,'cpu required'):run.worker('Multi')
            gate.assert_called_once()

    def fixture(self,r,arm):
        req=[dict(example_id=i,doc_hash=f'd{i}',prompt_hash=f'p{i}',target_hash=f't{i}',reference_answer='#### 2',prompt='test') for i in run.expected_ids()]
        write(r/'requests.json',dict(development=req,confirmation=[]));entries=[]
        for i in range(224):
            p=r/'masks'/f'{i}.bin';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'\x03')
            entries.append(dict(name=f'block{i//7}.p{i%7}',shape=[2,4],selected_mask=dict(path=str(p),file_sha256=sha(p),mask_sha256=f'm{i}')))
        m=dict(entries=entries,pruned=896);mp=r/'manifest.json';write(mp,m);identity=mask_identity(m)
        write(r/'config.json',dict(protocol_hash='protocol',sources={str(mp):sha(mp)},validation100=dict(candidates={arm:dict(manifest_path=str(mp),mask_identity=identity,sparse_model_sha256='sparse')})))
        write(r/'candidates'/arm/'model_identity.json',dict(config_sha256=sha(r/'config.json'),mask_identity=identity,sparse_model_sha256='sparse'))
        return req

    def test_actual_completion_and_tamper(self):
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)):
            r=Path(d);arm='Multi';req=self.fixture(r,arm)
            evaluate_requests(run.folder(arm),req,run.fingerprint(arm),'protocol',arm,lambda r:'answer 2',lambda r,t:dict(extracted_answer='2',correct=True),lambda *a,**k:None)
            run.completion(arm,save=True);run.completion(arm)
            p=run.folder(arm)/'examples'/f'{run.expected_ids()[0]:04d}.json';x=read(p);x['row']['generated_text']='changed';write(p,x)
            with self.assertRaises(RuntimeError):run.completion(arm)

    def test_fresh_ids_resume_only_missing(self):
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)):
            req=self.fixture(Path(d),'Multi')[:4];fp=run.fingerprint('Multi');count=[0]
            def gen(r):
                count[0]+=1
                if count[0]==3:raise RuntimeError('interrupted')
                return 'answer 2'
            args=(run.folder('Multi'),req,fp,'protocol','Multi');grade=lambda r,t:dict(extracted_answer='2',correct=True);progress=lambda *a,**k:None
            with self.assertRaises(RuntimeError):evaluate_requests(*args,gen,grade,progress)
            seen=[];result=evaluate_requests(*args,lambda r:seen.append(r['example_id']) or 'answer 2',grade,progress)
            self.assertEqual(seen,run.expected_ids()[2:4]);self.assertEqual(result['correct'],4)

    def test_cpu_receipt_gate(self):
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)),patch.object(run,'validate'):
            r=Path(d);write(r/'config.json',{});t=r/'test.py';t.write_text('pass');log=r/'cpu_validation.log';log.write_text('OK')
            receipt=dict(status='passed',returncode=0,config_sha256=sha(r/'config.json'),tests_run=7,test_sources={str(t):sha(t)},log_sha256=sha(log));write(r/'cpu_validation.json',receipt)
            run.gates();t.write_text('changed')
            with self.assertRaises(RuntimeError):run.gates()

    def test_controller_adapter_and_failure_cleanup(self):
        ctl=run.controller
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)),patch.dict(os.environ,TMUX='test'):
            run.configure_controller();self.assertEqual(ctl.MODULE,run.MODULE);self.assertIs(ctl.completion,run.completion)
            self.assertIs(ctl.launch_gates,run.gates);self.assertEqual(ctl.ARMS,('Multi','A','Uniform'))
            write(Path(d)/'config.json',{});spawned=[];real=subprocess.Popen
            def popen(*a,**kw):
                p=real(*a,**kw);spawned.append(p);return p
            def command(*args):
                return [sys.executable,'-c','import time,sys;time.sleep(.1);sys.exit(2)' if args[-1]=='Multi' else 'import time;time.sleep(60)']
            with patch.object(ctl,'launch_gates'),patch.object(ctl,'idle_devices'),patch.object(ctl,'command',command),patch.object(ctl.subprocess,'Popen',popen):
                with self.assertRaises(RuntimeError):ctl.pipeline(['1','2','3'])
            self.assertEqual(read(Path(d)/'execution.json')['status'],'failed');self.assertTrue(all(p.poll() is not None for p in spawned))
        run.configure_controller()

if __name__=='__main__':unittest.main()
