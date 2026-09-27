"""CPU fake workers exercise the real controller DAG, failure and resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from . import run
from experiments.dlm_multiscale_ac50.artifacts import read,write
from .runtime_integrity import write_job_receipt,validate_job_receipt,expected_outputs

class ControllerIntegrationTests(unittest.TestCase):
    def test_bootstrap_is_recognized_as_owned_controller(self):
        self.assertTrue(run._owned_controller_command('/usr/bin/python3 -B -u -m experiments.dlm_ac_screen50.resume_legacy'))
        self.assertFalse(run._owned_controller_command('/usr/bin/python3 unrelated_experiment.py'))

    def test_failure_resume_full_ten_arm_dag(self):
        real_popen=subprocess.Popen
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'output';root.mkdir();multi=Path(td)/'legacy';multi.mkdir()
            write(root/'manifest.json',{'test':'frozen'})
            flag=root/'fail_once';flag.write_text('Vector-AC')
            fake=Path(td)/'fake.py'
            fake.write_text("""import sys,json
from pathlib import Path
r=Path(sys.argv[1]);multi=Path(sys.argv[2]);job=json.loads(sys.argv[3]);paths=[Path(x) for x in json.loads(sys.argv[4])];name=job['id']
f=r/'fail_once'
if f.exists() and f.read_text()==name:
 f.unlink();sys.exit(23)
p=r/(name+'.fake')
if p.exists():raise RuntimeError('duplicate job')
p.write_text(name)
for out in paths:
 out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists():out.write_text('{}')
""")
            def popen(argv,**kwargs):
                if '--method' in argv:
                    name=argv[argv.index('--method')+1];name='MS-A' if name=='A' else name
                    job=next(j for j in run.job_graph() if j['id']==name)
                else:
                    name=argv[argv.index('--job')+1]
                    job=next(j for j in run.job_graph() if j['id']==name)
                from .runtime_integrity import expected_outputs
                paths=[str(x) for x in expected_outputs(root,multi,job)]
                return real_popen([sys.executable,'-B',str(fake),str(root),str(multi),json.dumps(job),json.dumps(paths)],**kwargs)
            def fake_report():
                self.assertTrue(all((root/(arm+'.fake')).exists() for arm in run.ARMS))
                return {'status':'complete'}
            with patch.object(run,'ROOT',root),patch.object(run,'MULTI',multi),patch.object(run,'REPO',Path(td)),patch.dict(os.environ,{'TMUX':'fake-test-only'}),patch('experiments.dlm_ac_screen50.prepare.validate',return_value={}),patch('experiments.dlm_multiscale_ac50.run.idle_devices'),patch('experiments.dlm_ac_screen50.allocation.allocate',side_effect=lambda family: (root/'allocations').mkdir(parents=True,exist_ok=True) or (root/'allocations'/f'{family}.json').write_text('{}')),patch.object(run,'rows_for',return_value=[]),patch.object(run,'report',side_effect=fake_report),patch.object(run.subprocess,'Popen',side_effect=popen),patch.object(run.time,'sleep',lambda _:None):
                with self.assertRaisesRegex(RuntimeError,'exit 23'):run.pipeline(['3'])
                self.assertEqual(read(root/'execution.json')['status'],'failed')
                before={p.name:p.read_bytes() for p in root.glob('*.fake')}
                run.pipeline(['3'])
                self.assertEqual(read(root/'execution.json')['status'],'complete')
                self.assertEqual(len(list((root/'jobs_done').glob('*.json'))),len(run.job_graph()))
                self.assertTrue(all((root/k).read_bytes()==v for k,v in before.items()))

    def test_receipt_rejects_terminal_child_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'output';root.mkdir();multi=Path(td)/'legacy';multi.mkdir()
            job=next(j for j in run.job_graph() if j['id']=='smoke')
            output=expected_outputs(root,multi,job)[0];output.parent.mkdir(parents=True,exist_ok=True);output.write_text('{"status":"passed"}')
            receipt=root/'jobs_done'/'smoke.json';write_job_receipt(receipt,root,multi,job,'config','jobs')
            output.write_text('{"status":"tampered"}')
            with self.assertRaisesRegex(RuntimeError,'output identity'):
                validate_job_receipt(receipt,root,multi,job,'config','jobs')

    def test_other_family_outputs_do_not_change_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'output';root.mkdir();multi=Path(td)/'legacy';multi.mkdir()
            job=next(j for j in run.job_graph() if j['id']=='Square_uniform')
            for p in expected_outputs(root,multi,job):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            receipt=root/'jobs_done'/'Square_uniform.json'
            write_job_receipt(receipt,root,multi,job,'config','jobs')
            other=root/'pair_metrics/Vector/uniform/calibration/000.json'
            other.parent.mkdir(parents=True);other.write_text('{}')
            validate_job_receipt(receipt,root,multi,job,'config','jobs')

    def test_terminate_owned_kills_private_descendants_only(self):
        script='import subprocess,sys,time; c=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); print(c.pid,flush=True); time.sleep(60)'
        proc=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,text=True,start_new_session=True)
        child=int(proc.stdout.readline().strip())
        info=run._proc_info(proc.pid)
        row=dict(proc=proc,pid=proc.pid,start_ticks=info['start_ticks'],pgid=info['pgid'])
        try:
            run.terminate_owned([row],grace=2)
            self.assertIsNotNone(proc.poll())
            proc.stdout.close()
            with self.assertRaises(ProcessLookupError):os.kill(child,0)
        finally:
            if proc.poll() is None:run.terminate_owned([row],grace=1)

if __name__=='__main__':unittest.main()
