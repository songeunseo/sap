"""Storage/checkpoint-only maintenance; frozen scientific code remains unchanged."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STORE=Path('/DATA/tmluser1/sap_storage_recovery_20260914/dlm_allocation_baselines65')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def rng_equal(a,b):
    import numpy as np
    import torch
    return (a['python']==b['python'] and a['numpy'][0]==b['numpy'][0]
        and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]
        and torch.equal(a['torch'],b['torch']) and len(a['cuda'])==len(b['cuda'])
        and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda'])))

def restore_rng(s):
    import numpy as np
    import torch
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])

def atomic_tensor(path,value):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def checkpoint_generate(original,harness,requests,directory,provenance):
    """Replay the exact original single-request generation and preserve RNG sequence."""
    import torch
    directory=Path(directory);out=[]
    for i,request in enumerate(requests):
        key=hashlib.sha256(json.dumps(request.args,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        path=directory/f'{i:04d}.pt';before=rng_state()
        if path.exists():
            c=torch.load(path,map_location='cpu',weights_only=False)
            if c['request_sha256']!=key or c['provenance']!=provenance or not rng_equal(before,c['before']):
                raise RuntimeError(f'checkpoint input/provenance/RNG mismatch at {i}')
            text=c['text'];restore_rng(c['after']);cached=True
        else:
            generated=original(harness,[request])
            if len(generated)!=1 or not isinstance(generated[0],str):raise RuntimeError('invalid generation result')
            text=generated[0]
            atomic_tensor(path,dict(request_sha256=key,provenance=provenance,before=before,after=rng_state(),text=text))
            cached=False
        out.append(text)
        atomic_json(directory/'progress.json',dict(completed=i+1,total=len(requests),cached=cached,time=time.time(),pid=os.getpid()))
        print(f'checkpoint {i+1}/{len(requests)} cached={cached}',flush=True)
    return out

def preflight(method):
    if not (ROOT/method).is_symlink() or (ROOT/method).resolve()!=STORE/method:
        raise RuntimeError('results must be the verified DATA symlink')
    if shutil.disk_usage(STORE).free<2*1024**3:raise RuntimeError('DATA has less than 2GiB free')
    if shutil.disk_usage(ROOT).free<512*1024**2:raise RuntimeError('root has less than 512MiB free')

def amendment():
    from experiments.dlm_allocation_baselines65.run import validate
    validate()
    files=[ROOT/x for x in ('recovery.py','status_safe.py','run_recovery.sh','test_recovery.py')]
    value=dict(original_config_sha256=digest(ROOT/'config.json'),files={str(p):digest(p) for p in files},
        change='Storage/exit reporting/checkpoint wrapper only. Original scoring, allocation, generation and evaluator code unchanged.',
        checkpoint='One original generate_until call per request, save/restore Python/NumPy/Torch/CUDA RNG; exact input and provenance guard.')
    path=STORE/'maintenance_amendment.json'
    if path.exists():
        if json.loads(path.read_text())!=value:raise RuntimeError('maintenance amendment mismatch')
    else:atomic_json(path,value)
    return digest(path)

def worker(method):
    import eval_llada
    from experiments.dlm_allocation_baselines65.run import harness
    preflight(method);amendment_sha=amendment();h=harness(method)
    with (h.ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=eval_llada.LLaDAEvalHarness.generate_until
        provenance=dict(config=digest(h.ROOT/'config.json'),manifest=digest(h.ROOT/'mask_manifest.json'),amendment=amendment_sha)
        directory=STORE/method/'checkpoints'
        def patched(obj,requests):return checkpoint_generate(original,obj,requests,directory,provenance)
        eval_llada.LLaDAEvalHarness.generate_until=patched
        try:
            h.run()
            atomic_json(STORE/method/'recovery_receipt.json',dict(amendment_sha256=amendment_sha,
                original_config_sha256=provenance['config'],results_sha256=digest(h.ROOT/'results.json')))
        except BaseException as e:
            h.event('failed',error=f'{type(e).__name__}: {e}');raise
        finally:eval_llada.LLaDAEvalHarness.generate_until=original

def supervise(method):
    preflight(method);amendment_sha=amendment()
    # Separate attempt logs never overwrite the interrupted original log.
    directory=STORE/method/'attempts'/str(time.time_ns());directory.mkdir(parents=True)
    with (STORE/method/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (directory/'run.log').open('wb',buffering=0) as log:
            child=subprocess.Popen([sys.executable,'-u','-m','experiments.dlm_allocation_baselines65.recovery','worker',method],
                cwd=str(ROOT.parents[1]),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            info=dict(pid=child.pid,supervisor_pid=os.getpid(),method=method,started=time.time(),log=str(directory/'run.log'),amendment=amendment_sha)
            atomic_json(STORE/method/'active_attempt.json',info)
            def forward(sig,frame):
                if child.poll() is None:os.killpg(child.pid,sig)
            signal.signal(signal.SIGTERM,forward);signal.signal(signal.SIGINT,forward)
            code=child.wait();os.fsync(log.fileno())
        exit_info=dict(**info,finished=time.time(),returncode=code,signal=(-code if code<0 else None))
        atomic_json(directory/'exit.json',exit_info);atomic_json(STORE/method/'last_exit.json',exit_info)
        if code:raise SystemExit(code if code>0 else 128-code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','supervise','worker']);p.add_argument('method',choices=['alpha','dlp','dsa','lsa']);a=p.parse_args()
    if a.phase=='preflight':preflight(a.method);print('preflight OK; amendment '+amendment())
    elif a.phase=='supervise':supervise(a.method)
    else:worker(a.method)
