"""One tmux controller; two explicitly assigned, initially idle GPUs."""
from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .prepare import ROOT, REPO, validate, read, write, freeze, sha
from experiments.dlm_multiscale_ac50.artifacts import lock


def idle_gpus():
    lines = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return {r[0] for line in lines if len(r:=[x.strip() for x in line.split(',')])==3
            and int(r[1])<2048 and int(r[2])<5}


def controller(gpus):
    if not os.environ.get('TMUX'):
        raise RuntimeError('Start the controller inside tmux')
    if len(gpus)!=2 or len(set(gpus))!=2 or not all(x.isdigit() for x in gpus):
        raise RuntimeError('Exactly two distinct GPU IDs required')
    with lock(ROOT / 'controller.lock'):
        validate()
        receipt=read(ROOT / 'launch_audit.json')
        if receipt.get('status')!='passed' or receipt['config_sha256']!=sha(ROOT / 'config.json'):
            raise RuntimeError('Launch audit missing or stale')
        if receipt['code_receipt_sha256']!=sha(ROOT / 'code_receipt.json'):
            raise RuntimeError('Code receipt differs from launch audit')
        # The other controller can reserve currently idle devices for pending jobs.
        other=REPO / 'experiments/dlm_crosschain_wikitext50/output/controller.json'
        if other.exists():
            state=read(other)
            if state.get('status')=='running':
                if state.get('pending') or set(state.get('active',{})) & set(gpus):
                    raise RuntimeError('WikiText controller still owns requested GPU capacity')
        if not set(gpus)<=idle_gpus():
            raise RuntimeError('A requested GPU is occupied; no worker was launched')
        started=time.time()
        if not (ROOT / 'started.json').exists():
            freeze(ROOT / 'started.json',dict(started=started,gpus=gpus,
                config_sha256=sha(ROOT / 'config.json'),code_receipt_sha256=sha(ROOT / 'code_receipt.json')))
        attempt=ROOT / 'controller_attempts' / f'{time.time_ns()}_{os.getpid()}.json'
        active={}
        def stop(signum,frame):
            raise RuntimeError(f'Controller stopped by signal {signum}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            for arm,gpu in zip(('Dense','A'),gpus,strict=True):
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
                log=ROOT / 'logs' / f'{arm}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as stream:
                    p=subprocess.Popen([sys.executable,'-B','-u','-m','experiments.dlm_pruning_nfe50.runner',
                                        'worker','--arm',arm],cwd=REPO,env=env,stdout=stream,
                                        stderr=subprocess.STDOUT,start_new_session=True)
                active[arm]=(gpu,p)
            write(ROOT / 'controller.json',dict(status='running',started=started,pid=os.getpid(),
                  workers={a:dict(gpu=g,pid=p.pid) for a,(g,p) in active.items()}))
            while active:
                for arm,(gpu,proc) in list(active.items()):
                    code=proc.poll()
                    if code is None:
                        continue
                    if code!=0:
                        raise RuntimeError(f'{arm} worker failed (exit {code}); inspect logs/{arm}.log')
                    del active[arm]
                write(ROOT / 'controller.json',dict(status='running',started=started,pid=os.getpid(),
                      updated=time.time(),workers={a:dict(gpu=g,pid=p.pid) for a,(g,p) in active.items()}))
                if active:
                    time.sleep(3)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
            subprocess.run([sys.executable,'-B','-u','-m','experiments.dlm_pruning_nfe50.analysis'],
                           cwd=REPO,env=env,check=True)
            report=read(ROOT / 'report.json')
            if report.get('status')!='complete':
                raise RuntimeError('Final analysis did not verify completion')
            result=dict(status='complete',started=started,ended=time.time(),gpus=gpus,
                        config_sha256=sha(ROOT / 'config.json'))
            write(attempt,result);write(ROOT / 'controller.json',result)
        except BaseException as exc:
            for _,proc in active.values():
                if proc.poll() is None:
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for _,proc in active.values():
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait(timeout=10)
            result=dict(status='failed',started=started,ended=time.time(),error=repr(exc),gpus=gpus)
            write(attempt,result);write(ROOT / 'controller.json',result)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpus',default='1,3')
    args=parser.parse_args();controller(args.gpus.split(','))
