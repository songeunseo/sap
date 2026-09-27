"""One frozen Multi32 cell; native generation with checkpoint and failure costs."""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import time
import torch
from experiments.dlm_multiscale_ac50.artifacts import read,write,freeze,sha,mask_identity,lock,Progress
from experiments.dlm_multiscale_ac50.evaluation import task_and_protocol
from experiments.dlm_pruning_nfe50.runner import run_cell, ensure_environment, _interrupt
from experiments.dlm_pruning_nfe50.control import idle_gpus
from .prepare import ROOT,REPO,MULTI_SHA,MULTI_MASK,PRUNED,validate,requests_for

def check_gpu(gpu):
    if gpu in idle_gpus():
        return dict(mode='exclusive',gpu=gpu)
    if os.environ.get('MULTI_ALLOW_SAME_USER_GPU') != '1':
        raise RuntimeError('Requested GPU is occupied')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    selected=[line.split(',') for line in rows if line.split(',')[0].strip()==gpu]
    if len(selected)!=1 or int(selected[0][2].strip())<49152:
        raise RuntimeError('Shared GPU requires at least48GiB free at launch')
    uuid=selected[0][1].strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True).strip().splitlines()
    pids=[int(line.split(',')[1]) for line in processes if line.split(',')[0].strip()==uuid]
    for pid in pids:
        try:owner=int(subprocess.check_output(['ps','-p',str(pid),'-o','uid='],text=True).strip())
        except (subprocess.CalledProcessError,ValueError):
            raise RuntimeError('GPU ownership changed; retry the check')
        if owner!=os.getuid():raise RuntimeError('Shared GPU is owned by a different user')
    return dict(mode='shared_same_user',gpu=gpu,free_mib=int(selected[0][2]),
                colocated_pids=pids,uid=os.getuid(),time=time.time())


def worker():
    gpu=ensure_environment()
    with lock(ROOT/'worker.lock'):
        cfg=validate()
        if not (ROOT/'code_receipt.json').exists():
            raise RuntimeError('Seal code before launch')
        audit=read(ROOT/'launch_audit.json')
        if audit['status']!='passed' or audit['config_sha256']!=sha(ROOT/'config.json') or audit['code_receipt_sha256']!=sha(ROOT/'code_receipt.json'):
            raise RuntimeError('Launch audit missing or stale')
        if (ROOT/'complete_Multi.json').exists():
            subprocess.run([sys.executable,'-B','-m','experiments.dlm_multi_nfe32.analysis'],cwd=REPO,check=True)
            if read(ROOT/'report.json')['status'] != 'complete':
                raise RuntimeError('Final analysis incomplete')
            write(ROOT/'controller.json',dict(status='complete',stage='complete',
                started=read(ROOT/'started.json')['started'],ended=time.time(),gpu=gpu,pid=os.getpid(),error=None))
            return
        allocation=check_gpu(gpu)
        write(ROOT/'gpu_allocation.json',allocation)
        signal.signal(signal.SIGTERM,_interrupt);signal.signal(signal.SIGINT,_interrupt)
        started=time.time(); counter=dict(forwards=0,forward_tokens=0)
        run_path=ROOT/'runs'/f'Multi_{time.time_ns()}.json'
        if not (ROOT/'started.json').exists():
            freeze(ROOT/'started.json',dict(started=started,gpu=gpu,config_sha256=sha(ROOT/'config.json'),
                                           code_receipt_sha256=sha(ROOT/'code_receipt.json')))
        elif read(ROOT/'started.json')['config_sha256'] != sha(ROOT/'config.json'):
            raise RuntimeError('Started config differs')
        status='failed';stage='loading';load_seconds=None;error=None
        progress=Progress(ROOT,'Multi')
        write(ROOT/'controller.json',dict(status='running',started=started,pid=os.getpid(),gpu=gpu))
        try:
            progress('loading',completed=0,total=1,candidate='Multi')
            load_start=time.monotonic()
            from experiments.projection_capacity_allocation_65.run import load_dense
            from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
            from experiments.wanda_failure_characterization.run_failure_map import model_sha
            from transformers import AutoTokenizer
            model,mapping=load_dense()
            if model_sha(model)!=cfg['identity']['model_sha256_dense']:
                raise RuntimeError('Dense physical model hash differs')
            manifest=read(cfg['source']['manifest_path'])
            if mask_identity(manifest)!=MULTI_MASK or manifest['pruned']!=PRUNED:
                raise RuntimeError('Multi manifest differs')
            if apply_manifest(model,mapping,manifest)!=PRUNED or model_sha(model)!=MULTI_SHA:
                raise RuntimeError('Multi physical mask/model differs')
            tokenizer=AutoTokenizer.from_pretrained(cfg['model']['id'],revision=cfg['model']['revision'],
                                                    trust_remote_code=True,local_files_only=True)
            load_seconds=time.monotonic()-load_start
            freeze(ROOT/'models/Multi.json',dict(arm='Multi',model_sha256=MULTI_SHA,mask_hash=MULTI_MASK,
                pruned_count=PRUNED,config_identity=cfg['config_identity']))
            task,_,_=task_and_protocol(read(cfg['source']['config_path']))
            stage='generation'
            run_cell(ROOT,cfg,'Multi',32,requests_for(cfg),task,model,tokenizer,counter,progress)
            attempts=[read(p) for p in sorted((ROOT/'attempts').glob('Multi_*.json'))]
            freeze(ROOT/'complete_Multi.json',dict(status='complete',arm='Multi',model_sha256=MULTI_SHA,
                mask_hash=MULTI_MASK,config_identity=cfg['config_identity'],gpu=gpu,
                gpu_name=torch.cuda.get_device_name(0),load_seconds=load_seconds,attempts=len(attempts),
                forwards=sum(a['forwards'] for a in attempts),forward_tokens=sum(a['forward_tokens'] for a in attempts),
                attempted_seconds=sum(a['seconds'] for a in attempts)))
            stage='analysis'
            del model,mapping
            import gc
            gc.collect();torch.cuda.empty_cache()
            subprocess.run([sys.executable,'-B','-m','experiments.dlm_multi_nfe32.analysis'],cwd=REPO,check=True)
            if read(ROOT/'report.json')['status']!='complete':raise RuntimeError('Final analysis incomplete')
            status='complete';stage='complete'
            progress('complete',completed=1,total=1,candidate='Multi')
        except BaseException as exc:
            error=repr(exc);raise
        finally:
            write(run_path,dict(status=status,stage=stage,started=started,ended=time.time(),gpu=gpu,
                config_identity=cfg['config_identity'],load_seconds=load_seconds,error=error,**counter))
            write(ROOT/'controller.json',dict(status=status,stage=stage,started=started,ended=time.time(),
                gpu=gpu,pid=os.getpid(),error=error))

if __name__=='__main__':worker()
