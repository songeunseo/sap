"""Isolated fixed three-arm follow-up controller and worker."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from experiments.dlm_multiscale_ac50.artifacts import read, write, freeze, sha, digest, lock, mask_identity, REPO
from experiments.dlm_multiscale_ac50.evaluation import read_predictions
from experiments.dlm_multiscale_ac50.core import metrics, paired, holm
from experiments.dlm_multiscale_ac50.run import idle_devices
from experiments.dlm_ac_screen50.run import _proc_info, proc_identity, _owned_group, terminate_owned, _worker_state
from .core import ARMS
from .prepare import ROOT, OLD, validate

SESSION = 'multi_followup50'
MODULE = 'experiments.dlm_multi_followup50.run'

def command(*args):
    return [sys.executable, '-B', '-u', '-m', MODULE, *args]

def rows_for(arm, root=None):
    root = Path(ROOT if root is None else root)
    folder = root/'gsm8k/development'/arm
    if not (folder/'identity.json').exists(): return []
    c = read(root/'config.json'); fp = read(folder/'identity.json')['fingerprint']
    mi = read(root/'candidates'/arm/'model_identity.json')
    manifest = read(root/'candidates'/arm/'mask_manifest.json')
    expected = dict(config_sha256=sha(root/'config.json'), mask_identity=mask_identity(manifest),
                    sparse_model_sha256=mi['sparse_model_sha256'], split='development',
                    requests_sha256=sha(root/'requests.json'), protocol_hash=c['protocol_hash'])
    if fp != expected or mi['mask_identity'] != expected['mask_identity'] or mi['config_sha256'] != expected['config_sha256']:
        raise ValueError('Candidate evaluation identity differs')
    return read_predictions(folder, read(root/'requests.json')['development'], fp, c['protocol_hash'], allow_partial=True)

def outputs_for(arm):
    c = read(ROOT/'config.json'); allocation = read(ROOT/'allocation.json')
    folder = ROOT/'candidates'/arm; manifest = read(folder/'mask_manifest.json')
    counts = allocation['allocations'][arm]['row_counts']
    if manifest['config_sha256'] != sha(ROOT/'config.json') or manifest['allocation_sha256'] != sha(ROOT/'allocation.json'):
        raise ValueError('Mask manifest source differs')
    if [r['selected_mask']['prune_per_row'] for r in manifest['entries']] != counts or manifest['pruned'] != c['pruning']['pruned']:
        raise ValueError('Wrong physical mask quota')
    paths = [ROOT/'allocation.json', folder/'mask_manifest.json', folder/'model_identity.json']
    for entry in manifest['entries']:
        meta = entry['selected_mask']; p = Path(meta['path'])
        if sha(p) != meta['file_sha256']: raise ValueError('Mask file changed')
        paths.append(p)
    result_folder = ROOT/'gsm8k/development'/arm
    rows = rows_for(arm); result = read(result_folder/'results.json')
    if result['status'] != 'complete' or result['total'] != 100 or len(rows) != 100 or result['correct'] != sum(r['correct'] for r in rows):
        raise ValueError('Incomplete result')
    if result['fingerprint'] != read(result_folder/'identity.json')['fingerprint'] or result['predictions_sha256'] != sha(result_folder/'predictions.json') or read(result_folder/'predictions.json') != rows:
        raise ValueError('Aggregate predictions differ from document checkpoints')
    paths.extend(sorted(result_folder.rglob('*.json')))
    for split in ('calibration','diagnostic'):
        p = ROOT/'readouts'/arm/f'{split}.json'; t = ROOT/'readouts/dense'/f'{split}.json'
        b = read(c['banks'][split]['path']); rr = read(p)
        expected = dict(config_sha256=sha(ROOT/'config.json'), mask_identity=mask_identity(manifest), bank_sha256=c['banks'][split]['sha256'])
        if rr['fingerprint'] != expected or not rr['complete']:
            raise ValueError('Candidate readout identity differs')
        actual = metrics(rr['values'], read(t)['values'], b)
        diagnostic = ROOT/'diagnostics'/arm/f'{split}.json'; saved = read(diagnostic)
        if saved != dict(config_sha256=sha(ROOT/'config.json'),mask_identity=expected['mask_identity'],readout_sha256=sha(p),teacher_sha256=sha(t),**actual):
            raise ValueError('Diagnostic metrics differ from raw readouts')
        paths.extend((p,t,diagnostic))
    return [dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths))]

def completion(arm, save=False):
    row = dict(status='complete',arm=arm,config_sha256=sha(ROOT/'config.json'),outputs=outputs_for(arm))
    path = ROOT/'done'/f'{arm}.json'
    if save: freeze(path,row)
    elif read(path) != row: raise ValueError('Completed artifacts changed')
    return row

def checked_generate(generate, calls, expected):
    def one(request):
        before = calls[0]
        result = generate(request)
        if calls[0]-before != expected:
            raise RuntimeError('Generation forward count differs from frozen steps')
        return result
    return one

def cost_accounting():
    costs=[read(p) for p in sorted((ROOT/'costs').glob('*.json'))]
    missing=[read(p) for p in sorted((ROOT/'attempts').glob('*.json')) if not (ROOT/'costs'/p.name).exists()]
    states=sum(s['forward_calls'] for r in costs for s in r.get('stages',[]) if s['stage'] in ('calibration','diagnostic'))
    generation=sum(s['forward_calls'] for r in costs for s in r.get('stages',[]) if s['stage']=='gsm8k')
    return dict(attempts=costs,incomplete_attempts=missing,complete_accounting=not missing,
                cumulative_state_forwards=states,cumulative_generation_forwards=generation,
                nominal_state_ceiling=768,nominal_generation_ceiling=76800,
                state_excess_over_nominal=max(0,states-768),generation_excess_over_nominal=max(0,generation-76800),
                counts_are_lower_bounds=bool(missing))

def worker(arm):
    launch_gates()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('1','2','3'):
        raise RuntimeError('Worker requires one authorized GPU')
    from experiments.dlm_multiscale_ac50.gpu import runtime
    from experiments.dlm_multiscale_ac50.evaluation import task_and_protocol, generator, grade, evaluate_requests
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    from transformers import AutoTokenizer
    import torch
    import resource
    started = time.monotonic(); attempt = str(time.time_ns()); stages = []; calls = [0]; outcome = 'failed'
    write(ROOT/'attempts'/f'{arm}_{attempt}.json',dict(arm=arm,attempt=attempt,pid=os.getpid(),started=time.time(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    def interrupted(*_):raise KeyboardInterrupt('Worker interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        rt = runtime(ROOT,arm)
    except BaseException as exc:
        write(ROOT/'costs'/f'{arm}_{attempt}.json',dict(arm=arm,status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',stage='initialization',error=str(exc),wall_seconds=time.monotonic()-started,forward_calls=0,stages=[],gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
        raise
    active_stage=[None,0]
    def tick(*_):
        calls[0] += 1
        name,before=active_stage
        cap={'build':0,'calibration':128,'diagnostic':128,'gsm8k':25600}.get(name)
        if cap is not None and calls[0]-before>cap:raise RuntimeError('Stage forward ceiling exceeded')
        rt.memory_check()
    hook = rt.model.register_forward_hook(tick)
    @contextlib.contextmanager
    def stage(name):
        before = calls[0]; begin = time.monotonic(); active_stage[:]=[name,before]
        try: yield
        finally:
            stages.append(dict(stage=name,wall_seconds=time.monotonic()-begin,forward_calls=calls[0]-before))
            active_stage[:]=[None,calls[0]]
    try:
        with stage('build'):
            manifest = rt.build(arm); identity = mask_identity(manifest); sparse_sha = model_sha(rt.model)
            freeze(ROOT/'candidates'/arm/'model_identity.json',dict(config_sha256=rt.config_hash,mask_identity=identity,sparse_model_sha256=sparse_sha))
        for split in ('calibration','diagnostic'):
            with stage(split): rt.distortion(arm,identity,split)
        c = rt.config
        fp = dict(config_sha256=rt.config_hash,mask_identity=identity,sparse_model_sha256=sparse_sha,
                  requests_sha256=sha(ROOT/'requests.json'),split='development',protocol_hash=c['protocol_hash'])
        task,_,_ = task_and_protocol(read(c['legacy_config']))
        tokenizer = AutoTokenizer.from_pretrained(c['model']['id'],revision=c['model']['revision'],trust_remote_code=True,local_files_only=True)
        generate = generator(rt.model,tokenizer,c['evaluation'],rt.banks['calibration']['mask_id'])
        generate = checked_generate(generate,calls,c['evaluation']['denoising_steps'])
        with stage('gsm8k'):
            result = evaluate_requests(ROOT/'gsm8k/development'/arm,read(ROOT/'requests.json')['development'],fp,c['protocol_hash'],arm,generate,lambda req,text:grade(task,req,text),rt.progress)
        if model_sha(rt.model) != sparse_sha: raise RuntimeError('Weights changed during evaluation')
        outcome = 'complete';rt.progress('complete',completed=100,total=100,correct=result['correct'])
    except KeyboardInterrupt:
        outcome='interrupted'
        raise
    finally:
        hook.remove()
        write(ROOT/'costs'/f'{arm}_{attempt}.json',dict(arm=arm,status=outcome,wall_seconds=time.monotonic()-started,
              forward_calls=calls[0],stages=stages,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))

def report():
    reference = rows_for('Multi',OLD)
    if len(reference) != 100: raise RuntimeError('Reference incomplete')
    scores = {'Multi':dict(correct=sum(r['correct'] for r in reference),total=100)}
    comparisons, diagnostics = {}, {}
    for arm in ARMS:
        rr = rows_for(arm);scores[arm] = dict(correct=sum(r['correct'] for r in rr),total=len(rr))
        if len(rr)==100:
            completion(arm)
            comparisons[arm] = paired([r['correct'] for r in reference],[r['correct'] for r in rr])
        diagnostics[arm] = {s:read(p)['mean'] for s in ('calibration','diagnostic') if (p:=ROOT/'diagnostics'/arm/f'{s}.json').exists()}
    if len(comparisons)==3:holm(comparisons)
    result = dict(status='complete' if len(comparisons)==3 else 'partial',scores=scores,paired_vs_Multi=comparisons,
                  diagnostics=diagnostics,allocations=read(ROOT/'allocation.json')['allocations'],reference_diagnostics={s:read(OLD/'diagnostics/Multi'/f'{s}.json')['mean'] for s in ('calibration','diagnostic')},
                  cost_accounting=cost_accounting(),costs_note='Attempts include loading; stage totals overlap and exclude loading. Reused probe/teacher costs excluded.',
                  limitations=['Repeated development mini100; not confirmation','No loss or score based arm selection','No guarantee of downstream improvement'])
    write(ROOT/'report.json',result)
    return result

def status():
    state = read(ROOT/'execution.json') if (ROOT/'execution.json').exists() else dict(status='prepared')
    print('Multi allocation follow-up |',state['status'])
    print('Reference Multi: 63/100 (reused)')
    for arm in ARMS:
        rows = rows_for(arm);text = f'{arm:13} {sum(r["correct"] for r in rows):3}/{len(rows):3}'
        p=ROOT/'progress'/f'{arm}.json'
        if p.exists():
            q=read(p);text+=f'  {q["stage"]} {q["completed"]}/{q["total"]} GPU {q.get("gpu")}';eta=q.get('eta_seconds')
            if eta is not None:text+=f' ETA {eta/60:.1f} min'
        print(text)
    if state.get('error'):print('Error:',state['error'])

def launch_gates():
    validate()
    audit=read(ROOT/'audit_pass.json'); cpu=read(ROOT/'cpu_validation.json')
    ch=sha(ROOT/'config.json')
    if audit.get('status')!='passed' or audit.get('config_sha256')!=ch or audit.get('cpu_validation_sha256')!=sha(ROOT/'cpu_validation.json'):
        raise RuntimeError('Fresh independent audit required')
    if cpu.get('status')!='passed' or cpu.get('returncode')!=0 or cpu.get('config_sha256')!=ch or cpu.get('tests_run',0)<8:
        raise RuntimeError('Successful CPU validation required')
    if not cpu.get('test_sources'):raise RuntimeError('Test identities missing')
    for p,h in cpu['test_sources'].items():
        if sha(p)!=h:raise RuntimeError('Validated tests changed')


def pipeline(gpus):
    if not gpus or len(set(gpus))!=len(gpus) or not set(gpus)<=set(('1','2','3')):raise ValueError('Authorized GPUs: 1,2,3')
    if not os.environ.get('TMUX'):raise RuntimeError('Use tmux launch')
    active=[];done=[]
    def interrupted(*_):raise KeyboardInterrupt('Controller interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with lock(ROOT/'pipeline.lock'):
        launch_gates();freeze(ROOT/'started.json',dict(config_sha256=sha(ROOT/'config.json')))
        base=dict(pid=os.getpid(),start_ticks=proc_identity(os.getpid()),started=time.time(),gpus=gpus,config_sha256=sha(ROOT/'config.json'))
        try:
            for arm in ARMS:
                if (ROOT/'done'/f'{arm}.json').exists():completion(arm);done.append(arm)
            while len(done)<len(ARMS):
                for gpu in gpus:
                    if any(r['gpu']==gpu for r in active):continue
                    arm=next((a for a in ARMS if a not in done and all(r['job']!=a for r in active)),None)
                    if arm is None:break
                    idle_devices([gpu])
                    log=ROOT/'logs'/f'{arm}.log';log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('a')
                    p=subprocess.Popen(command('worker','--arm',arm),cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    info=_proc_info(p.pid)
                    active.append(dict(proc=p,pid=p.pid,start_ticks=info['start_ticks'],pgid=info['pgid'],session=info['session'],gpu=gpu,job=arm,stream=stream,log=str(log),started=time.time()))
                    write(ROOT/'execution.json',dict(base,status='running',completed=done,workers=[_worker_state(r) for r in active]))
                for r in list(active):
                    code=r['proc'].poll()
                    if code is None:continue
                    if code:raise RuntimeError(f'{r["job"]} failed ({code}); see {r["log"]}')
                    completion(r['job'],save=True);r['stream'].close();active.remove(r);done.append(r['job'])
                write(ROOT/'execution.json',dict(base,status='running',completed=done,workers=[_worker_state(r) for r in active]))
                if active:time.sleep(2)
            final=report();write(ROOT/'execution.json',dict(base,status=final['status'],completed=done,workers=[],ended=time.time(),obsidian_sync='pending'))
        except BaseException as exc:
            terminate_owned(active)
            write(ROOT/'execution.json',dict(base,status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',completed=done,workers=[],error=str(exc),ended=time.time(),cost_accounting=cost_accounting()))
            raise
        finally:
            for r in active:r['stream'].close()

def launch(gpus,dry=False):
    if not gpus or len(set(gpus))!=len(gpus) or not set(gpus)<=set(('1','2','3')):raise ValueError('Authorized GPUs: 1,2,3')
    if dry:print(json.dumps(dict(arms=ARMS,gpus=gpus,max_new_generations=300,max_candidate_state_forwards=768,gpu_queried=False)));return
    launch_gates()
    with lock(ROOT/'launch.lock'):
        with lock(ROOT/'pipeline.lock'):pass
        if subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise RuntimeError('Session exists')
        old=read(ROOT/'execution.json') if (ROOT/'execution.json').exists() else {}
        if any(_owned_group(r) for r in old.get('workers',[])):raise RuntimeError('Recorded workers still alive; stop first')
        idle_devices(gpus)
        env={k:os.environ.get(k,'') for k in ['PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE']}
        log=ROOT/'logs/controller.log';log.parent.mkdir(parents=True,exist_ok=True)
        cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=',*[f'{k}={v}' for k,v in env.items()],*command('pipeline','--gpus',','.join(gpus))])+' >> '+shlex.quote(str(log))+' 2>&1'
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(REPO),cmd],check=True)
    print('Launched',SESSION,'GPUs',','.join(gpus))

def stop():
    p=ROOT/'execution.json'
    if not p.exists():return
    state=read(p);pid=state.get('pid')
    if pid and proc_identity(pid)==state.get('start_ticks'):
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        if MODULE not in cmd or 'pipeline' not in cmd:raise RuntimeError('Unrelated controller')
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline and proc_identity(pid)==state['start_ticks']:time.sleep(.2)
        if proc_identity(pid)==state['start_ticks']:raise RuntimeError('Controller did not exit; inspect before escalation')
    terminate_owned(state.get('workers',[]))
    if read(p).get('status')=='running':write(p,dict(state,status='interrupted',workers=[],ended=time.time()))

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='cmd',required=True)
    for name in ('prepare','validate','status','report','stop'):sub.add_parser(name)
    for name in ('launch','pipeline'):
        p=sub.add_parser(name);p.add_argument('--gpus',required=True);p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('worker');p.add_argument('--arm',choices=ARMS,required=True)
    args=parser.parse_args()
    if args.cmd!='worker':os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.cmd=='prepare':
        from .prepare import prepare
        print(json.dumps(prepare(),indent=2))
    elif args.cmd=='validate':validate();print('Validation passed')
    elif args.cmd=='status':status()
    elif args.cmd=='report':print(json.dumps(report(),indent=2))
    elif args.cmd=='stop':stop()
    elif args.cmd=='launch':launch(args.gpus.split(','),args.dry_run)
    elif args.cmd=='pipeline':pipeline(args.gpus.split(','))
    else:
        with lock(ROOT/'locks'/f'{args.arm}.lock'):worker(args.arm)

if __name__=='__main__':main()
