"""Fixed-mask separate100 evaluation with the previously audited tmux controller."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import signal
import time
from experiments.dlm_multiscale_ac50.artifacts import DEFAULT_ROOT as OLD,read,write,freeze,sha,digest,checked,mask_identity,lock,REPO
from experiments.dlm_multiscale_ac50.prepare import validate as validate_base
from experiments.dlm_multiscale_ac50.evaluation import read_predictions
from experiments.dlm_multiscale_ac50.core import paired,holm
from experiments.dlm_multi_followup50 import run as controller

HERE=Path(__file__).resolve().parent
ROOT=HERE/'output'
ARMS=('Multi','A','Uniform')
MODULE='experiments.dlm_multi_validation10050.run'
BASE_COST_ACCOUNTING=controller.cost_accounting

def expected_ids():
    return sorted(sorted(range(200,1319),key=lambda i:hashlib.sha256(f'multiscale-ac-confirm:20260922:{i}'.encode()).hexdigest())[:100])

def validate_requests(requests,old_requests):
    rr=requests['development']
    if [r['example_id'] for r in rr]!=expected_ids() or len({r['doc_hash'] for r in rr})!=100:
        raise ValueError('Wrong or duplicate validation sample')
    if requests['confirmation'] or rr!=old_requests['confirmation']:
        raise ValueError('Frozen requests changed')
    if set(r['doc_hash'] for r in rr)&set(r['doc_hash'] for r in old_requests['development']):
        raise ValueError('Development overlap')
    if requests['protocol_hash']!=old_requests['protocol_hash'] or requests['protocol']!=old_requests['protocol']:
        raise ValueError('Protocol differs')

def prepare():
    if (ROOT/'started.json').exists():raise RuntimeError('Started experiment immutable')
    c=copy.deepcopy(validate_base(OLD));original=read(OLD/'requests.json')
    requests=copy.deepcopy(original);requests['development']=original['confirmation'];requests['confirmation']=[]
    validate_requests(requests,original)
    # No confirmation outputs exist for these candidates in the source run.
    prior=list((OLD/'gsm8k/confirmation').rglob('*.json'))
    if prior:raise RuntimeError('Source confirmation already has artifacts; review before fresh validation claim')
    freeze(ROOT/'requests.json',requests);freeze(ROOT/'cached_development.json',read(OLD/'cached_development.json'))
    c['requests_sha256']=sha(ROOT/'requests.json')
    sources=c['sources']
    for p in (OLD/'config.json',OLD/'requests.json',OLD/'allocation.json'):
        sources[str(p)]=sha(p)
    for folder in (HERE,REPO/'experiments/dlm_multi_followup50',REPO/'experiments/dlm_ac_screen50'):
        for p in folder.glob('*.py'):
            if not p.name.startswith('test_'):sources[str(p)]=sha(p)
    for p in (HERE/'PLAN.md',HERE/'run.sh'):sources[str(p)]=sha(p)
    candidates={}
    import torch
    from experiments.dlm_loss_aggregation.core import mask_sha256,unpack_mask
    for arm in ARMS:
        if arm=='Uniform':
            item=c['legacy_manifests']['uniform'];p=Path(item['path']);identity=item['identity'];model_sha=item['sparse_model_sha256']
        else:
            p=OLD/'candidates'/arm/'mask_manifest.json';mi=OLD/'candidates'/arm/'model_identity.json';v=read(mi)
            sources[str(mi)]=sha(mi)
            if v['config_sha256']!=sha(OLD/'config.json'):raise ValueError('Source model config mismatch')
            identity=v['mask_identity'];model_sha=v['sparse_model_sha256']
        m=read(p);sources[str(p)]=sha(p)
        if mask_identity(m)!=identity or m['pruned']!=3489660928 or len(m['entries'])!=224:
            raise ValueError('Wrong source mask identity/budget')
        total=0
        for e in m['entries']:
            meta=e['selected_mask'];checked(meta['path'],meta['file_sha256'])
            packed=torch.load(meta['path'],map_location='cpu',weights_only=False)
            if mask_sha256(packed)!=meta['mask_sha256']:raise ValueError('Packed mask hash mismatch')
            mask=unpack_mask(packed)
            if list(mask.shape)!=e['shape'] or not bool((mask.sum(1)==meta['prune_per_row']).all()) or int(mask.sum())!=meta['pruned']:
                raise ValueError('Physical mask counts differ')
            del mask
            total+=meta['pruned'];sources[meta['path']]=meta['file_sha256']
        if total!=3489660928:raise ValueError('Mask counts differ')
        candidates[arm]=dict(manifest_path=str(p),mask_identity=identity,sparse_model_sha256=model_sha)
    c['validation100']=dict(arms=list(ARMS),ids=expected_ids(),candidates=candidates,
        contrasts=[['A','Multi'],['Uniform','Multi'],['Uniform','A']],max_forwards=76800,
        scope='Previously frozen sample held aside from current method selection; historical project use not ruled out')
    freeze(ROOT/'config.json',c);validate()
    return dict(config_sha256=sha(ROOT/'config.json'),ids=expected_ids(),candidates=candidates)

def validate():
    c=validate_base(ROOT)
    if c['validation100']['arms']!=list(ARMS) or c['validation100']['ids']!=expected_ids():raise ValueError('Contract changed')
    if c['validation100']['contrasts']!=[['A','Multi'],['Uniform','Multi'],['Uniform','A']]:raise ValueError('Contrasts changed')
    validate_requests(read(ROOT/'requests.json'),read(OLD/'requests.json'))
    return c

def gates():
    validate();r=read(ROOT/'cpu_validation.json')
    if r.get('status')!='passed' or r.get('returncode')!=0 or r.get('config_sha256')!=sha(ROOT/'config.json') or r.get('tests_run',0)<5:
        raise RuntimeError('Matching CPU validation required')
    if not r.get('test_sources'):raise RuntimeError('Test source hashes missing')
    for p,h in r['test_sources'].items():checked(p,h)
    checked(ROOT/'cpu_validation.log',r['log_sha256'])

def fingerprint(arm):
    c=read(ROOT/'config.json');source=c['validation100']['candidates'][arm]
    return dict(config_sha256=sha(ROOT/'config.json'),mask_identity=source['mask_identity'],sparse_model_sha256=source['sparse_model_sha256'],requests_sha256=sha(ROOT/'requests.json'),split='validation100',protocol_hash=c['protocol_hash'])

def folder(arm):return ROOT/'gsm8k/validation100'/arm

def rows_for(arm):
    f=folder(arm)
    if not (f/'identity.json').exists():return []
    fp=fingerprint(arm)
    if read(f/'identity.json')!=dict(fingerprint=fp,document_ids=expected_ids()):raise ValueError('Evaluation identity differs')
    mi=read(ROOT/'candidates'/arm/'model_identity.json')
    if mi!=dict(config_sha256=fp['config_sha256'],mask_identity=fp['mask_identity'],sparse_model_sha256=fp['sparse_model_sha256']):raise ValueError('Physical model identity differs')
    return read_predictions(f,read(ROOT/'requests.json')['development'],fp,fp['protocol_hash'],allow_partial=True)

def completion(arm,save=False):
    rr=rows_for(arm);f=folder(arm);r=read(f/'results.json')
    if len(rr)!=100 or r['status']!='complete' or r['total']!=100 or r['correct']!=sum(x['correct'] for x in rr):raise ValueError('Incomplete result')
    if r['fingerprint']!=fingerprint(arm) or read(f/'predictions.json')!=rr or r['predictions_sha256']!=sha(f/'predictions.json'):raise ValueError('Aggregate changed')
    src=read(ROOT/'config.json')['validation100']['candidates'][arm];mp=Path(src['manifest_path']);m=read(mp)
    checked(mp,read(ROOT/'config.json')['sources'][str(mp)])
    if mask_identity(m)!=src['mask_identity']:raise ValueError('Source mask identity changed')
    paths=list(f.rglob('*.json'))+[ROOT/'candidates'/arm/'model_identity.json',mp]
    for e in m['entries']:
        meta=e['selected_mask'];checked(meta['path'],meta['file_sha256']);paths.append(Path(meta['path']))
    receipt=dict(status='complete',arm=arm,config_sha256=sha(ROOT/'config.json'),outputs=[dict(path=str(p),sha256=sha(p)) for p in sorted(set(paths))])
    if save:freeze(ROOT/'done'/f'{arm}.json',receipt)
    elif read(ROOT/'done'/f'{arm}.json')!=receipt:raise ValueError('Completion changed')
    return receipt

def costs():
    r=BASE_COST_ACCOUNTING();r['nominal_state_ceiling']=0
    r['state_excess_over_nominal']=r['cumulative_state_forwards'];return r

def report():
    rr={a:rows_for(a) for a in ARMS};comparisons={}
    complete=all(len(v)==100 for v in rr.values())
    for a in ARMS:
        if len(rr[a])==100:completion(a)
    if complete:
        for ref,a in read(ROOT/'config.json')['validation100']['contrasts']:
            comparisons[f'{a}_minus_{ref}']=paired([x['correct'] for x in rr[ref]],[x['correct'] for x in rr[a]])
        holm(comparisons)
    out=dict(status='complete' if complete else 'partial',sample='previously frozen separate100',document_ids=expected_ids(),scores={a:dict(correct=sum(x['correct'] for x in v),total=len(v)) for a,v in rr.items()},paired=comparisons,cost_accounting=costs(),limitations=['Historical project exposure not ruled out','Masks selected using earlier development mini100','Do not interpret non-significance as equivalence'])
    write(ROOT/'report.json',out);return out

def worker(arm):
    gates()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('1','2','3'):raise RuntimeError('Authorized single GPU required')
    from experiments.dlm_multiscale_ac50.gpu import runtime
    from experiments.dlm_multiscale_ac50.evaluation import task_and_protocol,generator,grade,evaluate_requests
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    from transformers import AutoTokenizer
    import torch,resource
    begin=time.monotonic();attempt=str(time.time_ns());calls=[0];status='failed';stages=[];hook=None
    write(ROOT/'attempts'/f'{arm}_{attempt}.json',dict(arm=arm,pid=os.getpid(),started=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    def interrupted(*_):raise KeyboardInterrupt('Worker interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        rt=runtime(ROOT,arm);source=rt.config['validation100']['candidates'][arm];stage=time.monotonic()
        rt.apply_manifest(read(source['manifest_path']))
        actual=model_sha(rt.model)
        if actual!=source['sparse_model_sha256']:raise RuntimeError('Reused physical sparse model differs')
        freeze(ROOT/'candidates'/arm/'model_identity.json',dict(config_sha256=rt.config_hash,mask_identity=source['mask_identity'],sparse_model_sha256=actual))
        stages.append(dict(stage='apply_existing_mask',wall_seconds=time.monotonic()-stage,forward_calls=0))
        def tick(*_):
            calls[0]+=1
            if calls[0]>25600:raise RuntimeError('Generation ceiling exceeded')
            rt.memory_check()
        hook=rt.model.register_forward_hook(tick)
        task,_,_=task_and_protocol(read(rt.config['legacy_config']))
        tok=AutoTokenizer.from_pretrained(rt.config['model']['id'],revision=rt.config['model']['revision'],trust_remote_code=True,local_files_only=True)
        gen=controller.checked_generate(generator(rt.model,tok,rt.config['evaluation'],rt.banks['calibration']['mask_id']),calls,256)
        stage=time.monotonic()
        try:result=evaluate_requests(folder(arm),read(ROOT/'requests.json')['development'],fingerprint(arm),rt.config['protocol_hash'],arm,gen,lambda r,t:grade(task,r,t),rt.progress)
        finally:stages.append(dict(stage='gsm8k',wall_seconds=time.monotonic()-stage,forward_calls=calls[0]))
        if model_sha(rt.model)!=actual:raise RuntimeError('Weights changed during evaluation')
        status='complete';rt.progress('complete',completed=100,total=100,correct=result['correct'])
    except KeyboardInterrupt:
        status='interrupted';raise
    finally:
        if hook is not None:hook.remove()
        write(ROOT/'costs'/f'{arm}_{attempt}.json',dict(arm=arm,status=status,wall_seconds=time.monotonic()-begin,forward_calls=calls[0],stages=stages,peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,gpu=os.environ['CUDA_VISIBLE_DEVICES']))

def status():
    state=read(ROOT/'execution.json') if (ROOT/'execution.json').exists() else {'status':'prepared'}
    print('Multi / A / Uniform | separate100 |',state['status'])
    for arm in ARMS:
        rr=rows_for(arm);p=ROOT/'progress'/f'{arm}.json';q=read(p) if p.exists() else {}
        eta=q.get('eta_seconds');end=f' ETA {eta/60:.1f}min' if eta is not None else ''
        print(f'{arm:8} {sum(x["correct"] for x in rr):3}/{len(rr):3}  {q.get("stage","pending")} GPU {q.get("gpu","-")}{end}')
    if state.get('error'):print(state['error'])

def configure_controller():
    # Explicit adapter: reusable ownership/tmux scheduling, study-specific gates/results.
    controller.ROOT=ROOT;controller.ARMS=ARMS;controller.MODULE=MODULE;controller.SESSION='multi_validation10050'
    controller.validate=validate;controller.launch_gates=gates;controller.completion=completion;controller.report=report;controller.cost_accounting=costs


def main():
    configure_controller()
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('prepare','validate','status','report','stop'):sub.add_parser(name)
    for name in ('launch','pipeline'):
        x=sub.add_parser(name);x.add_argument('--gpus',required=True)
    x=sub.add_parser('worker');x.add_argument('--arm',choices=ARMS,required=True)
    a=p.parse_args()
    if a.cmd!='worker':os.environ['CUDA_VISIBLE_DEVICES']=''
    if a.cmd=='prepare':print(prepare())
    elif a.cmd=='validate':validate();print('Validation passed')
    elif a.cmd=='status':status()
    elif a.cmd=='report':print(report())
    elif a.cmd=='stop':controller.stop()
    elif a.cmd=='launch':controller.launch(a.gpus.split(','))
    elif a.cmd=='pipeline':controller.pipeline(a.gpus.split(','))
    else:
        with lock(ROOT/'locks'/f'{a.arm}.lock'):worker(a.arm)

if __name__=='__main__':main()
