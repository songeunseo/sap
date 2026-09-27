"""Shared dense-state collection, frozen allocation, up to two mini100 evaluations."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.dlm_dual_role_allocation.io import load_frozen_inputs, file_sha256 as sha, atomic_write_json as write
from experiments.dlm_role_token_geometry.core import METHODS, sufficient_statistics, pool

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / 'experiments/dlm_dual_role_mini100'
RAW = REPO / 'experiments/dlm_dual_role_allocation/role_reconstruction_raw.json'
EVAL = REPO / 'experiments/dlm_loss_aggregation/exp002/config.yaml'


def read(p): return json.loads(Path(p).read_text())


def frozen(p, value):
    if Path(p).exists():
        if read(p) != value: raise RuntimeError(f'frozen artifact changed: {p}')
    else: write(p, value)


def event(method, stage, **kw):
    v = dict(method=method, stage=stage, time=time.time(), pid=os.getpid(), **kw)
    write(ROOT/method/'progress.json', v)
    print(json.dumps(v), flush=True)


def validate():
    c = read(ROOT/'config.json')
    for p, h in c['sources'].items():
        if sha(p) != h: raise RuntimeError(f'frozen input changed: {p}')
    return c


def freeze():
    from experiments.dlm_loss_aggregation.exp002.run import load_config, _evaluation_config_hash
    from experiments.dlm_capacity_predictor.audit_existing import audit_prediction_file
    if (ROOT/'config.json').exists(): return validate()
    inputs = load_frozen_inputs()
    ph, protocol = _evaluation_config_hash(load_config(EVAL))
    baseline = OLD/'gsm8k/role_100_predictions.jsonl'
    audit = audit_prediction_file(baseline, 100)
    receipt = read(baseline.with_suffix('.receipt.json'))
    if (audit['correct'] != 24 or receipt['status'] != 'complete' or receipt['predictions_sha256'] != sha(baseline)
        or receipt['fingerprint']['manifest_sha256'] != sha(OLD/'role65_mask_manifest.json')
        or receipt['fingerprint']['protocol_sha256'] != ph or audit['evaluation_config_hash'] != ph):
        raise RuntimeError('historical Role receipt mismatch')
    sources = dict(inputs.receipt['files'])
    files = [RAW, EVAL, OLD/'config.json', baseline, baseline.with_suffix('.receipt.json'), OLD/'role65_mask_manifest.json',
        REPO/'experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl',
        REPO/'experiments/dlm_loss_aggregation/config.yaml', REPO/'eval_llada.py', REPO/'generate.py',
        REPO/'experiments/dlm_loss_aggregation/exp002/run.py', REPO/'experiments/dlm_loss_aggregation/run.py',
        REPO/'experiments/projection_capacity_allocation_65/core.py', REPO/'experiments/projection_capacity_allocation_65/run.py',
        REPO/'experiments/projection_capacity_followup_65/run_heldout.py', REPO/'experiments/projection_capacity_followup_65/core.py',
        REPO/'experiments/dlm_dual_role_allocation/io.py', REPO/'experiments/dlm_dual_role_allocation/core.py',
        *ROOT.glob('*.py'), ROOT/'run.sh', ROOT/'README.md']
    sources.update({str(p):sha(p) for p in files})
    c = dict(model=inputs.config['model'], states=80, state_digest=inputs.metadata['state_digest'],
        grid=[.50,.55,.60,.65,.70,.75], pruned=4536008704, weights=6979321856, methods=list(METHODS),
        sources=sources, protocol_hash=ph, protocol=protocol, baseline=str(baseline), baseline_correct=24,
        definitions=dict(token_relative='role pooled sum_t ||z-y||^2/||y||^2 divided by role token count',
                         token_angular='role pooled sum_t (1-cos(z,y)) divided by role token count'),
        fixed='dense background; original Standard Wanda candidates; max of role levels; raw marginal greedy; exact row-floor65 budget',
        zero_dense='stop', zero_candidate_angular=1, clip_cosine=[-1,1], smoothing=False, backward=False,
        selection='no fitted coefficients or rank cutoff; identical masks reuse only; no automatic full evaluation',
        stats='paired mini100 McNemar vs Role; Holm family2; angular-vs-relative descriptive, no independent confirmation',
        baseline_eval_seconds=receipt['metrics']['eval_seconds'])
    frozen(ROOT/'config.json', c)
    return c


@torch.inference_mode()
def collect():
    from experiments.projection_capacity_allocation_65.run import load_dense, read_mask, DENSE_SHA
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    c = validate()
    if (ROOT/'collection.json').exists():
        r=read(ROOT/'collection.json')
        if r['config_sha256'] != sha(ROOT/'config.json') or r['stats_sha256'] != sha(ROOT/'statistics.json'):
            raise RuntimeError('collection receipt mismatch')
        return
    inputs=load_frozen_inputs(); names=inputs.metadata['module_names']
    event('collection','loading_dense')
    model,mapping=load_dense()
    if list(mapping)!=names: raise RuntimeError('module ordering changed')
    device=next(model.parameters()).device; masks={}
    for i,e in enumerate(inputs.candidate['entries']):
        masks[e['name']]=[read_mask(e,k,device) for k in range(6)]
        if (i+1)%32==0: event('collection','loading_masks',completed=i+1,total=224)
    state={}; handles=[]
    for i,name in enumerate(names):
        def hook(module, inp, out, i=i, name=name):
            x=inp[0]
            if x.shape[0]!=1 or not torch.equal(F.linear(x,module.weight,module.bias),out):
                raise RuntimeError('dense batch1 same-path mismatch')
            for k,mask in enumerate(masks[name]):
                z=F.linear(x,module.weight.masked_fill(mask,0),module.bias)
                state['stats'][i,k]=sufficient_statistics(z,out,state['mask'])
            state['seen'].add(name)
        handles.append(mapping[name].register_forward_hook(hook))
    records=[]; started=time.monotonic(); new=0
    try:
        for i,s in enumerate(inputs.states['states']):
            path=ROOT/'states'/f'{i:03d}.json'
            if path.exists():
                row=read(path)
                if row['config_sha256']!=sha(ROOT/'config.json') or row['state_index']!=i:
                    raise RuntimeError('state checkpoint mismatch')
            else:
                state.clear(); state.update(stats=np.zeros((224,6,2,6)),seen=set(),mask=torch.tensor(s['mask'],device=device,dtype=torch.bool))
                model(torch.tensor(s['noisy_ids'],device=device,dtype=torch.long))
                if state['seen']!=set(names): raise RuntimeError('missing hook')
                row=dict(state_index=i,config_sha256=sha(ROOT/'config.json'),statistics=state['stats'].tolist())
                write(path,row); new+=1
            records.append(row['statistics'])
            event('collection','collecting',completed=i+1,total=80,eta_seconds=(79-i)*(time.monotonic()-started)/new if new else None)
    finally:
        for h in handles:h.remove()
    if model_sha(model)!=DENSE_SHA: raise RuntimeError('dense weights mutated')
    validate()
    frozen(ROOT/'statistics.json',dict(names=names,records=records))
    frozen(ROOT/'collection.json',dict(status='complete',config_sha256=sha(ROOT/'config.json'),stats_sha256=sha(ROOT/'statistics.json'),dense_sha256=DENSE_SHA))
    event('collection','complete',completed=80,total=80)


def prepare():
    from scipy.stats import spearmanr
    from experiments.projection_capacity_allocation_65.core import allocate
    from experiments.projection_capacity_followup_65.core import build_selected_manifest
    from experiments.dlm_dual_role_allocation.core import allocation_mask_xor
    from experiments.dlm_dual_role_mini100.core import validate_manifest
    c=validate(); inputs=load_frozen_inputs(); saved=read(ROOT/'statistics.json'); receipt=read(ROOT/'collection.json')
    if receipt['config_sha256']!=sha(ROOT/'config.json') or receipt['stats_sha256']!=sha(ROOT/'statistics.json'):
        raise RuntimeError('statistics provenance mismatch')
    raw=read(RAW)['projections']; names=inputs.metadata['module_names']
    if saved['names']!=names or [p['name'] for p in raw]!=names: raise RuntimeError('names mismatch')
    den=np.array([[sum(s['levels'][0]['den_'+r] for s in p['states']) for r in ('masked','unmasked')] for p in raw])
    curves=pool(saved['records'],den); shapes=[e['shape'] for e in inputs.candidate['entries']]
    old=read(OLD/'role65_mask_manifest.json'); control=allocate(curves['control'].max(-1),shapes)
    gate=dict(passed=control['levels']==old['allocation_levels'],allocation=control)
    frozen(ROOT/'dense_control.json',gate)
    if not gate['passed']: raise RuntimeError('batch1 control differs from historical Role; stop before downstream')
    comparison={}
    for method in METHODS:
        values=curves[method]; score=values.max(-1); allocation=allocate(score,shapes)
        if allocation['pruned']!=c['pruned'] or allocation['budget_error']: raise RuntimeError('exact budget failed')
        manifest=build_selected_manifest(method,inputs.candidate['entries'],allocation['sparsities'],c['grid'])
        manifest.update(config_sha256=sha(ROOT/'config.json'),allocation_levels=allocation['levels'])
        validate_manifest(manifest,names,c['pruned'],c['weights'])
        frozen(ROOT/method/'mask_manifest.json',manifest)
        frozen(ROOT/method/'curves.json',dict(masked=values[:,:,0].tolist(),unmasked=values[:,:,1].tolist(),role=score.tolist()))
        margins=np.diff(score,axis=1); original=np.diff(curves['control'].max(-1),axis=1)
        comparison[method]=dict(allocation=allocation,mask_difference=allocation_mask_xor(allocation['levels'],old['allocation_levels'],inputs.candidate['entries']),
            negative_marginal_count=int((margins<0).sum()),
            marginal_spearman_by_increment=[float(spearmanr(margins[:,k],original[:,k]).statistic) for k in range(5)],
            manifest_sha256=sha(ROOT/method/'mask_manifest.json'))
        frozen(ROOT/method/'allocation.json',comparison[method])
    frozen(ROOT/'comparison.json',comparison)
    event('collection','prepared')


def validate_predictions(data,c):
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    _validate_rows(data,read_jsonl(c['baseline']),c['protocol_hash'])


def read_jsonl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def completed(method,c):
    dest=ROOT/method
    if not (dest/'results.json').exists():
        if (dest/'predictions.jsonl').exists():raise RuntimeError('partial output exists; preserve and investigate')
        return None
    r=read(dest/'results.json')
    if r['config_sha256']!=sha(ROOT/'config.json') or r['predictions_sha256']!=sha(dest/'predictions.jsonl') or r['manifest_sha256']!=sha(dest/'mask_manifest.json'):
        raise RuntimeError('result provenance mismatch')
    data=read_jsonl(dest/'predictions.jsonl'); validate_predictions(data,c)
    return data


@torch.inference_mode()
def evaluate(method):
    from transformers import AutoTokenizer
    from experiments.dlm_loss_aggregation.exp002.run import _evaluate_gsm8k,_write_jsonl,load_config,_evaluation_config_hash
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    from experiments.dlm_dual_role_mini100.core import same_selected_masks
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    c=validate(); dest=ROOT/method
    if completed(method,c) is not None:return
    manifest=read(dest/'mask_manifest.json')
    if read(dest/'allocation.json')['manifest_sha256']!=sha(dest/'mask_manifest.json'):raise RuntimeError('allocation receipt mismatch')
    data=None; reused=None; metrics={}; sparse_hash=None
    if same_selected_masks(manifest,read(OLD/'role65_mask_manifest.json')):
        data=read_jsonl(c['baseline']);reused='historical_role'
    elif method=='token_angular' and same_selected_masks(manifest,read(ROOT/'token_relative/mask_manifest.json')):
        data=completed('token_relative',c)
        if data is None: raise RuntimeError('identical prior candidate not complete')
        reused='token_relative'
    if data is None:
        event(method,'loading_dense'); model,mapping=load_dense()
        if apply_manifest(model,mapping,manifest)!=c['pruned']:raise RuntimeError('applied budget mismatch')
        sparse_hash=model_sha(model); cfg=load_config(EVAL); ph,_=_evaluation_config_hash(cfg)
        if ph!=c['protocol_hash']:raise RuntimeError('protocol changed')
        tokenizer=AutoTokenizer.from_pretrained(cfg['model']['id'],revision=cfg['model']['revision'],trust_remote_code=True)
        event(method,'gsm8k',completed=0,total=100)
        metrics,data=_evaluate_gsm8k(model,tokenizer,cfg,method,100,ph)
        if model_sha(model)!=sparse_hash:raise RuntimeError('sparse model mutated')
    validate_predictions(data,c);validate()
    metadata=dict(status='complete',method=method,correct=sum(r['correct'] for r in data),total=100,
        config_sha256=sha(ROOT/'config.json'),manifest_sha256=sha(dest/'mask_manifest.json'),protocol_hash=c['protocol_hash'],
        sparse_model_sha256=sparse_hash,metrics=metrics,reused_identical_masks=reused,
        paired_vs_role=paired_binary_comparison([r['correct'] for r in read_jsonl(c['baseline'])],[r['correct'] for r in data]))
    frozen(dest/'pending_eval.json',metadata);_write_jsonl(dest/'predictions.jsonl',data)
    metadata['predictions_sha256']=sha(dest/'predictions.jsonl');frozen(dest/'results.json',metadata)
    event(method,'complete',correct=metadata['correct'],total=100)


def finish():
    from experiments.dlm_role_bundle_mini100.core import holm
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    c=validate(); data={m:completed(m,c) for m in METHODS}
    if any(x is None for x in data.values()):raise RuntimeError('missing candidate results')
    values={m:read(ROOT/m/'results.json') for m in METHODS}
    adjustment=holm({m:v['paired_vs_role']['exact_mcnemar_p'] for m,v in values.items()})
    frozen(ROOT/'results.json',dict(status='complete',config_sha256=sha(ROOT/'config.json'),methods=values,holm_family2=adjustment,
        angular_vs_relative_descriptive=paired_binary_comparison([r['correct'] for r in data['token_relative']],[r['correct'] for r in data['token_angular']]),
        decision='development screen only; no automatic full/retuning'))
    event('pipeline','complete')


def child(phase,method=None,gpu=0):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)); name=method or phase
    args=[sys.executable,'-u','-m','experiments.dlm_role_token_geometry.run',phase]
    if method:args+=['--method',method]
    log=ROOT/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        return subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=REPO)


def pipeline():
    from experiments.dlm_dual_role_mini100.core import same_selected_masks
    validate()
    for phase in ('collect','prepare'):
        event('pipeline',phase)
        if child(phase).wait():raise RuntimeError(f'{phase} failed; pipeline stopped')
    identical=same_selected_masks(read(ROOT/METHODS[0]/'mask_manifest.json'),read(ROOT/METHODS[1]/'mask_manifest.json'))
    event('pipeline','evaluation')
    first=child('evaluate',METHODS[0],0)
    if identical:
        if first.wait():raise RuntimeError('first evaluation failed')
        if child('evaluate',METHODS[1],1).wait():raise RuntimeError('identical-mask reuse failed')
    else:
        second=child('evaluate',METHODS[1],1)
        codes=[first.wait(),second.wait()]
        if any(codes):raise RuntimeError(f'evaluation failed: {codes}')
    finish()


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','collect','prepare','evaluate','pipeline','finish']);p.add_argument('--method',choices=METHODS)
    args=p.parse_args();ROOT.mkdir(exist_ok=True)
    if args.phase=='freeze':freeze();return
    identity=args.method or args.phase
    with (ROOT/f'{identity}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.phase=='evaluate':
                if not args.method:p.error('--method required')
                evaluate(args.method)
            else:globals()[args.phase]()
        except BaseException as e:
            event(identity,'failed',error=f'{type(e).__name__}: {e}');raise


if __name__=='__main__':main()
