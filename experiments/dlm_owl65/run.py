"""Frozen OWL allocation transfer and a single historical GSM8K mini100 run."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import torch

from experiments.dlm_dual_role_allocation.io import file_sha256 as sha, atomic_write_json as write
from experiments.dlm_role_token_geometry.run import read, read_jsonl
from experiments.dlm_owl65.core import owl_sparsities, exact_row_counts

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
SOURCE=REPO/'experiments/projection_capacity_allocation_65'
OLD=REPO/'experiments/dlm_dual_role_mini100'
STATS=REPO/'experiments/cgq_wanda_structured_diagnostic/sufficient_statistics.pt'
EVAL=REPO/'experiments/dlm_loss_aggregation/exp002/config.yaml'
BASELINES={'uniform':SOURCE/'gsm8k/uniform_100_predictions.jsonl',
           'role':OLD/'gsm8k/role_100_predictions.jsonl'}


def frozen(path,value):
    if path.exists():
        if read(path)!=value: raise RuntimeError(f'frozen artifact mismatch: {path}')
    else: write(path,value)


def event(stage,**kw):
    value=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw)
    write(ROOT/'progress.json',value);print(json.dumps(value),flush=True)


def validate():
    c=read(ROOT/'config.json')
    for p,h in c['sources'].items():
        if sha(p)!=h:raise RuntimeError(f'changed source: {p}')
    return c


def freeze():
    from experiments.dlm_role_token_geometry.run import validate as validate_previous
    from experiments.dlm_loss_aggregation.exp002.run import load_config,_evaluation_config_hash
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    if (ROOT/'config.json').exists():return validate()
    prior=validate_previous()
    parent=read(SOURCE/'config.json')
    if parent['sources'][str(STATS.relative_to(REPO))]!=sha(STATS):
        raise RuntimeError('historical activation hash mismatch')
    rows={k:read_jsonl(p) for k,p in BASELINES.items()}
    ph,protocol=_evaluation_config_hash(load_config(EVAL))
    _validate_rows(rows['uniform'],rows['role'],ph)
    if [sum(r['correct'] for r in rows[k]) for k in ('uniform','role')]!=[12,24]:
        raise RuntimeError('baseline accuracy mismatch')
    sources=dict(prior['sources'])
    files=[STATS,*BASELINES.values(),SOURCE/'config.json',SOURCE/'uniform65_mask_manifest.json',
        SOURCE/'candidate_mask_manifest.json',REPO/'experiments/dlm_role_token_geometry/run.py',
        REPO/'experiments/dlm_dual_role_mini100/run.py',
        REPO/'experiments/wanda_failure_characterization/run_failure_map.py',
        REPO/'experiments/dlm_loss_aggregation/core.py',*ROOT.glob('*.py'),ROOT/'README.md',ROOT/'run.sh']
    sources.update({str(p):sha(p) for p in files})
    c=dict(method='owl_allocation_frozen_dlm_wanda',model=prior['model'],M=5,lam=.08,target=.65,
        pruned=4536008704,weights=6979321856,states=80,state_digest=prior['state_digest'],
        protocol_hash=ph,protocol=protocol,sources=sources,
        upstream_commit='dddb7a4bffe27c73e4c8cf692b3a5e36401532c8',
        upstream_prune_all_sha256='3803aee75bfc15de92b7d79d643c6014fdd07898c9e0fc51d32b6a6c76d59a55',
        hyperparameters='official README unstructured example; no sweep',
        adaptation='OWL block allocation only; historical dense DLM calibration/ranking; exact integer budget',
        primary='paired mini100 vs Uniform; Role comparison descriptive; no automatic full')
    frozen(ROOT/'config.json',c);return c


def verify_masks(manifest):
    from experiments.projection_capacity_followup_65.run_heldout import selected_mask
    total=0
    expected=read(SOURCE/'candidate_mask_manifest.json')['entries']
    if len(manifest['entries'])!=224:raise RuntimeError('wrong target count')
    for e,ref in zip(manifest['entries'],expected):
        if e['name']!=ref['name'] or e['shape']!=ref['shape']:raise RuntimeError('target ordering mismatch')
        m=selected_mask(e,'cpu');k=e['selected_mask']['prune_per_row']
        if list(m.shape)!=e['shape'] or not (m.sum(1)==k).all():raise RuntimeError('row count mismatch')
        count=int(m.sum());total+=count
        if count!=e['selected_mask']['pruned']:raise RuntimeError('payload count mismatch')
    if total!=manifest['pruned'] or total!=4536008704:raise RuntimeError('budget mismatch')


@torch.inference_mode()
def build(model,mapping,c):
    from experiments.dlm_loss_aggregation.core import pack_mask,mask_sha256
    from experiments.projection_capacity_allocation_65.run import save_tensor,DENSE_SHA
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    if (ROOT/'build_receipt.json').exists():
        r=read(ROOT/'build_receipt.json')
        if r['config_sha256']!=sha(ROOT/'config.json') or r['manifest_sha256']!=sha(ROOT/'mask_manifest.json'):
            raise RuntimeError('build provenance mismatch')
        manifest=read(ROOT/'mask_manifest.json');verify_masks(manifest);return manifest
    ref=read(SOURCE/'candidate_mask_manifest.json')['entries']
    if list(mapping)!=[r['name'] for r in ref]:raise RuntimeError('module order mismatch')
    stats=torch.load(STATS,map_location='cpu',weights_only=False)['statistics']
    ratios=[];blocks=[]
    def score(name):
        w=mapping[name].weight
        a=stats[name]['overall_uniform'].float().to(w.device)
        if a.shape!=(w.shape[1],) or not torch.isfinite(a).all() or (a<0).any():raise RuntimeError('invalid activation')
        return w.float().abs()*a.sqrt()[None,:]
    for b in range(32):
        event('outlier_statistics',completed=b,total=32)
        # Exact upstream pooled block threshold: NOT the average of per-module ratios.
        pooled=torch.cat([score(r['name']).flatten().cpu() for r in ref[b*7:(b+1)*7]])
        threshold=pooled.mean()*c['M']; n=int((pooled>threshold).sum())
        ratio=n/pooled.numel()*100;ratios.append(ratio)
        blocks.append(dict(block=b,weights=pooled.numel(),outlier_count=n,outlier_percent=ratio,threshold=float(threshold)))
        del pooled
    sparsities=owl_sparsities(ratios,c['target'],c['lam'])
    counts,budget=exact_row_counts(ref,sparsities,c['pruned'])
    block_weights=[sum(r['weights'] for r in ref[b*7:(b+1)*7]) for b in range(32)]
    if len(set(block_weights))!=1:raise RuntimeError('equal block weight assumption violated')
    frozen(ROOT/'allocation.json',dict(blocks=blocks,ideal_block_sparsities=sparsities.tolist(),
        row_counts=counts,budget=budget,actual_global_sparsity=c['pruned']/c['weights']))
    entries=[]
    for i,r in enumerate(ref):
        metric=score(r['name']);order=torch.argsort(metric,dim=1,stable=True)
        mask=torch.zeros_like(metric,dtype=torch.bool).scatter_(1,order[:,:int(metric.shape[1]*.65)],True)
        if mask_sha256(pack_mask(mask.cpu()))!=r['masks'][3]['mask_sha256']:
            raise RuntimeError(f'historical Uniform65 ranking mismatch: {r["name"]}')
        mask.zero_().scatter_(1,order[:,:counts[i]],True)
        packed=pack_mask(mask.cpu());path=ROOT/'masks'/f'{r["name"]}.pt'
        if path.exists():
            old=torch.load(path,map_location='cpu',weights_only=False)
            if mask_sha256(old)!=mask_sha256(packed):raise RuntimeError('partial mask differs')
        else:save_tensor(path,packed)
        selected=dict(path=str(path),file_sha256=sha(path),mask_sha256=mask_sha256(packed),
            pruned=counts[i]*r['shape'][0],prune_per_row=counts[i])
        entries.append(dict(module_index=i,name=r['name'],shape=r['shape'],weights=r['weights'],
            assigned_sparsity=counts[i]/r['shape'][1],ideal_sparsity=float(sparsities[i//7]),selected_mask=selected))
        event('masks',completed=i+1,total=224)
        del metric,order,mask,packed
    manifest=dict(method=c['method'],config_sha256=sha(ROOT/'config.json'),entries=entries,
        pruned=sum(e['selected_mask']['pruned'] for e in entries),weights=sum(e['weights'] for e in entries))
    if model_sha(model)!=DENSE_SHA:raise RuntimeError('dense model changed during scoring')
    frozen(ROOT/'mask_manifest.json',manifest);verify_masks(manifest);validate()
    frozen(ROOT/'build_receipt.json',dict(config_sha256=sha(ROOT/'config.json'),manifest_sha256=sha(ROOT/'mask_manifest.json'),
        allocation_sha256=sha(ROOT/'allocation.json'),all_224_uniform_masks_reproduced=True))
    return manifest


def completed(c):
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    if not (ROOT/'results.json').exists():
        if (ROOT/'predictions.jsonl').exists():raise RuntimeError('partial predictions: preserve and investigate')
        return False
    r=read(ROOT/'results.json')
    if r['config_sha256']!=sha(ROOT/'config.json') or r['predictions_sha256']!=sha(ROOT/'predictions.jsonl') or r['manifest_sha256']!=sha(ROOT/'mask_manifest.json'):
        raise RuntimeError('result hashes changed')
    rows=read_jsonl(ROOT/'predictions.jsonl');_validate_rows(rows,read_jsonl(BASELINES['uniform']),c['protocol_hash'])
    if len(rows)!=100 or sum(x['correct'] for x in rows)!=r['correct']:raise RuntimeError('result count mismatch')
    return True


@torch.inference_mode()
def run():
    from transformers import AutoTokenizer
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    from experiments.dlm_loss_aggregation.exp002.run import load_config,_evaluate_gsm8k,_evaluation_config_hash,_write_jsonl
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    c=validate()
    if completed(c):event('complete',correct=read(ROOT/'results.json')['correct']);return
    event('loading_dense');model,mapping=load_dense();manifest=build(model,mapping,c)
    event('apply_masks')
    if apply_manifest(model,mapping,manifest)!=c['pruned']:raise RuntimeError('applied count mismatch')
    sh=model_sha(model);cfg=load_config(EVAL);ph,_=_evaluation_config_hash(cfg)
    if ph!=c['protocol_hash']:raise RuntimeError('protocol mismatch')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model']['id'],revision=cfg['model']['revision'],trust_remote_code=True)
    event('gsm8k',completed=0,total=100)
    metrics,rows=_evaluate_gsm8k(model,tokenizer,cfg,c['method'],100,ph)
    if model_sha(model)!=sh:raise RuntimeError('sparse weights changed')
    _validate_rows(rows,read_jsonl(BASELINES['uniform']),ph);validate()
    result=dict(status='complete',correct=sum(r['correct'] for r in rows),total=100,metrics=metrics,
        config_sha256=sha(ROOT/'config.json'),manifest_sha256=sha(ROOT/'mask_manifest.json'),
        sparse_model_sha256=sh,protocol_hash=ph,pruned=c['pruned'],
        comparisons={k:paired_binary_comparison([x['correct'] for x in read_jsonl(p)],[x['correct'] for x in rows]) for k,p in BASELINES.items()})
    frozen(ROOT/'pending_eval.json',result);_write_jsonl(ROOT/'predictions.jsonl',rows)
    result['predictions_sha256']=sha(ROOT/'predictions.jsonl');frozen(ROOT/'results.json',result)
    event('complete',correct=result['correct'],total=100)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);a=p.parse_args()
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:freeze() if a.phase=='freeze' else run()
        except BaseException as e:event('failed',error=f'{type(e).__name__}: {e}');raise
