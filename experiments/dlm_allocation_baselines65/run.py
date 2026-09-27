"""Four pinned allocation transfers; reuse the audited OWL evaluation harness."""
import argparse
import fcntl
import importlib.util
import json
import math
import os
import time
from pathlib import Path
import torch

from experiments.dlm_allocation_baselines65 import core
from experiments.dlm_owl65 import run as parent
from experiments.dlm_owl65.core import exact_row_counts
from experiments.dlm_dual_role_allocation.io import file_sha256 as sha, atomic_write_json as write

ROOT=Path(__file__).resolve().parent
DATA=Path('/DATA/tmluser1/dlm_allocation_baselines65')
METHODS=('dlp','dsa','alpha','lsa')
UPSTREAM={
 'dlp':('sap-dlp-reference','31d480fd38fc64ce4a91a23412f115c9bd14d5c7',['lib/prune.py','run.py']),
 'dsa':('sap-dsa-reference','a5ebe65b468860c7884cde9a88b8ced606f5a04f',['lib/autolayer.py','lib/prune_all.py','main.py']),
 'alpha':('sap-alphapruning-reference','5f0e9845549ec8ee6dc395f1410566f26cc9e54e',['lib/esd_utils.py','lib/prune.py','main.py']),
 'lsa':('sap-lsa-reference','1c28cae299cf942b0eb2ef5ca582972506dc0d0c',['layersp/blk.py','pruner/wanda.py','main.py']),
}

def validate():
    c=parent.read(ROOT/'config.json')
    for p,h in c['sources'].items():
        if sha(p)!=h:raise RuntimeError(f'changed frozen source: {p}')
    return c

def freeze():
    if (ROOT/'config.json').exists():return validate()
    previous=parent.validate()
    sources=dict(previous['sources'])
    files=[Path(parent.__file__),ROOT/'core.py',ROOT/'run.py',ROOT/'run.sh',ROOT/'README.md',ROOT/'test_core.py',ROOT/'status.py']
    for folder,commit,paths in UPSTREAM.values():
        files.extend(DATA/folder/p for p in paths)
    sources.update({str(p):sha(p) for p in files})
    c=dict(model=previous['model'],target=.65,pruned=previous['pruned'],weights=previous['weights'],
        states=80,state_digest=previous['state_digest'],protocol_hash=previous['protocol_hash'],protocol=previous['protocol'],
        sources=sources,upstream=UPSTREAM,methods=list(METHODS),
        parameters=dict(dlp={'alpha':.15,'statistic':'get_dlp_ratios mean'},
            dsa={'graph':'W:(ABSLOG)-(VAR)-(ATAN,ASIN)-(7)','Lamda':.08,'search':False},
            alpha={'metric':'alpha_peak','mapping':'block_wise','epsilon':.3,'svd':'FP32 CUDA gesvd; exact full spectrum'},
            lsa={'layer':'lsa','group_size':128,'resp':.5,'alpha':.1}),
        adaptation='Block allocation transfer only; fixed dense 80-state DLM calibration and historical Standard Wanda ranking; no sparse-prefix recalibration.',
        budget_rule='OWL exact row-count DP; no 5% grid; same global count',
        primary='Historical fixed mini100 strict EM; paired vs Uniform and Role; no automatic full or tuning',
        invalid_rule='Undefined allocation fails candidate; no clipping, graph search, or hyperparameter rescue')
    parent.frozen(ROOT/'config.json',c)
    for method in METHODS:
        parent.frozen(ROOT/method/'config.json',{**c,'method':method+'_allocation_frozen_dlm_wanda'})
    return c

def harness(method):
    # Independent module namespace: existing historical module and files stay untouched.
    spec=importlib.util.spec_from_file_location('allocation_eval_'+method,parent.__file__)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.ROOT=ROOT/method
    h.build=lambda model,mapping,c:build(method,h,model,mapping,c)
    return h

@torch.inference_mode()
def statistics(method,h,model,mapping,c,ref,stats):
    path=h.ROOT/'statistics.json'
    ch=sha(h.ROOT/'config.json')
    if path.exists():
        saved=parent.read(path)
        if saved['config_sha256']!=ch:raise RuntimeError('statistics provenance changed')
        rows=saved['rows']
    else:rows=[]
    def save():write(path,dict(config_sha256=ch,rows=rows))
    def score(name):
        w=mapping[name].weight;a=stats[name]['overall_uniform'].float().to(w.device)
        if a.shape!=(w.shape[1],) or not torch.isfinite(a).all() or (a<0).any():raise RuntimeError('activation mismatch')
        return w.float().abs()*a.sqrt()[None,:]
    if method in ('dlp','dsa'):
        for b in range(len(rows),32):
            h.event('block_statistics',completed=b,total=32)
            pooled=torch.cat([score(r['name']).flatten().cpu() for r in ref[b*7:b*7+7]])
            value={'mean':float(pooled.mean().abs())} if method=='dlp' else core.dsa_score(pooled)
            rows.append(dict(block=b,**value));save();del pooled
        return core.dlp_rates([r['mean'] for r in rows]) if method=='dlp' else core.dsa_rates([r['value'] for r in rows])
    if method=='alpha':
        for i in range(len(rows),224):
            name=ref[i]['name'];h.event('spectra',completed=i,total=224,module=name)
            eigs=torch.linalg.svdvals(mapping[name].weight.float(),driver='gesvd').square().cpu()
            rows.append(dict(name=name,**core.alpha_from_eigs(eigs)));save();del eigs
        return core.alpha_rates([r['alpha'] for r in rows],[r['weights'] for r in ref])
    if len(rows)<224:
        from experiments.projection_capacity_allocation_65.verify_states import CAL
        states=parent.read(CAL)['states']
        if len(states)!=80:raise RuntimeError('LSA calibration count mismatch')
        H={};calls={};handles=[]
        for r in ref[len(rows):]:
            name=r['name'];w=mapping[name].weight
            H[name]=torch.zeros((w.shape[1],w.shape[1]),device=w.device,dtype=torch.float32);calls[name]=0
            def hook(mod,inp,out,key=name):
                x=inp[0]
                if x.ndim!=3 or x.shape[0]!=1:raise RuntimeError('LSA requires frozen batch-one path')
                n=calls[key];H[key]*=n/(n+1);calls[key]+=1
                a=x.reshape(-1,x.shape[-1]).T.float()*math.sqrt(2/(n+1))
                H[key].add_(a@a.T)
            handles.append(mapping[name].register_forward_hook(hook))
        try:
            dev=next(model.parameters()).device
            for i,state in enumerate(states):
                model(torch.tensor(state['noisy_ids'],device=dev))
                h.event('lsa_dense_calibration',completed=i+1,total=80)
        finally:
            for handle in handles:handle.remove()
        diag_checks=[]
        for i in range(len(rows),224):
            name=ref[i]['name'];h.event('lsa_metrics',completed=i,total=224,module=name)
            if calls[name]!=80:raise RuntimeError('missing LSA observations')
            expected=2*stats[name]['overall_uniform'].float().to(H[name].device)
            relative=float((H[name].diag()-expected).norm()/expected.norm())
            if not math.isfinite(relative) or relative>1e-4:raise RuntimeError(f'LSA activation semantics mismatch {name}: {relative}')
            value=float(core.lsa_metric(mapping[name].weight,H[name]))
            if not math.isfinite(value):raise RuntimeError('nonfinite LSA statistic')
            rows.append(dict(name=name,metric=value,activation_diagonal_relative_error=relative));save()
            del H[name]
    return core.lsa_rates([r['metric'] for r in rows])

@torch.inference_mode()
def build(method,h,model,mapping,c):
    from experiments.dlm_loss_aggregation.core import pack_mask,mask_sha256
    from experiments.projection_capacity_allocation_65.run import save_tensor,DENSE_SHA
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    if (h.ROOT/'build_receipt.json').exists():
        r=parent.read(h.ROOT/'build_receipt.json')
        if r['config_sha256']!=sha(h.ROOT/'config.json') or r['manifest_sha256']!=sha(h.ROOT/'mask_manifest.json'):raise RuntimeError('build provenance mismatch')
        m=parent.read(h.ROOT/'mask_manifest.json');h.verify_masks(m);return m
    ref=parent.read(parent.SOURCE/'candidate_mask_manifest.json')['entries']
    if list(mapping)!=[r['name'] for r in ref]:raise RuntimeError('target order mismatch')
    stats=torch.load(parent.STATS,map_location='cpu',weights_only=False)['statistics']
    rates=core.validate_rates(statistics(method,h,model,mapping,c,ref,stats))
    counts,budget=exact_row_counts(ref,rates,c['pruned'])
    parent.frozen(h.ROOT/'allocation.json',dict(ideal_block_sparsities=rates.tolist(),row_counts=counts,budget=budget,
        actual_global_sparsity=c['pruned']/c['weights']))
    entries=[]
    for i,r in enumerate(ref):
        w=mapping[r['name']].weight;a=stats[r['name']]['overall_uniform'].float().to(w.device)
        metric=w.float().abs()*a.sqrt()[None,:];order=torch.argsort(metric,dim=1,stable=True)
        mask=torch.zeros_like(metric,dtype=torch.bool).scatter_(1,order[:,:int(w.shape[1]*.65)],True)
        if mask_sha256(pack_mask(mask.cpu()))!=r['masks'][3]['mask_sha256']:raise RuntimeError('historical Uniform ranking mismatch: '+r['name'])
        mask.zero_().scatter_(1,order[:,:counts[i]],True);packed=pack_mask(mask.cpu())
        path=DATA/method/'masks'/f'{r["name"]}.pt'
        if path.exists():
            if mask_sha256(torch.load(path,map_location='cpu',weights_only=False))!=mask_sha256(packed):raise RuntimeError('partial mask mismatch')
        else:save_tensor(path,packed)
        selected=dict(path=str(path),file_sha256=sha(path),mask_sha256=mask_sha256(packed),pruned=counts[i]*w.shape[0],prune_per_row=counts[i])
        entries.append(dict(module_index=i,name=r['name'],shape=r['shape'],weights=r['weights'],assigned_sparsity=counts[i]/w.shape[1],ideal_sparsity=float(rates[i//7]),selected_mask=selected))
        h.event('masks',completed=i+1,total=224);del metric,order,mask,packed
    manifest=dict(method=c['method'],config_sha256=sha(h.ROOT/'config.json'),entries=entries,
        pruned=sum(e['selected_mask']['pruned'] for e in entries),weights=sum(e['weights'] for e in entries))
    if model_sha(model)!=DENSE_SHA:raise RuntimeError('scoring mutated weights')
    parent.frozen(h.ROOT/'mask_manifest.json',manifest);h.verify_masks(manifest);h.validate()
    parent.frozen(h.ROOT/'build_receipt.json',dict(config_sha256=sha(h.ROOT/'config.json'),manifest_sha256=sha(h.ROOT/'mask_manifest.json'),
        allocation_sha256=sha(h.ROOT/'allocation.json'),all_224_uniform_masks_reproduced=True))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=['freeze',*METHODS]);a=p.parse_args()
    if a.method=='freeze':freeze()
    else:
        h=harness(a.method)
        with (h.ROOT/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:validate();h.run()
            except BaseException as e:h.event('failed',error=f'{type(e).__name__}: {e}');raise
