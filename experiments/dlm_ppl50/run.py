"""Frozen 50% DLM pruning comparison on the agreed WikiText likelihood protocol."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.dlm_ppl50 import sequential as seq
from experiments.dlm_allocation_baselines65 import core as formulas
from experiments.dlm_allocation_baselines65 import run as baseline
from experiments.dlm_owl65.core import owl_sparsities, exact_row_counts
from experiments.dlm_lsa_projection65.core import projection_rates, exact_projection_rows
from experiments.dlm_wikitext_ppl import run as protocol
from experiments.dlm_wikitext_ppl import evaluate as evaluator
from experiments.dlm_wikitext_ppl.core import digest, summarize

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
METHODS=('uniform','owl','dlp','alpha','lsa_layer','lsa_projection','dsa','evopress')
TARGET=3489660928
read,write,sha=seq.read,seq.write,seq.sha


def event(method,stage,**kw):
    value=dict(method=method,stage=stage,time=time.time(),pid=os.getpid(),**kw)
    write(ROOT/method/'progress.json',value)
    print(value,flush=True)


def layer_lsa_rates(metrics):
    # BLK.alpha[int(.5*10)] == .04 in the pinned upstream, unlike .1 at 65%.
    v=torch.tensor(metrics,dtype=torch.float32).reshape(32,7).mean(1).abs()
    z=1-v/v.sum();z=(z-z.min())/(z.max()-z.min())*.04*2
    return (.5+z.mean()-z).double().numpy()


def validate():
    cfg=read(ROOT/'config.json')
    for path,value in cfg['sources'].items():
        if sha(path)!=value:raise RuntimeError('frozen source changed: '+path)
    protocol.validate()
    return cfg


def freeze():
    if (ROOT/'config.json').exists():return validate()
    parent=seq.old.validate(); baseline.validate(); pcfg=protocol.validate()
    refs=read(seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
    if sum(r['weights'] for r in refs)!=TARGET*2:raise RuntimeError('projection budget mismatch')
    paths=[seq.CAL,seq.DEV,seq.HELD,seq.VER,seq.old.STATS,
           seq.old.SOURCE/'candidate_mask_manifest.json',protocol.ROOT/'config.json',
           protocol.ROOT/'corpus_manifest.json',Path(evaluator.__file__),
           Path(formulas.__file__),REPO/'experiments/dlm_owl65/core.py',
           REPO/'experiments/dlm_lsa_projection65/core.py']
    alloc={}
    alloc['uniform']=dict(row_counts=[r['shape'][1]//2 for r in refs])
    paths.append(seq.old.ROOT/'allocation.json')
    owl=read(paths[-1]); rates=owl_sparsities([r['outlier_percent'] for r in owl['blocks']],target=.5)
    counts,budget=exact_row_counts(refs,rates,TARGET)
    alloc['owl']=dict(row_counts=counts,ideal_block_sparsities=rates.tolist(),budget=budget)
    stats={}
    for name in ('dlp','alpha','lsa'):
        path=baseline.ROOT/name/'statistics.json'; paths.extend([path,baseline.ROOT/name/'config.json'])
        saved=read(path)
        if saved['config_sha256']!=sha(baseline.ROOT/name/'config.json'):
            raise RuntimeError('statistics provenance mismatch: '+name)
        stats[name]=saved['rows']
    for method,rates in (
        ('dlp',formulas.dlp_rates([r['mean'] for r in stats['dlp']],target=.5)),
        ('alpha',formulas.alpha_rates([r['alpha'] for r in stats['alpha']],[r['weights'] for r in refs],target=.5)),
        ('lsa_layer',layer_lsa_rates([r['metric'] for r in stats['lsa']]))):
        formulas.validate_rates(rates)
        counts,budget=exact_row_counts(refs,rates,TARGET)
        alloc[method]=dict(row_counts=counts,ideal_block_sparsities=rates.tolist(),budget=budget)
    rates=projection_rates([r['metric'] for r in stats['lsa']],[r['weights'] for r in refs],target=.5)
    counts,budget=exact_projection_rows(refs,rates,TARGET)
    alloc['lsa_projection']=dict(row_counts=counts,ideal_projection_sparsities=rates.tolist(),budget=budget)
    ver=read(seq.VER)
    if not ver['all_three_splits_disjoint'] or ver['manifest_sha256']!=sha(seq.DEV):
        raise RuntimeError('development verification failed')
    for path,h in ver['prior_manifest_sha256'].items():
        if sha(REPO/path)!=h:raise RuntimeError('state source changed')
    paths.extend(ROOT.glob('*.py'));paths.extend([ROOT/'run.sh',ROOT/'README.md'])
    paths.extend(p for p in (ROOT/'upstream').rglob('*') if p.is_file())
    paths.extend(baseline.DATA/folder/p for folder,commit,files in baseline.UPSTREAM.values() for p in files)
    cfg=dict(model=parent['model'],target=.5,pruned=TARGET,weights=TARGET*2,
        methods=list(METHODS),allocations=alloc,split='validation',phase='development',
        calibration='same frozen80 corrupted WikiText train states; native batch1 sparse-prefix block-wise integration',
        evaluation={k:v for k,v in pcfg.items() if k not in ('sources','pruning','pilot')},
        allocation_statistics='Provenance-verified dense surveys reused; no new outcome-dependent coefficient tuning',
        parameters=dict(owl_lam=.08,dlp_alpha=.15,alpha_epsilon=.3,lsa_layer_lam=.04,lsa_projection_lam=.07,lsa_probe=.5),
        dsa=dict(seed=0,population=8,generations=4,elites=2,parents=4,mutation_probability=.5,lam=.08,
                 fitness='existing disjoint development40 masked gold CE; custom bounded public-operator search, not official full controller'),
        evopress=dict(seed=0,generations=400,offspring=64,survivors=[8,2,1],selection_tokens=[2048,8192,10240],
                      num_levels=8,weights_diff=524288,rel_damp=.01,block_size=128,
                      fitness='same-position masked-token Dense||Sparse KL on frozen development40; original equal-weight-count level transfers',
                      adaptation='Native LLaDA sparse-prefix full replay; original FastOBC database and elitist mutations; no AR loss shifting',
                      budget='nominal50%; original FastOBC block thresholds and dead columns may cause actual zero count deviations; report actual'),
        test_policy='Final test held untouched; no GSM8K',sources={str(p):sha(p) for p in paths})
    protocol.frozen(ROOT/'config.json',cfg)
    return cfg


@torch.inference_mode()
def score(model,method,cfg,manifest_path):
    folder=ROOT/method/'validation';folder.mkdir(parents=True,exist_ok=True)
    pcfg=protocol.validate()
    run_cfg=dict(method=method,split='validation',config_sha256=sha(ROOT/'config.json'),
                 mask_manifest_sha256=sha(manifest_path),protocol_config_sha256=sha(protocol.ROOT/'config.json'),
                 mc_samples=pcfg['mc_samples'])
    protocol.frozen(folder/'config.json',run_cfg); ch=sha(folder/'config.json')
    if (folder/'results.json').exists():
        result=read(folder/'results.json')
        if result['config_sha256']!=ch:raise RuntimeError('result mismatch')
        return result
    corpus=read(protocol.ROOT/'corpus_manifest.json')['splits']['validation']
    rows=[];started=time.monotonic();fresh=0
    for source in corpus:
        path=folder/'blocks'/f"{digest(source['block_id'])[:20]}.json"
        if path.exists():row=read(path)
        else:
            def tick(current):
                elapsed=time.monotonic()-started
                event(method,'evaluating_validation',blocks_completed=len(rows),blocks_total=len(corpus),
                      completed=len(rows)*128+current,total=len(corpus)*128,
                      remaining_seconds=(len(corpus)*128-len(rows)*128-current)*elapsed/(fresh+current))
            row=evaluator.seal(dict(evaluator.evaluate_block(model,source,pcfg,tick),config_sha256=ch))
            evaluator.verify_row(row,source,pcfg,ch);protocol.frozen(path,row);fresh+=128
        evaluator.verify_row(row,source,pcfg,ch);rows.append(row)
    if len({r['block_id'] for r in rows})!=len(corpus):raise RuntimeError('duplicate coverage')
    validate()
    result=dict(status='complete',method=method,split='validation',summary=summarize(rows),
        config_sha256=ch,mask_manifest_sha256=run_cfg['mask_manifest_sha256'],
        evaluation_seconds=sum(r['evaluation_seconds'] for r in rows),
        block_checkpoint_digests={r['block_id']:r['row_sha256'] for r in rows})
    protocol.frozen(folder/'results.json',result)
    return result


@torch.inference_mode()
def run(method):
    folder=ROOT/method;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=validate()
        try:
            if (folder/'validation/results.json').exists():
                event(method,'complete',summary=read(folder/'validation/results.json')['summary']);return
            from experiments.projection_capacity_allocation_65.run import load_dense
            from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
            event(method,'loading_dense');model,mapping=load_dense()
            refs=read(seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
            if list(mapping)!=[r['name'] for r in refs]:raise RuntimeError('projection order mismatch')
            states=read(seq.CAL)['states'];seq.ROOT=ROOT;seq.event=event
            mp=folder/'mask_manifest.json'
            if mp.exists():
                manifest=read(mp)
                if manifest['config_sha256']!=sha(ROOT/'config.json'):raise RuntimeError('manifest config mismatch')
                if method=='evopress':
                    from experiments.dlm_ppl50.evo import apply_result
                    apply_result(model,mapping,manifest)
                else:apply_manifest(model,mapping,manifest)
            elif method=='evopress':
                from experiments.dlm_ppl50.evo import build
                manifest=build(model,mapping,refs,states,cfg,event)
                protocol.frozen(mp,manifest)
            else:
                counts=seq.search(model,mapping,refs,states,cfg) if method=='dsa' else cfg['allocations'][method]['row_counts']
                write(folder/'allocation.json',dict(row_counts=counts,pruned=TARGET))
                rows=seq.sequential(model,mapping,refs,states,counts,method,folder)
                manifest=dict(method=method+'_sequential_wanda50',config_sha256=sha(ROOT/'config.json'),
                              entries=rows,pruned=TARGET,weights=TARGET*2)
                protocol.frozen(mp,manifest);verify_masks(manifest)
            result=score(model,method,cfg,mp)
            event(method,'complete',summary=result['summary'])
        except BaseException as exc:
            event(method,'failed',error=f'{type(exc).__name__}: {exc}');raise


def verify_masks(manifest):
    from experiments.projection_capacity_followup_65.run_heldout import selected_mask
    refs=read(seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
    if len(manifest['entries'])!=224:raise RuntimeError('wrong matrix count')
    total=0
    for row,ref in zip(manifest['entries'],refs):
        if row['name']!=ref['name'] or row['shape']!=ref['shape']:raise RuntimeError('projection identity mismatch')
        mask=selected_mask(row,'cpu');k=row['selected_mask']['prune_per_row']
        if list(mask.shape)!=ref['shape'] or not (mask.sum(1)==k).all():raise RuntimeError('mask row budget mismatch')
        count=int(mask.sum());total+=count
        if count!=row['selected_mask']['pruned']:raise RuntimeError('mask count mismatch')
    if total!=TARGET or manifest['pruned']!=TARGET:raise RuntimeError('wrong50 budget')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);p.add_argument('--method',choices=METHODS)
    a=p.parse_args();freeze() if a.phase=='freeze' else run(a.method)
