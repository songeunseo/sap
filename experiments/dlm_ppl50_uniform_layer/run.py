"""Projection-agnostic layer-global Wanda50 with frozen DLM likelihood evaluation."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.dlm_ppl50 import sequential as seq
from experiments.dlm_wikitext_ppl import run as protocol
from experiments.dlm_wikitext_ppl import evaluate as evaluator
from experiments.dlm_wikitext_ppl.core import digest, summarize
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write, file_sha256 as sha

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
STORE=Path('/DATA/tmluser1/dlm-ppl50-uniform-layer')
TARGET=3_489_660_928
METHOD='uniform_layer_global'


def read(path):return json.loads(Path(path).read_text())

def event(stage,**kw):
    row=dict(method=METHOD,stage=stage,time=time.time(),pid=os.getpid(),**kw)
    write(ROOT/'progress.json',row);print(json.dumps(row),flush=True)


def exact_global_masks(scores):
    """Bottom half over one flattened layer; deterministic tie order is module/row/column."""
    if not scores or any(s.ndim!=2 or not torch.isfinite(s).all() for s in scores):
        raise RuntimeError('finite score matrices required')
    sizes=[s.numel() for s in scores];total=sum(sizes)
    if total%2:raise RuntimeError('layer weight count must be even')
    pooled=torch.cat([s.reshape(-1) for s in scores]);k=total//2
    threshold=torch.kthvalue(pooled,k).values
    flat=pooled<threshold;remaining=k-int(flat.sum())
    if remaining:
        ties=torch.nonzero(pooled==threshold,as_tuple=False).flatten()
        if remaining>len(ties):raise RuntimeError('threshold tie accounting failed')
        flat[ties[:remaining]]=True
    if int(flat.sum())!=k:raise RuntimeError('layer exact budget failed')
    masks=[];start=0
    for score,size in zip(scores,sizes):
        masks.append(flat[start:start+size].reshape(score.shape));start+=size
    return masks,float(threshold),k


def validate():
    cfg=read(ROOT/'config.json')
    for path,h in cfg['sources'].items():
        if sha(path)!=h:raise RuntimeError('frozen source changed: '+path)
    protocol.validate();return cfg


def freeze():
    if (ROOT/'config.json').exists():return validate()
    parent=read(REPO/'experiments/dlm_ppl50/config.json')
    refs=read(seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
    if sum(r['weights'] for r in refs)!=TARGET*2:raise RuntimeError('wrong scope')
    for b in range(32):
        if sum(r['weights'] for r in refs[b*7:(b+1)*7])%2:raise RuntimeError('odd layer budget')
    paths=[Path(__file__),ROOT/'test_run.py',REPO/'experiments/dlm_ppl50/sequential.py',
           REPO/'experiments/dlm_ppl50/config.json',seq.CAL,seq.old.STATS,
           seq.old.SOURCE/'candidate_mask_manifest.json',protocol.ROOT/'config.json',
           protocol.ROOT/'corpus_manifest.json',Path(evaluator.__file__)]
    cfg=dict(model=parent['model'],method=METHOD,target=.5,pruned=TARGET,weights=TARGET*2,
        hypothesis='Layer-global Wanda may differ from projection-wise row-quota Wanda; no direction assumed',
        calibration='same frozen80 corrupted states; native batch1 sparse-prefix replay',
        mask_rule='For each transformer block, flatten Wanda |W|*sqrt(input energy) over all seven prunable matrices; prune exact bottom50%; projection and row boundaries impose no quota',
        tie_rule='stable flattened order: frozen module order then row-major; only used at exact threshold ties',
        comparison='existing projection-wise rowwise Uniform-Wanda50 on identical validation protocol',
        evaluation='frozen filtered WikiText2 validation551 chunks, token NELBO/exp, exact-k MC128 seed2025 shared draws',
        test_policy='validation development only; final test and GSM8K untouched',
        sources={str(p):sha(p) for p in paths})
    protocol.frozen(ROOT/'config.json',cfg);return cfg


@torch.inference_mode()
def build(model,mapping,refs,states,cfg):
    from experiments.dlm_loss_aggregation.core import pack_mask,mask_sha256
    from experiments.projection_capacity_allocation_65.run import save_tensor
    from experiments.projection_capacity_followup_65.run_heldout import selected_mask
    previous=torch.load(seq.old.STATS,map_location='cpu',weights_only=False)['statistics']
    entries=[]
    for block in range(32):
        checkpoint=ROOT/'blocks'/f'block_{block:02d}.json'
        blockrefs=refs[block*7:(block+1)*7]
        if checkpoint.exists():
            rows=read(checkpoint)
            for row,ref in zip(rows,blockrefs):
                if row['name']!=ref['name']:raise RuntimeError('resume identity mismatch')
                mask=selected_mask(row,mapping[ref['name']].weight.device)
                mapping[ref['name']].weight.masked_fill_(mask,0)
            entries.extend(rows);continue
        event('calibration',block=block,completed=0,total=len(states))
        activation=seq.collect_block(model,mapping,refs,states,block,
            lambda i:event('calibration',block=block,completed=i,total=len(states)))
        scores=[];errors=[]
        for ref in blockrefs:
            name=ref['name'];w=mapping[name].weight;a=activation[name]
            dense=previous[name]['overall_uniform'].float().to(a.device)
            error=float((a-dense).norm()/dense.norm());errors.append(error)
            if block==0 and error>1e-4:raise RuntimeError(f'dense calibration control failed: {name}: {error}')
            scores.append(w.float().abs()*a.sqrt()[None,:])
        masks,threshold,count=exact_global_masks(scores)
        if count!=sum(r['weights'] for r in blockrefs)//2:raise RuntimeError('block count mismatch')
        rows=[]
        for index,(ref,mask,error) in enumerate(zip(blockrefs,masks,errors),start=block*7):
            name=ref['name'];packed=pack_mask(mask.cpu());path=STORE/'masks'/f'{name}.pt'
            if path.exists():
                if mask_sha256(torch.load(path,map_location='cpu',weights_only=False))!=mask_sha256(packed):
                    raise RuntimeError('partial mask differs')
            else:save_tensor(path,packed)
            pruned=int(mask.sum())
            row=dict(name=name,shape=ref['shape'],weights=ref['weights'],module_index=index,
                assigned_sparsity=pruned/ref['weights'],dense_activation_relative_change=error,
                layer_global_threshold=threshold,selected_mask=dict(path=str(path),file_sha256=sha(path),
                    mask_sha256=mask_sha256(packed),pruned=pruned))
            mapping[name].weight.masked_fill_(mask,0);rows.append(row)
        if sum(r['selected_mask']['pruned'] for r in rows)!=count:raise RuntimeError('stored block count mismatch')
        write(checkpoint,rows);entries.extend(rows);event('pruned_block',completed=block+1,total=32)
    if sum(r['selected_mask']['pruned'] for r in entries)!=TARGET:raise RuntimeError('global budget mismatch')
    return dict(method=METHOD,config_sha256=sha(ROOT/'config.json'),entries=entries,pruned=TARGET,weights=TARGET*2)


@torch.inference_mode()
def evaluate(model,manifest,cfg):
    folder=ROOT/'validation';folder.mkdir(parents=True,exist_ok=True)
    pcfg=protocol.validate();run_cfg=dict(method=METHOD,split='validation',config_sha256=sha(ROOT/'config.json'),
        mask_manifest_sha256=sha(ROOT/'mask_manifest.json'),protocol_config_sha256=sha(protocol.ROOT/'config.json'),mc_samples=128)
    protocol.frozen(folder/'config.json',run_cfg);ch=sha(folder/'config.json')
    if (folder/'results.json').exists():return read(folder/'results.json')
    corpus=read(protocol.ROOT/'corpus_manifest.json')['splits']['validation'];rows=[];started=time.monotonic();fresh=0
    for source in corpus:
        path=folder/'blocks'/f"{digest(source['block_id'])[:20]}.json"
        if path.exists():row=read(path)
        else:
            def tick(current):
                elapsed=time.monotonic()-started;done=len(rows)*128+current
                event('evaluating_validation',blocks_completed=len(rows),blocks_total=len(corpus),completed=done,total=len(corpus)*128,
                      remaining_seconds=(len(corpus)*128-done)*elapsed/(fresh+current))
            row=evaluator.seal(dict(evaluator.evaluate_block(model,source,pcfg,tick),config_sha256=ch))
            evaluator.verify_row(row,source,pcfg,ch);protocol.frozen(path,row);fresh+=128
        evaluator.verify_row(row,source,pcfg,ch);rows.append(row)
    result=dict(status='complete',method=METHOD,split='validation',summary=summarize(rows),config_sha256=ch,
        mask_manifest_sha256=run_cfg['mask_manifest_sha256'],evaluation_seconds=sum(r['evaluation_seconds'] for r in rows),
        block_checkpoint_digests={r['block_id']:r['row_sha256'] for r in rows})
    validate();protocol.frozen(folder/'results.json',result);return result


@torch.inference_mode()
def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=validate()
        try:
            from experiments.projection_capacity_allocation_65.run import load_dense
            from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
            event('loading_dense');model,mapping=load_dense();refs=read(seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
            if list(mapping)!=[r['name'] for r in refs]:raise RuntimeError('projection order mismatch')
            mp=ROOT/'mask_manifest.json'
            if mp.exists():manifest=read(mp);apply_manifest(model,mapping,manifest)
            else:manifest=build(model,mapping,refs,read(seq.CAL)['states'],cfg);protocol.frozen(mp,manifest)
            result=evaluate(model,manifest,cfg);event('complete',summary=result['summary'])
        except BaseException as exc:event('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':
    freeze();run()
