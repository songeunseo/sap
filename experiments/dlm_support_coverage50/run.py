"""Frozen state-support coverage allocation versus pooled-energy control."""
import argparse, fcntl, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from experiments.dlm_ppl50 import sequential as seq
from experiments.dlm_owl65 import run as old
from experiments.dlm_lsa_projection65.core import exact_projection_rows
from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
from experiments.projection_capacity_allocation_65.run import load_dense, save_tensor
from experiments.projection_capacity_followup_65.run_heldout import selected_mask
from experiments.wanda_failure_characterization.run_failure_map import model_sha
from experiments.dlm_support_coverage50.core import support_metrics, budget_rates
from experiments.dlm_wikitext_ppl import evaluate as ppl

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
PRIOR=REPO/'experiments/dlm_context_response50'
SURVEY=REPO/'experiments/dlm_scale_shape50'
BASE=REPO/'experiments/dlm_ppl50/uniform/mask_manifest.json'
IDENTITY=REPO/'experiments/dlm_allocation_sequential65/uniform/predictions.jsonl'
PRUNED=3489660928
MODES=('pooled','coverage')
read,write,sha=seq.read,seq.write,seq.sha


def event(stage,**kw):
    row=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw)
    write(ROOT/'progress.json',row);print(json.dumps(row),flush=True)


def frozen(path,obj):
    if path.exists():
        if read(path)!=obj: raise RuntimeError('Frozen artifact mismatch: '+str(path))
    else: write(path,obj)


def validate():
    c=read(ROOT/'config.json')
    for p,h in c['sources'].items():
        if sha(p)!=h: raise RuntimeError('Source changed: '+p)
    return c


def freeze():
    from experiments.dlm_loss_aggregation.exp002.run import load_config,_evaluation_config_hash
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    if (ROOT/'config.json').exists(): return validate()
    parent=read(PRIOR/'config.json'); receipt=read(PRIOR/'collection_receipt.json')
    assert receipt['config_sha256']==sha(PRIOR/'config.json')
    survey=read(SURVEY/'config.json'); input_receipt=read(SURVEY/'input_collection.json')
    assert input_receipt['config_sha256']==sha(SURVEY/'config.json')
    assert sha(input_receipt['path'])==input_receipt['sha256']
    assert survey['sources'][str(seq.CAL)]==sha(seq.CAL)
    for b in range(32):
        p=PRIOR/'activations'/f'block{b:02d}.pt'
        assert receipt['files'][str(p)]==sha(p)
    refs=read(old.SOURCE/'candidate_mask_manifest.json')['entries']
    assert len(refs)==224 and sum(r['weights'] for r in refs)==2*PRUNED
    cfg=load_config(old.EVAL); ph,_=_evaluation_config_hash(cfg)
    assert ph==parent['protocol_hash']
    previous=read(PRIOR/'uniform/results.json')
    assert previous['protocol_hash']==ph and previous['config_sha256']==sha(PRIOR/'config.json')
    assert previous['predictions_sha256']==sha(PRIOR/'uniform/predictions.jsonl')
    assert previous['manifest_sha256']==sha(PRIOR/'uniform/mask_manifest.json')
    predictions=old.read_jsonl(PRIOR/'uniform/predictions.jsonl')
    _validate_rows(predictions,old.read_jsonl(IDENTITY),ph)
    assert len(predictions)==100 and sum(x['correct'] for x in predictions)==previous['correct']
    base=read(BASE)['entries']; earlier=read(PRIOR/'uniform/mask_manifest.json')['entries']
    assert len(base)==len(earlier)==224
    for x,y in zip(base,earlier):
        assert x['name']==y['name'] and x['selected_mask']['mask_sha256']==y['selected_mask']['mask_sha256']
    paths=[*ROOT.glob('*.py'),ROOT/'run.sh',ROOT/'pipeline.sh',ROOT/'README.md',
           seq.CAL,old.EVAL,IDENTITY,BASE,old.SOURCE/'candidate_mask_manifest.json',
           SURVEY/'config.json',SURVEY/'input_collection.json',Path(input_receipt['path']),
           PRIOR/'config.json',PRIOR/'collection_receipt.json',PRIOR/'uniform/results.json',
           PRIOR/'uniform/predictions.jsonl',PRIOR/'uniform/mask_manifest.json',
           *sorted((PRIOR/'activations').glob('*.pt')),
           REPO/'experiments/dlm_lsa_projection65/core.py',Path(seq.__file__),Path(ppl.__file__),
           REPO/'experiments/dlm_wikitext_ppl/core.py',
           REPO/'experiments/projection_capacity_allocation_65/run.py',
           REPO/'experiments/projection_capacity_followup_65/run_heldout.py',
           REPO/'experiments/projection_capacity_followup_65/core.py',
           REPO/'experiments/dlm_loss_aggregation/core.py',
           REPO/'experiments/dlm_loss_aggregation/run.py',REPO/'experiments/dlm_loss_aggregation/config.yaml',
           REPO/'experiments/dlm_loss_aggregation/exp002/run.py',
           REPO/'experiments/dlm_dual_role_mini100/run.py',
           REPO/'experiments/wanda_failure_characterization/run_failure_map.py',
           REPO/'generate.py',REPO/'eval_llada.py']
    for row in base:
        m=row['selected_mask'];assert sha(m['path'])==m['file_sha256'];paths.append(Path(m['path']))
    from experiments.dlm_wikitext_ppl.core import digest
    for source in survey['evaluation_blocks']:
        paths.append(SURVEY/'evaluation/baseline'/f'{digest(source["block_id"])[:20]}.json')
    c=dict(model=parent['model'],seed=2026,target=.5,pruned=PRUNED,weights=2*PRUNED,
           coverage=.90,allocation_bounds=[.45,.55],modes=list(MODES),
           calibration='same8 WikiText train spans x10 frozen corrupted states,256tokens; dense input energies reused with receipt; not generation trajectories',
           score='e_sj=sum_out W_ij^2 * sum_token X_stj^2; diagonal energy; one descending mean_s e_sj order; required prefix fraction',
           pooled='shortest prefix covering90% pooled energy',
           coverage_rule='shortest SAME pooled-order prefix covering90% EACH of80 states; not globally optimal arbitrary support',
           mapping='224projection average-rank; larger support -> less pruning; parameter weighted centered, clipped45..55; exact projection row DP',
           ranking='frozen Uniform50 sparse-prefix Wanda ranking for both arms; reproduce224 baseline masks; no new candidate prefix recalibration',
           input_path=input_receipt['path'],protocol_hash=ph,evaluation=cfg['evaluation'],
           uniform_model_sha=previous['sparse_model_sha256'],uniform_correct=previous['correct'],
           nelbo=survey['evaluation'],evaluation_blocks=survey['evaluation_blocks'],
           primary='coverage vs pooled paired GSM8Kmini100 McNemar; fixed16article NELBO paired development diagnostic',
           secondary='both vs exact reused Uniform50 mini, Holm2; no full/test/tuning',
           limits=['channel support is only proxy for unstructured weight pruning','diagonal energy ignores cancellation/propagation','worst-state requirement and .90 fixed design choice','8 calibration spans only; existing mini is development','rank allocation is heuristic; no DLM-specificity or minimum weight-capacity proof'],
           sources={str(p.resolve()):sha(p) for p in paths})
    frozen(ROOT/'config.json',c);print('Frozen',sha(ROOT/'config.json'),flush=True)
    return c


@torch.inference_mode()
def prepare():
    c=validate()
    if (ROOT/'allocation.json').exists(): validate_allocation(); return
    event('loading_dense_for_support');torch.manual_seed(c['seed'])
    model,mapping=load_dense();model.eval()
    refs=read(old.SOURCE/'candidate_mask_manifest.json')['entries']
    assert list(mapping)==[r['name'] for r in refs]
    all_inputs=torch.load(c['input_path'],map_location='cpu',weights_only=False)
    inputs=all_inputs['corrupted'];states=read(seq.CAL)['states']
    assert len(states)==80 and len({s['sequence_index'] for s in states})==8
    # Cross-device/path sanity on independently replayed first dense state.
    observed={};handles=[]
    for name,module in mapping.items():
        def hook(mod,inp,out,key=name):
            x=inp[0];assert x.ndim==3 and x.shape[0]==1
            observed[key]=x[0].float().square().sum(0).cpu()
        handles.append(module.register_forward_hook(hook))
    try: model(torch.tensor(states[0]['noisy_ids'],device=next(model.parameters()).device))
    finally:
        for h in handles:h.remove()
    errors={}
    rows=[]
    for i,ref in enumerate(refs):
        name=ref['name'];item=inputs[name];energy=item['state_energy']
        assert tuple(energy.shape)==(80,ref['shape'][1])
        assert item['sequence_indices']==[s['sequence_index'] for s in states]
        assert item['mask_probabilities']==[s['p_mask'] for s in states]
        error=float((observed[name]-energy[0]).norm()/energy[0].norm())
        assert error<1e-5,(name,error);errors[name]=error
        w=mapping[name].weight
        w2=w.float().square().sum(0,dtype=torch.float64).cpu().numpy()
        metrics=support_metrics(energy.numpy(),w2,c['coverage'])
        rows.append(dict(name=name,index=i,layer=i//7,type=name.split('.')[1],
                         shape=ref['shape'],weights=ref['weights'],**metrics))
        event('support_metrics',completed=i+1,total=224)
    sizes=[r['weights'] for r in refs];allocations={}
    for mode in MODES:
        scores=[r[mode+'_score'] for r in rows]
        rates=budget_rates(scores,sizes)
        counts,budget=exact_projection_rows(refs,rates,PRUNED)
        assert all(.45-1/r['shape'][1]<=k/r['shape'][1]<=.55+1/r['shape'][1] for r,k in zip(refs,counts))
        allocations[mode]=dict(scores=scores,rates=rates.tolist(),row_counts=counts,budget=budget)
    write(ROOT/'support_metrics.json',dict(config_sha256=sha(ROOT/'config.json'),rows=rows,
         first_state_input_relative_errors=errors,mean_excess_fraction=float(np.mean([r['excess_fraction'] for r in rows]))))
    frozen(ROOT/'allocation.json',dict(config_sha256=sha(ROOT/'config.json'),allocations=allocations,
          metrics_sha256=sha(ROOT/'support_metrics.json')))
    event('allocation_complete',changed_projection_counts=sum(a!=b for a,b in zip(
          allocations['pooled']['row_counts'],allocations['coverage']['row_counts'])))


def validate_allocation():
    a=read(ROOT/'allocation.json')
    assert a['config_sha256']==sha(ROOT/'config.json')
    assert a['metrics_sha256']==sha(ROOT/'support_metrics.json')
    refs=read(old.SOURCE/'candidate_mask_manifest.json')['entries']
    for mode in MODES:
        x=a['allocations'][mode]
        rates=budget_rates(x['scores'],[r['weights'] for r in refs])
        assert np.array_equal(rates,np.array(x['rates']))
        counts,_=exact_projection_rows(refs,rates,PRUNED)
        assert counts==x['row_counts']
    return a


@torch.inference_mode()
def nelbo(model,mode,c):
    from experiments.dlm_wikitext_ppl.core import digest
    folder=ROOT/mode/'nelbo';rows=[]
    for i,source in enumerate(c['evaluation_blocks']):
        path=folder/f'{digest(source["block_id"])[:20]}.json'
        if path.exists(): r=read(path)
        else:
            def tick(n): event('mini_nelbo',method=mode,completed=i*128+n,total=16*128)
            r=ppl.evaluate_block(model,source,c['nelbo'],tick)
            r=ppl.seal(dict(r,config_sha256=sha(ROOT/'config.json')));frozen(path,r)
        ppl.verify_row(r,source,c['nelbo'],sha(ROOT/'config.json'));rows.append(r)
    result=dict(status='complete',config_sha256=sha(ROOT/'config.json'),rows=rows,
        mean_nelbo=float(np.mean([r['token_nelbo'] for r in rows])),
        ppl_bound=float(math.exp(np.mean([r['token_nelbo'] for r in rows]))),
        note='same16 development articles, not full WikiText or untouched test')
    frozen(folder/'results.json',result);return result


@torch.inference_mode()
def baseline():
    from experiments.dlm_wikitext_ppl.core import digest
    c=validate();validate_allocation();folder=ROOT/'uniform'
    if (folder/'results.json').exists():return
    event('baseline_loading');model,mapping=load_dense();model.eval()
    entries=read(BASE)['entries'];total=0
    for e in entries:
        mask=selected_mask(e,next(model.parameters()).device)
        k=e['selected_mask']['prune_per_row'];assert (mask.sum(1)==k).all()
        total+=int(mask.sum());mapping[e['name']].weight.masked_fill_(mask,0)
    assert total==PRUNED and model_sha(model)==c['uniform_model_sha']
    result=nelbo(model,'uniform',c);diff=0.
    for source,r in zip(c['evaluation_blocks'],result['rows']):
        oldrow=read(SURVEY/'evaluation/baseline'/f'{digest(source["block_id"])[:20]}.json')
        assert oldrow['mask_sha256']==r['mask_sha256']
        diff=max(diff,float(np.max(np.abs(np.array(oldrow['sample_token_nelbo'])-r['sample_token_nelbo']))))
    assert diff<=2e-6,('Uniform per-draw reproduction failed',diff)
    frozen(folder/'results.json',dict(status='complete',config_sha256=sha(ROOT/'config.json'),
        correct=c['uniform_correct'],total=100,protocol_hash=c['protocol_hash'],
        predictions_path=str(PRIOR/'uniform/predictions.jsonl'),
        predictions_sha256=sha(PRIOR/'uniform/predictions.jsonl'),
        sparse_model_sha256=model_sha(model),baseline_max_nelbo_draw_difference=diff,
        origin='exact Uniform50 mini reused after model,224mask,identity,protocol checks'))
    event('baseline_verified',correct=c['uniform_correct'],max_nelbo_draw_difference=diff)


def verify(mode):
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    c=validate();folder=ROOT/mode;r=read(folder/'results.json')
    assert r['config_sha256']==sha(ROOT/'config.json') and r['protocol_hash']==c['protocol_hash']
    path=Path(r['predictions_path'])
    assert sha(path)==r['predictions_sha256']
    rows=old.read_jsonl(path);_validate_rows(rows,old.read_jsonl(IDENTITY),c['protocol_hash'])
    assert len(rows)==100 and sum(x['correct'] for x in rows)==r['correct']
    if mode!='uniform':
        assert r['manifest_sha256']==sha(folder/'mask_manifest.json')
        manifest=read(folder/'mask_manifest.json');assert len(manifest['entries'])==224
        total=0
        for e in manifest['entries']:
            mask=selected_mask(e,'cpu');k=e['selected_mask']['prune_per_row']
            assert (mask.sum(1)==k).all();total+=int(mask.sum())
        assert total==PRUNED
    n=read(folder/'nelbo/results.json')
    assert n['config_sha256']==sha(ROOT/'config.json') and len(n['rows'])==16
    for s,v in zip(c['evaluation_blocks'],n['rows']):ppl.verify_row(v,s,c['nelbo'],sha(ROOT/'config.json'))
    return rows


@torch.inference_mode()
def evaluate(mode):
    from experiments.dlm_loss_aggregation.exp002.run import load_config,_evaluation_config_hash,_evaluate_gsm8k,_write_jsonl
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    from transformers import AutoTokenizer
    c=validate();a=validate_allocation();assert (ROOT/'uniform/results.json').exists()
    folder=ROOT/mode;folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():verify(mode);return
    refs=read(old.SOURCE/'candidate_mask_manifest.json')['entries'];base=read(BASE)['entries']
    counts=a['allocations'][mode]['row_counts']
    event('candidate_loading',method=mode);model,mapping=load_dense();model.eval();entries=[]
    assert list(mapping)==[r['name'] for r in refs]
    for b in range(32):
        activation=torch.load(PRIOR/'activations'/f'block{b:02d}.pt',map_location='cpu',weights_only=False)
        for i in range(b*7,(b+1)*7):
            ref=refs[i];name=ref['name'];w=mapping[name].weight
            v=activation[name].to(w.device)
            uniform=seq.wanda_mask(w,v,w.shape[1]//2)
            assert mask_sha256(pack_mask(uniform.cpu()))==base[i]['selected_mask']['mask_sha256'],name
            mask=seq.wanda_mask(w,v,counts[i]);packed=pack_mask(mask.cpu())
            assert (mask.sum(1)==counts[i]).all()
            path=folder/'masks'/f'{name}.pt';save_tensor(path,packed)
            entries.append(dict(name=name,shape=ref['shape'],weights=ref['weights'],
                selected_mask=dict(path=str(path),file_sha256=sha(path),mask_sha256=mask_sha256(packed),
                    prune_per_row=counts[i],pruned=counts[i]*ref['shape'][0])))
            w.masked_fill_(mask,0)
        event('build',method=mode,completed=b+1,total=32)
    assert sum(e['selected_mask']['pruned'] for e in entries)==PRUNED
    frozen(folder/'mask_manifest.json',dict(entries=entries,pruned=PRUNED,
        config_sha256=sha(ROOT/'config.json'),allocation_sha256=sha(ROOT/'allocation.json'),
        uniform_masks_reproduced=224))
    sparse_sha=model_sha(model);nelbo(model,mode,c)
    cfg=load_config(old.EVAL);ph,_=_evaluation_config_hash(cfg);assert ph==c['protocol_hash']
    tok=AutoTokenizer.from_pretrained(c['model']['id'],revision=c['model']['revision'],trust_remote_code=True)
    calls=[0];start=time.monotonic()
    def tick(mod,inp,out):
        calls[0]+=1
        if calls[0]%256==0:
            n=calls[0]//256
            event('mini100',method=mode,completed=n,total=100,
                  remaining_seconds=(100-n)*(time.monotonic()-start)/n)
    handle=model.register_forward_hook(tick);event('mini100',method=mode,completed=0,total=100)
    try:metrics,predictions=_evaluate_gsm8k(model,tok,cfg,'support_coverage50_'+mode,100,ph)
    finally:handle.remove()
    _validate_rows(predictions,old.read_jsonl(IDENTITY),ph)
    assert len(predictions)==100 and calls[0]==25600
    _write_jsonl(folder/'predictions.jsonl',predictions)
    assert model_sha(model)==sparse_sha;validate()
    frozen(folder/'results.json',dict(status='complete',config_sha256=sha(ROOT/'config.json'),
         correct=sum(x['correct'] for x in predictions),total=100,metrics=metrics,protocol_hash=ph,
         predictions_path=str(folder/'predictions.jsonl'),predictions_sha256=sha(folder/'predictions.jsonl'),
         manifest_sha256=sha(folder/'mask_manifest.json'),sparse_model_sha256=sparse_sha,forward_calls=calls[0]))
    verify(mode);event('candidate_complete',method=mode,correct=sum(x['correct'] for x in predictions))


def summarize():
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    c=validate();a=validate_allocation()
    values={m:[r['correct'] for r in verify(m)] for m in ('uniform',*MODES)}
    comparisons={m:paired_binary_comparison(values['uniform'],values[m]) for m in MODES}
    last=0
    for i,m in enumerate(sorted(comparisons,key=lambda m:comparisons[m]['exact_mcnemar_p'])):
        last=max(last,min(1,(2-i)*comparisons[m]['exact_mcnemar_p']))
        comparisons[m]['holm_two_vs_uniform']=last
    losses={m:np.array([r['token_nelbo'] for r in read(ROOT/m/'nelbo/results.json')['rows']]) for m in values}
    rng=np.random.default_rng(2026);indices=rng.integers(0,16,size=(5000,16))
    paired={}
    for control,treatment in [('pooled','coverage'),('uniform','pooled'),('uniform','coverage')]:
        d=losses[treatment]-losses[control]
        paired[treatment+'_minus_'+control]=dict(delta_nelbo=float(d.mean()),
            article_bootstrap95=np.quantile(d[indices].mean(1),[.025,.975]).tolist(),
            improved_articles=int((d<0).sum()),articles=16,
            note='unadjusted descriptive mini development interval, includes fixed MC noise')
    result=dict(status='complete',config_sha256=sha(ROOT/'config.json'),
        scores={m:sum(v) for m,v in values.items()},
        primary_coverage_vs_pooled=paired_binary_comparison(values['pooled'],values['coverage']),
        secondary_vs_uniform=comparisons,
        mini_nelbo={m:dict(nelbo=float(v.mean()),ppl_bound=float(np.exp(v.mean()))) for m,v in losses.items()},
        paired_mini_nelbo=paired,
        changed_projection_counts=sum(x!=y for x,y in zip(a['allocations']['pooled']['row_counts'],a['allocations']['coverage']['row_counts'])),
        limits=c['limits'])
    frozen(ROOT/'results.json',result)
    (ROOT/'report.md').write_text('# Support coverage50 mini experiment\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    event('complete',scores=result['scores'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','baseline','evaluate','summarize'])
    p.add_argument('--method',choices=MODES);args=p.parse_args()
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.action=='evaluate': assert args.method;evaluate(args.method)
            else:globals()[args.action]()
        except BaseException as exc:event('failed',action=args.action,method=args.method,error=repr(exc));raise
