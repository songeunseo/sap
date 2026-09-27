"""Context-response allocation50: paired A+C versus endpoint A, GSM8K mini100."""
import argparse, fcntl, json, os, time
from pathlib import Path
import numpy as np
import torch
from experiments.dlm_ppl50 import sequential as seq
from experiments.dlm_owl65 import run as old
from experiments.dlm_owl65.core import exact_row_counts
from experiments.dlm_loss_aggregation.core import pack_mask,mask_sha256
from experiments.projection_capacity_allocation_65.run import load_dense,save_tensor
from experiments.projection_capacity_followup_65.run_heldout import selected_mask
from experiments.wanda_failure_characterization.run_failure_map import model_sha
from experiments.dlm_context_response50.core import make_pairs,log_odds,distortion,rank_rates
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
BASE=REPO/'experiments/dlm_ppl50/uniform/mask_manifest.json'
IDENTITY=REPO/'experiments/dlm_allocation_sequential65/uniform/predictions.jsonl'
read,write,sha=seq.read,seq.write,seq.sha
PRUNED=3489660928
MODES=('uniform','A','AC')

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
    cal=read(seq.CAL)
    pairs=make_pairs(cal['states'],cal['mask_id'])
    assert len(pairs)==80 and len({p['sequence_index'] for p in pairs})==8
    for q in range(8): assert sum(p['sequence_index']==q for p in pairs)==10
    frozen(ROOT/'pairs.json',dict(source_sha256=sha(seq.CAL),pairs=pairs))
    base=read(BASE);refs=read(old.SOURCE/'candidate_mask_manifest.json')['entries']
    assert len(base['entries'])==224 and len(refs)==224
    assert sum(r['weights'] for r in refs)==2*PRUNED
    assert len({sum(r['weights'] for r in refs[b*7:(b+1)*7]) for b in range(32)})==1
    cfg=load_config(old.EVAL);ph,_=_evaluation_config_hash(cfg)
    paths=[*ROOT.glob('*.py'),ROOT/'run.sh',ROOT/'pipeline.sh',ROOT/'README.md',
           ROOT/'pairs.json',seq.CAL,BASE,IDENTITY,old.EVAL,old.STATS,
           old.SOURCE/'candidate_mask_manifest.json',Path(seq.__file__),Path(old.__file__),
           REPO/'experiments/dlm_owl65/core.py',
           REPO/'experiments/projection_capacity_allocation_65/run.py',
           REPO/'experiments/projection_capacity_followup_65/run_heldout.py',
           REPO/'experiments/dlm_loss_aggregation/core.py',
           REPO/'experiments/dlm_loss_aggregation/exp002/run.py',
           REPO/'experiments/dlm_dual_role_mini100/run.py',
           REPO/'generate.py',REPO/'eval_llada.py']
    for r in base['entries']:
        m=r['selected_mask'];assert sha(m['path'])==m['file_sha256']
        paths.append(Path(m['path']))
    c=dict(model=read(REPO/'experiments/dlm_ppl50/config.json')['model'],
           target=.5,pruned=PRUNED,weights=2*PRUNED,seed=2026,
           probe_rates=[.48,.52],allocation_range=[.45,.55],
           calibration='same8 WikiText train spans x10 existing mask states; native batch1,256tokens',
           pairs='each old state plus gold reveal min(round(.05*256),max(1,masked_count//4)); shared remaining masked queries; seed2026+state_index',
           objective='gold-vs-rest FP32 logodds; FP64 A=mean((e_before^2+e_after^2)/2), C=mean((e_after-e_before)^2); AC=A+C; equal state averages; no coefficient tuning',
           probes='one block48/52 in frozen Uniform50 background; exact native physical masks; cost difference per actual additional pruned parameter; preserve signed values',
           ranking='reproduce all224 Uniform50 sparse-prefix Wanda masks, freeze those activation vectors/rankings for probes AND both final allocations; no candidate-dependent recalibration',
           mapping='cost average-rank -> .5-.10*(rank01-meanrank01); grouped row-count DP exact50; no clamping signed cost',
           modes=list(MODES),protocol_hash=ph,evaluation=cfg['evaluation'],
           primary='AC vs A paired exact McNemar; reused100 GSM8K development examples',
           secondary='both vs freshly evaluated same-mask Uniform50; Holm2',
           limits=['paired gold reveal is not generated context','scalar gold-vs-rest omits wrong-vs-wrong ranking','state response is surrogate, not NELBO','finite marginal may not compose jointly','no fresh test/generalization claim'],
           sources={str(p.resolve()):sha(p) for p in paths})
    frozen(ROOT/'config.json',c);return validate()

@torch.inference_mode()
def margins(model,pair):
    device=next(model.parameters()).device;out=[]
    for key in ('before','after'):
        logits=model(torch.tensor(pair[key],device=device)).logits[0,pair['query']]
        out.append(log_odds(logits,pair['gold']).cpu());del logits
    return torch.stack(out)

@torch.inference_mode()
def dense_reference(model,pairs):
    path=ROOT/'dense.pt';receipt=ROOT/'dense_receipt.json'
    if receipt.exists():
        r=read(receipt);assert r['config_sha256']==sha(ROOT/'config.json') and sha(path)==r['sha256']
        return torch.load(path,weights_only=False)
    out=[]
    for i,p in enumerate(pairs):
        out.append(margins(model,p));event('dense_pairs',completed=i+1,total=len(pairs))
    assert torch.equal(margins(model,pairs[0]),out[0]),'Dense repeat mismatch'
    save_tensor(path,out);write(receipt,dict(config_sha256=sha(ROOT/'config.json'),sha256=sha(path)))
    return out

@torch.inference_mode()
def baseline(model,mapping,refs):
    reference=read(BASE)['entries'];states=read(seq.CAL)['states'];receipts=[]
    for b in range(32):
        path=ROOT/'activations'/f'block{b:02d}.pt'
        rec=path.with_suffix('.json')
        if rec.exists():
            r=read(rec);assert r['config_sha256']==sha(ROOT/'config.json') and sha(path)==r['sha256']
            activation=torch.load(path,weights_only=False)
        else:
            event('uniform_calibration',block=b,completed=0,total=80)
            activation=seq.collect_block(model,mapping,refs,states,b,
                  lambda n:event('uniform_calibration',block=b,completed=n,total=80))
            activation={k:v.cpu() for k,v in activation.items()}
            save_tensor(path,activation)
        masks=[]
        for i in range(b*7,(b+1)*7):
            name=refs[i]['name'];w=mapping[name].weight
            mask=seq.wanda_mask(w,activation[name].to(w.device),w.shape[1]//2)
            h=mask_sha256(pack_mask(mask.cpu()))
            assert h==reference[i]['selected_mask']['mask_sha256'],name
            w.masked_fill_(mask,0);masks.append(h)
        row=dict(config_sha256=sha(ROOT/'config.json'),sha256=sha(path),masks=masks)
        frozen(rec,row);receipts.append(row);event('uniform_masks',completed=b+1,total=32)
    return receipts

@torch.inference_mode()
def score(model,pairs,teacher,label):
    rows=[];errors=[]
    for i,(p,d) in enumerate(zip(pairs,teacher)):
        v=margins(model,p);r=distortion(v,d);rows.append(r);errors.append(v-d)
        event('probe_pairs',candidate=label,completed=i+1,total=len(pairs))
    return dict(rows=rows,mean={k:float(np.mean([r[k] for r in rows])) for k in ('A','C','AC')}),errors

@torch.inference_mode()
def collect():
    c=validate();torch.manual_seed(c['seed'])
    pairs=read(ROOT/'pairs.json')['pairs']
    refs=read(old.SOURCE/'candidate_mask_manifest.json')['entries']
    event('loading_dense');model,mapping=load_dense();model.eval()
    assert list(mapping)==[r['name'] for r in refs]
    teacher=dense_reference(model,pairs)
    dense={n:m.weight.detach().cpu().clone() for n,m in mapping.items()}
    receipts=baseline(model,mapping,refs)
    baseline_sha=model_sha(model);reference=margins(model,pairs[0])
    uniform,errors=score(model,pairs,teacher,'uniform')
    frozen(ROOT/'uniform_distortion.json',uniform)
    save_tensor(ROOT/'uniform_errors.pt',errors)
    base=read(BASE)['entries'];allprobes=[]
    for b in range(32):
        path=ROOT/'probes'/f'block{b:02d}.json'
        if path.exists():
            r=read(path);assert r['config_sha256']==sha(ROOT/'config.json')
            for v in r['conditions'].values(): assert sha(v['errors_path'])==v['errors_sha256']
            allprobes.append(r);continue
        activation=torch.load(ROOT/'activations'/f'block{b:02d}.pt',weights_only=False)
        conditions={}
        try:
            for rate in c['probe_rates']:
                pruned=0
                for i in range(b*7,(b+1)*7):
                    name=refs[i]['name'];w=mapping[name].weight
                    w.copy_(dense[name]);k=int(w.shape[1]*rate)
                    mask=seq.wanda_mask(w,activation[name].to(w.device),k)
                    w.masked_fill_(mask,0);pruned+=k*w.shape[0]
                s,e=score(model,pairs,teacher,f'block{b:02d}_{rate}')
                ep=ROOT/'probes'/f'block{b:02d}_{rate}_errors.pt'
                save_tensor(ep,e)
                conditions[str(rate)]=dict(**s,pruned=pruned,errors_path=str(ep),errors_sha256=sha(ep))
        finally:
            for i in range(b*7,(b+1)*7):
                name=refs[i]['name'];w=mapping[name].weight
                w.copy_(dense[name]);w.masked_fill_(selected_mask(base[i],w.device),0)
        assert torch.equal(margins(model,pairs[0]),reference),'Uniform restore mismatch'
        low,high=[conditions[str(r)] for r in c['probe_rates']]
        denom=high['pruned']-low['pruned'];assert denom>0
        costs={k:(high['mean'][k]-low['mean'][k])/denom for k in ('A','C','AC')}
        r=dict(block=b,config_sha256=sha(ROOT/'config.json'),conditions=conditions,
               costs=costs,baseline_restore_exact=True)
        write(path,r);allprobes.append(r);event('probe_blocks',completed=b+1,total=32)
    assert model_sha(model)==baseline_sha,'Final baseline hash changed'
    allocations={'uniform':dict(row_counts=[r['shape'][1]//2 for r in refs],rates=[.5]*32)}
    for mode in ('A','AC'):
        scores=[r['costs'][mode] for r in allprobes];rates=rank_rates(scores)
        counts,budget=exact_row_counts(refs,rates,PRUNED)
        allocations[mode]=dict(scores=scores,rates=rates.tolist(),row_counts=counts,budget=budget)
    frozen(ROOT/'allocation.json',dict(config_sha256=sha(ROOT/'config.json'),allocations=allocations))
    write(ROOT/'collection_receipt.json',dict(config_sha256=sha(ROOT/'config.json'),
          baseline_sha256=baseline_sha,verified_masks=224,probe_blocks=32,
          files={str(p):sha(p) for p in [ROOT/'allocation.json',ROOT/'dense.pt',
                  *sorted((ROOT/'activations').glob('*.pt')),*sorted((ROOT/'probes').glob('*.json'))]}))
    event('collection_complete')

def verify(mode):
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    c=validate();folder=ROOT/mode;r=read(folder/'results.json')
    rows=old.read_jsonl(folder/'predictions.jsonl')
    assert r['config_sha256']==sha(ROOT/'config.json')
    assert r['predictions_sha256']==sha(folder/'predictions.jsonl')
    assert r['manifest_sha256']==sha(folder/'mask_manifest.json')
    assert len(rows)==100 and r['correct']==sum(x['correct'] for x in rows)
    _validate_rows(rows,old.read_jsonl(IDENTITY),c['protocol_hash'])
    manifest=read(folder/'mask_manifest.json');total=0
    for e in manifest['entries']:
        mask=selected_mask(e,'cpu');k=e['selected_mask']['prune_per_row']
        assert (mask.sum(1)==k).all();total+=int(mask.sum())
    assert total==PRUNED
    return rows

@torch.inference_mode()
def evaluate(mode):
    from experiments.dlm_loss_aggregation.exp002.run import load_config,_evaluation_config_hash,_evaluate_gsm8k,_write_jsonl
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    from transformers import AutoTokenizer
    c=validate();folder=ROOT/mode;folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():verify(mode);return
    receipt=read(ROOT/'collection_receipt.json')
    assert receipt['config_sha256']==sha(ROOT/'config.json')
    for p,h in receipt['files'].items(): assert sha(p)==h,p
    allocation=read(ROOT/'allocation.json')
    assert allocation['config_sha256']==sha(ROOT/'config.json')
    counts=allocation['allocations'][mode]['row_counts']
    refs=read(old.SOURCE/'candidate_mask_manifest.json')['entries']
    event('evaluation_loading',method=mode);model,mapping=load_dense();model.eval();entries=[]
    base=read(BASE)['entries']
    for b in range(32):
        activation=torch.load(ROOT/'activations'/f'block{b:02d}.pt',weights_only=False)
        for i in range(b*7,(b+1)*7):
            ref=refs[i];name=ref['name'];w=mapping[name].weight
            mask=seq.wanda_mask(w,activation[name].to(w.device),counts[i])
            packed=pack_mask(mask.cpu());mh=mask_sha256(packed)
            if mode=='uniform': assert mh==base[i]['selected_mask']['mask_sha256']
            path=folder/'masks'/f'{name}.pt';save_tensor(path,packed)
            entries.append(dict(name=name,shape=ref['shape'],weights=ref['weights'],
                 selected_mask=dict(path=str(path),file_sha256=sha(path),mask_sha256=mh,
                     prune_per_row=counts[i],pruned=counts[i]*ref['shape'][0])))
            w.masked_fill_(mask,0)
        event('build',method=mode,completed=b+1,total=32)
    assert sum(e['selected_mask']['pruned'] for e in entries)==PRUNED
    manifest=dict(entries=entries,pruned=PRUNED,config_sha256=sha(ROOT/'config.json'),
                  allocation_sha256=sha(ROOT/'allocation.json'))
    frozen(folder/'mask_manifest.json',manifest)
    sparse_sha=model_sha(model)
    if mode=='uniform': assert sparse_sha==receipt['baseline_sha256']
    pairs=read(ROOT/'pairs.json')['pairs'];teacher=torch.load(ROOT/'dense.pt',weights_only=False)
    joint,errors=score(model,pairs,teacher,mode+'_joint');write(folder/'joint_distortion.json',joint)
    cfg=load_config(old.EVAL);ph,_=_evaluation_config_hash(cfg);assert ph==c['protocol_hash']
    tok=AutoTokenizer.from_pretrained(c['model']['id'],revision=c['model']['revision'],trust_remote_code=True)
    calls=[0]
    def tick(module,inp,out):
        calls[0]+=1
        if calls[0]%256==0:event('mini100',method=mode,completed=calls[0]//256,total=100)
    handle=model.register_forward_hook(tick)
    event('mini100',method=mode,completed=0,total=100)
    try:metrics,predictions=_evaluate_gsm8k(model,tok,cfg,'context_response50_'+mode,100,ph)
    finally:handle.remove()
    _validate_rows(predictions,old.read_jsonl(IDENTITY),ph)
    _write_jsonl(folder/'predictions.jsonl',predictions)
    assert model_sha(model)==sparse_sha;validate()
    write(folder/'results.json',dict(status='complete',config_sha256=sha(ROOT/'config.json'),
            manifest_sha256=sha(folder/'mask_manifest.json'),predictions_sha256=sha(folder/'predictions.jsonl'),
            sparse_model_sha256=sparse_sha,protocol_hash=ph,correct=sum(r['correct'] for r in predictions),
            total=100,metrics=metrics,forward_calls=calls[0]))
    verify(mode);event('candidate_complete',method=mode,correct=sum(r['correct'] for r in predictions))

def summarize():
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    values={m:verify(m) for m in MODES}
    correct={m:[r['correct'] for r in rows] for m,rows in values.items()}
    comparisons={m:paired_binary_comparison(correct['uniform'],correct[m]) for m in ('A','AC')}
    previous=0
    for i,m in enumerate(sorted(comparisons,key=lambda m:comparisons[m]['exact_mcnemar_p'])):
        previous=max(previous,min(1,(2-i)*comparisons[m]['exact_mcnemar_p']))
        comparisons[m]['holm_two_vs_uniform']=previous
    result=dict(status='complete',config_sha256=sha(ROOT/'config.json'),
         scores={m:sum(v) for m,v in correct.items()},primary_AC_vs_A=paired_binary_comparison(correct['A'],correct['AC']),
         secondary_vs_uniform=comparisons,limits=validate()['limits'])
    write(ROOT/'results.json',result)
    (ROOT/'report.md').write_text('# Context response50 mini100\n\n'+json.dumps(result,indent=2)+'\n')
    event('complete',scores=result['scores'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','collect','evaluate','summarize'])
    p.add_argument('--method',choices=MODES);args=p.parse_args()
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.action=='evaluate':assert args.method;evaluate(args.method)
            else:globals()[args.action]()
        except BaseException as exc:event('failed',action=args.action,method=args.method,error=repr(exc));raise
