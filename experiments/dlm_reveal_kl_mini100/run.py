"""Trajectory reveal-KL vs all-masked KL, fixed65 sequential-Wanda mini100."""
import argparse,fcntl,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from experiments.dlm_allocation_sequential65 import run as seq
from experiments.dlm_owl65 import run as old
from experiments.dlm_loss_aggregation.core import unpack_mask,mask_sha256
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write,file_sha256 as sha
from experiments.dlm_reveal_kl_mini100.core import rank_rates,select_reveal,token_kl,pool_scores
ROOT=Path(__file__).resolve().parent
MODES=('reveal','all_masked')
STEPS=list(range(8,256,16))
def read(p):return json.loads(Path(p).read_text())
def event(stage,**kw):
    obj=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw);write(ROOT/'progress.json',obj);print(json.dumps(obj),flush=True)
def frozen(p,obj):
    if Path(p).exists():assert read(p)==obj,p
    else:write(p,obj)
def source_config():return read(ROOT/'config.json')
def validate():
    c=source_config()
    for p,h in c['sources'].items():assert sha(p)==h,p
    return c

def freeze():
    c=seq.validate();ref=read(old.SOURCE/'candidate_mask_manifest.json')['entries'];cal=read(seq.CAL);prompts=[]
    for q in range(8):
        spans=[s['clean_ids'][0] for s in cal['states'] if s['sequence_index']==q]
        assert len(spans)==10 and all(x==spans[0] for x in spans)
        prompts.append(dict(sequence_index=q,prompt_ids=spans[0][:128]))
    assert len({tuple(x['prompt_ids']) for x in prompts})==8
    assert all(len(x['prompt_ids'])==128 and cal['mask_id'] not in x['prompt_ids'] for x in prompts)
    baseline=seq.ROOT/'uniform';result=read(baseline/'results.json')
    assert result['predictions_sha256']==sha(baseline/'predictions.jsonl') and result['manifest_sha256']==sha(baseline/'mask_manifest.json')
    assert result['protocol_hash']==c['protocol_hash'] and result['correct']==12
    from experiments.dlm_loss_aggregation.exp002.run import load_config,_evaluation_config_hash
    assert _evaluation_config_hash(load_config(old.EVAL))[0]==c['protocol_hash']
    files=[*ROOT.glob('*.py'),ROOT/'run.sh',ROOT/'pipeline.sh',ROOT/'README.md',seq.CAL,old.EVAL,
           seq.ROOT/'config.json',Path(seq.__file__),Path(old.__file__),old.STATS,old.SOURCE/'candidate_mask_manifest.json',
           Path('generate.py'),Path('eval_llada.py'),Path('experiments/dlm_loss_aggregation/exp002/run.py'),Path('experiments/dlm_loss_aggregation/core.py'),
           baseline/'predictions.jsonl',baseline/'results.json',baseline/'mask_manifest.json']
    for r in ref:
        m=r['masks'][3];assert m['nominal_sparsity']==.65;assert m['file_sha256']==sha(m['path']);files.append(Path(m['path']))
    cfg=dict(model=c['model'],target=.65,pruned=c['pruned'],weights=c['weights'],protocol_hash=c['protocol_hash'],modes=list(MODES),
        hypothesis='Reveal KL at target sparsity selects better budgets than same-state all-masked KL.',
        trajectories=dict(prompts=prompts,steps=256,generation_length=256,block_length=256,temperature=0,cfg_scale=0,mask_id=cal['mask_id'],capture_steps=STEPS,states=128,seed=0,source='first128 tokens of eight frozen WikiText train calibration spans'),
        probe='one entire block rowwise65 in otherwise dense model; verified historical dense80-calibrated StandardWanda65 masks; native batch1 full-forward',
        objective='FP32 KL(dense||pruned), uniform token mean over dense next reveal (K1) or all currently masked; equal16 states/prompt, equal8 prompts',
        allocation='average ranks across32 scores; s=.65-.10*((rank-1)/31-meanrank); block/inputwidth floor±1 exactbudget DP; no tuning',
        final_pruning='both candidates native sparse-prefix block-wise StandardWanda on original80 corruptionstates; not trajectory-recalibrated weight ranking',
        approximation='probe masks are dense-background local probes; final sparse-prefix ranking depends on preceding sparse blocks. Same protocol in both arms, not exact global marginal utility.',
        evaluation='historical GSM8K first100,5shot,temp0,256steps,strictEM; Uniform12 reused after hash checks; no automatic full/PPL/75%',
        primary='paired reveal versus all_masked exact McNemar; secondary both versus sequential Uniform with Holm2; repeated mini development screen',
        limits=['only8 WikiText-prefix prompts; decoding-domain/context-length mismatch to five-shot GSM8K','K1 reveal statistic may have high variance; block grouping does not guarantee low noise','single-target loss level is a heuristic, not marginal damage; no additive optimality claim'],
        sources={str(p.resolve()):sha(p) for p in files})
    frozen(ROOT/'config.json',cfg);return cfg


def tensor_save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def verify_trajectory(c):
    meta=read(ROOT/'trajectory_manifest.json');assert meta['config_sha256']==sha(ROOT/'config.json') and len(meta['states'])==128
    assert {(s['sequence_index'],s['step']) for s in meta['states']}=={(q,t) for q in range(8) for t in STEPS}
    for row in meta['states']:assert sha(row['path'])==row['sha256']
    return meta

@torch.inference_mode()
def trajectories(model,c):
    if (ROOT/'trajectory_manifest.json').exists():return verify_trajectory(c)
    from generate import generate
    dev=next(model.parameters()).device;tc=c['trajectories'];maskid=tc['mask_id'];allrows=[]
    for prompt in tc['prompts']:
        q=prompt['sequence_index'];receipt=ROOT/'trajectory'/f'prompt{q:02d}.json'
        if receipt.exists():
            saved=read(receipt);assert saved['config_sha256']==sha(ROOT/'config.json')
            for row in saved['states']:assert sha(row['path'])==row['sha256']
            allrows.extend(saved['states']);continue
        ids=torch.full((1,384),maskid,dtype=torch.long,device=dev);ids[0,:128]=torch.tensor(prompt['prompt_ids'],device=dev);rows=[]
        for step in range(256):
            masked=ids.eq(maskid)[0];assert int(masked.sum())==256-step
            out=model(ids).logits;tokens,chosen=select_reveal(out,ids,maskid,1)
            if step in STEPS:
                positions=masked.nonzero().flatten();ridx=(positions==chosen[0]).nonzero().flatten();assert len(ridx)==1
                path=ROOT/'trajectory'/f'q{q:02d}_t{step:03d}.pt'
                payload=dict(noisy_ids=ids.cpu().clone(),masked_positions=positions.cpu(),reveal_positions=chosen.cpu(),reveal_row_indices=ridx.cpu(),dense_logits=out[0,positions].cpu(),sequence_index=q,step=step)
                tensor_save(path,payload);rows.append(dict(sequence_index=q,step=step,path=str(path),sha256=sha(path),masked_count=len(positions),reveal_count=1))
            ids[0,chosen]=tokens[0,chosen];del out,tokens
        assert not ids.eq(maskid).any()
        # Independently reproduce first complete rollout using the original decoder.
        if q==0:
            original=generate(model,torch.tensor([prompt['prompt_ids']],device=dev),steps=256,gen_length=256,block_length=256,temperature=0,cfg_scale=0,remasking='low_confidence',mask_id=maskid)
            assert torch.equal(ids,original),'trajectory differs from native generate'
        saved=dict(config_sha256=sha(ROOT/'config.json'),states=rows,final_ids=ids.cpu().tolist(),native_decoder_parity=(q==0))
        write(receipt,saved);allrows.extend(rows);event('trajectory',completed=q+1,total=8)
    frozen(ROOT/'trajectory_manifest.json',dict(config_sha256=sha(ROOT/'config.json'),states=allrows,native_first_prompt_output_exact=True))
    return verify_trajectory(c)

@torch.inference_mode()
def collect():
    from experiments.projection_capacity_allocation_65.run import load_dense,DENSE_SHA
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    c=validate();torch.manual_seed(0);event('loading_dense');model,mapping=load_dense();model.eval();dev=next(model.parameters()).device
    ref=read(old.SOURCE/'candidate_mask_manifest.json')['entries'];assert list(mapping)==[r['name'] for r in ref]
    meta=trajectories(model,c);first=torch.load(meta['states'][0]['path'],map_location='cpu',weights_only=False)
    restored_reference=first['dense_logits'].to(dev)
    for b in range(32):
        path=ROOT/'probe'/f'block{b:02d}.json'
        if path.exists():
            r=read(path);assert r['config_sha256']==sha(ROOT/'config.json') and r['trajectory_sha256']==sha(ROOT/'trajectory_manifest.json');pool_scores(r['rows']);continue
        originals={};hashes={}
        try:
            for i in range(b*7,b*7+7):
                r=ref[i];mod=mapping[r['name']];originals[i]=mod.weight.detach().clone();m=r['masks'][3];packed=torch.load(m['path'],map_location='cpu',weights_only=False)
                assert mask_sha256(packed)==m['mask_sha256'];mask=unpack_mask(packed).to(dev);assert (mask.sum(1)==int(mask.shape[1]*.65)).all()
                mod.weight.masked_fill_(mask,0);hashes[r['name']]=m['mask_sha256']
            rows=[]
            for state in meta['states']:
                data=torch.load(state['path'],map_location='cpu',weights_only=False);ids=data['noisy_ids'].to(dev);positions=data['masked_positions'].to(dev)
                sparse=model(ids).logits[0,positions];dense=data['dense_logits'].to(dev);loss=token_kl(dense,sparse)
                rows.append(dict(sequence_index=state['sequence_index'],step=state['step'],reveal=float(loss[data['reveal_row_indices'].to(dev)].mean()),all_masked=float(loss.mean()),masked_count=len(loss)))
                del sparse,dense,loss
        finally:
            for i,w in originals.items():mapping[ref[i]['name']].weight.copy_(w);assert torch.equal(mapping[ref[i]['name']].weight,w)
        restored=model(first['noisy_ids'].to(dev)).logits[0,first['masked_positions'].to(dev)]
        assert torch.equal(restored,restored_reference),'dense restore/cache parity failed'
        write(path,dict(config_sha256=sha(ROOT/'config.json'),trajectory_sha256=sha(ROOT/'trajectory_manifest.json'),block=b,rows=rows,scores=pool_scores(rows),mask_hashes=hashes,physical_pruning=True,restored_dense_exact=True))
        del originals,restored;event('probe',completed=b+1,total=32)
    assert model_sha(model)==DENSE_SHA;write(ROOT/'collection_receipt.json',dict(config_sha256=sha(ROOT/'config.json'),trajectory_sha256=sha(ROOT/'trajectory_manifest.json'),dense_sha256=DENSE_SHA,probe_sha256={str(p):sha(p) for p in sorted((ROOT/'probe').glob('*.json'))}))
    allocate();event('collection_complete')


def allocate():
    c=validate();receipt=read(ROOT/'collection_receipt.json');assert receipt['config_sha256']==sha(ROOT/'config.json')
    for p,h in receipt['probe_sha256'].items():assert sha(p)==h
    refs=read(old.SOURCE/'candidate_mask_manifest.json')['entries'];out={}
    assert len(set(sum(r['weights'] for r in refs[b*7:b*7+7]) for b in range(32)))==1
    for mode in MODES:
        scores=[pool_scores(read(ROOT/'probe'/f'block{b:02d}.json')['rows'])[mode]['score'] for b in range(32)];rates=rank_rates(scores)
        counts,budget=seq.exact_row_counts(refs,rates,c['pruned'])
        out[mode]=dict(scores=scores,ideal_block_sparsities=rates.tolist(),row_counts=counts,budget=budget)
    frozen(ROOT/'allocation.json',dict(config_sha256=sha(ROOT/'config.json'),collection_receipt_sha256=sha(ROOT/'collection_receipt.json'),allocations=out))


def eval_harness():
    spec=importlib.util.spec_from_file_location('reveal_kl_sequential',seq.__file__);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    h.ROOT=ROOT;h.STORE=ROOT/'runtime'
    return h


def evaluate(mode):
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    from experiments.dlm_loss_aggregation.exp002.run import load_config,_evaluation_config_hash,_evaluate_gsm8k,_write_jsonl
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    from transformers import AutoTokenizer
    c=validate();folder=ROOT/mode;folder.mkdir(exist_ok=True);alloc=read(ROOT/'allocation.json')
    assert alloc['config_sha256']==sha(ROOT/'config.json') and alloc['collection_receipt_sha256']==sha(ROOT/'collection_receipt.json')
    if (folder/'results.json').exists():verify_one(mode);return
    event('evaluation_loading',method=mode);model,mapping=load_dense();ref=read(old.SOURCE/'candidate_mask_manifest.json')['entries'];assert list(mapping)==[r['name'] for r in ref]
    counts=alloc['allocations'][mode]['row_counts'];frozen(folder/'allocation.json',alloc['allocations'][mode]);h=eval_harness()
    rows=h.sequential(model,mapping,ref,read(seq.CAL)['states'],counts,mode,folder)
    manifest=dict(method=mode+'_kl_sequential_wanda65',config_sha256=sha(ROOT/'config.json'),allocation_sha256=sha(ROOT/'allocation.json'),entries=rows,pruned=c['pruned'],weights=c['weights'])
    frozen(folder/'mask_manifest.json',manifest);old.verify_masks(manifest)
    sparse_sha=model_sha(model);frozen(folder/'build_receipt.json',dict(config_sha256=sha(ROOT/'config.json'),allocation_sha256=sha(ROOT/'allocation.json'),mask_manifest_sha256=sha(folder/'mask_manifest.json'),sparse_model_sha256=sparse_sha))
    cfg=load_config(old.EVAL);ph,_=_evaluation_config_hash(cfg);assert ph==c['protocol_hash']
    tok=AutoTokenizer.from_pretrained(c['model']['id'],revision=c['model']['revision'],trust_remote_code=True)
    event('mini100',method=mode,completed=0,total=100)
    metrics,predictions=_evaluate_gsm8k(model,tok,cfg,mode+'_kl_sequential_wanda65',100,ph)
    _validate_rows(predictions,old.read_jsonl(seq.ROOT/'uniform/predictions.jsonl'),ph)
    _write_jsonl(folder/'predictions.jsonl',predictions);assert model_sha(model)==sparse_sha;validate()
    write(folder/'results.json',dict(status='complete',config_sha256=sha(ROOT/'config.json'),allocation_sha256=sha(ROOT/'allocation.json'),manifest_sha256=sha(folder/'mask_manifest.json'),predictions_sha256=sha(folder/'predictions.jsonl'),sparse_model_sha256=sparse_sha,protocol_hash=ph,correct=sum(r['correct'] for r in predictions),total=100,metrics=metrics))
    verify_one(mode);event('candidate_complete',method=mode,correct=sum(r['correct'] for r in predictions))


def verify_one(mode):
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    c=validate();folder=ROOT/mode;r=read(folder/'results.json');rows=old.read_jsonl(folder/'predictions.jsonl');base=old.read_jsonl(seq.ROOT/'uniform/predictions.jsonl')
    assert r['config_sha256']==sha(ROOT/'config.json') and r['allocation_sha256']==sha(ROOT/'allocation.json')
    assert r['manifest_sha256']==sha(folder/'mask_manifest.json') and r['predictions_sha256']==sha(folder/'predictions.jsonl')
    assert r['correct']==sum(x['correct'] for x in rows) and len(rows)==100 and r['protocol_hash']==c['protocol_hash']
    _validate_rows(rows,base,c['protocol_hash']);old.verify_masks(read(folder/'mask_manifest.json'))
    return rows


def summarize():
    from experiments.projection_capacity_followup_65.core import paired_binary_comparison
    from scipy.stats import spearmanr
    c=validate();values={m:verify_one(m) for m in MODES};base=old.read_jsonl(seq.ROOT/'uniform/predictions.jsonl');a=read(ROOT/'allocation.json')['allocations'];comparisons={}
    for mode in MODES:comparisons[mode]=paired_binary_comparison([r['correct'] for r in base],[r['correct'] for r in values[mode]])
    keys=sorted(MODES,key=lambda k:comparisons[k]['exact_mcnemar_p']);previous=0
    for j,k in enumerate(keys):
        previous=max(previous,min(1,(2-j)*comparisons[k]['exact_mcnemar_p']));comparisons[k]['holm_two_vs_uniform']=previous
    primary=paired_binary_comparison([r['correct'] for r in values['all_masked']],[r['correct'] for r in values['reveal']])
    out=dict(status='complete',config_sha256=sha(ROOT/'config.json'),scores={**{m:sum(r['correct'] for r in values[m]) for m in MODES},'uniform':12},primary_reveal_vs_all_masked=primary,secondary_vs_uniform=comparisons,
        score_spearman=float(spearmanr(a['reveal']['scores'],a['all_masked']['scores']).statistic),changed_projection_budgets=sum(x!=y for x,y in zip(a['reveal']['row_counts'],a['all_masked']['row_counts'])),
        mean_absolute_block_rate_difference_pp=float(100*np.mean(abs(np.array(a['reveal']['ideal_block_sparsities'])-np.array(a['all_masked']['ideal_block_sparsities'])))),
        sources={str(p):sha(p) for p in [ROOT/'allocation.json',ROOT/'collection_receipt.json',*[ROOT/m/'results.json' for m in MODES],*[ROOT/m/'predictions.jsonl' for m in MODES]]},limits=c['limits'])
    write(ROOT/'results.json',out)
    lines=['# Reveal-KL allocation mini100','','## Setup','',c['probe'],c['objective'],c['allocation'],c['final_pruning'],'','## Results','',json.dumps(out,indent=2),'','## Interpretation / Decision','','Primary is reveal versus all-masked under matched conditions. This is a reused mini100 development screen, not a new held-out confirmation. No full/PPL/75% automatically launched.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');event('complete',scores=out['scores'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','collect','evaluate','summarize']);p.add_argument('--method',choices=MODES);args=p.parse_args()
    lockpath=ROOT/(('eval_'+str(args.method)) if args.action=='evaluate' else args.action)
    with lockpath.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.action=='evaluate':assert args.method is not None;evaluate(args.method)
            else:globals()[args.action]()
        except BaseException as exc:event('failed',action=args.action,method=args.method,error=repr(exc));raise
