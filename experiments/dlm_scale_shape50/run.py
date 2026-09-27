"""Distribution survey and paired exact-budget PPL50 interventions."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from experiments.dlm_ppl50 import run as prior, sequential as seq
from experiments.dlm_wikitext_ppl import run as protocol, evaluate as evaluator
from experiments.dlm_wikitext_ppl.core import digest
from experiments.projection_capacity_allocation_65.run import load_dense, save_tensor
from experiments.projection_capacity_followup_65.run_heldout import selected_mask
from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
from experiments.dlm_scale_shape50.core import summarize_tensor, select_pairs, boundary_changes, apply_change

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
STORE=Path('/DATA/tmluser1/dlm-scale-shape50')
BASE=REPO/'experiments/dlm_ppl50/uniform'
read,write,sha=seq.read,seq.write,seq.sha


def event(stage,**kwargs):
    row=dict(stage=stage,time=time.time(),pid=os.getpid(),**kwargs)
    write(ROOT/'progress.json',row);print(json.dumps(row),flush=True)


def validate():
    cfg=read(ROOT/'config.json')
    for path,value in cfg['sources'].items():
        if sha(path)!=value:raise RuntimeError('Frozen source changed: '+path)
    return cfg


def freeze():
    if (ROOT/'config.json').exists():return validate()
    base=prior.validate(); pcfg=protocol.validate()
    corpus=read(protocol.ROOT/'corpus_manifest.json')['splits']['validation']
    articles={}
    for row in corpus:
        if len(row['clean_ids'])==512:articles.setdefault(row['article_sha256'],row)
    selected=sorted(articles.values(),key=lambda r:digest(['scale-shape50',r['article_sha256']]))[:16]
    if len(selected)!=16:raise RuntimeError('16 distinct validation articles required')
    paths=[seq.CAL,seq.old.STATS,seq.old.SOURCE/'candidate_mask_manifest.json',
           prior.ROOT/'config.json',BASE/'mask_manifest.json',BASE/'validation/results.json',
           protocol.ROOT/'config.json',protocol.ROOT/'corpus_manifest.json',
           Path(seq.__file__),Path(evaluator.__file__),REPO/'experiments/dlm_wikitext_ppl/core.py',
           REPO/'experiments/projection_capacity_followup_65/run_heldout.py',
           REPO/'experiments/dlm_allocation_backtrace/collection.json',
           *ROOT.glob('*.py'),ROOT/'run.sh']
    cfg=dict(model=base['model'],seed=2026,target=.5,pruned=3489660928,weights=6979321856,
        calibration='same80 frozen corrupted train states; add same8 clean inputs solely for distribution control',
        distributions='all224: exact moments/zero fraction/log variance and sampled quantiles/histograms of weights, channel RMS, dense/clean Wanda scores; raw activation samples1024/projection/state',
        selection='mean-score vs positive-log-score-variance disagreement; feature-only; four disjoint pairs each cross-depth/same-type, nearby-depth/same-type, same-layer/same-shape',
        move_weights=786432,per_stratum=4,
        ranking='frozen Uniform50 sparse-prefix Wanda order; all224 reconstructed masks must equal saved Uniform50 masks',
        intervention='baseline plus two opposite physical budget exchanges per pair; no recalibration between interventions; revert and verify weights after each',
        evaluation={k:pcfg[k] for k in ['sequence_length','mc_samples','seed','mask_id','conditioning','estimator','batch_size']},
        evaluation_blocks=selected,
        article_groups={'exploratory_A':[r['block_id'] for r in selected[:8]],'replication_B':[r['block_id'] for r in selected[8:]]},
        split_caveat='16 selected articles are disjoint from calibration but belong to previously used validation; two8-document groups are descriptive consistency checks, not untouched final data',
        primary='per-article NELBO(raw-direction)-NELBO(shape-direction), averaged equally across preselected pairs; paired article bootstrap5000 seed2026; report baseline deltas and stratum splits without discarding negative changes',
        limits='existing shape baseline; no AR control/DLM-specific claim, new pruning method, full validation search, test or GSM8K',
        sources={str(p):sha(p) for p in paths})
    protocol.frozen(ROOT/'config.json',cfg);return cfg


@torch.inference_mode()
def collect_inputs(model,mapping,states,cfg):
    receipt=ROOT/'input_collection.json'; path=STORE/'input_samples.pt'
    if receipt.exists():
        saved=read(receipt)
        if saved['config_sha256']!=sha(ROOT/'config.json') or saved['sha256']!=sha(path):raise RuntimeError('Input receipt mismatch')
        return torch.load(path,map_location='cpu',weights_only=False)
    device=next(model.parameters()).device
    clean={}
    for s in states:clean.setdefault(s['sequence_index'],s)
    conditions={'corrupted':states,'clean':list(clean.values())}
    output={}; old=torch.load(seq.old.STATS,map_location='cpu',weights_only=False)['statistics']
    for condition,rows in conditions.items():
        energies={name:[] for name in mapping};samples={name:[] for name in mapping}
        generators={name:torch.Generator(device='cpu').manual_seed(cfg['seed']+i) for i,name in enumerate(mapping)}
        handles=[]
        for name,module in mapping.items():
            def hook(mod,inp,out,key=name):
                x=inp[0]
                if x.ndim!=3 or x.shape[0]!=1:raise RuntimeError('Native batch1 required')
                energies[key].append(x.reshape(-1,x.shape[-1]).float().square().sum(0).cpu())
                index=torch.randint(x.numel(),(1024,),generator=generators[key]).to(x.device)
                samples[key].append(x.reshape(-1)[index].cpu())
            handles.append(module.register_forward_hook(hook))
        try:
            for i,state in enumerate(rows):
                ids=state['noisy_ids'] if condition=='corrupted' else state['clean_ids']
                model(torch.tensor(ids,device=device))
                event('collecting_inputs',condition=condition,completed=i+1,total=len(rows))
        finally:
            for handle in handles:handle.remove()
        output[condition]={}
        for name in mapping:
            if len(energies[name])!=len(rows):raise RuntimeError('Missing input observations')
            energy=torch.stack(energies[name]);mean=energy.mean(0)
            relative=None
            if condition=='corrupted':
                reference=old[name]['overall_uniform'].float()
                relative=float((mean-reference).norm()/reference.norm())
                if relative>1e-4:raise RuntimeError('Historical activation control mismatch: '+name)
            output[condition][name]=dict(mean_energy=mean,state_energy=energy,samples=torch.stack(samples[name]),
                dense_control_relative_error=relative,
                sequence_indices=[r['sequence_index'] for r in rows],
                mask_probabilities=[r['p_mask'] if condition=='corrupted' else 0. for r in rows])
    save_tensor(path,output)
    write(receipt,dict(path=str(path),sha256=sha(path),config_sha256=sha(ROOT/'config.json')))
    return output


@torch.inference_mode()
def survey(mapping,refs,inputs,cfg):
    rows=[]
    historical={r['name']:r for r in read(REPO/'experiments/dlm_allocation_backtrace/collection.json')['features']}
    for i,ref in enumerate(refs):
        path=ROOT/'statistics'/f'{i:03d}.json';name=ref['name']
        if path.exists():
            row=read(path)
            if row['config_sha256']!=sha(ROOT/'config.json') or row['name']!=name:raise RuntimeError('Statistics resume mismatch')
            rows.append(row);continue
        w=mapping[name].weight
        row=dict(name=name,layer=i//7,type=name.split('.')[1],shape=list(w.shape),weights=w.numel(),
            config_sha256=sha(ROOT/'config.json'),weight=summarize_tensor(w,cfg['seed']+i))
        for condition,label in [('corrupted','dense'),('clean','clean')]:
            item=inputs[condition][name]
            rms=item['mean_energy'].float().sqrt().to(w.device)
            row['channel_rms_'+label]=summarize_tensor(rms,cfg['seed']+i)
            row['activation_samples_'+label]=summarize_tensor(item['samples'],cfg['seed']+i)
            row['activation_samples_'+label]['measurement']='uniform scalar sample1024 per state; not all input activations'
            row['score_'+label]=summarize_tensor(w.float().abs()*rms[None,:],cfg['seed']+i)
        if not np.isclose(row['score_dense']['mean_abs'],historical[name]['mean_score'],rtol=2e-6):
            raise RuntimeError('Historical score control mismatch: '+name)
        write(path,row);rows.append(row);event('distribution_survey',completed=i+1,total=len(refs))
    protocol.frozen(ROOT/'statistics.json',dict(rows=rows,config_sha256=sha(ROOT/'config.json')))
    return rows


@torch.inference_mode()
def build_baseline_and_changes(model,mapping,refs,states,pairs,cfg):
    wanted={p[key] for p in pairs for key in ('prune_by_raw','prune_by_shape')}
    reference=read(BASE/'mask_manifest.json')['entries']
    receipts=[]
    for b in range(32):
        checkpoint=ROOT/'ranking'/f'block_{b:02d}.json'
        if checkpoint.exists():
            rows=read(checkpoint)
            for row in rows:
                name=row['name'];mask=selected_mask(reference[row['index']],mapping[name].weight.device)
                mapping[name].weight.masked_fill_(mask,0)
                if row.get('change_path') and sha(row['change_path'])!=row['change_sha256']:raise RuntimeError('Changed delta file')
            receipts.extend(rows);continue
        activation=seq.collect_block(model,mapping,refs,states,b)
        rows=[]
        for i in range(b*7,(b+1)*7):
            ref=refs[i];name=ref['name'];w=mapping[name].weight
            score=w.float().abs()*activation[name].sqrt()[None,:]
            order=torch.argsort(score,dim=1,stable=True);k=w.shape[1]//2
            mask=torch.zeros_like(w,dtype=torch.bool).scatter_(1,order[:,:k],True)
            expected=reference[i]['selected_mask']
            if mask_sha256(pack_mask(mask.cpu()))!=expected['mask_sha256']:raise RuntimeError('Uniform50 rank mismatch: '+name)
            row=dict(index=i,name=name,mask_sha256=expected['mask_sha256'],config_sha256=sha(ROOT/'config.json'))
            if name in wanted:
                changes=boundary_changes(w,order,cfg['move_weights'])
                path=STORE/'changes'/f'{name}.pt';save_tensor(path,changes)
                row.update(change_path=str(path),change_sha256=sha(path))
            w.masked_fill_(mask,0);rows.append(row)
        write(checkpoint,rows);receipts.extend(rows);event('uniform_rank_reproduction',completed=b+1,total=32)
    protocol.frozen(ROOT/'ranking_receipt.json',dict(rows=receipts,verified_masks=224,pruned=cfg['pruned']))
    return {r['name']:torch.load(r['change_path'],map_location='cpu',weights_only=False) for r in receipts if r.get('change_path')}


@torch.inference_mode()
def evaluate(model,label,cfg):
    out=[];start=time.monotonic();new=0
    for index,source in enumerate(cfg['evaluation_blocks']):
        path=ROOT/'evaluation'/label/f'{digest(source["block_id"])[:20]}.json'
        if path.exists():row=read(path)
        else:
            def tick(done):
                elapsed=time.monotonic()-start
                event('evaluating',candidate=label,blocks_completed=index,blocks_total=16,
                    completed=index*128+done,total=2048,
                    candidate_remaining_seconds=(2048-index*128-done)*elapsed/(new+done))
            value=evaluator.evaluate_block(model,source,cfg['evaluation'],tick)
            row=evaluator.seal(dict(value,config_sha256=sha(ROOT/'config.json')))
            evaluator.verify_row(row,source,cfg['evaluation'],sha(ROOT/'config.json'))
            protocol.frozen(path,row);new+=128
        evaluator.verify_row(row,source,cfg['evaluation'],sha(ROOT/'config.json'));out.append(row)
    protocol.frozen(ROOT/'evaluation'/label/'results.json',dict(status='complete',label=label,
        mean_nelbo=float(np.mean([r['token_nelbo'] for r in out])),
        block_ids=[r['block_id'] for r in out],config_sha256=sha(ROOT/'config.json')))
    return out


@torch.inference_mode()
def run():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=validate()
        try:
            event('loading_dense');model,mapping=load_dense()
            refs=read(seq.old.SOURCE/'candidate_mask_manifest.json')['entries']
            if list(mapping)!=[r['name'] for r in refs]:raise RuntimeError('Wrong module order')
            states=read(seq.CAL)['states']
            inputs=collect_inputs(model,mapping,states,cfg)
            stats=survey(mapping,refs,inputs,cfg);del inputs
            pairs=select_pairs(stats,cfg['per_stratum'],cfg['move_weights'])
            protocol.frozen(ROOT/'selection.json',dict(pairs=pairs,statistics_sha256=sha(ROOT/'statistics.json'),
                rule=cfg['selection'],config_sha256=sha(ROOT/'config.json')))
            from experiments.dlm_scale_shape50.analyze import distribution_report
            distribution_report()
            changes=build_baseline_and_changes(model,mapping,refs,states,pairs,cfg)
            baseline=evaluate(model,'baseline',cfg)
            differences=[]
            for row in baseline:
                historical=read(BASE/'validation/blocks'/f'{digest(row["block_id"])[:20]}.json')
                err=float(np.max(np.abs(np.array(row['sample_token_nelbo'])-historical['sample_token_nelbo'])))
                differences.append(err)
            if max(differences)>2e-6:raise RuntimeError('Baseline likelihood control failed')
            protocol.frozen(ROOT/'baseline_control.json',dict(max_per_draw_difference=max(differences),threshold=2e-6,passed=True))
            sham_source=cfg['evaluation_blocks'][0]
            from experiments.dlm_wikitext_ppl.core import masks,normalized_nelbo
            device=next(model.parameters()).device
            clean=torch.tensor(sham_source['clean_ids'],device=device)
            mask=next(masks(len(clean),128,sham_source['mask_seed'])).to(device)
            def sham():
                logits=model(clean.masked_fill(mask,cfg['evaluation']['mask_id']).unsqueeze(0)).logits[0]
                return float(normalized_nelbo(logits,clean,mask))
            sham_value=sham()
            for index,pair in enumerate(pairs):
                a,b=pair['prune_by_raw'],pair['prune_by_shape']
                for direction in ('raw','shape'):
                    label=pair['id']+'_'+direction
                    actions={a:'prune' if direction=='raw' else 'protect',b:'protect' if direction=='raw' else 'prune'}
                    before={n:mapping[n].weight.detach().cpu().clone() for n in actions}
                    try:
                        for name,action in actions.items():apply_change(mapping[name].weight,changes[name],action)
                        evaluate(model,label,cfg)
                    finally:
                        for name,action in actions.items():
                            apply_change(mapping[name].weight,changes[name],action,restore=True)
                            if not torch.equal(mapping[name].weight.cpu(),before[name]):raise RuntimeError('Weight restoration failed: '+name)
                    if sham()!=sham_value:raise RuntimeError('Baseline sham changed')
                    protocol.frozen(ROOT/'restoration'/f'{label}.json',dict(exact_weights_restored=True,baseline_sham_equal=True,
                        exchanged_weights=cfg['move_weights'],total_mask_pruned=cfg['pruned'],config_sha256=sha(ROOT/'config.json')))
                    event('candidate_complete',candidate=label,completed=index*2+(1 if direction=='raw' else 2),total=len(pairs)*2)
            validate()
            from experiments.dlm_scale_shape50.analyze import main
            main();event('complete',report=str(ROOT/'report.md'))
        except BaseException as exc:
            event('failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.phase=='freeze' else run()
