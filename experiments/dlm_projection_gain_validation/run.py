"""Projection Wanda-square gain and physical joint-pruning diagnostics."""
import argparse, fcntl, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write, file_sha256 as sha
from experiments.dlm_lsa_mask_mismatch import run as prior
from experiments.dlm_owl65.run import STATS
from experiments.projection_capacity_allocation_65.run import load_dense,DENSE_SHA
from experiments.wanda_failure_characterization.run_failure_map import model_sha
from experiments.dlm_loss_aggregation.core import unpack_mask,mask_sha256
ROOT=Path(__file__).resolve().parent
BLOCKS=[0,1,15,16,30,31]
SELECT=[(0,'q_proj'),(1,'k_proj'),(15,'ff_proj'),(16,'q_proj'),(30,'ff_out'),(31,'ff_out')]
def read(p):return json.loads(Path(p).read_text())
def event(stage,**kw):
    d=dict(stage=stage,time=time.time(),**kw);write(ROOT/'progress.json',d);print(json.dumps(d),flush=True)
def freeze():
    prior.validate();ref=read(prior.MAN)['entries'];names=[r['name'] for r in ref]
    select=[names.index(f'block_{b:02d}.{p}') for b,p in SELECT]
    jobs=[dict(id=f'p{i:03d}_{int(s*100)}',indices=[i],sparsity=s) for i in select for s in [.3,.7]]
    jobs += [dict(id=f'b{b:02d}_50',indices=list(range(b*7,b*7+7)),sparsity=.5) for b in BLOCKS]
    jobs += [dict(id=f'pair{b:02d}_50',indices=list(range(b*7,(b+2)*7)),sparsity=.5) for b in [0,15,30]]
    files=[Path(__file__),ROOT/'run.sh',prior.ROOT/'config.json',prior.ROOT/'verification.json',prior.ROOT/'dense_losses.json',prior.MAN,prior.CAL,STATS,Path(prior.__file__),Path('experiments/projection_capacity_allocation_65/run.py'),Path('experiments/dlm_loss_aggregation/core.py')]
    files += [prior.ROOT/'damage'/f'{i:03d}.json' for i in range(224)]
    files += [Path(r['masks'][0]['path']) for r in ref]
    c=dict(status='frozen',model=read(prior.CAL)['model'],dense_sha=DENSE_SHA,states=80,sequences=8,length=256,blocks=BLOCKS,selected=select,jobs=jobs,
        proxy='E=sum removed FP32 Wanda score squared, stable row sort, floor(width*s); g=mean signed projection deltaCE50/E50',
        target='native batch1 physical pruning dense background; CE sum/(p_mask*length); uniform state mean',
        features='dense block mean token ||hout-hin||2/||hin||2 and mean 1-cos(hin,hout), collected during mandatory dense CE forwards; no extra forward',
        analysis='sequence-cluster paired bootstrap2000 seed0; exploratory simultaneous Bonferroni95% intervals per family (12 curve,6 block,3 pair); depth versus feature leave-one-block-out OLS prediction of signed g; depth+type baseline',
        policy='no clipping signed gains; no automatic allocation/mini; reconstruction or interaction are follow-up candidates, not proven causes',sources={str(p.resolve()):sha(p) for p in files})
    if (ROOT/'config.json').exists():assert read(ROOT/'config.json')==c
    else:write(ROOT/'config.json',c)
    return c

def validate():
    c=read(ROOT/'config.json')
    for p,h in c['sources'].items():assert sha(p)==h,p
    return c

def analyze():
    c=validate();ref=read(prior.MAN)['entries'];states=read(prior.CAL)['states'];seq=np.array([s['sequence_index'] for s in states]);rng=np.random.default_rng(0);draws=rng.integers(0,8,(2000,8))
    damage=np.array([read(prior.ROOT/'damage'/f'{i:03d}.json')['delta_ce'] for i in range(224)])
    curves=read(ROOT/'curves.json')['rows'];g=np.array([r['g'] for r in curves]);results=dict(status='complete',config_sha256=sha(ROOT/'config.json'),negative_g_indices=np.where(g<0)[0].tolist(),curve=[],block=[],pair=[])
    def summary(v,n):
        means=np.array([v[seq==q].mean() for q in range(8)]);boot=means[draws].mean(1);a=.05/(2*n)
        return dict(mean=float(v.mean()),per_sequence=means.tolist(),simultaneous_conditional_ci=np.quantile(boot,[a,1-a]).tolist())
    jobvals={j['id']:np.array(read(ROOT/'measurements'/f"{j['id']}.json")['delta_ce']) for j in c['jobs']}
    for j in c['jobs']:
        y=jobvals[j['id']];s=j['sparsity'];idx=j['indices']
        if len(idx)==1:
            i=idx[0];ratio=curves[i]['E'][str(s)]/curves[i]['E']['0.5'];pred=damage[i]*ratio
            results['curve'].append(dict(id=j['id'],name=ref[i]['name'],sparsity=s,observed=summary(y,12),predicted=summary(pred,12),residual=summary(y-pred,12),gain_ratio=float(y.mean()/pred.mean()) if abs(pred.mean())>1e-12 else None))
        elif len(idx)==7:
            pred=damage[idx].sum(0);results['block'].append(dict(id=j['id'],observed=summary(y,6),sum_projection=summary(pred,6),interaction=summary(y-pred,6)))
        else:
            b=idx[0]//7;pred=jobvals[f'b{b:02d}_50']+jobvals[f'b{b+1:02d}_50'];results['pair'].append(dict(id=j['id'],observed=summary(y,3),sum_block=summary(pred,3),interaction=summary(y-pred,3)))
    feats=np.array(read(ROOT/'dense_features.json')['values']).mean(0);depth=np.arange(224)//7;types=np.tile(np.eye(7),(32,1))[:,1:]
    designs={'depth':np.column_stack([np.ones(224),depth,types])}
    for k,label in enumerate(['update_ratio','cosine_change']):designs[label]=np.column_stack([designs['depth'],np.repeat(feats[:,k],7)])
    predictors={}
    for label,x in designs.items():
        pred=np.zeros(224)
        for b in range(32):
            train=depth!=b;xx=x[train];mu=xx[:,1:].mean(0);sd=xx[:,1:].std(0);sd[sd==0]=1
            z=x.copy();z[:,1:]=(x[:,1:]-mu)/sd;coef=np.linalg.lstsq(z[train],g[train],rcond=None)[0];pred[~train]=z[~train]@coef
        predictors[label]=dict(leave_one_block_out_mse=float(np.mean((pred-g)**2)),predictions=pred.tolist())
    results['predictors']=predictors;results['feature_correlations']={label:float(spearmanr(np.repeat(feats[:,k],7),g).statistic) for k,label in enumerate(['update_ratio','cosine_change'])}
    results['limitations']=['in-calibration 8 sequences, exploratory identity-selected projections; no heldout or DLM vs AR control','physical simultaneous50 interventions in dense background do not establish65 sparse-context utility','negative gains break convex nondecreasing mu-bisection assumptions; preserved, not clipped','block features repeated across7 types; use leave-one-block-out, not224 independent layers','gain-ratio instability near zero and correlation do not justify replacement alone']
    write(ROOT/'results.json',results)
    lines=['# Projection gain validation','','## Results','',f"Negative signed gains: {results['negative_g_indices']}",'','|Condition|Observed CE delta|Predicted/summed CE delta|Residual|Simultaneous CI|','|---|---:|---:|---:|---|']
    for family in ['curve','block','pair']:
        for r in results[family]:
            target=r.get('predicted',r.get('sum_projection',r.get('sum_block')));res=r.get('residual',r.get('interaction'))
            lines.append(f"|{r['id']}|{r['observed']['mean']:.6g}|{target['mean']:.6g}|{res['mean']:.6g}|{res['simultaneous_conditional_ci']}|")
    lines += ['','## Predictor check','',json.dumps({k:{'LOBO MSE':v['leave_one_block_out_mse']} for k,v in predictors.items()},indent=2),'','## Interpretation / Decision','','Inspect curve residuals and signed-g validity before choosing a proxy or allocator. Pair interactions at50 in dense background are not universal redundancy. No mini launched.','',*results['limitations']]
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');event('complete')

@torch.inference_mode()
def collect():
    c=validate();event('loading');model,mapping=load_dense();model.eval();dev=next(model.parameters()).device;ref=read(prior.MAN)['entries'];assert list(mapping)==[r['name'] for r in ref]
    states=read(prior.CAL)['states'];tensors=[(torch.tensor(s['noisy_ids'],device=dev),torch.tensor(s['clean_ids'],device=dev),torch.tensor(s['mask'],device=dev,dtype=torch.bool)[0],s['p_mask']) for s in states]
    stats=torch.load(STATS,map_location='cpu',weights_only=False)['statistics'];rows=[]
    for i,r in enumerate(ref):
        mod=mapping[r['name']];score=mod.weight.float().abs()*stats[r['name']]['overall_uniform'].float().to(dev).sqrt()[None,:];order=torch.argsort(score,dim=1,stable=True);sorted_score=score.gather(1,order)
        marginal=sorted_score.double().square().sum(0).cpu().numpy();E={str(s):float(marginal[:int(score.shape[1]*s)].sum()) for s in [.3,.5,.7]}
        mask=torch.zeros_like(score,dtype=torch.bool).scatter_(1,order[:,:int(score.shape[1]*.5)],True);packed=torch.load(r['masks'][0]['path'],map_location='cpu',weights_only=False);assert mask_sha256(packed)==r['masks'][0]['mask_sha256'];assert torch.equal(mask.cpu(),unpack_mask(packed)),r['name']
        old=read(prior.ROOT/'damage'/f'{i:03d}.json');assert old['name']==r['name'] and old['mask_sha256']==r['masks'][0]['mask_sha256'];assert E['0.5']>0
        rows.append(dict(name=r['name'],E=E,g=float(np.mean(old['delta_ce'])/E['0.5']),weights=r['weights']))
        np.save(ROOT/f'marginal_{i:03d}.npy',marginal);del score,order,sorted_score,mask,packed
        if (i+1)%7==0:event('curve_build',completed=i+1,total=224)
    write(ROOT/'curves.json',dict(config_sha256=sha(ROOT/'config.json'),all_224_masks50_reproduced=True,rows=rows))
    values=[];captured={};handles=[]
    for b,block in enumerate(model.model.transformer.blocks):
        def hook(mod,inp,out,key=b):
            x=inp[0].float();y=(out[0] if isinstance(out,tuple) else out).float();den=x.norm(dim=-1).clamp_min(1e-30)
            captured[key]=[float(((y-x).norm(dim=-1)/den).mean()),float((1-F.cosine_similarity(x,y,dim=-1)).mean())]
        handles.append(block.register_forward_hook(hook))
    try:
        base=[]
        for ids,target,pos,p in tensors:
            captured.clear();base.append(prior.loss(prior.logits(model,ids),target,pos,p));assert len(captured)==32;values.append([captured[b] for b in range(32)])
    finally:
        for h in handles:h.remove()
    assert np.allclose(base,read(prior.ROOT/'dense_losses.json')['losses'],atol=1e-7,rtol=0)
    write(ROOT/'dense_features.json',dict(config_sha256=sha(ROOT/'config.json'),values=values));write(ROOT/'dense_losses.json',dict(losses=base))
    for j in c['jobs']:
        path=ROOT/'measurements'/f"{j['id']}.json"
        if path.exists():assert read(path)['config_sha256']==sha(ROOT/'config.json');continue
        saved={};masks={}
        try:
            for i in j['indices']:
                r=ref[i];mod=mapping[r['name']];saved[i]=mod.weight.detach().clone()
                if j['sparsity']==.5:
                    masks[i]=unpack_mask(torch.load(r['masks'][0]['path'],map_location='cpu',weights_only=False)).to(dev)
                else:
                    score=mod.weight.float().abs()*stats[r['name']]['overall_uniform'].float().to(dev).sqrt()[None,:];order=torch.argsort(score,dim=1,stable=True);masks[i]=torch.zeros_like(score,dtype=torch.bool).scatter_(1,order[:,:int(score.shape[1]*j['sparsity'])],True);del score,order
                mod.weight.masked_fill_(masks[i],0)
            delta=[prior.loss(prior.logits(model,ids),target,pos,p)-base[q] for q,(ids,target,pos,p) in enumerate(tensors)]
        finally:
            for i,w in saved.items():mapping[ref[i]['name']].weight.copy_(w);assert torch.equal(mapping[ref[i]['name']].weight,w)
        restored=prior.loss(prior.logits(model,tensors[0][0]),*tensors[0][1:]);assert abs(restored-base[0])<1e-7
        write(path,dict(config_sha256=sha(ROOT/'config.json'),job=j,delta_ce=delta,restoration_delta=restored-base[0],physical_pruning=True))
        del saved,masks;event('measurement',id=j['id'],completed=len(list((ROOT/'measurements').glob('*.json'))),total=21)
    assert model_sha(model)==DENSE_SHA;validate();analyze()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','collect','analyze']);a=p.parse_args()
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:globals()[a.action]()
        except BaseException as e:event('failed',error=repr(e));raise
