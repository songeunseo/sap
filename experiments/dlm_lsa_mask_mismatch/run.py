"""Clean/corrupted LSA proxy audit with fixed masks and native batch-one forwards."""
import argparse, fcntl, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, rankdata
from experiments.dlm_dual_role_allocation.io import file_sha256 as sha, atomic_write_json as write
from experiments.dlm_allocation_baselines65.core import lsa_metric
from experiments.dlm_allocation_baselines65 import run as baseline
from experiments.dlm_owl65 import run as old
from experiments.projection_capacity_allocation_65.verify_states import CAL
from experiments.dlm_loss_aggregation.core import unpack_mask, mask_sha256
ROOT=Path(__file__).resolve().parent
CACHE=baseline.ROOT/'lsa/statistics.json'
MAN=old.SOURCE/'candidate_mask_manifest.json'

def read(p): return json.loads(Path(p).read_text())
def event(stage,**kw):
    v=dict(stage=stage,time=time.time(),**kw);write(ROOT/'progress.json',v);print(json.dumps(v),flush=True)
def freeze():
    baseline.validate()
    assert read(CACHE)['config_sha256']==sha(baseline.ROOT/'lsa/config.json')
    doc=read(CAL); ref=read(MAN)['entries']; rows=read(CACHE)['rows']
    assert len(rows)==224 and [r['name'] for r in rows]==[r['name'] for r in ref]
    assert len(doc['states'])==80 and len({s['sequence_index'] for s in doc['states']})==8
    files=[Path(__file__),CACHE,MAN,CAL,baseline.ROOT/'lsa/config.json',Path(baseline.__file__),Path(old.__file__),Path('experiments/dlm_allocation_baselines65/core.py')]
    for r in ref:
        assert r['masks'][0]['nominal_sparsity']==.5
        files.append(Path(r['masks'][0]['path']))
    c=dict(model=doc['model'],states=80,sequences=8,sequence_length=256,probe=.5,group=128,
       primary='paired Spearman difference: corrupted LSA vs clean LSA predicting SAME corrupted masked gold CE damage; projection and block allocation proxy',
       target='CE sum/(p_mask*sequence_length), uniform state mean; delta against native dense model',
       clean_control='8 unique identical clean spans; never score clean input as valid DLM likelihood',
       intervention='fixed historical corrupted-calibrated Standard Wanda rowwise50 mask, one projection in dense background; functional Linear hook, native batch1 full-forward',
       limits=['in-calibration diagnostic, eight sequences; not AR comparison or full PPL','LSA surrogate selector differs from Wanda intervention','block damage is mean of projection-only damage, not joint block intervention'],
       bootstrap='paired sequence-cluster 2000 resamples seed0; fixed proxy stats, conditional CIs; two primary granularities Bonferroni 97.5% CIs',
       sources={str(p.resolve()):sha(p) for p in files})
    if (ROOT/'config.json').exists(): assert read(ROOT/'config.json')==c
    else: write(ROOT/'config.json',c)
    return c

def validate():
    c=read(ROOT/'config.json')
    for p,h in c['sources'].items(): assert sha(p)==h,p
    return c

def logits(model,ids):
    o=model(ids);return o.logits if hasattr(o,'logits') else o

def loss(z,target,pos,p):
    assert pos.any()
    return float(F.cross_entropy(z[0,pos].float(),target[0,pos],reduction='sum')/(p*target.shape[1]))

def corr(x,y): return float(spearmanr(x,y).statistic)

def analyze():
    validate();states=read(CAL)['states'];ref=read(MAN)['entries']
    clean=np.array([r['metric'] for b in range(32) for r in read(ROOT/'clean'/f'{b:02d}.json')['rows']])
    noisy=np.array([r['metric'] for r in read(CACHE)['rows']])
    damage=np.array([read(ROOT/'damage'/f'{i:03d}.json')['delta_ce'] for i in range(224)])
    seq=np.array([s['sequence_index'] for s in states]); byseq=np.stack([damage[:,seq==q].mean(1) for q in range(8)],axis=1)
    rng=np.random.default_rng(0);draws=rng.integers(0,8,(2000,8));result={}
    for gran in ('projection','block'):
        a,b,y=clean,noisy,byseq
        if gran=='block':a=abs(a.reshape(32,7).mean(1));b=abs(b.reshape(32,7).mean(1));y=y.reshape(32,7,8).mean(1)
        target=y.mean(1);point=corr(b,target)-corr(a,target)
        boot=[corr(b,y[:,d].mean(1))-corr(a,y[:,d].mean(1)) for d in draws]
        result[gran]=dict(rho_clean=corr(a,target),rho_corrupted=corr(b,target),rho_proxy_clean_corrupted=corr(a,b),rho_difference=point,conditional_ci_975=np.quantile(boot,[.0125,.9875]).tolist(),sequence_differences=[corr(b,y[:,q])-corr(a,y[:,q]) for q in range(8)])
    # Generic depth/type structure control (exploratory), rank residuals.
    design=np.column_stack([np.ones(224),np.repeat(np.eye(32),7,axis=0)[:,1:],np.tile(np.eye(7),(32,1))[:,1:]])
    def residual(v):
        r=rankdata(v);return r-design@np.linalg.lstsq(design,r,rcond=None)[0]
    target=byseq.mean(1)
    result['exploratory_layer_type_partial_rank']=dict(clean=float(np.corrcoef(residual(clean),residual(target))[0,1]),corrupted=float(np.corrcoef(residual(noisy),residual(target))[0,1]))
    result['config_sha256']=sha(ROOT/'config.json');result['status']='complete';write(ROOT/'results.json',result)
    lines=['# LSA clean/corrupted proxy audit','',json.dumps(result,indent=2),'','Interpretation: higher corrupted correlation supports mask-conditioned sensitivity information, not proof of DLM-specific failure. A nonpositive difference does not support that hypothesis. CIs condition on fixed calibration proxy estimates. No clean-input likelihood, full PPL, generation or new allocation evaluated.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');event('complete',results=result)

@torch.inference_mode()
def collect():
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    c=validate();event('loading');model,mapping=load_dense();model.eval();dev=next(model.parameters()).device
    ref=read(MAN)['entries'];states=read(CAL)['states'];assert list(mapping)==[r['name'] for r in ref]
    tensors=[(torch.tensor(s['noisy_ids'],device=dev),torch.tensor(s['clean_ids'],device=dev),torch.tensor(s['mask'],device=dev,dtype=torch.bool)[0],s['p_mask']) for s in states]
    unique={s['sequence_index']:t[1] for s,t in zip(states,tensors)}
    # Only seven projection covariances resident at a time.
    for block in range(32):
        path=ROOT/'clean'/f'{block:02d}.json'
        if path.exists():assert read(path)['config_sha256']==sha(ROOT/'config.json');continue
        names=list(mapping)[block*7:block*7+7];H={n:torch.zeros((mapping[n].weight.shape[1],)*2,device=dev) for n in names};handles=[]
        for n in names:
            def hook(mod,inp,out,key=n):
                x=inp[0];assert x.ndim==3 and x.shape[0]==1
                a=x.reshape(-1,x.shape[-1]).float();H[key].add_(a.T@a,alpha=2/8)
            handles.append(mapping[n].register_forward_hook(hook))
        try:
            for q in sorted(unique):logits(model,unique[q])
        finally:
            for h in handles:h.remove()
        rows=[dict(name=n,metric=float(lsa_metric(mapping[n].weight,H[n]))) for n in names]
        assert all(math.isfinite(r['metric']) for r in rows)
        write(path,dict(config_sha256=sha(ROOT/'config.json'),rows=rows));del H;event('clean_statistics',completed=block+1,total=32)
    # CE uses masked gold labels; dense background and physical smoke validation.
    base=[loss(logits(model,ids),target,pos,p) for ids,target,pos,p in tensors]
    bp=ROOT/'dense_losses.json'
    if bp.exists():assert np.allclose(read(bp)['losses'],base,rtol=0,atol=1e-7)
    else:write(bp,dict(config_sha256=sha(ROOT/'config.json'),losses=base))
    for i,r in enumerate(ref):
        path=ROOT/'damage'/f'{i:03d}.json'
        if path.exists():assert read(path)['config_sha256']==sha(ROOT/'config.json');continue
        mod=mapping[r['name']];saved=mod.weight.detach().clone();packed=torch.load(r['masks'][0]['path'],map_location='cpu',weights_only=False)
        assert mask_sha256(packed)==r['masks'][0]['mask_sha256'];mask=unpack_mask(packed).to(dev);assert mask.shape==saved.shape
        sparse=saved.masked_fill(mask,0);calls=[0]
        def replace(module,inp,out):
            calls[0]+=1;return F.linear(inp[0],sparse,module.bias)
        handle=mod.register_forward_hook(replace)
        try:
            values=[loss(logits(model,ids),target,pos,p)-b for (ids,target,pos,p),b in zip(tensors,base)]
        finally:handle.remove()
        assert calls[0]==80 and torch.equal(mod.weight,saved)
        # Each module independently checked against actual physical weight pruning on one state.
        ids,target,pos,p=tensors[0]
        try:
            mod.weight.copy_(sparse);physical=loss(logits(model,ids),target,pos,p)-base[0]
        finally:mod.weight.copy_(saved)
        assert abs(physical-values[0])<1e-6,(r['name'],physical,values[0])
        restored=loss(logits(model,ids),target,pos,p);assert abs(restored-base[0])<1e-7
        write(path,dict(config_sha256=sha(ROOT/'config.json'),name=r['name'],mask_sha256=r['masks'][0]['mask_sha256'],delta_ce=values,physical_hook_delta=physical-values[0],restoration_delta=restored-base[0]))
        del saved,sparse,mask;event('damage',completed=i+1,total=224)
    assert model_sha(model)==__import__('experiments.projection_capacity_allocation_65.run',fromlist=['DENSE_SHA']).DENSE_SHA
    analyze()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','collect','analyze']);a=p.parse_args()
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: {'freeze':freeze,'collect':collect,'analyze':analyze}[a.action]()
        except BaseException as e:event('failed',error=repr(e));raise
