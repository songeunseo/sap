"""Paired-context response distortion; fixed-budget block allocation."""
import numpy as np
import torch
from scipy.stats import rankdata

def make_pairs(states, mask_id, seed=2026):
    pairs=[]
    for i,s in enumerate(states):
        before=torch.tensor(s['noisy_ids'],dtype=torch.long)
        clean=torch.tensor(s['clean_ids'],dtype=torch.long)
        masked=before[0].eq(mask_id).nonzero().flatten()
        if len(masked)<2: raise ValueError('Need masked query and context')
        g=torch.Generator().manual_seed(seed+i)
        n=min(max(1,round(before.shape[1]*.05)),max(1,len(masked)//4))
        reveal=masked[torch.randperm(len(masked),generator=g)[:n]]
        after=before.clone();after[0,reveal]=clean[0,reveal]
        query=after[0].eq(mask_id).nonzero().flatten()
        assert len(query)>0 and torch.all(before[0,query]==mask_id)
        assert not torch.isin(query,reveal).any()
        assert int((before!=after).sum())==n
        pairs.append(dict(index=i,sequence_index=s['sequence_index'],p_mask=s['p_mask'],
                          before=before.tolist(),after=after.tolist(),query=query.tolist(),
                          gold=clean[0,query].tolist(),reveal=reveal.tolist()))
    return pairs

def log_odds(logits, gold):
    z=logits.float().clone()
    y=torch.as_tensor(gold,device=z.device,dtype=torch.long)
    own=z.gather(1,y[:,None]).squeeze(1)
    z.scatter_(1,y[:,None],-torch.inf)
    out=own-torch.logsumexp(z,dim=1)
    if not torch.isfinite(out).all(): raise ValueError('Nonfinite log odds')
    return out

def distortion(pred, teacher):
    e=(pred.double()-teacher.double())
    if e.ndim!=2 or e.shape[0]!=2: raise ValueError('Expected 2 x query')
    a=float(e.square().mean())
    c=float((e[1]-e[0]).square().mean())
    return dict(A=a,C=c,AC=a+c)

def rank_rates(scores):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.shape!=(32,) or not np.isfinite(scores).all(): raise ValueError('32 finite scores')
    # A 10pp full span: half-width .05; equal block parameter counts checked by caller.
    ranks=(rankdata(scores,method='average')-1)/31
    return .5-.10*(ranks-ranks.mean())
