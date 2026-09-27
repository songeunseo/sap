"""Single-probe KL pooling and fixed-width sensitivity-to-sparsity mapping."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata


def rank_rates(scores, target=.65, width=.10):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.shape!=(32,) or not np.isfinite(scores).all():raise ValueError('32 finite block scores required')
    ranks=(rankdata(scores,method='average')-1)/31
    rates=target-width*(ranks-ranks.mean())
    assert np.all(rates>=target-width/2-1e-12) and np.all(rates<=target+width/2+1e-12)
    return rates


def select_reveal(logits,ids,mask_id,k):
    """Exact native temp0 low_confidence policy, including original softmax dtype."""
    masked=ids.eq(mask_id)
    if k<1 or k>int(masked.sum()):raise ValueError('invalid reveal count')
    tokens=logits.argmax(-1)
    probs=F.softmax(logits,dim=-1)
    conf=probs.gather(-1,tokens.unsqueeze(-1)).squeeze(-1)
    conf=torch.where(masked,conf,-torch.inf)
    chosen=torch.topk(conf[0],k=k).indices
    return tokens,chosen


def token_kl(dense,pruned,chunk=16):
    if dense.shape!=pruned.shape or dense.ndim!=2:raise ValueError('masked-token logits shape mismatch')
    values=[]
    for d,p in zip(dense.split(chunk),pruned.split(chunk)):
        ld=F.log_softmax(d.float(),dim=-1);lp=F.log_softmax(p.float(),dim=-1)
        values.append((ld.exp()*(ld-lp)).sum(-1))
    out=torch.cat(values)
    if not torch.isfinite(out).all() or float(out.min()) < -2e-6:raise ValueError('invalid KL')
    # Preserve tiny floating-point negatives; no signed CE / fitted-g transformation.
    return out


def pool_scores(rows):
    if len(rows)!=128 or len({(r['sequence_index'],r['step']) for r in rows})!=128:
        raise ValueError('incomplete or duplicate state coverage')
    result={}
    for mode in ('reveal','all_masked'):
        grouped=[]
        for seq in range(8):
            subset=[r for r in rows if r['sequence_index']==seq]
            if len(subset)!=16:raise ValueError('unequal prompt coverage')
            grouped.append(np.mean([r[mode] for r in subset]))
        result[mode]=dict(score=float(np.mean(grouped)),per_prompt=grouped)
    return result
