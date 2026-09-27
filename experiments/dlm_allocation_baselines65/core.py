"""Commit-pinned baseline math. Preserve upstream quirks; never tune on outcomes."""
import math
import numpy as np
import torch


def centered_density(importance, width, target=.65):
    a=np.asarray(importance,dtype=np.float64)
    if not np.isfinite(a).all() or np.ptp(a)==0:raise ValueError('undefined minmax allocation')
    a=(a-a.min())*(1/(a.max()-a.min())*width*2)
    return 1-(a-np.mean(a)+(1-target))


def dlp_rates(means,target=.65,alpha=.15):
    # get_dlp_ratios uses mean, not the separate structure function's median.
    values=[torch.tensor(x,dtype=torch.float32) for x in means]
    total=sum(values)
    r=torch.tensor([1-x/total for x in values])
    z=(r-r.min())*(1/(r.max()-r.min())*alpha*2)
    return (1-(z-z.mean()+(1-target))).double().numpy()


def dsa_score(pooled):
    # README graph W:(ABSLOG)-(VAR)-(ATAN,ASIN)-(7), applied to WANDA scores.
    x=pooled.clone();x[x==0]=1
    v=torch.var(torch.log(torch.abs(x)))
    a=torch.atan(v);y=torch.asin(a)
    # Official compute_importance maps NaN to zero. Persist this, don't clip asin.
    value=0. if math.isnan(float(y)) else (-1. if math.isinf(float(y)) else float(y))
    return dict(value=value,log_variance=float(v),atan=float(a),nan_mapped_to_zero=bool(torch.isnan(y)))


def dsa_rates(values,target=.65):
    v=np.asarray(values,dtype=np.float64)
    if not np.isfinite(v).all() or np.ptp(v)==0:raise ValueError('DSA fixed graph cannot define allocation')
    v=(v-v.min())/(v.max()-v.min())
    # CLI uses args.Lamda=.08; graph's 7 isn't consulted by allocation caller.
    return centered_density(1-v,.08,target)


def alpha_from_eigs(eigs):
    # Literal xmin_peak estimator, including upstream log10 lower-bound comparison.
    e=torch.sort(eigs.float().cpu().flatten())[0]
    if not torch.isfinite(e).all() or (e<=0).any():raise ValueError('nonpositive spectrum')
    n=len(e);logs=torch.log(e);alphas=torch.zeros(n-1);ds=torch.ones(n-1)
    hist=torch.log10(e);lo,hi=hist.min(),hist.max()
    counts=torch.histc(hist,100,min=lo,max=hi);edges=torch.linspace(lo,hi,101)
    peak=10**edges[torch.argmax(counts)]
    xmin_min=torch.log10(.95*peak);xmin_max=1.5*peak
    for i,xmin in enumerate(e[:-1]):
        if xmin<xmin_min:continue
        if xmin>xmin_max:break
        count=float(n-i);seq=torch.arange(count)
        a=1+count/(torch.sum(logs[i:])-count*logs[i]);alphas[i]=a
        if a>1:ds[i]=torch.max(torch.abs(1-(e[i:]/xmin)**(-a+1)-seq/count))
    idx=int(torch.argmin(ds));a=float(alphas[idx])
    if not math.isfinite(a) or a<=1:raise ValueError('upstream alpha_peak fit invalid')
    return dict(alpha=a,D=float(ds[idx]),spectral_norm=float(e[-1]),fit_index=idx)


def alpha_rates(alphas,weights,target=.65,epsilon=.3):
    a=np.asarray(alphas).reshape(32,7).mean(1).repeat(7)
    if not np.isfinite(a).all() or np.ptp(a)==0:raise ValueError('undefined alpha mapping')
    a=torch.tensor(a);n=torch.tensor(weights)
    r=(a-a.min())/(a.max()-a.min())*(2*epsilon)+(1-epsilon)
    r=r*(torch.sum(n)*target/torch.sum(n*r))
    return r.numpy().reshape(32,7)[:,0]


def lsa_metric(weight,H,block_size=128,s=.5):
    # Port of BlockWanda.blk_s. Weights never modified; score updates are intentional.
    w=weight.float();sx=H.float();co,ci=w.shape
    score=(w**2)*torch.diag(sx);rec=0.
    for i1 in range(0,ci,block_size):
        i2=min(i1+block_size,ci);w1=w[:,i1:i2];w2=w[:,i2:]
        score1=score[:,i1:i2];score2=score[:,i2:]
        sx1=sx[i1:i2,i1:i2];sx2=sx[i1:i2,i2:];err=torch.zeros_like(w1)
        for _ in range(int(block_size*s)):
            idx=torch.argmin(score1,dim=1).unsqueeze(1)
            rec+=torch.sum(score1.gather(1,idx));v=w1.gather(1,idx)
            change=w1*v*sx1[idx.squeeze(1)]
            score1+=2*change;score1.scatter_(1,idx,torch.inf);err.scatter_(1,idx,v)
        score2+=2*(w2*(err@sx2))
    return rec


def lsa_rates(metrics,target=.65):
    # BLK.get_layer_sp alpha[int(target*10)] = .1 at 65%, basic layer='lsa'.
    v=torch.tensor(metrics,dtype=torch.float32).reshape(32,7).mean(1).abs()
    z=1-v/v.sum();z=(z-z.min())/(z.max()-z.min())*.1*2
    return (target+z.mean()-z).double().numpy()


def validate_rates(rates):
    s=np.asarray(rates)
    if s.shape!=(32,) or not np.isfinite(s).all() or (s<=0).any() or (s>=1).any():
        raise ValueError('invalid baseline rates; do not silently clip or retune')
    return s
