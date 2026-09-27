"""Frozen descriptive statistics and exact-budget disagreement selection."""
import itertools
import numpy as np
import torch
from scipy.stats import rankdata


def summarize_tensor(value, seed, sample_size=131072):
    x = value.detach().float().flatten()
    if not torch.isfinite(x).all() or x.numel() < 2:
        raise ValueError('Finite nontrivial observations required')
    a = x.abs()
    positive = a[a > 0].double()
    logx = positive.log()
    g = torch.Generator(device='cpu').manual_seed(seed)
    if x.numel() <= sample_size:
        sample = a.cpu()
    else:
        index = torch.randint(x.numel(), (sample_size,), generator=g).to(x.device)
        sample = a[index].cpu()
    quantiles = [.01, .1, .5, .9, .99, .999]
    mean = float(a.double().mean())
    edge = np.linspace(-12., 6., 145)
    normedge = np.linspace(-8., 4., 145)
    sample_np = sample.double().numpy()
    logs = np.log10(sample_np[sample_np > 0])
    normlogs = logs - np.log10(mean) if mean > 0 else logs
    return dict(count=x.numel(), mean_abs=mean, signed_mean=float(x.double().mean()),
        rms=float(x.double().square().mean().sqrt()), zero_fraction=float((a == 0).double().mean()),
        max_abs=float(a.max()), positive_log_mean=float(logx.mean()) if len(logx) else None,
        positive_log_variance=float(logx.var(unbiased=True)) if len(logx)>1 else None,
        outlier5_fraction=float((a > mean*5).double().mean()),
        outlier5_energy_fraction=float(a[a>mean*5].double().square().sum()/a.double().square().sum()) if bool((a>0).any()) else 0.,
        quantile_probabilities=quantiles, quantiles_abs=torch.quantile(sample.double(), torch.tensor(quantiles,dtype=torch.float64)).tolist(),
        quantile_hist_sample_count=len(sample), quantile_hist_sampling='all values if <=131072; otherwise deterministic uniform with replacement',
        log10_abs_edges=edge.tolist(), log10_abs_hist=np.histogram(logs, bins=edge)[0].tolist(),
        log10_abs_underflow=int((logs<edge[0]).sum()), log10_abs_overflow=int((logs>edge[-1]).sum()),
        log10_mean_normalized_edges=normedge.tolist(), log10_mean_normalized_hist=np.histogram(normlogs,bins=normedge)[0].tolist(),
        normalized_underflow=int((normlogs<normedge[0]).sum()), normalized_overflow=int((normlogs>normedge[-1]).sum()))


def select_pairs(rows, per_stratum=4, move=786432):
    """Feature-only deterministic selection; positive means raw and shape disagree."""
    raw = np.array([r['score_dense']['mean_abs'] for r in rows])
    shape = np.array([r['score_dense']['positive_log_variance'] for r in rows])
    if not np.isfinite(raw).all() or not np.isfinite(shape).all():
        raise ValueError('Undefined statistics')
    # Type-conditional percentiles prevent the score used to select pairs being
    # dominated by a projection type's different numerical units.
    qr, qs = np.zeros(len(rows)), np.zeros(len(rows))
    for typ in sorted({r['type'] for r in rows}):
        ids = [i for i,r in enumerate(rows) if r['type']==typ]
        qr[ids]=rankdata(raw[ids])/len(ids);qs[ids]=rankdata(shape[ids])/len(ids)
    used = set(); output=[]
    for stratum in ('cross_depth_same_type','near_depth_same_type','same_layer_same_shape'):
        pool=[]
        for i,j in itertools.combinations(range(len(rows)),2):
            a,b=rows[i],rows[j]
            if a['shape'] != b['shape'] or move % a['shape'][0]: continue
            if (raw[i]-raw[j])*(shape[i]-shape[j]) >= 0: continue
            if stratum=='cross_depth_same_type':
                eligible=a['type']==b['type'] and min(a['layer'],b['layer'])<8 and max(a['layer'],b['layer'])>=24
            elif stratum=='near_depth_same_type':
                eligible=a['type']==b['type'] and 0<abs(a['layer']-b['layer'])<=3 and a['layer']//8==b['layer']//8
            else:
                eligible=a['layer']==b['layer'] and a['type']!=b['type']
            if not eligible: continue
            # Within-layer cross-type contrasts use relative, dimensionless gaps.
            strength=(abs(raw[i]-raw[j])/(raw[i]+raw[j]))*(abs(shape[i]-shape[j])/(shape[i]+shape[j])) if stratum=='same_layer_same_shape' else abs(qr[i]-qr[j])*abs(qs[i]-qs[j])
            first,second=(i,j) if raw[i]>raw[j] else (j,i)
            pool.append((-strength,rows[first]['name'],rows[second]['name'],first,second))
        selected=[]
        for neg,_,__,i,j in sorted(pool):
            if i in used or j in used: continue
            a,b=rows[i],rows[j]
            d=move//a['shape'][0];k=a['shape'][1]//2
            if not 0<k-d<k+d<a['shape'][1]: raise ValueError('Invalid exchange range')
            record=dict(id=f'pair_{len(output)+len(selected):02d}',stratum=stratum,
                        prune_by_raw=a['name'],prune_by_shape=b['name'],move_weights=move,
                        rows=a['shape'][0],width=a['shape'][1],row_delta=d,
                        raw_values=[float(raw[i]),float(raw[j])],shape_values=[float(shape[i]),float(shape[j])],
                        selection_strength=-neg)
            selected.append(record);used.update([i,j])
            if len(selected)==per_stratum:break
        if len(selected)!=per_stratum:
            raise RuntimeError(f'{stratum}: only {len(selected)} eligible disjoint pairs; stop before loss evaluation')
        output.extend(selected)
    return output


def boundary_changes(weight, order, move):
    rows,width=weight.shape
    if move%rows: raise ValueError('Move must fit row budgets')
    d=move//rows;k=width//2
    if not 0<k-d<k+d<width:raise ValueError('Invalid row range')
    offsets=torch.arange(rows,device=order.device)[:,None]*width
    protect=(offsets+order[:,k-d:k]).flatten()
    prune=(offsets+order[:,k:k+d]).flatten()
    flat=weight.flatten()
    return dict(protect_indices=protect.cpu(),prune_indices=prune.cpu(),
                protect_values=flat[protect].cpu(),prune_values=flat[prune].cpu(),
                row_delta=d,move_weights=move,shape=list(weight.shape))


def apply_change(weight, change, action, restore=False):
    idx=change[action+'_indices'].to(weight.device)
    values=change[action+'_values'].to(weight.device)
    fill_values=(action=='protect') != restore
    if fill_values:weight.flatten()[idx]=values
    else:weight.flatten()[idx]=0
