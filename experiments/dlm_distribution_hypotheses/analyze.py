"""Exploratory CPU audit of frozen distributions; no model execution or policy fitting."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / 'experiments/dlm_scale_shape50'
torch.set_num_threads(4)


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def rho(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def energy_features(e):
    p = e / e.sum(-1, keepdims=True)
    s = np.sort(p, axis=-1)
    return dict(effective_fraction=1 / (p.shape[-1] * (p*p).sum(-1)),
                top1=s[..., -1], top1pct=s[..., -max(1, int(np.ceil(p.shape[-1]*.01))):].sum(-1))


def main():
    sources = [OLD / 'statistics.json', OLD / 'input_collection.json', OLD / 'results.json',
               REPO / 'experiments/dlm_lsa_mask_mismatch/config.json',
               REPO / 'experiments/dlm_ppl50_mechanism_review/projection_allocations.csv',
               REPO / 'experiments/dlm_allocation_backtrace/projection_features.csv']
    receipt = read(OLD / 'input_collection.json')
    source = Path(receipt['path'])
    assert sha(source) == receipt['sha256']
    tensor = torch.load(source, map_location='cpu', weights_only=False)
    rows = read(OLD / 'statistics.json')['rows']
    assert len(rows) == 224
    damage_files = sorted((REPO / 'experiments/dlm_lsa_mask_mismatch/damage').glob('*.json'))
    damage = {d['name']: d for d in map(read, damage_files)}
    assert len(damage) == 224
    cfg = read(REPO / 'experiments/dlm_lsa_mask_mismatch/config.json')
    print('state schema', str(cfg['states'])[:500], flush=True)
    flat, sequence_detail = [], []
    for r in rows:
        name = r['name']
        c, u = tensor['corrupted'][name], tensor['clean'][name]
        e = c['state_energy'].double().numpy()
        clean = u['state_energy'].double().numpy()
        si, ts = np.array(c['sequence_indices']), np.array(c['mask_probabilities'])
        us = np.array(u['sequence_indices'])
        assert e.shape[0] == 80 and clean.shape[0] == 8
        assert np.all(e > 0) and np.all(clean > 0)
        if isinstance(cfg['states'], list) and isinstance(cfg['states'][0], dict):
            assert [s['sequence_index'] for s in cfg['states']] == si.tolist()
            assert np.allclose([s['p_mask'] for s in cfg['states']], ts)
        mu = e.mean(0)
        assert np.allclose(mu, c['mean_energy'].double().numpy(), rtol=1e-6)
        ef, cf = energy_features(mu), energy_features(clean.mean(0))
        p, pc = mu/mu.sum(), clean.mean(0)/clean.mean(0).sum()
        f = dict(name=name, layer=r['layer'], type=r['type'], weights=r['weights'],
                 damage_ce=float(np.mean(damage[name]['delta_ce'])),
                 channel_tv_clean_corrupt=float(np.abs(p-pc).sum()/2),
                 channel_rank_clean_corrupt=rho(mu, clean.mean(0)),
                 channel_energy_ratio=float(mu.sum()/clean.mean(0).sum()),
                 channel_effective_fraction=float(ef['effective_fraction']),
                 clean_channel_effective_fraction=float(cf['effective_fraction']),
                 channel_top1=float(ef['top1']), channel_top1pct=float(ef['top1pct']),
                 channel_top1pct_clean=float(cf['top1pct']))
        # A fixed channel energy profile plus multiplicative state scale would give zero.
        pp = e/e.sum(1, keepdims=True)
        f['state_profile_tv'] = float(np.abs(pp-pp.mean(0)).sum(1).mean()/2)
        clean_pp = clean/clean.sum(1, keepdims=True)
        f['clean_sequence_profile_tv'] = float(np.abs(clean_pp-clean_pp.mean(0)).sum(1).mean()/2)
        seq_tvs, seq_neff_ratios, seq_cos = [], [], []
        for j, seq in enumerate(us):
            mean = e[si == seq].mean(0)
            ps, cs = mean/mean.sum(), clean[j]/clean[j].sum()
            tv = float(np.abs(ps-cs).sum()/2)
            ratio = float(energy_features(mean)['effective_fraction']/energy_features(clean[j])['effective_fraction'])
            seq_tvs.append(tv);seq_neff_ratios.append(ratio)
            seq_cos.append(float(np.dot(ps, cs)/(np.linalg.norm(ps)*np.linalg.norm(cs))))
            sequence_detail.append(dict(name=name, sequence=int(seq), channel_tv=tv, effective_fraction_ratio=ratio))
        f['paired_sequence_tv'] = float(np.mean(seq_tvs))
        f['paired_sequence_neff_ratio_median'] = float(np.median(seq_neff_ratios))
        f['paired_sequence_neff_ratio_min'] = float(np.min(seq_neff_ratios))
        f['paired_sequence_neff_ratio_max'] = float(np.max(seq_neff_ratios))
        # Balanced two-way decomposition of log channel energy, after state total removal.
        # Descriptive sums of squares, not causal variance or a proposed timestep proxy.
        z = np.log(pp); z -= z.mean(0)
        denom = float((z*z).sum())
        for label, groups in [('sequence', si), ('mask', ts)]:
            component = np.zeros_like(z)
            for g in np.unique(groups):
                component[groups == g] = z[groups == g].mean(0)
            f['profile_log_ss_' + label] = float((component*component).sum()/denom)
        f['profile_log_ss_remaining'] = 1-f['profile_log_ss_sequence']-f['profile_log_ss_mask']
        low, high = e[ts == ts.min()].mean(0), e[ts == ts.max()].mean(0)
        f['low_high_profile_tv'] = float(np.abs(low/low.sum()-high/high.sum()).sum()/2)
        f['low_high_neff_ratio'] = float(energy_features(high)['effective_fraction']/energy_features(low)['effective_fraction'])
        for field, prefix in [('weight','w'), ('channel_rms_dense','a'), ('score_dense','s'),
                              ('score_clean','sc'), ('activation_samples_dense','x')]:
            d = r[field]
            for k in ['mean_abs','rms','positive_log_variance','zero_fraction','outlier5_fraction','outlier5_energy_fraction']:
                f[prefix+'_'+k] = d[k]
            for i, q in enumerate(['p01','p10','p50','p90','p99','p999']):
                f[prefix+'_'+q] = d['quantiles_abs'][i]
            f[prefix+'_p99_over_p50'] = d['quantiles_abs'][4]/d['quantiles_abs'][2]
            f[prefix+'_p50_over_mean'] = d['quantiles_abs'][2]/d['mean_abs']
        flat.append(f)
    with (ROOT/'projection_features.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write('sequence_details.json',sequence_detail)
    types=sorted({r['type'] for r in flat})
    metrics=[k for k in flat[0] if k not in ('name','type','layer','weights')]
    summaries={}
    for typ in ['ALL']+types:
        subset=[r for r in flat if typ=='ALL' or r['type']==typ]
        summaries[typ]={}
        for key in metrics:
            summaries[typ][key]=dict(quartiles=[float(np.mean([r[key] for r in subset if r['layer']//8==q])) for q in range(4)],
                                    depth_rho=rho([r['layer'] for r in subset],[r[key] for r in subset]))
    # Type fixed effects + flexible depth fixed effects, descriptive partial ranks only.
    layers=np.array([r['layer'] for r in flat]);typv=np.array([r['type'] for r in flat])
    design=np.column_stack([np.ones(224)]+[(layers==l).astype(float) for l in range(1,32)]+
                           [(typv==t).astype(float) for t in types[1:]])
    qmat=np.linalg.qr(design)[0]
    resid=lambda v: rankdata(v)-qmat@(qmat.T@rankdata(v))
    target=np.array([r['damage_ce'] for r in flat]);target_res=resid(target)
    correlations={}
    for key in metrics:
        if key=='damage_ce':continue
        x=np.array([r[key] for r in flat]);xr=resid(x)
        correlations[key]=dict(pooled_rho=rho(x,target),
            layer_type_partial_rank=float(np.corrcoef(xr,target_res)[0,1]) if np.std(xr)>1e-10 else None,
            by_type={t:rho(x[typv==t],target[typv==t]) for t in types})
    # No feature selection from these exploratory correlations; every statistic is exported.
    allocations=list(csv.DictReader((REPO/'experiments/dlm_ppl50_mechanism_review/projection_allocations.csv').open()))
    allocation_summary={}
    for method in sorted({r['method'] for r in allocations}):
        rr=[r for r in allocations if r['method']==method]
        assert len(rr)==224
        allocation_summary[method]={t:[float(np.mean([float(r['sparsity']) for r in rr if r['projection_type']==t and int(r['layer'])//8==q])) for q in range(4)] for t in types}
    # Robustness of the small exchange set: pair-specific article signs, leave-one-pair means.
    ex=read(OLD/'results.json');exchange={}
    for st in ex['strata']:
        pairs=[p for p in ex['pairs'] if p['stratum']==st]
        vals=np.array([p['raw_minus_shape'] for p in pairs])
        exchange[st]=dict(n=len(vals),values=vals.tolist(),leave_one_pair_out=[float(np.delete(vals,i).mean()) for i in range(len(vals))])
    result=dict(scope='Exploratory reuse, no model forwards; no p-value selection or confirmed novelty',
                created=datetime.now().isoformat(),summaries=summaries,damage_correlations=correlations,
                allocation_by_type_quartile=allocation_summary,exchange_robustness=exchange,
                source_sha256={str(p):sha(p) for p in sources+damage_files},
                tensor_sha256=receipt['sha256'],script_sha256=sha(Path(__file__)))
    write('results.json',result)
    # Scientific static figures use equal-projection aggregation; types separate coordinate systems.
    fig, axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    specs=[('s_mean_abs','Mean Wanda score',True),('s_p50_over_mean','Median / mean Wanda score (sample median)',False),
           ('s_outlier5_energy_fraction','Score energy above 5 x mean (exact)',False),
           ('channel_effective_fraction','Effective channel fraction (energy participation)',True),
           ('channel_tv_clean_corrupt','Clean-corrupted normalized channel energy TV',False),
           ('paired_sequence_neff_ratio_median','Within-sequence corrupted / clean effective fraction',True)]
    for ax,(key,title,log) in zip(axes.flat,specs):
        for typ in types:
            rr=sorted([r for r in flat if r['type']==typ],key=lambda r:r['layer'])
            ax.plot([r['layer'] for r in rr],[r[key] for r in rr],label=typ,lw=1.5)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Block');ax.grid(alpha=.2)
        if log:ax.set_yscale('log')
    axes.flat[-1].legend(fontsize=7,ncol=2)
    fig.savefig(ROOT/'distribution_structure.png',dpi=160);plt.close(fig)
    for typ in types:
        print('\nTYPE',typ,flush=True)
        for key in ['s_mean_abs','s_p50_over_mean','s_outlier5_fraction','s_outlier5_energy_fraction',
                    'channel_effective_fraction','channel_top1','channel_top1pct','channel_tv_clean_corrupt',
                    'paired_sequence_neff_ratio_median','low_high_profile_tv','profile_log_ss_mask']:
            print(key, np.round(summaries[typ][key]['quartiles'],5),flush=True)
    print('DONE',flush=True)


if __name__=='__main__':
    main()
