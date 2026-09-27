"""Independent source, identity, formula and paired-article checks."""
import csv
from pathlib import Path
import numpy as np
from experiments.dlm_distribution_hypotheses.analyze import ROOT,REPO,OLD,read,sha,write

result=read(ROOT/'results.json');boundary=read(ROOT/'boundary_statistics.json')
for p,h in result['source_sha256'].items():assert sha(Path(p))==h
assert sha(ROOT/'analyze.py')==result['script_sha256']
for p,h in boundary['source_sha256'].items():assert sha(Path(p))==h
for p,h in boundary['change_sha256'].items():assert sha(Path(p))==h
stats=read(OLD/'statistics.json')['rows']
features={r['name']:r for r in csv.DictReader((ROOT/'projection_features.csv').open())}
assert len(features)==224
for r in stats:
    f=features[r['name']]
    assert np.isclose(float(f['s_p50_over_mean']),r['score_dense']['quantiles_abs'][2]/r['score_dense']['mean_abs'])
    for q in ['p01','p10','p50','p90','p99','p999']:
        assert float(f['s_'+q])>=0
    assert 0<float(f['channel_effective_fraction'])<=1
    assert 0<=float(f['channel_tv_clean_corrupt'])<=1
    assert np.isclose(sum(float(f['profile_log_ss_'+k]) for k in ['mask','sequence','remaining']),1)
    assert float(f['profile_log_ss_remaining'])>=-1e-12
    d=read(REPO/f'experiments/dlm_lsa_mask_mismatch/damage/{stats.index(r):03d}.json')
    assert d['name']==r['name'] and np.isclose(np.mean(d['delta_ce']),float(f['damage_ce']))
cfg=read(OLD/'config.json');ex=read(OLD/'results.json')
ids=[r['block_id'] for r in cfg['evaluation_blocks']]
def load(label):
    rr=[read(p) for p in (OLD/'evaluation'/label).glob('*.json') if p.name!='results.json']
    byid={r['block_id']:r for r in rr};assert len(rr)==len(byid)==16 and set(byid)==set(ids)
    for r in rr:
        assert r['tokens']==512 and len(r['sample_token_nelbo'])==128
        assert np.isclose(np.mean(r['sample_token_nelbo']),r['token_nelbo'],rtol=0,atol=1e-12)
    return np.array([byid[k]['sample_token_nelbo'] for k in ids])
base=load('baseline');delta=[]
for p in ex['pairs']:
    raw,shape=load(p['id']+'_raw'),load(p['id']+'_shape')
    d=(raw-shape).mean(1);delta.append(d)
    assert np.isclose(d.mean(),p['raw_minus_shape'],atol=1e-15,rtol=0)
delta=np.array(delta)
rng=np.random.default_rng(cfg['seed']);draws=rng.integers(0,16,size=(5000,16))
ci=lambda d,alpha:np.quantile(d[draws].mean(1),[alpha/2,1-alpha/2]).tolist()
assert np.allclose(ci(delta.mean(0),.05),ex['primary']['paired_article_ci'],atol=1e-15,rtol=0)
pairs=[]
for i,p in enumerate(ex['pairs']):
    pairs.append(dict(pair=p['id'],mean=float(delta[i].mean()),article_ci95=ci(delta[i],.05),
        article_ci_bonferroni12=ci(delta[i],.05/12),positive_articles=int((delta[i]>0).sum()),
        half_A=float(delta[i,:8].mean()),half_B=float(delta[i,8:].mean())))
loo={}
for st in ex['strata']:
    indices=[i for i,p in enumerate(ex['pairs']) if p['stratum']==st]
    loo[st]=[]
    for omitted in indices:
        d=delta[[i for i in indices if i!=omitted]].mean(0)
        loo[st].append(dict(omitted=ex['pairs'][omitted]['id'],mean=float(d.mean()),article_ci95=ci(d,.05)))
write('pair_uncertainty.json',dict(pairs=pairs,leave_one_pair_out=loo,
    caveat='Exploratory post-hoc intervals conditional on fixed selected pairs and fixed measured features; article resampling only, no new holdout. Pair signs are not all statistically resolved.'))
write('verification.json',dict(status='passed',projection_count=224,source_count=len(result['source_sha256']),
    boundary_file_count=len(boundary['change_sha256']),input_identity=boundary['input_identity_verified'],
    shared_projection_inputs=boundary['shared_qkv_and_mlp_inputs_verified'],
    original_primary_bootstrap_reproduced=True,paired_blocks_verified=400,
    output_sha256={p.name:sha(p) for p in ROOT.iterdir() if p.suffix in ['.json','.csv','.png'] and p.name!='verification.json'},
    script_sha256=sha(Path(__file__))))
print('PASS',pairs,loo)
