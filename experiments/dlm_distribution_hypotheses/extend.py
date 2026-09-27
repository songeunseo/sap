"""Boundary and routing audit, with explicit input identity checks."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from experiments.dlm_distribution_hypotheses.analyze import ROOT, REPO, OLD, read, sha, write

torch.set_num_threads(4)
cfg=read(REPO/'experiments/dlm_lsa_mask_mismatch/config.json')
state_path=REPO/'experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json'
states=read(state_path)['states']
assert sha(state_path)==cfg['sources'][str(state_path)]
assert sha(state_path)==read(OLD/'config.json')['sources'][str(state_path)]
receipt=read(OLD/'input_collection.json');tensor=torch.load(receipt['path'],map_location='cpu',weights_only=False)
for name,c in tensor['corrupted'].items():
    si=np.array(c['sequence_indices']);ts=np.array(c['mask_probabilities'])
    assert [s['sequence_index'] for s in states]==si.tolist()
    assert np.allclose([s['p_mask'] for s in states],ts)
    assert len(np.unique(si))==8 and len(np.unique(ts))==10
    assert all(np.sum((si==s)&(ts==t))==1 for s in np.unique(si) for t in np.unique(ts))
for layer in range(32):
    names=[f'block_{layer:02}.{t}' for t in ['q_proj','k_proj','v_proj']]
    assert all(torch.equal(tensor['corrupted'][names[0]]['state_energy'],tensor['corrupted'][n]['state_energy']) for n in names[1:])
    names=[f'block_{layer:02}.{t}' for t in ['ff_proj','up_proj']]
    assert torch.equal(tensor['corrupted'][names[0]]['state_energy'],tensor['corrupted'][names[1]]['state_energy'])
rows=read(OLD/'statistics.json')['rows'];lookup={r['name']:r for r in rows}
extra=[]
for r in rows:
    s,w,a=r['score_dense'],r['weight'],r['channel_rms_dense']
    extra.append(dict(name=r['name'],layer=r['layer'],type=r['type'],
        energy_alignment=(s['rms']/(w['rms']*a['rms']))**2,
        log_variance_remainder=s['positive_log_variance']-w['positive_log_variance']-a['positive_log_variance'],
        weight_zero_fraction=w['zero_fraction']))
boundary={};rank_path=OLD/'ranking_receipt.json';change_hashes={}
for receipt_row in read(rank_path)['rows']:
    if 'change_path' not in receipt_row:continue
    name=receipt_row['name'];p=Path(receipt_row['change_path'])
    change_hashes[str(p)]=sha(p);assert change_hashes[str(p)]==receipt_row['change_sha256']
    v=torch.load(p,map_location='cpu',weights_only=False)
    energy=tensor['corrupted'][name]['mean_energy'].double().numpy()
    r=lookup[name];out={}
    for action in ['prune','protect']:
        idx=v[action+'_indices'].numpy();values=v[action+'_values'].double().numpy()
        assert len(idx)==786432 and len(np.unique(idx))==len(idx)
        score=np.abs(values)*np.sqrt(energy[idx%r['shape'][1]])
        out[action]=dict(count=len(idx),mean_score=float(score.mean()),score_sq_sum=float(np.dot(score,score)),
            above_dense_5mean_fraction=float(np.mean(score>5*r['score_dense']['mean_abs'])),
            mean_over_dense_mean=float(score.mean()/r['score_dense']['mean_abs']),
            dense_total_score_energy_fraction=float(np.dot(score,score)/(r['weights']*r['score_dense']['rms']**2)))
    boundary[name]=out
ex=read(OLD/'results.json');exchange=[]
for p in ex['pairs']:
    a,b=boundary[p['prune_by_raw']],boundary[p['prune_by_shape']]
    raw=a['prune']['score_sq_sum']-b['protect']['score_sq_sum']
    shape=b['prune']['score_sq_sum']-a['protect']['score_sq_sum']
    exchange.append(dict(pair=p['id'],stratum=p['stratum'],raw_minus_shape_dense_boundary_energy=raw-shape,
        raw_minus_shape_nelbo=p['raw_minus_shape'],same_direction=bool((raw-shape)*p['raw_minus_shape']>0)))
old_path=REPO/'experiments/dlm_super_outlier_statistics/projection_statistics.csv'
old_rows=list(csv.DictReader(old_path.open()))
old_summary={t:{k:[float(np.mean([float(r[k]) for r in old_rows if r['projection_type']==t and int(r['layer'])//8==q and r[k]]))
    for q in range(4)] for k in ['input_super_share','read_super_component_over_output','read_cross_over_output','input_excluded_effective_fraction']}
    for t in ['q_proj','k_proj','v_proj','ff_proj','up_proj']}
types=sorted({r['type'] for r in rows})
align_summary={t:[float(np.mean([r['energy_alignment'] for r in extra if r['type']==t and r['layer']//8==q])) for q in range(4)] for t in types}
result=dict(projections=boundary,exchanges=exchange,energy_alignment=extra,energy_alignment_quartiles=align_summary,
    old_read_decomposition=old_summary,input_identity_verified=True,shared_qkv_and_mlp_inputs_verified=True,
    caveat='Dense activation weighted energy of actual sparse-prefix-selected boundary weights; excludes off-diagonal reconstruction and downstream propagation. Selected12 pairs only. Alignment is coordinate-wise second-moment weighting, not output reconstruction or semantic importance.',
    source_sha256={str(p):sha(p) for p in [Path(__file__),state_path,rank_path,old_path]},change_sha256=change_hashes)
write('boundary_statistics.json',result)
print('alignment',align_summary)
print('old read decomposition',old_summary)
print('boundary ranges', {k:[min(r[a][k] for r in boundary.values() for a in ['prune','protect']),max(r[a][k] for r in boundary.values() for a in ['prune','protect'])] for k in ['above_dense_5mean_fraction','mean_over_dense_mean','dense_total_score_energy_fraction']})
print('energy vs NELBO',exchange)
print('DONE')
