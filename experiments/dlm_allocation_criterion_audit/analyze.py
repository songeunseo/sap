"""Read-only source audit and exploratory criterion diagnostics; no new masks."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXP = REPO / 'experiments'
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    SOURCES[str(path)] = sha(path)
    return json.loads(path.read_text())


def rho(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def partial(x, y, controls):
    x, y = rankdata(x), rankdata(y)
    design = np.column_stack([np.ones(len(x)), controls])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.linalg.norm(rx) < 1e-8 or np.linalg.norm(ry) < 1e-8:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def rows(path):
    SOURCES[str(path)] = sha(path)
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    from experiments.projection_capacity_followup_65.core import summarize_curve_records
    base = EXP / 'dlm_allocation_baselines65'
    old = read(base / 'allocation_analysis/analysis.json')
    # Verify original data sources; an analysis script edit is not a data change.
    for p, expected in old['sources'].items():
        if not p.endswith('.py'):
            assert sha(p) == expected, p
    reference = read(EXP / 'projection_capacity_allocation_65/uniform65_mask_manifest.json')['entries']
    names = [r['name'] for r in reference]
    weights = np.array([r['weights'] for r in reference])
    shapes = np.array([r['shape'] for r in reference])
    extra = (np.floor(shapes[:, 1] * .70) - np.floor(shapes[:, 1] * .65)) * shapes[:, 0]
    records = read(EXP / 'projection_capacity_allocation_65/capacity_curves_raw.json')['projections']
    data = summarize_curve_records(records, (.50, .55, .60, .65, .70, .75))
    assert data['names'] == names and len(data['state_keys']) == 80
    increment = data['damage_states'][:, :, 4] - data['damage_states'][:, :, 3]
    marginal = increment.mean(1) / extra
    block_states = increment.reshape(32, 7, 80).sum(1) / extra.reshape(32, 7).sum(1)[:, None]
    block_cost = block_states.mean(1)
    baseline = rows(EXP / 'projection_capacity_allocation_65/gsm8k/uniform_100_predictions.jsonl')
    assert len(baseline) == 100
    manifests = {'Uniform': EXP / 'projection_capacity_allocation_65/uniform65_mask_manifest.json',
                 'Role': EXP / 'dlm_dual_role_mini100/role65_mask_manifest.json',
                 'Aggregate': EXP / 'dlm_dual_role_mini100/aggregate65_mask_manifest.json',
                 'OWL': EXP / 'dlm_owl65/mask_manifest.json',
                 **{m: base / m / 'mask_manifest.json' for m in ('dlp', 'dsa', 'alpha', 'lsa')}}
    predictions = {'Uniform': EXP / 'projection_capacity_allocation_65/gsm8k/uniform_100_predictions.jsonl',
                   'Role': EXP / 'dlm_dual_role_mini100/gsm8k/role_100_predictions.jsonl',
                   'Aggregate': EXP / 'dlm_dual_role_mini100/gsm8k/aggregate_100_predictions.jsonl',
                   'OWL': EXP / 'dlm_owl65/predictions.jsonl',
                   **{m: base / m / 'predictions.jsonl' for m in ('dlp', 'dsa', 'alpha', 'lsa')}}
    methods = {}
    for method, path in manifests.items():
        manifest = read(path)
        entries = manifest['entries']
        assert [r['name'] for r in entries] == names
        counts = np.array([r['selected_mask']['pruned'] for r in entries])
        assert sum(counts) == manifest['pruned'] == 4536008704
        assert sum(weights) == manifest['weights'] == 6979321856
        assert all(r['shape'] == ref['shape'] and r['selected_mask']['prune_per_row'] * r['shape'][0] == r['selected_mask']['pruned'] for r, ref in zip(entries, reference))
        prediction = rows(predictions[method])
        assert len(prediction) == len(baseline)
        for row, ref in zip(prediction, baseline):
            for key in ('example_id', 'doc_hash', 'prompt_hash', 'target_hash', 'evaluation_config_hash'):
                assert row[key] == ref[key], (method, key)
        correct = sum(r['correct'] for r in prediction)
        assert correct == old['methods'][method]['correct']
        if method not in ('Uniform', 'Role', 'Aggregate'):
            folder = EXP / 'dlm_owl65' if method == 'OWL' else base / method
            receipt = read(folder / 'results.json')
            assert receipt['manifest_sha256'] == sha(path)
            assert receipt['predictions_sha256'] == sha(predictions[method])
            assert receipt['config_sha256'] == sha(folder / 'config.json')
            assert receipt['correct'] == correct
        density = 1 - counts.reshape(32, 7).sum(1) / weights.reshape(32, 7).sum(1)
        sequence_rhos = [rho(density, block_states[:, [i for i, key in enumerate(data['state_keys']) if key[0] == seq]].mean(1)) for seq in range(8)]
        methods[method] = {'correct': correct, 'quartile_sparsity': (1-density).reshape(4,8).mean(1).tolist(),
            'density_vs_block_marginal': rho(density, block_cost),
            'density_vs_block_marginal_excluding_B31': rho(density[:31], block_cost[:31]),
            'density_vs_block_marginal_linear_depth_partial': partial(density, block_cost, np.arange(32)),
            'per_sequence_rho': sequence_rhos, 'block_density': density.tolist()}
    dlp = np.array([r['mean'] for r in read(base / 'dlp/statistics.json')['rows']])
    assert rho(dlp, 1-np.array(methods['dlp']['block_density'])) > .999
    stat_path = EXP / 'cgq_wanda_structured_diagnostic/sufficient_statistics.pt'
    SOURCES[str(stat_path)] = sha(stat_path)
    config = read(EXP / 'projection_capacity_allocation_65/config.json')
    assert config['sources'][str(stat_path.relative_to(REPO))] == sha(stat_path)
    stats = torch.load(stat_path, map_location='cpu', weights_only=True)
    calibration = read(EXP / 'dlm_loss_aggregation/calibration_manifest.json')['states']
    times = np.array([next(s['timestep'] for s in calibration if s['timestep_index']==t) for t in range(10)])
    mask_rates = np.array([np.mean([s['p_mask'] for s in calibration if s['timestep_index']==t]) for t in range(10)])
    assert len(calibration)==80 and all(sum(s['timestep_index']==t for s in calibration)==8 for t in range(10))
    assert sorted(set(times))==sorted(set(t for _,t in data['state_keys']))
    temporal = []
    features = []
    for i, name in enumerate(names):
        a = stats['statistics'][name]['overall_uniform'].double().numpy()
        by_t = stats['statistics'][name]['timestep_uniform'].double().numpy()
        assert np.allclose(by_t.mean(0), a, rtol=1e-5, atol=1e-8)
        effective_t = by_t.sum(1)**2 / (len(a)*np.square(by_t).sum(1))
        rms_t = np.sqrt(by_t).mean(1)
        temporal.append({'name':name, 'effective_fraction_by_t':effective_t.tolist(),
            'rms_by_t':rms_t.tolist(), 'mask_vs_effective_rho':rho(mask_rates,effective_t),
            'mask_vs_rms_rho':rho(mask_rates,rms_t)})
        # Effective support of the energy distribution; no channel is removed.
        features.append({'name': name, 'layer': i//7, 'type': name.split('.')[1],
            'energy_per_channel': float(a.mean()),
            'mean_channel_rms': float(np.sqrt(a).mean()),
            'effective_fraction': float(a.sum()**2 / (len(a)*np.square(a).sum())),
            'dominant_share': float(a.max()/a.sum()),
            'timestep_energy_cv': float(by_t.sum(1).std()/by_t.sum(1).mean()),
            'marginal65_70': float(marginal[i])})
    types = np.array([x['type'] for x in features])
    layers = np.arange(224)//7
    controls = np.column_stack([np.eye(32)[layers,1:], np.column_stack([types==t for t in sorted(set(types))[1:]])])
    diagnostics = {}
    for key in ('energy_per_channel','mean_channel_rms','effective_fraction','dominant_share','timestep_energy_cv'):
        values = np.array([r[key] for r in features])
        block_values = (values*weights).reshape(32,7).sum(1)/weights.reshape(32,7).sum(1)
        diagnostics[key] = {'projection_vs_marginal': rho(values,marginal),
            'projection_vs_marginal_layer_type_partial': partial(values,marginal,controls),
            'block_vs_dlp': rho(block_values,dlp),
            'block_vs_marginal':rho(block_values,block_cost),
            'by_type_vs_marginal':{t:rho(values[types==t],marginal[types==t]) for t in sorted(set(types))}}
    dlp_diag = {'mean_score_vs_marginal':rho(dlp,block_cost),
        'mean_score_vs_marginal_linear_depth_partial':partial(dlp,block_cost,np.arange(32)),
        'mean_score_vs_depth':rho(dlp,np.arange(32)),
        'early_late_score_ratio':float(dlp[24:].mean()/dlp[:8].mean()),
        'quartile_marginal':block_cost.reshape(4,8).mean(1).tolist()}
    temporal_summary = {'times':times.tolist(),'mask_rates':mask_rates.tolist(), 'by_type':{}}
    for typ in sorted(set(types)):
        selected = [r for r,t in zip(temporal,types) if t==typ]
        temporal_summary['by_type'][typ] = {
            key:[float(np.median([r[key] for r in selected[q*8:(q+1)*8]])) for q in range(4)]
            for key in ('mask_vs_effective_rho','mask_vs_rms_rho')}
    rms_array = np.array([r['rms_by_t'] for r in temporal])
    block_rms = (rms_array*weights[:,None]).reshape(32,7,10).sum(1)/weights.reshape(32,7).sum(1)[:,None]
    temporal_summary['block_rms_low_high_mask_rho'] = rho(block_rms[:,0],block_rms[:,-1])
    temporal_summary['block_rms_vs_depth_by_t'] = [rho(block_rms[:,t],np.arange(32)) for t in range(10)]
    SOURCES[str(Path(__file__))] = sha(__file__)
    result = {'methods':methods,'dlp':dlp_diag,'activation_diagnostics':diagnostics,'mask_rate_diagnostics':temporal_summary,
        'sources':SOURCES,'source_state_sha256':stats['source_state_sha256'],
        'capacity_measurement_path':config['capacity_path'],
        'limitations':['Post-hoc exploratory; no new downstream or masks.',
            'Block marginal sums individual-projection interventions; it is not measured whole-block damage.',
            'Historical capacity uses batched suffix BF16; not corrected physical batch-one evidence.',
            'Linear depth partial does not remove all nonlinear depth dependence.',
            'No matched clean/AR activations or full-vector token means; cannot establish a DLM-specific mechanism.',
            'Timestep CV is descriptive; no timestep allocation proposal.',
            'Prediction/manifest provenance checked; physical mask payloads not reloaded; strict EM uses saved audited correct flags.']}
    ROOT.mkdir(exist_ok=True)
    (ROOT/'timestep_features.json').write_text(json.dumps(temporal,indent=2,allow_nan=False)+'\n')
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (ROOT/'projection_features.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(features[0]));writer.writeheader();writer.writerows(features)
    with (ROOT/'block_diagnostics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['block','dlp_mean','additive_marginal65_70',*methods])
        for b in range(32):writer.writerow([b,dlp[b],block_cost[b],*[methods[m]['block_density'][b] for m in methods]])
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','methods')},indent=2))
    print(json.dumps({m:{k:v for k,v in d.items() if k not in ('block_density','per_sequence_rho')} for m,d in methods.items()},indent=2))


if __name__ == '__main__':
    main()
