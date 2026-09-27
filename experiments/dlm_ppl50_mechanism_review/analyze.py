"""Read-only source audit of completed PPL50 allocations; no model forwards."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCES = {}


def read(path):
    path = REPO / path
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    refs = read('experiments/projection_capacity_allocation_65/candidate_mask_manifest.json')['entries']
    names = [r['name'] for r in refs]
    weights = np.array([r['weights'] for r in refs], dtype=np.int64)
    layers = np.array([int(n.split('.')[0].split('_')[1]) for n in names])
    types = [n.split('.')[1] for n in names]
    assert len(names) == len(set(names)) == 224
    assert weights.sum() == 6979321856
    methods = [('uniform', 'Uniform / row quota'), ('owl', 'OWL / layer'),
               ('dlp', 'DLP / layer'), ('alpha', 'Alpha / layer'),
               ('lsa_layer', 'LSA / layer'), ('lsa_projection', 'LSA / projection'),
               ('dsa', 'DSA / layer'), ('evopress', 'EvoPress')]
    locations = [(key, label, f'experiments/dlm_ppl50/{key}') for key, label in methods]
    locations += [(key, label, f'experiments/dlm_ppl50_projection/{key}') for key, label in
                  [('owl_projection', 'OWL / projection'), ('dsa_projection', 'DSA / projection')]]
    locations += [('uniform_layer_global', 'Uniform / layer-global', 'experiments/dlm_ppl50_uniform_layer')]
    all_rows, summaries = [], []
    for method, label, folder in locations:
        manifest = read(folder + '/mask_manifest.json')
        result = read(folder + '/validation/results.json')
        by_name = {r['name']: r for r in manifest['entries']}
        assert len(by_name) == 224 and set(by_name) == set(names)
        counts = []
        for ref in refs:
            row = by_name[ref['name']]
            assert row['shape'] == ref['shape'] and row['weights'] == ref['weights']
            count = row['actual_zeros'] if method == 'evopress' else row['selected_mask']['pruned']
            assert 0 <= count <= row['weights']
            if 'prune_per_row' in row.get('selected_mask', {}):
                assert count == row['selected_mask']['prune_per_row'] * row['shape'][0]
            counts.append(count)
        counts = np.array(counts, dtype=np.int64)
        assert int(counts.sum()) == manifest['pruned']
        assert manifest['weights'] == int(weights.sum())
        assert int(counts.sum()) == (3489662724 if method == 'evopress' else 3489660928)
        s = result['summary']
        assert s['blocks'] == 551 and s['tokens'] == 268163 and result['status'] == 'complete'
        for i, name in enumerate(names):
            all_rows.append(dict(method=method, name=name, layer=int(layers[i]), projection_type=types[i],
                                 weights=int(weights[i]), pruned=int(counts[i]), sparsity=float(counts[i]/weights[i])))
        depth = [float(counts[layers == b].sum()/weights[layers == b].sum()) for b in range(32)]
        quartiles = [float(counts[layers//8 == b].sum()/weights[layers//8 == b].sum()) for b in range(4)]
        type_rates = {t: float(counts[np.array(types) == t].sum()/weights[np.array(types) == t].sum()) for t in sorted(set(types))}
        summaries.append(dict(method=method, label=label, token_nelbo=s['token_nelbo'],
            ppl=s['ppl_upper_bound_estimate'], quartiles=quartiles, layers=depth, types=type_rates,
            late_minus_early_pp=100*(quartiles[-1]-quartiles[0]),
            projection_sparsity_min=float((counts/weights).min()), projection_sparsity_max=float((counts/weights).max())))
    summaries.sort(key=lambda x: x['token_nelbo'])
    save_csv(ROOT/'projection_allocations.csv', all_rows)
    save_csv(ROOT/'summary.csv', [dict(method=r['method'], token_nelbo=r['token_nelbo'],
        ppl=r['ppl'], early8=100*r['quartiles'][0], middle8a=100*r['quartiles'][1],
        middle8b=100*r['quartiles'][2], late8=100*r['quartiles'][3], late_minus_early_pp=r['late_minus_early_pp']) for r in summaries])

    collection = read('experiments/dlm_allocation_backtrace/collection.json')
    prior_analysis = read('experiments/dlm_allocation_backtrace/results.json')
    read('experiments/dlm_lsa_mask_mismatch/results.json')
    dsa_graphs = {name: read(path)['graph'] for name, path in [
        ('layer', 'experiments/dlm_ppl50/dsa/selection.json'),
        ('projection', 'experiments/dlm_ppl50_projection/dsa_projection/selection.json')]}
    features = collection['features']
    assert [x['name'] for x in features] == names
    # Recheck the recorded algebraic factorization, not a statistical fit.
    for f in features:
        assert np.isclose(f['mean_abs_weight']*f['mean_channel_rms']*f['alignment'], f['mean_score'], rtol=1e-10)
    block_features = {}
    for key in ['mean_abs_weight', 'mean_channel_rms', 'mean_score']:
        block_features[key] = [float(np.average([f[key] for f in features if f['layer'] == b],
            weights=[f['weights'] for f in features if f['layer'] == b])) for b in range(32)]
    fig, ax = plt.subplots(figsize=(12.5, 5.7), layout='constrained')
    values = np.array([r['layers'] for r in summaries])*100
    img = ax.imshow(values, aspect='auto', cmap='RdBu_r', vmin=30, vmax=70)
    ax.set_yticks(range(len(summaries)), [r['label'] for r in summaries])
    ax.set_xticks([0,7,15,23,31], ['0','7','15','23','31'])
    for x in [7.5,15.5,23.5]: ax.axvline(x, color='white', alpha=.7, lw=1)
    ax.set_xlabel('Transformer block (0-based)')
    ax.set_title('Actual layer sparsity at nominal 50% | rows ordered by validation NELBO')
    fig.colorbar(img, ax=ax, label='Pruned weights (%)')
    fig.savefig(ROOT/'allocation_depth.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), layout='constrained')
    for key, label in [('mean_abs_weight','Mean |W|'),('mean_channel_rms','Mean input RMS'),('mean_score','Mean Wanda score')]:
        v = np.array(block_features[key]); axes[0].plot(v/np.mean(v[:8]), label=label, lw=2)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Relative to mean of blocks 0-7')
    axes[0].set_title('Scale profiles from frozen dense calibration'); axes[0].legend()
    for method in ['dsa','evopress','uniform','owl','dlp','alpha','lsa_layer']:
        row = next(r for r in summaries if r['method'] == method)
        axes[1].plot(np.arange(32), np.array(row['layers'])*100, label=row['label'], lw=1.6)
    axes[1].axhline(50, color='gray', lw=.8, ls='--'); axes[1].set_ylabel('Pruned weights (%)')
    axes[1].set_title('Allocation direction differs across methods'); axes[1].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.set_xlabel('Transformer block (0-based)'); ax.grid(alpha=.2)
    fig.savefig(ROOT/'scale_and_allocation.png', dpi=180)
    plt.close(fig)
    rowwise = next(r for r in summaries if r['method']=='uniform')
    layerglobal = next(r for r in summaries if r['method']=='uniform_layer_global')
    payload = dict(status='complete', scope='CPU descriptive review, no new model forwards or masks',
        methods=summaries, dsa_graphs=dsa_graphs, block_features=block_features,
        prior_backtrace_scope=prior_analysis['scope'], source_sha256=SOURCES,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limits=['Saved manifest counts verified; mask payloads were not reloaded.',
                'Dense statistics summaries are not raw activation or weight histograms.',
                'Prior 65% damage curves are not 50% jointly-sparse ground truth.',
                'No causal ablation, AR control, or independent task evaluation in this review.'])
    (ROOT/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    table = '\n'.join(f"| {r['label']} | {100*r['quartiles'][0]:.2f}% | {100*r['quartiles'][-1]:.2f}% | {r['late_minus_early_pp']:+.2f} | {r['token_nelbo']:.6f} |" for r in summaries)
    report = f'''# PPL50 allocation mechanism review

## Observed results
Same LLaDA revision, 551 WikiText validation chunks, 268,163 tokens, shared exact-k MC128. Counts and projection identities verified from all 11 saved manifests. EvoPress actual sparsity is 50.0000257%; other candidates remove exactly 50%. This is a descriptive review of completed experiments.

| Method | Blocks 0-7 | Blocks 24-31 | Late minus early (pp) | NELBO |
|---|---:|---:|---:|---:|
{table}

![Actual allocations](allocation_depth.png)
![Scale and allocation](scale_and_allocation.png)

## What the current results change
- Late-heavy pruning is not common to all candidates. DSA layer and EvoPress protect later blocks on average. DSA was independently selected with the bounded DLM development-loss search; it is not the historical fixed DSA graph.
- Layer-global Uniform keeps every layer at 50% but allows projection sparsity {100*layerglobal['projection_sparsity_min']:.2f}–{100*layerglobal['projection_sparsity_max']:.2f}%. Its NELBO {layerglobal['token_nelbo']:.6f} versus row-quota Uniform {rowwise['token_nelbo']:.6f} shows a large difference can occur without changing any layer's total sparsity. Which row/type imbalance causes that difference remains untested.
- Depth direction, per-projection/per-row quota, allocator range, and the pruning engine must be separated. EvoPress uses FastOBC, whereas the Wanda variants use their specified Wanda masks; its advantage cannot be attributed to allocation alone.

## Existing evidence explaining the mapping
- DLP public mean path: larger mean Wanda score gives greater sparsity. The prior audit measured late/early score 7.405x, falling to 1.100x after removing activation RMS scale; depth rank still stayed high. The activation component accounted for 95.68% of the algebraic log-depth slope, not of the causal performance loss.
- OWL: more outliers means more protection. Early blocks have more outliers; channel3848 removal or projection-local thresholds do not erase the trend. Mean-relative thresholds already remove uniform multiplicative scale within a group, so mean activation scale alone is not the OWL explanation.
- LSA: larger pooled basic surrogate gives greater sparsity. The prior 42.23x late/early raw increase falls to 2.76x after output-energy normalization but does not disappear. The current 50% layer mapping uses lambda .04 (8pp ideal range), not the 65% lambda .10 (20pp range).
- Alpha: alpha grows with depth across all seven types; the implemented mapping gives larger alpha more pruning. Spectral tail exponent is not equivalent to stable rank or directly established redundancy.
- DSA layer winner is `{dsa_graphs['layer']}`: log-score dispersion rather than arithmetic score magnitude drives its allocation. For strictly positive scores, variance(log(score)) is invariant to a common multiplicative scale; the public zero substitution breaks exact invariance when zeros occur. DSA projection winner is `{dsa_graphs['projection']}` and also includes epsilon/zero handling. These mathematical properties do not establish why either model performs better or DLM specificity.

## Research questions to pursue
1. **Scale versus shape:** at each layer/type inspect raw and RMS-normalized distributions of |W|, channel RMS, and Wanda score: median/tail quantiles, log dispersion, outlier count AND energy mass, concentration. Keep parameter-weighted and equal-projection summaries separately. Existing files contain summaries, not full histograms; new distribution collection must retain this distinction.
2. **Statistic to decision:** reproduce each score, its sign/normalization, ideal rate and integer rate. Separate mapping direction from mapping width. Distinguish high statistic magnitude from evidence of spare capacity.
3. **DLM-relevant target:** compare a small, frozen set of candidate decisions to actual changes in the agreed exact-k masked NELBO near 50% pruning, rather than repurposing old dense-background 65-to-70 KL curves as ground truth. Fixed-budget protect/prune exchanges around a common sparse model measure allocation utility. Use held-out articles and shared mask draws; preserve signed changes.
4. **DLM attribution:** masked/unmasked and corruption probability can be diagnostic axes, not mandatory score components. Existing LSA clean/corrupted ranks (.992 projection/.9996 block) argue against assuming masking alone changes the criterion. A matched clean-input control only isolates corruption-conditioning; DLM-versus-AR specificity still requires a model/task control.

## Bounded next design (proposal only)
First prioritize the scale-versus-shape distinction exposed by DLP versus searched DSA. Compare a simple scale-invariant shape statistic to the existing magnitude statistic, under the same local Wanda ranking, calibration, allocation granularity, global budget and allowed sparsity spread. Include a depth-matched control so merely protecting late layers is not credited as a new proxy. Treat the DSA-selected log-dispersion family as an existing baseline, not a novel DLM proxy. Use disjoint development/validation articles to assess decision direction; select one correction, then compare full validation NELBO. Actual generation capability requires a separately fixed task evaluation. Do not require solving all sparse interactions before testing one falsifiable correction.

No new GPU jobs, masks, model scores, or tuned proxies were produced. Older super-outlier, timestep, role, reconstruction and gain-fit results retain their original limitations; none is promoted here as a proven DLM solution.
'''
    (ROOT/'report.md').write_text(report)
    print(table)
    print('Verified: 11 manifests, 224 identities each, exact recorded budgets, 551-chunk coverage summaries, factorization identities.')


if __name__ == '__main__':
    main()
