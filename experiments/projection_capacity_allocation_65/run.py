"""Frozen projection-capacity oracle experiment; expensive phases run in tmux."""
import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.dlm_loss_aggregation.core import pack_mask, unpack_mask, mask_sha256
from experiments.dlm_loss_aggregation.run import _load_model, _suffix_logits, load_config
from experiments.wanda_failure_characterization.run_failure_map import modules, model_sha, state_tensors
from experiments.wanda_failure_characterization.core import masked_linear_variants
from experiments.projection_capacity_allocation_65.verify_states import ROOT, CAL, HELD, sha, write_json
from experiments.projection_capacity_allocation_65.core import GRID, allocate, distribution, correlation, paired_gate

STATS = Path('experiments/cgq_wanda_structured_diagnostic/sufficient_statistics.pt')
SWEEP = Path('experiments/uniform_wanda_sparsity_sweep/mask_manifest.json')
DENSE_SHA = '2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc'
CONFIG = ROOT / 'config.json'
STORE = ROOT / 'runtime'


def event(event_type, **values):
    print(json.dumps({'event': event_type, **values}), flush=True)


def save_tensor(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def freeze():
    verification = json.loads((ROOT / 'state_verification.json').read_text())
    assert verification['status'] == 'verified' and verification['disjoint']
    for row in verification['splits']:
        assert sha(row['path']) == row['file_sha256']
    if CONFIG.exists():
        return require_config()
    reference = json.loads(SWEEP.read_text())
    config = {
        'status': 'frozen_before_capacity_outcomes', 'model': json.loads(CAL.read_text())['model'],
        'dense_model_sha256': DENSE_SHA, 'projection_count': 224, 'grid': list(GRID),
        'nominal_global_sparsity': .65,
        'selector': 'Standard Wanda: abs(W.float()) * sqrt(A.float()); stable ascending row-wise sort',
        'activation_semantics': 'Historical sweep DLM-Wanda: unweighted mean of sum_token X^2 across 80 frozen states; overall_uniform field only',
        'calibration_choice': 'sweep DLM calibration, as stated before construction',
        'allocation_metric': 'arithmetic mean across states of masked-token Dense||Sparse KL',
        'allocation': 'raw next-increment marginal KL / (0.05*N); repository-order ties; exact row-floor budget reachability constraint',
        'monotone_envelope': None, 'budget': reference['summaries']['V0_65_uniform'],
        'capacity_path': 'seven-variant suffix batch: dense same-path sham + all six grid masks; one target Linear intervention',
        'heldout_path': 'batch-one full forwards with physical masks; repeated dense sham must be exactly equal',
        'bootstrap': {'seed': 0, 'resamples': 20000, 'unit': 'paired state', 'interval': 'percentile 95%'},
        'gate': 'mean delta<0 AND CI upper<0 AND >4/8 sequences AND >2/5 timesteps improve',
        'downstream': 'only on DLM PASS: frozen GSM8K mini100; full1319 only if mini capacity correct > uniform correct',
        'sources': {str(p): sha(p) for p in (CAL, HELD, STATS, SWEEP,
                     Path('experiments/dlm_loss_aggregation/exp002/config.yaml'),
                     Path('experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl'))},
    }
    write_json(CONFIG, config)
    return config


def require_config():
    config = json.loads(CONFIG.read_text())
    for path, digest in config['sources'].items():
        assert sha(path) == digest, f'frozen input changed: {path}'
    return config


def load_dense():
    model, _ = _load_model(load_config('experiments/dlm_loss_aggregation/config.yaml'))
    assert model_sha(model) == DENSE_SHA, 'wrong dense weights'
    mapping = modules(model)
    assert len(mapping) == 224
    return model, mapping


@torch.inference_mode()
def prepare(model, mapping):
    destination = ROOT / 'candidate_mask_manifest.json'
    if destination.exists():
        result = json.loads(destination.read_text())
        assert result['config_sha256'] == sha(CONFIG)
        return result
    # Only the historical unweighted activation field is consumed.
    stats = torch.load(STATS, map_location='cpu', weights_only=False)
    activation = {k: v['overall_uniform'] for k, v in stats['statistics'].items()}
    assert set(activation) == set(mapping)
    ref = {r['canonical_name']: r for r in json.loads(SWEEP.read_text())['entries'] if r['method'] == 'V0_65_uniform'}
    entries = []
    for index, (name, module) in enumerate(mapping.items()):
        weight = module.weight
        a = activation[name].float().to(weight.device)
        assert a.shape == (weight.shape[1],) and torch.isfinite(a).all() and (a >= 0).all()
        score = weight.float().abs() * a.sqrt()[None, :]
        order = torch.argsort(score, dim=1, stable=True)
        row = {'module_index': index, 'name': name, 'shape': list(weight.shape), 'weights': weight.numel(), 'masks': []}
        for r in GRID:
            count = int(weight.shape[1] * r)
            mask = torch.zeros_like(weight, dtype=torch.bool).scatter_(1, order[:, :count], True)
            packed = pack_mask(mask.cpu())
            digest = mask_sha256(packed)
            if r == .65:
                assert digest == ref[name]['sha256'], f'historical Uniform65 mask mismatch: {name}'
            path = STORE / 'masks' / f'{name}.{round(r * 100)}.pt'
            save_tensor(path, packed)
            row['masks'].append({'nominal_sparsity': r, 'prune_per_row': count,
                                 'pruned': weight.shape[0] * count, 'mask_sha256': digest,
                                 'path': str(path), 'file_sha256': sha(path)})
        entries.append(row)
        del score, order, mask, packed
        event('masks', module=index + 1, total=224)
    result = {'config_sha256': sha(CONFIG), 'entries': entries, 'all_uniform65_masks_match_historical': True}
    write_json(destination, result)
    return result


def read_mask(row, level, device):
    meta = row['masks'][level]
    packed = torch.load(meta['path'], map_location='cpu', weights_only=False)
    assert mask_sha256(packed) == meta['mask_sha256']
    return unpack_mask(packed).to(device)


@torch.inference_mode()
def diagnostics(sparse, dense, target, p_mask, length=256):
    """Inputs are masked-token logits, same shape [tokens,vocab]."""
    lp, rp = F.log_softmax(sparse.float(), -1), F.log_softmax(dense.float(), -1)
    kl = (rp.exp() * (rp - lp)).sum(-1)
    ce_sparse = F.cross_entropy(sparse.float(), target, reduction='none')
    ce_dense = F.cross_entropy(dense.float(), target, reduction='none')
    delta = (ce_sparse - ce_dense).sum() / p_mask / length
    result = {'mean_kl': kl.mean().item(), 'median_kl': kl.median().item(),
              'top1_agreement': (sparse.argmax(-1) == dense.argmax(-1)).float().mean().item(),
              'confidence_mae': (lp.exp().amax(-1) - rp.exp().amax(-1)).abs().mean().item(),
              'delta_loss': delta.item(), 'positive_delta_loss': delta.clamp_min(0).item(),
              'mean_positive_token_delta_ce': (ce_sparse - ce_dense).clamp_min(0).mean().item(),
              'masked_count': len(target)}
    assert all(math.isfinite(v) for v in result.values())
    return result, kl.cpu()


def summarize_states(rows, token_kl):
    keys = ('mean_kl', 'top1_agreement', 'confidence_mae', 'positive_delta_loss', 'delta_loss', 'mean_positive_token_delta_ce')
    return {**{k: float(np.mean([r[k] for r in rows])) for k in keys},
            'median_kl': float(torch.cat(token_kl).quantile(.5)),
            'pooled_token_mean_kl': float(torch.cat(token_kl).mean()),
            'median_state_mean_kl': float(np.median([r['mean_kl'] for r in rows]))}


@torch.inference_mode()
def capacity(model, mapping, candidates):
    states = json.loads(CAL.read_text())['states']
    dev = next(model.parameters()).device
    # Cached dense prefixes are shared across all seven variants. At 80 states
    # they occupy about 5 GiB; no prefix comes from a sparse model.
    prefixes = {i: [] for i in range(32)}
    handles = []
    for i, block in enumerate(model.model.transformer.blocks):
        def hook(_, inp, index=i):
            prefixes[index].append(inp[0].detach().clone())
        handles.append(block.register_forward_pre_hook(hook))
    try:
        for state in states:
            model(torch.tensor(state['noisy_ids'], device=dev))
    finally:
        for handle in handles:
            handle.remove()
    event('dense_prefixes_complete', states=80)
    started = time.monotonic()
    for row in candidates['entries']:
        name, index = row['name'], row['module_index']
        path = ROOT / 'curves' / f'{name}.json'
        if path.exists():
            old = json.loads(path.read_text())
            assert old['config_sha256'] == sha(CONFIG) and len(old['curves']) == 6
            continue
        module = mapping[name]
        block = int(name.split('.')[0].removeprefix('block_'))
        masks = [None] + [read_mask(row, k, module.weight.device) for k in range(6)]
        records, token_kl = [[] for _ in GRID], [[] for _ in GRID]
        reconstruction = []
        def intervention(mod, inp, out):
            result = masked_linear_variants(inp[0], mod.weight, mod.bias, masks)
            den = result[0].float().square().sum().clamp_min(1e-30)
            reconstruction[:] = [float((result[k].float() - result[0].float()).square().sum() / den) for k in range(1, 7)]
            return result
        for si, state in enumerate(states):
            noisy, clean, masked = state_tensors(state, dev)
            h = module.register_forward_hook(intervention)
            try:
                logits = _suffix_logits(model, prefixes[block][si].expand(7, -1, -1).contiguous(), block)
                if index == 0 and si == 0:
                    full = model(noisy.expand(7, -1).contiguous()).logits
                    diff = (full - logits).abs().max().item()
                    write_json(ROOT / 'capacity_execution_sanity.json', {'suffix_full_max_abs': diff, 'batch_size': 7})
                    assert diff == 0, 'same-shape full/suffix mismatch'
                    del full
            finally:
                h.remove()
            selected = logits[:, masked[0]]
            for k in range(6):
                measured, kl = diagnostics(selected[k + 1], selected[0], clean[0, masked[0]], state['p_mask'])
                measured.update(state_index=si, sequence_index=state['sequence_index'], timestep=state['timestep'],
                                local_reconstruction_error=reconstruction[k])
                records[k].append(measured)
                token_kl[k].append(kl)
            del logits, selected
        output = {'name': name, 'module_index': index, 'shape': row['shape'], 'config_sha256': sha(CONFIG),
                  'curves': [{'sparsity': r, 'summary': summarize_states(records[k], token_kl[k]),
                              'per_state': records[k]} for k, r in enumerate(GRID)]}
        write_json(path, output)
        event('capacity_projection_complete', module=index + 1, total=224, name=name,
              mean_kl=[c['summary']['mean_kl'] for c in output['curves']], elapsed=time.monotonic() - started)
        del masks
    del prefixes
    assert model_sha(model) == DENSE_SHA, 'single-projection experiment changed dense weights'


def construct_allocation(candidates):
    all_curves = [json.loads((ROOT / 'curves' / f"{r['name']}.json").read_text()) for r in candidates['entries']]
    assert len(all_curves) == 224
    curves = np.array([[c['summary']['mean_kl'] for c in row['curves']] for row in all_curves])
    write_json(ROOT / 'capacity_curves_raw.json', {'metric': 'mean state masked-token Dense||Sparse KL',
               'grid': GRID, 'monotone_envelope_used': False, 'projections': all_curves})
    names = [r['name'] for r in candidates['entries']]
    shapes = [r['shape'] for r in candidates['entries']]
    marginals = []
    for i, name in enumerate(names):
        for k in range(5):
            cost = float(curves[i, k + 1] - curves[i, k])
            marginals.append({'name': name, 'module_index': i, 'from': GRID[k], 'to': GRID[k + 1],
                              'marginal_kl': cost, 'cost_per_nominal_parameter': cost / (.05 * math.prod(shapes[i]))})
    summary = {'damage_by_sparsity': {str(r): distribution(curves[:, k]) for k, r in enumerate(GRID)},
               'marginal_by_increment': {f'{GRID[k]}->{GRID[k+1]}': distribution(np.diff(curves, axis=1)[:, k]) for k in range(5)},
               'top20_expensive_raw': sorted(marginals, key=lambda x: -x['marginal_kl'])[:20],
               'top20_cheapest_raw': sorted(marginals, key=lambda x: x['marginal_kl'])[:20],
               'top20_expensive_per_parameter': sorted(marginals, key=lambda x: -x['cost_per_nominal_parameter'])[:20],
               'top20_cheapest_per_parameter': sorted(marginals, key=lambda x: x['cost_per_nominal_parameter'])[:20],
               'monotonicity_violations': [r for r in marginals if r['marginal_kl'] < 0],
               'per_projection_violation_counts': {n: int((np.diff(curves[i]) < 0).sum()) for i, n in enumerate(names)}}
    for grouping in ('layer', 'type'):
        groups = sorted({n.split('.')[0 if grouping == 'layer' else 1] for n in names})
        summary['by_' + grouping] = {g: {'damage': [distribution(curves[[i for i, n in enumerate(names) if n.split('.')[0 if grouping == 'layer' else 1] == g], k]) for k in range(6)],
                                                    'marginal': [distribution(np.diff(curves, axis=1)[[i for i, n in enumerate(names) if n.split('.')[0 if grouping == 'layer' else 1] == g], k]) for k in range(5)]} for g in groups}
    write_json(ROOT / 'marginal_costs.json', {'increments': marginals, 'summary': summary})
    event('capacity_premise_summary', damage_by_sparsity=summary['damage_by_sparsity'],
          nonmonotone_increments=len(summary['monotonicity_violations']))
    result = allocate(curves, shapes)
    assert result['pruned'] == json.loads(CONFIG.read_text())['budget']['total_pruned']
    assignments = [{**{k: row[k] for k in ('name', 'module_index', 'shape', 'weights')},
                    'assigned_sparsity': result['sparsities'][i], 'level': result['levels'][i],
                    'pruned': row['masks'][result['levels'][i]]['pruned']} for i, row in enumerate(candidates['entries'])]
    document = {**result, 'config_sha256': sha(CONFIG), 'assignments': assignments,
                'capacity_curves_sha256': sha(ROOT / 'capacity_curves_raw.json'),
                'frozen_before_heldout_evaluation': True}
    if (ROOT / 'allocation.json').exists():
        assert json.loads((ROOT / 'allocation.json').read_text()) == document, 'attempt to alter frozen allocation'
    else:
        write_json(ROOT / 'allocation.json', document)
    s = np.array(result['sparsities'])
    layers = [int(n.split('.')[0].removeprefix('block_')) for n in names]
    alloc_summary = {'global_parameter_weighted_sparsity': result['pruned'] / result['weights'],
                     'uniform_parameter_weighted_sparsity': result['uniform_pruned'] / result['weights'],
                     'nominal_projection_sparsity': distribution(s),
                     'counts': {str(r): int((s == r).sum()) for r in GRID},
                     'correlation_D65_assigned_density': correlation(curves[:, 3], 1 - s),
                     'correlation_mean_marginal_assigned_sparsity': correlation(np.diff(curves, axis=1).mean(1), s),
                     'correlation_mean_marginal_per_parameter_assigned_sparsity': correlation(np.diff(curves, axis=1).mean(1) / (.05 * np.prod(shapes, axis=1)), s),
                     'correlation_layer_sparsity': correlation(layers, s),
                     'quartile_mean_sparsity': [float(s[np.array(layers) // 8 == q].mean()) for q in range(4)],
                     'budget_error': result['budget_error'], 'feasibility_skip_count': len(result['feasibility_skips'])}
    for key in ('layer', 'type'):
        labels = layers if key == 'layer' else [n.split('.')[1] for n in names]
        alloc_summary['by_' + key] = {}
        for value in sorted(set(labels)):
            ids = [i for i, label in enumerate(labels) if label == value]
            alloc_summary['by_' + key][str(value)] = {'mean_nominal_sparsity': float(s[ids].mean()),
                'parameter_weighted_sparsity': sum(assignments[i]['pruned'] for i in ids) / sum(assignments[i]['weights'] for i in ids),
                'assigned_sparsities': {names[i]: float(s[i]) for i in ids}}
    write_json(ROOT / 'allocation_summary.json', alloc_summary)
    for method in ('uniform', 'capacity'):
        selected = [{**row, 'selected_mask': row['masks'][3 if method == 'uniform' else result['levels'][i]]}
                    for i, row in enumerate(candidates['entries'])]
        for row in selected:
            del row['masks']
        write_json(ROOT / f'{method}65_mask_manifest.json', {'method': method, 'config_sha256': sha(CONFIG),
            'allocation_sha256': sha(ROOT / 'allocation.json'), 'pruned': sum(r['selected_mask']['pruned'] for r in selected),
            'weights': result['weights'], 'entries': selected})
    event('allocation_frozen', **alloc_summary)
    return document


@torch.inference_mode()
def apply_full(model, method):
    manifest = json.loads((ROOT / f'{method}65_mask_manifest.json').read_text())
    assert manifest['allocation_sha256'] == sha(ROOT / 'allocation.json')
    mapping = modules(model)
    for row in manifest['entries']:
        meta = row['selected_mask']
        packed = torch.load(meta['path'], map_location='cpu', weights_only=False)
        assert mask_sha256(packed) == meta['mask_sha256']
        mask = unpack_mask(packed).to(mapping[row['name']].weight.device)
        assert int(mask.sum()) == meta['pruned']
        mapping[row['name']].weight.masked_fill_(mask, 0)
    return manifest


@torch.inference_mode()
def heldout():
    states = json.loads(HELD.read_text())['states']
    model, _ = load_dense()
    dev = next(model.parameters()).device
    refs = []
    sham_max = 0.
    for si, state in enumerate(states):
        noisy, clean, mask = state_tensors(state, dev)
        dense = model(noisy).logits
        sham = model(noisy).logits
        diff = (dense - sham).abs().max().item()
        sham_max = max(sham_max, diff)
        assert diff == 0, 'dense repeated same-path sham differs'
        refs.append(dense[0, mask[0]].cpu())
        del dense, sham
    assert model_sha(model) == DENSE_SHA
    dense_weights = {n: m.weight.detach().cpu().clone() for n, m in modules(model).items()}
    results = {}
    for method in ('uniform', 'capacity'):
        for name, module in modules(model).items():
            module.weight.copy_(dense_weights[name])
        assert model_sha(model) == DENSE_SHA
        manifest = apply_full(model, method)
        before = model_sha(model)
        rows, tokens = [], []
        for si, state in enumerate(states):
            noisy, clean, mask = state_tensors(state, dev)
            sparse = model(noisy).logits[0, mask[0]]
            measured, kl = diagnostics(sparse, refs[si].to(dev), clean[0, mask[0]], state['p_mask'])
            measured.update(state_index=si, sequence_index=state['sequence_index'], timestep=state['timestep'])
            rows.append(measured)
            tokens.append(kl)
            event('heldout_state', method=method, state=si + 1, mean_kl=measured['mean_kl'])
        assert model_sha(model) == before
        results[method] = {'summary': summarize_states(rows, tokens), 'per_state': rows,
                           'global_sparsity': manifest['pruned'] / manifest['weights'],
                           'pruned': manifest['pruned'], 'model_sha256': before}
        write_json(ROOT / f'heldout_{method}.json', results[method])
    assert results['uniform']['pruned'] == results['capacity']['pruned']
    gate = paired_gate(results['uniform']['per_state'], results['capacity']['per_state'])
    output = {'status': 'complete', 'config_sha256': sha(CONFIG), 'allocation_sha256': sha(ROOT / 'allocation.json'),
              'heldout_state_sha256': json.loads(HELD.read_text())['historical_state_sha256'],
              'dense_sham_max_abs': sham_max, 'methods': results, 'gate': gate}
    write_json(ROOT / 'heldout_dlm_results.json', output)
    event('kill_gate', **gate)
    del model, dense_weights, refs
    gc.collect()
    torch.cuda.empty_cache()
    return output


def report(result):
    allocation = json.loads((ROOT / 'allocation.json').read_text())
    summary = json.loads((ROOT / 'allocation_summary.json').read_text())
    marginal = json.loads((ROOT / 'marginal_costs.json').read_text())['summary']
    gate = result['gate']
    lines = ['# Projection-Wise Functional Capacity Allocation @ 65%', '', '## Question', '',
             'Can changing projection-level sparsity allocation improve pruning under an identical global parameter budget?', '',
             '## Frozen Setup', '',
             'GSAI-ML/LLaDA-8B-Base, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`; all 224 historical projections. '
             'Standard Wanda uses the exact unweighted 80-state DLM activation cache and stable row-wise ranking from the cited uniform sweep. '
             'All Uniform-65 mask hashes match that sweep. Grid: 50/55/60/65/70/75%. '
             'Allocation uses only the mean state masked-token Dense||Sparse KL. '
             'All masks, state digests and corpus intervals were verified; held-out 40 states are disjoint from allocation 80 states. '
             'Calibration uses a seven-variant same-path dense sham; held-out evaluation uses batch-one full forwards and exact repeated-dense sham equality.', '',
             '## Projection Capacity Curves', '',
             '| Sparsity | Mean damage | Median | p10 | p90 | CV |', '|---|---:|---:|---:|---:|---:|']
    for r, d in marginal['damage_by_sparsity'].items():
        lines.append(f"| {r} | {d['mean']:.6g} | {d['median']:.6g} | {d['p10']:.6g} | {d['p90']:.6g} | {d['coefficient_of_variation']:.6g} |")
    lines += ['', f"Raw curves contain {len(marginal['monotonicity_violations'])} negative increments. No envelope or manual correction was used.",
              'Full raw curves, per-state diagnostics, marginal distributions, layer/type breakdowns and top/bottom 20 increments are in `capacity_curves_raw.json` and `marginal_costs.json`.', '',
              '## Frozen Allocation', '', f"Both models prune exactly {allocation['pruned']:,} / {allocation['weights']:,} weights "
              f"({summary['global_parameter_weighted_sparsity']:.10%}); nominal target is 65%. The difference from mathematical 65% is historical row flooring. Budget difference between methods: 0.",
              f"Grid counts: {summary['counts']}. Exact-budget feasibility skips: {summary['feasibility_skip_count']}.",
              f"Layer-index Spearman: {summary['correlation_layer_sparsity']['spearman']}; layer-quartile means: {summary['quartile_mean_sparsity']}.",
              'These are descriptive patterns from the frozen rule. They did not alter it.', '',
              '| Projection | Assigned sparsity | Pruned weights |', '|---|---:|---:|']
    lines += [f"| {r['name']} | {r['assigned_sparsity']:.0%} | {r['pruned']} |" for r in allocation['assignments']]
    lines += ['', '## Full-Model Held-Out DLM Evaluation', '',
              '| Method | Global sparsity | Mean KL | Median KL | Top-1 agreement |',
              '|--------|----------------:|--------:|----------:|----------------:|']
    for method, row in result['methods'].items():
        d = row['summary']
        lines.append(f"| {method.title()} Wanda | {row['global_sparsity']:.8%} | {d['mean_kl']:.7g} | {d['median_kl']:.7g} | {d['top1_agreement']:.7g} |")
    lines += ['', 'Mean KL averages state-level masked-token means; median KL pools masked tokens.',
              f"Paired capacity-minus-uniform mean: {gate['mean_difference']:.7g}; median: {gate['median_difference']:.7g}; "
              f"95% bootstrap CI: {gate['bootstrap_95_ci']}. 20,000 paired state resamples, seed 0.",
              f"States improved/worsened/tied: {gate['states_improved']}/{gate['states_worsened']}/{gate['states_tied']}. "
              f"Sequence means improved: {gate['sequence_means_improved']}/8. Timestep means improved: {gate['timestep_means_improved']}/5.", '',
              '| Group | Uniform mean KL | Capacity mean KL | Difference |', '|---|---:|---:|---:|']
    for key in ('per_sequence', 'per_timestep'):
        for row in gate[key]:
            label = row.get('sequence_index', row.get('timestep'))
            lines.append(f"| {key}: {label} | {row['uniform_mean_kl']:.7g} | {row['capacity_mean_kl']:.7g} | {row['difference']:.7g} |")
    lines += ['', '## Kill-Gate Decision', '']
    if gate['passed']:
        lines += ['### SUPPORTED', '', 'Projection-wise functional capacity allocation improves held-out full-model DLM fidelity under the same global sparsity budget. Allocation has real decision-level leverage; the next task is to approximate this oracle with a cheap DLM-specific signal.']
    else:
        lines += ['### NOT SUPPORTED', '', 'Although individual projections show different single-module pruning damage, the resulting capacity allocation does not improve the jointly sparse model under the same global budget. Projection allocation is not pursued further.']
    downstream_file = ROOT / 'downstream.json'
    if gate['passed'] and downstream_file.exists():
        lines += ['', '## Downstream', '', '```json', downstream_file.read_text().strip(), '```']
    (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')


def downstream():
    from transformers import AutoTokenizer
    from experiments.dlm_loss_aggregation.exp002.run import (_evaluate_gsm8k, _evaluation_config_hash,
        _assert_same_examples, _read_jsonl, _write_jsonl, load_config as eval_config)
    config = eval_config('experiments/dlm_loss_aggregation/exp002/config.yaml')
    protocol_hash, protocol = _evaluation_config_hash(config)
    reference = _read_jsonl(Path('experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl'))
    tokenizer = AutoTokenizer.from_pretrained(config['model']['id'], revision=config['model']['revision'], trust_remote_code=True)
    output = {'protocol_hash': protocol_hash, 'protocol': protocol, 'evaluations': []}
    for limit in (100, 1319):
        outcomes = {}
        for method in ('uniform', 'capacity'):
            model, _ = load_dense()
            apply_full(model, method)
            before = model_sha(model)
            measured, records = _evaluate_gsm8k(model, tokenizer, config, f'{method}_capacity65_oracle', limit, protocol_hash)
            assert model_sha(model) == before
            assert len(records) == limit
            if len(reference) >= limit:
                _assert_same_examples(reference[:limit], records)
            path = ROOT / 'gsm8k' / f'{method}_{limit}_predictions.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_jsonl(path, records)
            outcomes[method] = {'correct': sum(r['correct'] for r in records), 'limit': limit,
                                'predictions': str(path), 'sha256': sha(path), 'metrics': measured}
            del model
            gc.collect()
            torch.cuda.empty_cache()
        a = _read_jsonl(Path(outcomes['uniform']['predictions']))
        b = _read_jsonl(Path(outcomes['capacity']['predictions']))
        _assert_same_examples(a, b)
        output['evaluations'].append(outcomes)
        output['mini_directionally_better'] = outcomes['capacity']['correct'] > outcomes['uniform']['correct'] if limit == 100 else output['mini_directionally_better']
        write_json(ROOT / 'downstream.json', output)
        if limit == 100 and not output['mini_directionally_better']:
            event('downstream_stop', reason='mini100 capacity not directionally better', outcomes=outcomes)
            break
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'all', 'analyze', 'heldout'])
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)
    freeze()
    if args.phase == 'freeze':
        return
    if args.phase == 'all':
        model, mapping = load_dense()
        candidates = prepare(model, mapping)
        capacity(model, mapping, candidates)
        del model, mapping
        gc.collect()
        torch.cuda.empty_cache()
    else:
        candidates = json.loads((ROOT / 'candidate_mask_manifest.json').read_text())
    construct_allocation(candidates)
    if args.phase == 'analyze':
        return
    result = heldout()
    report(result)
    if result['gate']['passed']:
        downstream()
        report(result)
    event('experiment_complete', supported=result['gate']['passed'])


if __name__ == '__main__':
    main()
