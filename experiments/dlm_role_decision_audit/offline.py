#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.dlm_dual_role_allocation.io import atomic_write_json, load_frozen_inputs
from experiments.dlm_dual_role_mini100.core import validate_manifest
from experiments.dlm_role_decision_audit.core import (
    enumerate_budget_bundles, json_digest, select_bundles, sha256,
)
from experiments.projection_capacity_followup_65.core import build_selected_manifest

ROOT = Path('experiments/dlm_role_decision_audit')
DP = Path('experiments/dlm_role_tradeoff/dp_results.json')
ROLE_CURVES = Path('experiments/dlm_role_validation/curves_random65.json')
CAPACITY = Path('experiments/projection_capacity_allocation_65/capacity_curves_raw.json')
CANDIDATES = Path('experiments/projection_capacity_allocation_65/candidate_mask_manifest.json')
A_MANIFEST = Path('experiments/dlm_dual_role_mini100/role65_mask_manifest.json')
VERIFY = Path('experiments/projection_capacity_allocation_65/state_verification.json')
HELDOUT = Path('experiments/wanda_failure_characterization/heldout_state_manifest.json')
MINI = Path('experiments/dlm_role_global_minimax_mini100/mini100_results.json')
GRID = (.50, .55, .60, .65, .70, .75)
TARGET = 4_536_008_704
WEIGHTS = 6_979_321_856


def frozen_sources():
    return [DP, ROLE_CURVES, CAPACITY, CANDIDATES, A_MANIFEST, VERIFY, HELDOUT, MINI]


def freeze():
    inputs = load_frozen_inputs()
    verification = json.loads(VERIFY.read_text())
    heldout = json.loads(HELDOUT.read_text())
    if verification.get('status') != 'verified' or not verification.get('disjoint') or verification.get('overlaps'):
        raise RuntimeError('historical splits are not verified disjoint')
    split = [x for x in verification['splits'] if int(x['states']) == 40]
    if len(split) != 1 or split[0]['state_sha256'] != heldout['historical_state_sha256']:
        raise RuntimeError('heldout state digest mismatch')
    if sha256(HELDOUT) != split[0]['file_sha256']:
        raise RuntimeError('heldout manifest hash mismatch')
    cal_pairs = {(int(x['sequence_index']), float(x['timestep'])) for x in inputs.states['states']}
    held_pairs = {(int(x['sequence_index']), float(x['timestep'])) for x in heldout['states']}
    if cal_pairs & held_pairs or len(held_pairs) != 40:
        raise RuntimeError('calibration/heldout state identifiers overlap or duplicate')
    config = {'status': 'frozen_before_offline_selection', 'model': inputs.config['model'],
              'dense_model_sha256': inputs.metadata['dense_model_sha256'],
              'calibration_state_digest': inputs.metadata['state_digest'],
              'heldout_state_digest': heldout['historical_state_sha256'],
              'grid': list(GRID), 'target_pruned': TARGET, 'weights': WEIGHTS,
              'allocations': {'A': 'current Role greedy', 'B': 'exact local-Max',
                              'C': 'supported Global Minimax'},
              'bundle_sizes': [2, 3, 4], 'bundles_per_source': 3,
              'selection': ['proxy_better_kl_worse', 'proxy_better_kl_better', 'kl_neutral'],
              'primary': 'batch1 full-forward change in masked-token Dense||candidate KL',
              'statistics': {'bootstrap_resamples': 20000, 'seed': 1234,
                             'sign_flip': 'enumerate 2^8 sequence signs', 'multiple_testing': 'Holm'},
              'no_gsm8k_or_allocation_tuning': True,
              'source_hashes': {str(x): sha256(x) for x in frozen_sources()}}
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError('frozen config changed')
    if not path.exists(): atomic_write_json(path, config)
    state_doc = {'status': 'verified', 'disjoint': True, 'overlap': [],
                 'calibration_pairs': len(cal_pairs), 'heldout_pairs': len(held_pairs),
                 'calibration_digest': inputs.metadata['state_digest'],
                 'heldout_digest': heldout['historical_state_sha256'],
                 'source_verification_sha256': sha256(VERIFY), 'heldout_manifest_sha256': sha256(HELDOUT)}
    atomic_write_json(ROOT / 'state_verification.json', state_doc)
    return config


def load_arrays():
    inputs = load_frozen_inputs()
    names = inputs.metadata['module_names']
    role = json.loads(ROLE_CURVES.read_text())
    capacity = json.loads(CAPACITY.read_text())
    if role['names'] != names or [x['name'] for x in capacity['projections']] != names:
        raise RuntimeError('projection ordering mismatch')
    if tuple(map(float, role['grid'])) != GRID or tuple(map(float, capacity['grid'])) != GRID:
        raise RuntimeError('grid mismatch')
    m = np.asarray(role['groups']['actual']['a'], float)
    u = np.asarray(role['groups']['actual']['b'], float)
    d = np.asarray([[float(level['summary']['mean_kl']) for level in row['curves']]
                    for row in capacity['projections']], float)
    ds = np.asarray([[[float(state['mean_kl']) for state in level['per_state']]
                      for level in row['curves']] for row in capacity['projections']], float)
    ds = ds.transpose(0, 2, 1)
    if m.shape != (224, 6) or u.shape != m.shape or d.shape != m.shape or ds.shape != (224, 80, 6):
        raise RuntimeError('curve shape mismatch')
    return inputs, m, u, d, ds


def make_manifest(label, levels, inputs):
    sparsities = [GRID[int(x)] for x in levels]
    manifest = build_selected_manifest(label, inputs.candidate['entries'], sparsities, GRID)
    manifest.update(allocation_levels=list(map(int, levels)), config_sha256=sha256(ROOT/'config.json'))
    validate_manifest(manifest, inputs.metadata['module_names'], TARGET, WEIGHTS)
    path = ROOT / f'manifest_{label}.json'
    atomic_write_json(path, manifest)
    return manifest


def offline():
    config = freeze()
    for path, digest in config['source_hashes'].items():
        if sha256(path) != digest: raise RuntimeError(f'frozen source changed: {path}')
    inputs, m, u, d, ds = load_arrays()
    dp = json.loads(DP.read_text())
    levels = {'A': list(map(int, dp['current_role']['levels'])),
              'B': list(map(int, dp['exact_local_max']['levels'])),
              'C': list(map(int, dp['supported_minimax']['levels']))}
    historical = json.loads(A_MANIFEST.read_text())
    if levels['A'] != list(map(int, historical['allocation_levels'])):
        raise RuntimeError('A levels do not match historical Role manifest')
    manifests = {'A': historical, 'B': make_manifest('B_exact_local_max', levels['B'], inputs),
                 'C': make_manifest('C_supported_global_minimax', levels['C'], inputs)}
    for label, manifest in manifests.items():
        if int(manifest['pruned']) != TARGET: raise RuntimeError(f'{label} budget mismatch')

    decisions = []
    sequences = np.asarray([int(x['sequence_index']) for x in inputs.states['states']])
    summaries = {}
    for source in ('B', 'C'):
        changed = []
        for i, (a, b) in enumerate(zip(levels['A'], levels[source])):
            if a == b: continue
            delta_states = ds[i, :, b] - ds[i, :, a]
            changed.append({'source': source, 'module_index': i, 'name': inputs.metadata['module_names'][i],
                            'base_level': a, 'new_level': b,
                            'delta_masked_reconstruction': float(m[i,b]-m[i,a]),
                            'delta_unmasked_reconstruction': float(u[i,b]-u[i,a]),
                            'delta_single_projection_kl': float(d[i,b]-d[i,a]),
                            'sequence_kl_deltas': [float(delta_states[sequences == s].mean())
                                                   for s in sorted(set(sequences))]})
        decisions.extend(changed)
        proxy = np.array([max(x['delta_masked_reconstruction'], x['delta_unmasked_reconstruction']) for x in changed])
        kl = np.array([x['delta_single_projection_kl'] for x in changed])
        summaries[source] = {'changed_projections': len(changed),
                             'proxy_vs_kl_spearman': float(spearmanr(proxy, kl).statistic),
                             'proxy_kl_sign_agreement': float(np.mean(np.sign(proxy) == np.sign(kl))),
                             'sum_delta_kl': float(kl.sum()), 'positive_delta_kl': float(kl[kl>0].sum()),
                             'negative_delta_kl': float(kl[kl<0].sum()),
                             'max_single_delta_kl': float(kl.max()), 'min_single_delta_kl': float(kl.min())}

    all_candidates = {}
    selected, used = [], set()
    for source in ('B', 'C'):
        candidates = enumerate_budget_bundles(levels['A'], levels[source], inputs.candidate['entries'],
                                              m, u, d, source)
        all_candidates[source] = {'count': len(candidates),
                                  'proxy_nonincrease': sum(x['proxy_nonincrease'] for x in candidates),
                                  'proxy_better_kl_worse': sum(x['proxy_nonincrease'] and x['delta_additive_single_projection_kl'] > 0 for x in candidates),
                                  'proxy_better_kl_better': sum(x['proxy_nonincrease'] and x['delta_additive_single_projection_kl'] < 0 for x in candidates)}
        selected.extend(select_bundles(candidates, used))
    for index, bundle in enumerate(selected):
        bundle['bundle_index'] = index
        bundle['bundle_id'] = f"{bundle['source']}{index % 3 + 1}_{bundle['selection_category']}"
    frozen = {'status': 'frozen_before_heldout_forward', 'config_sha256': sha256(ROOT/'config.json'),
              'allocation_levels': levels, 'manifest_sha256': {
                  'A': sha256(A_MANIFEST), 'B': sha256(ROOT/'manifest_B_exact_local_max.json'),
                  'C': sha256(ROOT/'manifest_C_supported_global_minimax.json')},
              'candidate_summary': all_candidates, 'bundles': selected}
    frozen['bundle_selection_sha256'] = json_digest(frozen['bundles'])
    atomic_write_json(ROOT / 'bundles.json', frozen)
    atomic_write_json(ROOT / 'offline_analysis.json', {'status': 'complete', 'summaries': summaries,
                                                       'decisions': decisions, 'bundle_selection_sha256': frozen['bundle_selection_sha256']})
    with (ROOT/'decision_table.csv').open('w', newline='') as stream:
        fields = [x for x in decisions[0] if x != 'sequence_kl_deltas']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({k:v for k,v in row.items() if k in fields} for row in decisions)
    print(json.dumps({'event':'offline_complete','summaries':summaries,
                      'bundle_candidates':all_candidates,'bundles':selected}, sort_keys=True), flush=True)
    return frozen


if __name__ == '__main__': offline()
