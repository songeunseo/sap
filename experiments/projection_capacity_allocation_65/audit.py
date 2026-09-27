"""Read-only scientific artifact checks; writes an audit result when complete."""
import json
from pathlib import Path

import numpy as np

from experiments.projection_capacity_allocation_65.core import GRID, paired_gate
from experiments.projection_capacity_allocation_65.verify_states import ROOT, sha, write_json


def audit():
    config = json.loads((ROOT / 'config.json').read_text())
    states = json.loads((ROOT / 'state_verification.json').read_text())
    assert states['status'] == 'verified' and states['disjoint']
    for split in states['splits']:
        spans = split['spans']
        assert all(max(a['start'], b['start']) >= min(a['end_exclusive'], b['end_exclusive'])
                   for i, a in enumerate(spans) for b in spans[i + 1:]), 'within-split span overlap'
    for path, digest in config['sources'].items():
        assert sha(path) == digest
    curves = json.loads((ROOT / 'capacity_curves_raw.json').read_text())['projections']
    assert len(curves) == 224
    for row in curves:
        assert [c['sparsity'] for c in row['curves']] == list(GRID)
        for c in row['curves']:
            assert len(c['per_state']) == 80
            assert np.isclose(c['summary']['mean_kl'], np.mean([s['mean_kl'] for s in c['per_state']]), rtol=0, atol=1e-12)
    allocation = json.loads((ROOT / 'allocation.json').read_text())
    assert len(allocation['assignments']) == 224
    assert allocation['budget_error'] == 0
    for method in ('uniform', 'capacity'):
        manifest = json.loads((ROOT / f'{method}65_mask_manifest.json').read_text())
        assert len(manifest['entries']) == 224
        count = sum(row['selected_mask']['pruned'] for row in manifest['entries'])
        assert count == config['budget']['total_pruned'] == manifest['pruned']
    result = json.loads((ROOT / 'heldout_dlm_results.json').read_text())
    assert result['allocation_sha256'] == sha(ROOT / 'allocation.json')
    assert result['dense_sham_max_abs'] == 0
    recomputed = paired_gate(result['methods']['uniform']['per_state'], result['methods']['capacity']['per_state'])
    assert recomputed == result['gate']
    if not recomputed['passed']:
        assert not (ROOT / 'downstream.json').exists(), 'forbidden downstream after failed gate'
    else:
        downstream = json.loads((ROOT / 'downstream.json').read_text())
        if not downstream['mini_directionally_better']:
            assert len(downstream['evaluations']) == 1
        else:
            assert len(downstream['evaluations']) == 2
    required = ['config.json', 'state_verification.json', 'capacity_curves_raw.json', 'marginal_costs.json',
                'allocation.json', 'allocation_summary.json', 'uniform65_mask_manifest.json',
                'capacity65_mask_manifest.json', 'heldout_dlm_results.json', 'report.md']
    output = {'status': 'verified', 'supported': recomputed['passed'],
              'artifact_sha256': {name: sha(ROOT / name) for name in required}}
    write_json(ROOT / 'final_audit.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    audit()
