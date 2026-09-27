from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


BACKGROUNDS = ('dense', 'role_sparse')
CONDITIONS = ('baseline', 'sham', 'bundle_mixed', 'bundle_direct',
              'masked_only', 'unmasked_only', 'single_0', 'single_1',
              'single_2', 'single_3')


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def enumerate_budget_bundles(base_levels, new_levels, entries, role_m, role_u,
                             functional, source, sizes=(2, 3, 4)):
    changed = [i for i, (a, b) in enumerate(zip(base_levels, new_levels)) if a != b]
    count_delta = {
        i: int(entries[i]['masks'][new_levels[i]]['pruned'])
        - int(entries[i]['masks'][base_levels[i]]['pruned']) for i in changed
    }
    result = []
    for size in sizes:
        for indices in itertools.combinations(changed, size):
            if sum(count_delta[i] for i in indices) != 0:
                continue
            dm = sum(float(role_m[i][new_levels[i]]) - float(role_m[i][base_levels[i]])
                     for i in indices)
            du = sum(float(role_u[i][new_levels[i]]) - float(role_u[i][base_levels[i]])
                     for i in indices)
            dd = sum(float(functional[i][new_levels[i]]) - float(functional[i][base_levels[i]])
                     for i in indices)
            changes = [{'module_index': i, 'name': entries[i]['name'],
                        'base_level': int(base_levels[i]), 'new_level': int(new_levels[i]),
                        'base_sparsity': float(entries[i]['masks'][base_levels[i]]['nominal_sparsity']),
                        'new_sparsity': float(entries[i]['masks'][new_levels[i]]['nominal_sparsity']),
                        'pruned_delta': count_delta[i],
                        'base_mask_sha256': entries[i]['masks'][base_levels[i]]['mask_sha256'],
                        'new_mask_sha256': entries[i]['masks'][new_levels[i]]['mask_sha256']}
                       for i in indices]
            signature = tuple((i, int(new_levels[i])) for i in indices)
            result.append({'source': source, 'size': size, 'indices': list(indices),
                           'signature': [list(x) for x in signature], 'changes': changes,
                           'delta_masked_reconstruction': dm,
                           'delta_unmasked_reconstruction': du,
                           'delta_additive_single_projection_kl': dd,
                           'proxy_nonincrease': dm <= 0 and du <= 0 and (dm < 0 or du < 0)})
    return result


def select_bundles(candidates, globally_used):
    pool = [x for x in candidates if tuple(map(tuple, x['signature'])) not in globally_used]
    selected = []
    rules = (
        ('proxy_better_kl_worse', lambda x: x['proxy_nonincrease'] and
         x['delta_additive_single_projection_kl'] > 0,
         lambda x: (-x['delta_additive_single_projection_kl'], tuple(x['indices']))),
        ('proxy_better_kl_better', lambda x: x['proxy_nonincrease'] and
         x['delta_additive_single_projection_kl'] < 0,
         lambda x: (x['delta_additive_single_projection_kl'], tuple(x['indices']))),
        ('kl_neutral', lambda x: True,
         lambda x: (abs(x['delta_additive_single_projection_kl']), tuple(x['indices']))),
    )
    for label, predicate, order in rules:
        eligible = [x for x in pool if x not in selected and predicate(x)]
        fallback = False
        if not eligible:
            eligible = [x for x in pool if x not in selected]
            fallback = True
        if not eligible:
            raise RuntimeError('not enough unique exact-budget bundles')
        chosen = sorted(eligible, key=order)[0].copy()
        chosen['selection_category'] = label
        chosen['selection_fallback'] = fallback
        selected.append(chosen)
        globally_used.add(tuple(map(tuple, chosen['signature'])))
    return selected


def sequence_bootstrap(values, sequence_ids, resamples=20000, seed=1234):
    values = np.asarray(values, float)
    sequence_ids = np.asarray(sequence_ids, int)
    unique = np.unique(sequence_ids)
    means = np.array([values[sequence_ids == x].mean() for x in unique])
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(0, len(means), size=(resamples, len(means)))].mean(1)
    return {'mean': float(means.mean()), 'sequence_means': means.tolist(),
            'bootstrap_95_ci': np.quantile(sampled, [.025, .975]).tolist(),
            'resamples': resamples, 'seed': seed}


def exact_sign_flip(values):
    values = np.asarray(values, float)
    observed = abs(values.mean())
    signs = np.array(list(itertools.product((-1., 1.), repeat=len(values))))
    distribution = np.abs((signs * values).mean(1))
    return float(np.mean(distribution >= observed - 1e-15))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.)
    return adjusted.tolist()
