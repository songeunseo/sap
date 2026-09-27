"""Frozen integer-budget water filling and paired oracle decision gate."""
import math
from functools import reduce

import numpy as np
from scipy.stats import spearmanr

GRID = (.50, .55, .60, .65, .70, .75)


def distribution(values):
    a = np.asarray(values, dtype=float)
    mean, std = float(a.mean()), float(a.std())
    return dict(mean=mean, std=std, coefficient_of_variation=std / abs(mean) if mean else None,
                **{k: float(v) for k, v in zip(('min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'),
                                               np.quantile(a, [0, .1, .25, .5, .75, .9, 1]))})


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {'spearman': None, 'pearson': None}
    return {'spearman': float(spearmanr(a, b).statistic), 'pearson': float(np.corrcoef(a, b)[0, 1])}


def _reachable(counts, levels, remaining):
    """Exact multiple-choice subset sum; a bit marks each attainable count."""
    if remaining < 0:
        return False
    if remaining == 0:
        return True
    reachable = 1
    limit = (1 << (remaining + 1)) - 1
    for row, level in zip(counts, levels):
        base = row[level]
        nxt = 0
        for value in row[level:]:
            delta = value - base
            if delta <= remaining:
                nxt |= reachable << delta
        reachable = nxt & limit
        if (reachable >> remaining) & 1:
            return True  # All subsequent projections can add zero.
    return False


def allocate(curves, shapes):
    damage = np.asarray(curves, dtype=float)
    if damage.shape != (len(shapes), 6) or not np.isfinite(damage).all():
        raise ValueError('complete finite six-point curves required')
    counts = [[rows * int(cols * r) for r in GRID] for rows, cols in shapes]
    unit = reduce(math.gcd, (v for row in counts for v in row))
    units = [[v // unit for v in row] for row in counts]
    target = sum(row[3] for row in units)
    levels = [0] * len(shapes)
    remaining = target - sum(row[0] for row in units)
    assert _reachable(units, levels, remaining)
    trace, skipped = [], []
    while remaining:
        candidates = []
        for i, level in enumerate(levels):
            if level < 5:
                cost = float(damage[i, level + 1] - damage[i, level])
                nominal_gain = .05 * math.prod(shapes[i])
                candidates.append((cost / nominal_gain, i, cost))
        for per_parameter, i, cost in sorted(candidates):
            old = levels[i]
            gain = units[i][old + 1] - units[i][old]
            if gain > remaining:
                continue
            levels[i] += 1
            if _reachable(units, levels, remaining - gain):
                remaining -= gain
                trace.append({'module_index': i, 'from': GRID[old], 'to': GRID[old + 1],
                              'marginal_kl': cost, 'cost_per_nominal_parameter': per_parameter,
                              'actual_parameter_gain': gain * unit, 'remaining': remaining * unit})
                break
            levels[i] -= 1
            skipped.append({'step': len(trace), 'module_index': i, 'from': GRID[old],
                            'reason': 'would make exact row-floor uniform budget unreachable'})
        else:
            raise RuntimeError('no feasible next increment despite reachability invariant')
    pruned = sum(row[level] for row, level in zip(counts, levels))
    return {'levels': levels, 'sparsities': [GRID[k] for k in levels], 'pruned': pruned,
            'uniform_pruned': target * unit, 'budget_error': pruned - target * unit,
            'weights': sum(math.prod(s) for s in shapes), 'trace': trace,
            'feasibility_skips': skipped, 'integer_budget_unit': unit}


def paired_gate(uniform, capacity):
    if len(uniform) != 40 or len(capacity) != 40:
        raise ValueError('gate requires exactly 40 paired held-out states')
    for a, b in zip(uniform, capacity):
        if (a['sequence_index'], a['timestep']) != (b['sequence_index'], b['timestep']):
            raise ValueError('state pairing mismatch')
    delta = np.array([b['mean_kl'] - a['mean_kl'] for a, b in zip(uniform, capacity)])
    if not np.isfinite(delta).all():
        raise ValueError('nonfinite paired result')
    rng = np.random.default_rng(0)
    means = delta[rng.integers(0, 40, size=(20000, 40))].mean(1)
    ci = np.quantile(means, [.025, .975]).tolist()
    groups = {}
    for key in ('sequence_index', 'timestep'):
        groups[key] = []
        for value in sorted({a[key] for a in uniform}):
            ids = [i for i, row in enumerate(uniform) if row[key] == value]
            groups[key].append({key: value, 'uniform_mean_kl': float(np.mean([uniform[i]['mean_kl'] for i in ids])),
                                'capacity_mean_kl': float(np.mean([capacity[i]['mean_kl'] for i in ids])),
                                'difference': float(delta[ids].mean())})
    if len(groups['sequence_index']) != 8 or len(groups['timestep']) != 5:
        raise ValueError('expected eight sequences and five timesteps')
    seq = sum(row['difference'] < 0 for row in groups['sequence_index'])
    times = sum(row['difference'] < 0 for row in groups['timestep'])
    return {'mean_difference': float(delta.mean()), 'median_difference': float(np.median(delta)),
            'bootstrap_95_ci': ci, 'bootstrap_resamples': 20000, 'bootstrap_seed': 0,
            'bootstrap_unit': 'paired held-out state', 'states_improved': int((delta < 0).sum()),
            'states_worsened': int((delta > 0).sum()), 'states_tied': int((delta == 0).sum()),
            'sequence_means_improved': seq, 'timestep_means_improved': times,
            'per_sequence': groups['sequence_index'], 'per_timestep': groups['timestep'],
            'passed': bool(delta.mean() < 0 and ci[1] < 0 and seq > 4 and times > 2)}
