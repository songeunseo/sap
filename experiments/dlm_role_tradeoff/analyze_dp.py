"""Exact-budget bi-objective tradeoff analysis via multiple-choice knapsack DP."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.dlm_role_mechanism_analysis.analyze import load_suite
from experiments.dlm_role_mechanism_analysis.core import pooled_curves, allocation_difference
from experiments.dlm_role_validation.core import allocate_grid

ROOT = Path('experiments/dlm_role_tradeoff')


def save(name: str, obj: object) -> None:
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def risks(curves: dict[str, np.ndarray], levels: list[int] | np.ndarray) -> np.ndarray:
    levels = np.asarray(levels, dtype=int)
    ix = np.arange(len(levels)), levels
    return np.array([curves['masked'][ix].sum(), curves['unmasked'][ix].sum()])


def exact_scalar(cost: np.ndarray, counts: np.ndarray, budget: int) -> dict:
    """Solve the exact-budget multiple-choice knapsack for one additive cost."""
    n, k = cost.shape
    base = counts[:, 0]
    gains = counts - base[:, None]
    remaining_raw = int(budget - base.sum())
    unit = math.gcd(remaining_raw, *gains.ravel().tolist())
    gains = gains // unit
    remaining = remaining_raw // unit
    inf = np.inf
    dp = np.full(remaining + 1, inf)
    dp[0] = 0.0
    choices = np.full((n, remaining + 1), 255, dtype=np.uint8)
    reachable_max = 0
    start = time.time()
    for module in range(n):
        new_max = min(remaining, reachable_max + int(gains[module].max()))
        nxt = np.full(remaining + 1, inf)
        selected = np.full(remaining + 1, 255, dtype=np.uint8)
        for level in range(k):
            gain = int(gains[module, level])
            stop = min(reachable_max, new_max - gain)
            if stop < 0:
                continue
            candidate = dp[:stop + 1] + float(cost[module, level])
            target = nxt[gain:gain + stop + 1]
            better = candidate < target
            target[better] = candidate[better]
            selected[gain:gain + stop + 1][better] = level
        dp, reachable_max = nxt, new_max
        choices[module] = selected
    if not np.isfinite(dp[remaining]):
        raise RuntimeError('exact budget is unreachable')
    levels = np.empty(n, dtype=int)
    cursor = remaining
    for module in range(n - 1, -1, -1):
        level = int(choices[module, cursor])
        if level == 255:
            raise RuntimeError('broken DP backpointer')
        levels[module] = level
        cursor -= int(gains[module, level])
    if cursor != 0 or int(counts[np.arange(n), levels].sum()) != budget:
        raise RuntimeError('DP reconstruction budget mismatch')
    objective = float(cost[np.arange(n), levels].sum())
    if abs(objective - dp[remaining]) > 1e-8 * max(abs(objective), 1):
        raise RuntimeError('DP objective mismatch')
    return {'levels': levels.tolist(), 'objective': objective, 'pruned': budget,
            'integer_budget_unit': unit, 'seconds': time.time() - start,
            'optimality': 'exact for the supplied additive scalar objective'}


def solve_alpha(curves, counts, budget, reference, alpha):
    cost = alpha * curves['masked'] / reference[0] + (1-alpha) * curves['unmasked'] / reference[1]
    result = exact_scalar(cost, counts, budget)
    result['alpha_masked'] = float(alpha)
    result['risks'] = risks(curves, result['levels']).tolist()
    result['relative_risks'] = (np.array(result['risks']) / reference).tolist()
    print('alpha', f'{alpha:.12g}', 'relative', result['relative_risks'],
          'seconds', round(result['seconds'], 2), flush=True)
    return result


def supported_frontier(curves, counts, budget, reference, limit=256):
    by_levels = {}
    endpoints = [solve_alpha(curves, counts, budget, reference, x) for x in (0., 1.)]
    for x in endpoints:
        by_levels[tuple(x['levels'])] = x
    pending = [(endpoints[0], endpoints[1])]
    while pending and len(by_levels) < limit:
        left, right = pending.pop()
        l = np.array(left['relative_risks']); r = np.array(right['relative_risks'])
        dm, du = l[0] - r[0], l[1] - r[1]
        denominator = dm - du
        if abs(denominator) < 1e-15:
            continue
        alpha = float(-du / denominator)
        if not 1e-12 < alpha < 1 - 1e-12:
            continue
        new = solve_alpha(curves, counts, budget, reference, alpha)
        key = tuple(new['levels'])
        if key in (tuple(left['levels']), tuple(right['levels'])) or key in by_levels:
            continue
        by_levels[key] = new
        pending.extend([(left, new), (new, right)])
        save('dp_frontier_partial.json', sorted(by_levels.values(), key=lambda x: x['risks'][0]))
    frontier = sorted(by_levels.values(), key=lambda x: x['risks'][0])
    return frontier, bool(pending)


def main():
    data = load_suite('random65')
    fields = data['fields']['actual']
    curves = pooled_curves(fields, list(range(80)))
    counts = np.array([[int(a) * int(int(b) * r) for r in data['grid']]
                       for a, b in data['shapes']], dtype=np.int64)
    budget = int(counts[:, 3].sum())
    old = data['allocation']['methods']['actual']
    reference = risks(curves, old['levels'])
    save('dp_status.json', {'status': 'running', 'stage': 'full_data'})

    frontier, capped = supported_frontier(curves, counts, budget, reference)
    local = exact_scalar(np.maximum(curves['masked'], curves['unmasked']), counts, budget)
    local['risks'] = risks(curves, local['levels']).tolist()
    local['relative_risks'] = (np.array(local['risks']) / reference).tolist()
    for row in frontier:
        row['max_relative_risk'] = max(row['relative_risks'])
        row['difference_from_current'] = allocation_difference(row, old, data['shapes'], data['grid'])
    minimax = min(frontier, key=lambda x: (x['max_relative_risk'], sum(x['relative_risks'])))
    local['difference_from_current'] = allocation_difference(local, old, data['shapes'], data['grid'])
    save('dp_status.json', {'status': 'running', 'stage': 'crossfit',
                            'frontier_points': len(frontier), 'frontier_capped': capped})

    folds = []
    policy_alpha = minimax['alpha_masked']
    for seq in sorted(set(data['sequences'])):
        train_ids = np.flatnonzero(data['sequences'] != seq).tolist()
        test_ids = np.flatnonzero(data['sequences'] == seq).tolist()
        train, test = pooled_curves(fields, train_ids), pooled_curves(fields, test_ids)
        baseline = allocate_grid(train['max'], data['shapes'], data['grid'])
        train_ref = risks(train, baseline['levels'])
        test_ref = risks(test, baseline['levels'])
        solutions = {}
        for name, alpha in [('unmasked', 0.), ('policy', policy_alpha), ('masked', 1.)]:
            solution = solve_alpha(train, counts, budget, train_ref, alpha)
            solution['test_risks'] = risks(test, solution['levels']).tolist()
            solution['test_relative_change_vs_fold_role'] = (np.array(solution['test_risks']) / test_ref - 1).tolist()
            solutions[name] = solution
        folds.append({'sequence': int(seq), 'fold_role_test_risks': test_ref.tolist(), 'solutions': solutions})
        save('dp_crossfit_partial.json', folds)

    summary = {}
    for name in ('unmasked', 'policy', 'masked'):
        values = np.array([x['solutions'][name]['test_relative_change_vs_fold_role'] for x in folds])
        summary[name] = {'mean_relative_change': values.mean(0).tolist(),
                         'median_relative_change': np.median(values, axis=0).tolist(),
                         'both_improve_sequences': int(np.all(values < 0, axis=1).sum()),
                         'improve_sequences_by_role': np.sum(values < 0, axis=0).tolist()}

    result = {'status': 'complete', 'budget': budget,
              'global_sparsity': budget / sum(a*b for a,b in data['shapes']),
              'current_role': {'levels': old['levels'], 'risks': reference.tolist()},
              'supported_frontier': frontier, 'frontier_search_capped': capped,
              'supported_minimax': minimax, 'exact_local_max': local,
              'crossfit_policy_alpha': policy_alpha, 'crossfit': folds,
              'crossfit_summary': summary,
              'limitations': ['supported Pareto points only; unsupported discrete Pareto points are not enumerated',
                              'crossfit calibration folds overlap and are descriptive',
                              'proxy improvement is not downstream improvement']}
    save('dp_results.json', result)
    lines = ['# Exact-budget Role Trade-off DP @65%', '',
             f'- Supported frontier points: {len(frontier)}; search capped: {capped}',
             f'- Global sparsity: {result["global_sparsity"]:.10f}',
             f'- Current Role risks: {reference.tolist()}',
             f'- Supported minimax relative risks: {minimax["relative_risks"]}',
             f'- Supported minimax changed projections: {minimax["difference_from_current"]["changed_projections"]}',
             f'- Exact local-Max relative risks: {local["relative_risks"]}',
             f'- Exact local-Max changed projections: {local["difference_from_current"]["changed_projections"]}',
             '', '## Crossfit', '```json', json.dumps(summary, indent=2), '```', '',
             '## Limits', '- Frontier contains all discovered supported points, not unsupported discrete Pareto points.',
             '- Exact optimality applies to every scalar DP solve and the exact local-Max solve.',
             '- Crossfit is descriptive and proxy-only; no downstream evaluation was run.']
    (ROOT / 'dp_report.md').write_text('\n'.join(lines) + '\n')
    save('dp_status.json', {'status': 'complete', 'frontier_points': len(frontier),
                            'frontier_capped': capped, 'crossfit_folds': len(folds)})


if __name__ == '__main__':
    main()
