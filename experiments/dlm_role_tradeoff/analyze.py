"""Offline discrete role tradeoff analysis; no model or generation calls."""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.optimize._highspy import _core as highs
from types import SimpleNamespace
from scipy.sparse import csr_matrix, eye, kron, vstack, hstack

from experiments.dlm_role_mechanism_analysis.analyze import load_suite
from experiments.dlm_role_mechanism_analysis.core import pooled_curves, allocation_difference
from experiments.dlm_role_validation.core import allocate_grid

ROOT = Path('experiments/dlm_role_tradeoff')


def save(name, obj):
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def totals(curves, levels):
    ix = np.arange(len(levels)), np.asarray(levels)
    return [float(curves[r][ix].sum()) for r in ('masked', 'unmasked')]


def solve(curves, counts, budget, kind, reference, cap=None, warm=None):
    n, k = counts.shape
    unit = math.gcd(*counts.ravel().tolist())
    base = vstack([kron(eye(n), np.ones((1, k))), csr_matrix(counts.reshape(1, -1) / unit)], format='csr')
    lo = np.r_[np.ones(n), budget / unit]
    hi = lo.copy()
    m = curves['masked'].ravel() / reference[0]
    u = curves['unmasked'].ravel() / reference[1]
    dim = n * k
    if kind == 'minimax':
        base = hstack([base, csr_matrix((n + 1, 1))], format='csr')
        base = vstack([base, csr_matrix(np.r_[m, -1][None]), csr_matrix(np.r_[u, -1][None])], format='csr')
        lo = np.r_[lo, -np.inf, -np.inf]
        hi = np.r_[hi, 0, 0]
        objective = np.r_[np.zeros(dim), 1.]
        integrality = np.r_[np.ones(dim), 0]
        bounds = Bounds(np.zeros(dim + 1), np.r_[np.ones(dim), np.inf])
    else:
        objective = {'masked': m, 'unmasked': u,
                     'local_max': np.maximum(curves['masked'], curves['unmasked']).ravel() / sum(reference)}[kind]
        integrality = np.ones(dim)
        bounds = Bounds(0, 1)
        if cap is not None:
            base = vstack([base, csr_matrix(u[None])], format='csr')
            lo = np.r_[lo, -np.inf]
            hi = np.r_[hi, cap / reference[1]]
    start = time.time()
    solver = highs._Highs()
    solver.setOptionValue('output_flag', False)
    solver.setOptionValue('threads', 1)
    solver.setOptionValue('mip_rel_gap', 1e-8)
    solver.setOptionValue('time_limit', 60.)
    size = len(objective)
    solver.addCols(size, objective, np.broadcast_to(bounds.lb,(size,)).copy(),
                   np.broadcast_to(bounds.ub,(size,)).copy(), 0, np.zeros(size+1,dtype=np.int32),
                   np.array([],dtype=np.int32), np.array([],dtype=float))
    solver.addRows(base.shape[0], lo, hi, base.nnz, base.indptr.astype(np.int32), base.indices.astype(np.int32), base.data)
    solver.changeColsIntegrality(size, np.arange(size,dtype=np.int32), integrality.astype(np.uint8))
    if warm is not None:
        seed = np.eye(k)[np.asarray(warm)].ravel()
        if kind == 'minimax': seed = np.r_[seed, max(np.array(totals(curves,warm))/reference)]
        assert np.all(base @ seed <= hi + 1e-7) and np.all(base @ seed >= lo - 1e-7)
        solver.setSolution(size, np.arange(size,dtype=np.int32), seed)
    solver.run()
    info, sol = solver.getInfo(), solver.getSolution()
    result = SimpleNamespace(x=np.array(sol.col_value) if sol.value_valid else None,
        status=int(solver.getModelStatus()), message=solver.modelStatusToString(solver.getModelStatus()),
        mip_gap=info.mip_gap, fun=info.objective_function_value, mip_dual_bound=info.mip_dual_bound)
    if result.x is None:
        raise RuntimeError(f'{kind}: no feasible solution: {result.message}')
    x = result.x[:dim].reshape(n, k)
    levels = x.argmax(axis=1)
    if not np.allclose(x, np.eye(k)[levels], atol=1e-5):
        raise RuntimeError('noninteger solution')
    pruned = int(counts[np.arange(n), levels].sum())
    if pruned != budget:
        raise RuntimeError(f'budget mismatch {pruned} != {budget}')
    values = totals(curves, levels)
    if cap is not None and values[1] > cap + 1e-6 * reference[1]:
        raise RuntimeError('epsilon constraint violation')
    doc = {'levels': levels.tolist(), 'risks': values, 'relative_to_reference': (np.array(values)/reference).tolist(),
           'pruned': pruned, 'status': int(result.status), 'message': result.message,
           'gap': float(result.mip_gap), 'objective': float(result.fun),
           'dual_bound': float(result.mip_dual_bound), 'seconds': time.time()-start}
    print(kind, 'risks', values, 'gap', doc['gap'], 'seconds', round(doc['seconds'], 2), flush=True)
    return doc


def self_test():
    import itertools
    c = {'masked': np.array([[0., 2., 4.], [0., 3., 5.]]),
         'unmasked': np.array([[0., 4., 6.], [0., 1., 3.]])}
    counts = np.array([[0, 1, 2], [0, 1, 2]])
    feasible = [x for x in itertools.product(range(3), repeat=2) if sum(x)==2]
    for kind in ['masked', 'unmasked', 'minimax', 'local_max']:
        s = solve(c, counts, 2, kind, [5., 5.])
        def score(x):
            r = totals(c, x)
            if kind == 'minimax': return max(r)/5
            if kind == 'local_max': return sum(max(c['masked'][g,l], c['unmasked'][g,l]) for g,l in enumerate(x))/10
            return r[0 if kind=='masked' else 1]/5
        assert abs(score(s['levels']) - min(map(score, feasible))) < 1e-8
    print('brute-force solver checks passed', flush=True)


def main():
    ROOT.mkdir(exist_ok=True)
    self_test()
    data = load_suite('random65')
    save('input_manifest.json', {'source_files': data['source_files'], 'invariant': data['invariant']})
    save('config.json', {'target': .65, 'grid': data['grid'], 'proxy': 'pooled dense-energy-normalized reconstruction',
          'solver_time_limit': 60, 'mip_rel_gap': 1e-8, 'epsilon_points': 11, 'sequence_folds': 8,
          'downstream': False, 'frontier': 'epsilon samples, not exhaustive discrete frontier',
          'ties': 'solver-selected optimum; allocation identity not claimed unique',
          'solver': 'SciPy bundled HiGHS direct interface with feasible incumbent warm start',
          'solver_status_semantics': 'HiGHS model status; inspect message and gap, not scipy status codes'})
    counts = np.array([[int(a)*int(int(b)*r) for r in data['grid']] for a,b in data['shapes']], dtype=np.int64)
    budget = int(counts[:,3].sum())
    fields = data['fields']['actual']
    c = pooled_curves(fields, list(range(80)))
    old = data['allocation']['methods']['actual']
    assert int(counts[np.arange(224), old['levels']].sum()) == budget
    ref = np.array(totals(c, old['levels']))
    results = {'current_role': {'levels': old['levels'], 'risks': ref.tolist(), 'pruned': budget}}
    for kind in ['masked', 'unmasked', 'local_max', 'minimax']:
        results[kind] = solve(c, counts, budget, kind, ref, warm=old['levels'])
        results[kind]['difference_from_current'] = allocation_difference(results[kind], old, data['shapes'], data['grid'])
        save('solutions.json', results)
    results['endpoint_difference'] = allocation_difference(results['masked'], results['unmasked'], data['shapes'], data['grid'])
    frontier = []
    for cap in np.linspace(results['unmasked']['risks'][1], results['masked']['risks'][1], 11):
        s = solve(c, counts, budget, 'masked', ref, float(cap), warm=results['unmasked']['levels'])
        s['unmasked_cap'] = float(cap)
        frontier.append(s)
        save('frontier.json', frontier)
    folds = []
    for seq in np.unique(data['sequences']):
        train = np.flatnonzero(data['sequences'] != seq).tolist()
        test = np.flatnonzero(data['sequences'] == seq).tolist()
        tr, te = pooled_curves(fields, train), pooled_curves(fields, test)
        baseline = allocate_grid(tr['max'], data['shapes'], data['grid'])
        train_ref = np.array(totals(tr, baseline['levels']))
        test_ref = np.array(totals(te, baseline['levels']))
        fold = {'sequence': int(seq), 'train_role': baseline['levels'], 'test_role_risks': test_ref.tolist(), 'solutions': {}}
        for kind in ['masked', 'unmasked', 'minimax']:
            s = solve(tr, counts, budget, kind, train_ref, warm=baseline['levels'])
            s['test_risks'] = totals(te, s['levels'])
            s['test_relative_change'] = (np.array(s['test_risks'])/test_ref-1).tolist()
            s['difference_from_full_solution'] = allocation_difference(s, results[kind], data['shapes'], data['grid'])
            fold['solutions'][kind] = s
        folds.append(fold)
        save('crossfit.json', folds)
        print('completed fold', seq, flush=True)
    rng = np.random.default_rng(1234)
    indices = rng.integers(0, 8, size=(20000,8))
    summary = {}
    for kind in ['masked','unmasked','minimax']:
        delta = np.array([f['solutions'][kind]['test_relative_change'] for f in folds])
        summary[kind] = {'mean_relative_change': delta.mean(0).tolist(),
                         'descriptive_bootstrap_95_ci': np.quantile(delta[indices].mean(1), [.025,.975], axis=0).tolist(),
                         'both_improve_sequences': int((delta < 0).all(1).sum()),
                         'role_improve_sequences': (delta < 0).sum(0).tolist()}
    results['crossfit_summary'] = summary
    results['budget'] = budget
    results['global_sparsity'] = budget / sum(a*b for a,b in data['shapes'])
    save('solutions.json', results)
    lines = ['# Role Allocation Trade-off @65%', '', '## 설정',
             '기존 224×6×80 role 통계. 동일 Wanda row-floor 제거 예산. 두 목적은 projection별 normalized reconstruction의 합이다. GPU/downstream 미실행.',
             f'정확한 제거 수: {budget}; global sparsity: {results["global_sparsity"]:.10f}.', '',
             '## 전체 데이터 결과', '|할당|Masked 합|Unmasked 합|기존 Role 대비 M/U 변화|변경 projections|solver gap|', '|---|---:|---:|---|---:|---:|']
    for kind in ['current_role','masked','unmasked','local_max','minimax']:
        s = results[kind]; change = (np.array(s['risks'])/ref-1)*100
        lines.append(f'|{kind}|{s["risks"][0]:.8g}|{s["risks"][1]:.8g}|{change[0]:+.4f}% / {change[1]:+.4f}%|{s.get("difference_from_current",{}).get("changed_projections",0)}|{s.get("gap",0):.3g}|')
    lines += ['', f'두 endpoint 사이 변경 projections: {results["endpoint_difference"]["changed_projections"]}; nested mask XOR: {results["endpoint_difference"]["xor_fraction"]:.6%}.',
              '', '## Sequence crossfit', '각 fold는 7개 sequence로 최적화하고 제외된 1개 sequence의 proxy로 평가한다. 기준 greedy Role도 train에서 다시 만든다.',
              '```json', json.dumps(summary, indent=2), '```', '', '## 해석 한계 및 결정',
              '- epsilon-constraint 11개 지점은 전체 discrete Pareto frontier가 아니다. Scalar endpoint는 ties 때문에 유일한 allocation 또는 strong Pareto point라 단정하지 않는다.',
              '- 솔버 gap/status를 확인하고 최적성 인증과 feasible 개선을 구분한다.',
              '- Crossfit bootstrap은 8개 fold 기술적 구간이다. Training overlap과 frozen 전체 calibration Wanda ranking 때문에 독립적인 새 데이터 검증은 아니다.',
              '- 두 proxy의 동시 개선은 실제 KL/GSM8K 개선을 의미하지 않는다. Projection 간 가산성은 검증되지 않았다.',
              '- 방법 선택 또는 downstream 실행 없이 allocation geometry 근거로 사용한다.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    save('status.json', {'status':'complete', 'folds':len(folds), 'frontier_samples':len(frontier)})


if __name__ == '__main__':
    main()
