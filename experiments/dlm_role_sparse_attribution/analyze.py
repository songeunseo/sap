"""Exploratory ablation of sparse-context inputs; both dense roles always retained."""
import fcntl
import json
import time
from pathlib import Path

import numpy as np

from experiments.dlm_role_proxy_units.analyze import feature as previous_feature, metric, weights
from experiments.dlm_role_exchange_prediction_v2.analyze import load
from experiments.dlm_role_exchange_prediction_v2.common import ridge, bootstrap_cells, validate_sources, atomic_json, sha256

ROOT = Path('experiments/dlm_role_sparse_attribution')
PREVIOUS = Path('experiments/dlm_role_proxy_units')
MODELS = ['dense_roles', 'plus_sparse_M', 'plus_sparse_U', 'plus_sparse_MU']
PAIRS = [('plus_sparse_M', 'dense_roles'), ('plus_sparse_U', 'dense_roles'),
         ('plus_sparse_MU', 'plus_sparse_M'), ('plus_sparse_MU', 'plus_sparse_U')]


def feature(row, name):
    if name not in MODELS:
        raise ValueError(name)
    values = previous_feature(row, 'D-both')
    roles = {'dense_roles': [], 'plus_sparse_M': ['masked'], 'plus_sparse_U': ['unmasked'],
             'plus_sparse_MU': ['masked', 'unmasked']}[name]
    values += [row['features']['sparse'][f'{role}_{unit}'] for unit in ('token', 'energy') for role in roles]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite input')
    return values


def progress(stage):
    value = dict(stage=stage, time=time.time())
    atomic_json(ROOT / 'progress.json', value)
    print(json.dumps(value), flush=True)


def main():
    with (ROOT / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'results.json').exists():
            raise RuntimeError('Preserve completed results; create a new version')
        validate_sources()
        old_config = json.loads((PREVIOUS / 'config.json').read_text())
        hashes = dict(old_config['sources'])
        for p in [PREVIOUS / 'config.json', PREVIOUS / 'results.json', PREVIOUS / 'predictions.npz',
                  PREVIOUS / 'prediction_keys.json', *ROOT.glob('*.py'), ROOT / 'run.sh']:
            hashes[str(p)] = sha256(p)
        for p, digest in hashes.items():
            if sha256(p) != digest:
                raise RuntimeError('Changed source: ' + p)
        config = dict(models=MODELS, comparisons=PAIRS, alpha=1, roles='Both dense roles always retained',
                      unit='exact-budget bundle/state', family=4, seed=20260912, bootstrap=20000,
                      scope='exploratory attribution of predictor inputs, not causal role mediation or fresh confirmation',
                      hashes=hashes)
        if (ROOT / 'config.json').exists() and json.loads((ROOT / 'config.json').read_text()) != config:
            raise RuntimeError('Frozen config mismatch')
        atomic_json(ROOT / 'config.json', config)
        progress('loading')
        train, _ = load('development')
        test, bundles = load('final')
        y = np.array([r['delta_kl'] for r in test])
        by = np.array([r['delta_kl'] for r in bundles])
        pred = {m: np.full(len(test), np.nan) for m in MODELS}
        fits = {}
        for fold in range(4):
            tr = [r for r in train if r['layer'] // 8 != fold]
            ids = [i for i, r in enumerate(test) if r['layer'] // 8 == fold]
            for m in MODELS:
                pred[m][ids], fits[f'{fold}:{m}'] = ridge([feature(r, m) for r in tr], [r['delta_kl'] for r in tr],
                                                         [feature(test[i], m) for i in ids], weights(tr))
            progress(f'fold_{fold+1}_of_4')
        if not all(np.isfinite(p).all() for p in pred.values()):
            raise AssertionError('Missing prediction')
        lookup = {(r['document'], r['timestep'], r['exchange_index']): i for i, r in enumerate(test)}
        indices = [[lookup[b['document'], b['timestep'], e] for e in b['exchange_indices']] for b in bundles]
        bp = {m: np.array([p[ix].sum() for ix in indices]) for m, p in pred.items()}
        old = np.load(PREVIOUS / 'predictions.npz')
        old_keys = json.loads((PREVIOUS / 'prediction_keys.json').read_text())
        if old_keys['single'] != [list(k) for k in lookup]:
            raise AssertionError('Prediction row order changed')
        if not np.array_equal(y, old['y']) or not np.array_equal(by, old['bundle_y']):
            raise AssertionError('Measurement target changed')
        for new, prior in [('dense_roles', 'D-both'), ('plus_sparse_MU', 'DS-both')]:
            if not np.allclose(pred[new], old['single_' + prior], rtol=1e-9, atol=1e-12):
                raise AssertionError('Existing predictor reproduction failed: ' + new)
            if not np.allclose(bp[new], old['bundle_' + prior], rtol=1e-9, atol=1e-12):
                raise AssertionError('Bundle reproduction failed')
        summary = {'single': {m: metric(y, p, test) for m, p in pred.items()},
                   'bundle': {m: metric(by, p, bundles) for m, p in bp.items()}}
        comparisons = {}
        for a, b in PAIRS:
            delta = (bp[a] - by)**2 - (bp[b] - by)**2
            ci = bootstrap_cells(delta, bundles, family=4)
            comparisons[f'{a}_minus_{b}'] = dict(**ci, relative_mse_reduction=float(-(weights(bundles) @ delta) / summary['bundle'][b]['mse']))
        progress('comparisons_complete')
        strata = {}
        for field in ('document', 'timestep', 'layer'):
            for value in sorted({r[field] for r in bundles}):
                ids = [i for i, r in enumerate(bundles) if r[field] == value]
                rows = [bundles[i] for i in ids]
                strata[f'{field}:{value}'] = {m: metric(by[ids], p[ids], rows) for m, p in bp.items()}
        mse = {m: summary['bundle'][m]['mse'] for m in MODELS}
        result = dict(status='complete', config_sha256=sha256(ROOT / 'config.json'), metrics=summary,
                      comparisons=comparisons, strata=strata, fits=fits,
                      descriptive_mse_interaction=mse['plus_sparse_MU']-mse['plus_sparse_M']-mse['plus_sparse_U']+mse['dense_roles'],
                      caveat='Interaction describes retrained predictor errors, not biological/causal token synergy. Fixed ridge can redistribute redundant inputs.',
                      decision='Role separation retained. No allocation/downstream/model selection is automated.')
        for p, digest in hashes.items():
            if sha256(p) != digest:
                raise RuntimeError('Source changed during analysis: ' + p)
        lines = ['# Sparse-context 역할별 정보 기여', '', '모든 후보가 dense masked/unmasked 절대·상대 통계를 보존한다.',
                 '기존 final 재사용 탐색 분석. 역할 개입의 인과효과가 아니라 예측기 입력 ablation이다.', '',
                 '| 후보 | Bundle MSE | 방향 정확도 |', '|---|---:|---:|']
        lines += [f"| {m} | {summary['bundle'][m]['mse']:.9g} | {summary['bundle'][m]['sign_accuracy']:.4f} |" for m in MODELS]
        lines += ['', '## 사전 지정 4비교', '']
        for pair, c in comparisons.items():
            lines.append(f"- {pair}: MSE 감소 {100*c['relative_mse_reduction']:.3f}%, 동시 조건부 CI {c['simultaneous_ci']}")
        lines += ['', '## 한계', '', '- 기존 final을 반복 사용한 탐색 분석이며 fresh confirmatory evidence가 아니다.',
                  '- 신뢰구간은 고정된 OOF 모델에 조건부. 학습 데이터 재표집/재학습 불확실성 미포함.',
                  '- 입력 추가는 정보와 ridge의 정규화 효과를 함께 바꾼다. 역할의 인과적 중요도 또는 necessity로 해석하지 않는다.',
                  '- 어느 sparse 입력의 추가 이득이 약해도 dense 역할 분리는 유지한다.',
                  '- KL 예측 개선은 GSM8K 개선이나 올바른 max/mean/minimax의 증명이 아니다.']
        (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')
        np.savez_compressed(ROOT / 'predictions.npz', **{f'single_{m}': p for m, p in pred.items()},
                            **{f'bundle_{m}': p for m, p in bp.items()})
        atomic_json(ROOT / 'results.json', result)
        progress('complete')


if __name__ == '__main__':
    main()
