"""Exploratory, role-preserving offline analysis of frozen V2 measurements."""
import fcntl
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from experiments.dlm_role_exchange_prediction.analyze import controls, TYPES
from experiments.dlm_role_exchange_prediction_v2.analyze import load, metric as source_metric
from experiments.dlm_role_exchange_prediction_v2.common import (
    ROOT as SOURCE, STORE, ridge, equal_layer_weights, bootstrap_cells,
    validate_sources, atomic_json, sha256,
)

ROOT = Path('experiments/dlm_role_proxy_units')
MODELS = [f'{c}-{u}' for c in ('D', 'DS') for u in ('raw', 'relative', 'both')]
COMPARISONS = [('D-relative', 'D-raw'), ('DS-relative', 'DS-raw'),
               ('DS-both', 'DS-raw'), ('DS-both', 'D-both')]


def metric(y, p, rows):
    # Constant predictions have undefined correlation/calibration, not zero skill by definition.
    if np.ptp(p) == 0:
        w = weights(rows)
        return dict(mse=float(w @ ((y-p)**2)), mae=float(w @ abs(y-p)),
                    sign_accuracy=float(w @ (np.sign(y) == np.sign(p))),
                    spearman=None, calibration_slope=None)
    return {k: float(v) if np.isfinite(v) else None for k, v in source_metric(y, p, rows).items()}


def feature(row, model):
    context, unit = model.split('-')
    contexts = ['dense'] if context == 'D' else ['dense', 'sparse']
    units = ['token'] if unit == 'raw' else ['energy'] if unit == 'relative' else ['token', 'energy']
    # Preserve P4's order (unit, context, role) to reproduce existing results.
    result = controls(row)
    result += [row['features'][c][f'{r}_{u}'] for u in units
               for c in contexts for r in ('masked', 'unmasked')]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite feature; do not silently remove observations')
    return result


def weights(rows):
    w = equal_layer_weights(rows)
    return w / w.sum()


def decomposition(actual, single_sum, predicted, rows):
    w = weights(rows)
    interaction = actual - single_sum
    prediction = single_sum - predicted
    a = float(w @ (interaction ** 2))
    b = float(w @ (prediction ** 2))
    cross = float(2 * (w @ (interaction * prediction)))
    mse = float(w @ ((actual - predicted) ** 2))
    if not np.isclose(mse, a + b + cross, rtol=1e-10, atol=1e-15):
        raise AssertionError('Error decomposition identity failed')
    return dict(mse=mse, nonadditivity_mse=a, single_prediction_sum_mse=b,
                cross_term=cross, mean_nonadditivity=float(w @ interaction),
                caution='correlated residual terms, not independent error shares or an oracle bound')


def progress(stage, **values):
    obj = dict(stage=stage, time=time.time(), **values)
    atomic_json(ROOT / 'progress.json', obj)
    print(json.dumps(obj), flush=True)


def rank_audit(rows):
    """Within-state/direction ranks; average changes per projection and document."""
    cells = defaultdict(list)
    for row in rows:
        cells[row['document'], row['timestep'], int(np.sign(row['parameter_delta']))].append(row)
    output = {}
    for context in ('dense', 'sparse'):
        for role in ('masked', 'unmasked'):
            records, by_module = [], defaultdict(list)
            by_doc_module = defaultdict(list)
            for (doc, timestep, direction), group in sorted(cells.items()):
                raw = [r['features'][context][role + '_token'] for r in group]
                relative = [r['features'][context][role + '_energy'] for r in group]
                delta = (rankdata(relative) - rankdata(raw)) / max(1, len(group) - 1)
                rho = spearmanr(raw, relative).statistic
                records.append(dict(document=doc, timestep=timestep, direction=direction,
                                    rho=float(rho) if np.isfinite(rho) else None,
                                    mean_absolute_percentile_change=float(np.abs(delta).mean())))
                for r, d in zip(group, delta):
                    by_module[r['name']].append((float(d), r['layer'], r['projection_type']))
                    by_doc_module[doc, r['name'], direction].append(float(d))
            modules = {name: dict(mean_signed=float(np.mean([x[0] for x in values])),
                                  mean_absolute=float(np.mean([abs(x[0]) for x in values])),
                                  layer=values[0][1], projection_type=values[0][2])
                       for name, values in by_module.items()}
            docs = sorted({r['document'] for r in rows})
            stability = []
            for doc in docs:
                keys = sorted((name, direction) for d, name, direction in by_doc_module if d == doc)
                own = [np.mean(by_doc_module[doc, n, s]) for n, s in keys]
                other = [np.mean([np.mean(by_doc_module[d, n, s]) for d in docs if d != doc]) for n, s in keys]
                rho = spearmanr(own, other).statistic
                stability.append(dict(document=doc, loo_spearman=float(rho) if np.isfinite(rho) else None))
            output[f'{context}:{role}'] = dict(cells=records, projections=modules, loo_stability=stability,
                by_type={t: float(np.mean([v['mean_absolute'] for v in modules.values() if v['projection_type'] == t])) for t in TYPES},
                by_layer={str(l): float(np.mean([v['mean_absolute'] for v in modules.values() if v['layer'] == l])) for l in range(32)})
    return output


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'analysis.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'results.json').exists():
            raise RuntimeError('Completed outputs already exist; use a new version, not overwrite')
        validate_sources()
        paths = [SOURCE / x for x in ('config.json', 'complete_development.json', 'complete_final.json', 'analysis.json', 'next_analysis_plan.md')]
        paths += list(ROOT.glob('*.py')) + [ROOT / 'run.sh']
        paths += sorted(STORE.glob('development/state_*.pt')) + sorted(STORE.glob('final/state_*.pt'))
        hashes = {str(p): sha256(p) for p in paths}
        config = dict(status='frozen', models=MODELS, comparisons=COMPARISONS, primary_unit='exact-budget bundle/state',
                      alpha=1, bootstrap_seed=20260912, resamples=20000, family=4,
                      validation='seen final: exploratory, no fresh confirmation; conditional fitted-model bootstrap',
                      roles='masked/unmasked preserved in all six candidates', sources=hashes)
        if (ROOT / 'config.json').exists() and json.loads((ROOT / 'config.json').read_text()) != config:
            raise RuntimeError('Frozen input/config changed')
        atomic_json(ROOT / 'config.json', config)
        progress('loading')
        train, train_bundles = load('development')
        test, bundles = load('final')
        if {r['document'] for r in train} & {r['document'] for r in test}:
            raise AssertionError('Document IDs overlap')
        # V2 config also records true-parent-article validation, not only these IDs.
        y = np.array([r['delta_kl'] for r in test])
        names = MODELS + ['structure', 'zero', 'train_mean']
        pred = {m: np.full(len(y), np.nan) for m in names}
        fits = {}
        for fold in range(4):
            tr = [r for r in train if r['layer'] // 8 != fold]
            ids = [i for i, r in enumerate(test) if r['layer'] // 8 == fold]
            te = [test[i] for i in ids]
            target = [r['delta_kl'] for r in tr]
            for m in MODELS + ['structure']:
                f = controls if m == 'structure' else lambda r: feature(r, m)
                pred[m][ids], fits[f'{fold}:{m}'] = ridge([f(r) for r in tr], target, [f(r) for r in te], weights(tr))
            pred['zero'][ids] = 0
            pred['train_mean'][ids] = weights(tr) @ np.array(target)
            progress('fitting', completed_folds=fold + 1, total_folds=4)
        if any(not np.isfinite(p).all() for p in pred.values()):
            raise AssertionError('Incomplete OOF predictions')
        # Reproduction gate: unchanged candidate specifications must match V2 MSE.
        previous = json.loads((SOURCE / 'analysis.json').read_text())
        metrics = {m: metric(y, p, test) for m, p in pred.items()}
        # Actual key checked in tests/launch; fail loudly on mismatched artifact schema.
        old_metrics = previous['models']
        for m, old in [('D-raw', 'P2_role_dense'), ('DS-raw', 'P3_role_dense_sparse'), ('DS-both', 'P4_role_energy')]:
            if not np.isclose(metrics[m]['mse'], old_metrics[old]['mse'], rtol=1e-9, atol=1e-15):
                raise AssertionError(f'V2 reproduction failed: {m}')
        lookup = {(r['document'], r['timestep'], r['exchange_index']): i for i, r in enumerate(test)}
        indices = [[lookup[b['document'], b['timestep'], e] for e in b['exchange_indices']] for b in bundles]
        by = np.array([r['delta_kl'] for r in bundles])
        summed = np.array([y[ix].sum() for ix in indices])
        bp = {m: np.array([p[ix].sum() for ix in indices]) for m, p in pred.items()}
        bundle_means = {}
        for fold in range(4):
            tb = [r for r in train_bundles if r['layer'] // 8 != fold]
            bundle_means[fold] = float(weights(tb) @ np.array([r['delta_kl'] for r in tb]))
        bp['bundle_train_mean'] = np.array([bundle_means[b['layer'] // 8] for b in bundles])
        bm = {m: metric(by, p, bundles) for m, p in bp.items()}
        comparisons = {'bundle_primary': {}, 'single_secondary': {}}
        for unit, target, predictions, rows in [('bundle_primary', by, bp, bundles), ('single_secondary', y, pred, test)]:
            for left, right in COMPARISONS:
                delta = (predictions[left] - target) ** 2 - (predictions[right] - target) ** 2
                ci = bootstrap_cells(delta, rows, family=4)
                denom = weights(rows) @ ((predictions[right] - target) ** 2)
                comparisons[unit][f'{left}_minus_{right}'] = dict(**ci, relative_mse_reduction=float(-(weights(rows) @ delta) / denom))
            progress('comparisons', completed=unit)
        strata = {}
        for unit, rows, target, predictions, fields in [('single', test, y, pred, ['document', 'timestep', 'layer', 'projection_type']),
                                                     ('bundle', bundles, by, bp, ['document', 'timestep', 'layer'])]:
            for field in fields:
                for value in sorted({r[field] for r in rows}):
                    ids = [i for i, r in enumerate(rows) if r[field] == value]
                    strata[f'{unit}:{field}:{value}'] = {m: metric(target[ids], p[ids], [rows[i] for i in ids]) for m, p in predictions.items()}
        progress('rank_audit')
        ranks = {'development': rank_audit(train), 'final_exploratory': rank_audit(test)}
        # role_feature saved normalized results only; do not divide two deltas to infer denominators.
        schema = {c: sorted({k for r in train + test for k in r['features'][c]}) for c in ('dense', 'sparse')}
        audit = dict(feature_keys=schema, output_energy_denominator='not saved by role_feature; no unstable reverse calculation',
                     limit='No direct test of output-scale dominance without separately persisted denominators/activations')
        result = dict(status='complete', config_sha256=sha256(ROOT / 'config.json'), single_metrics=metrics,
                      bundle_metrics=bm, comparisons=comparisons, strata=strata, fits=fits, denominator_audit=audit,
                      decomposition={m: decomposition(by, summed, p, bundles) for m, p in bp.items()},
                      counts=dict(development=len(train), final=len(test), final_bundles=len(bundles)),
                      decision='retain role separation; proxy diagnostics only, no downstream or aggregation selection')
        for p, digest in hashes.items():
            if sha256(p) != digest:
                raise RuntimeError('Input changed while running: ' + p)
        atomic_json(ROOT / 'rank_audit.json', ranks)
        np.savez_compressed(ROOT / 'predictions.npz', y=y, bundle_y=by, single_sum=summed,
                            **{f'single_{m}': p for m, p in pred.items()}, **{f'bundle_{m}': p for m, p in bp.items()})
        atomic_json(ROOT / 'prediction_keys.json', dict(single=[list(k) for k in lookup],
                         bundle=[dict(document=r['document'], timestep=r['timestep'], exchange_indices=r['exchange_indices']) for r in bundles]))
        lines = ['# 역할 보존 proxy 단위 분석', '', '기존 final 재사용 탐색 분석. 역할 분리는 고정하며 GSM8K를 평가하지 않았다.', '',
                 '| 후보 | Single MSE | Bundle MSE | Bundle 방향 정확도 |', '|---|---:|---:|---:|']
        lines += [f"| {m} | {metrics[m]['mse']:.8g} | {bm[m]['mse']:.8g} | {bm[m]['sign_accuracy']:.4f} |" for m in names]
        lines += ['', '## 동일 예산 bundle 주 비교', '']
        for name, c in comparisons['bundle_primary'].items():
            lines.append(f"- {name}: MSE 감소 {100*c['relative_mse_reduction']:.3f}%, 동시 CI {c['simultaneous_ci']}")
        lines += ['', '## 한계', '', '- 신뢰구간은 학습된 OOF 모델에 조건부이며 재학습 불확실성은 포함하지 않는다.',
                  '- 출력 에너지 분모는 저장되지 않아 스케일 원인 분석은 보류. 오차 변화 비율을 역산하지 않았다.',
                  '- 잔차 분해의 cross term을 포함했다. 가산성 잔차를 독립적 기여율이나 달성 가능한 상한으로 해석하지 않는다.',
                  '- Bundle의 layer는 4-layer group 대표값이다. Bundle projection-type별 attribution은 정의되지 않아 single에서만 보고한다.',
                  '- Raw/relative/both 비교는 정규화 및 정보 추가의 효과이며 max/mean/minimax의 정당화가 아니다.']
        (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')
        atomic_json(ROOT / 'results.json', result)
        progress('complete')


if __name__ == '__main__':
    main()
