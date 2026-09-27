#!/usr/bin/env python3
"""CPU diagnostics first; at most one frozen allocation per preregistered path."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.projection_capacity_allocation_65.core import allocate, distribution, correlation
from experiments.projection_capacity_followup_65.core import (
    additive_damage, summarize_curve_records, build_selected_manifest)
from experiments.projection_capacity_followup_65.analyze_existing import residualized_rank_corr
from experiments.dlm_capacity_predictor.core import (
    ROOT, SOURCE, CALIBRATION, FEATURES, GRID, TARGET, sha, write_json,
    state_indices, anchor_prediction, common_shape, fit_log_alpha, predict_alpha)


def freeze_config():
    config = dict(model='GSAI-ML/LLaDA-8B-Base',
                  revision='0f2787f2d87eac5eed8a087d5ecd24277e6255b2',
                  target_pruned=TARGET, total_weights=6_979_321_856, grid=GRID,
                  selector='historical unweighted Standard Wanda; existing candidate masks',
                  calibration_sha256=sha(CALIBRATION),
                  plan_sha256=sha(ROOT / 'PLAN.md'),
                  source_sha256={str(p): sha(p) for p in (
                      SOURCE/'capacity_curves_raw.json', SOURCE/'candidate_mask_manifest.json',
                      SOURCE/'allocation.json', SOURCE/'config.json')},
                  probe_timesteps=[.15, .85], probe_states=16,
                  sequence_folds=[[0, 1, 2, 3], [4, 5, 6, 7]],
                  layer_folds=[list(range(i, i+8)) for i in (0, 8, 16, 24)],
                  dense_features=FEATURES, sketch_size=64, sketch_seed=20260910,
                  anchor_level=.65, regression='train-standardized univariate OLS log alpha',
                  predictor_selection='must beat reconstruction in all 8 crossed folds; '
                  'minimum mean test additive damage; feature declaration order tie-break',
                  fresh_split={'seed': 3, 'sequences': list(range(24, 32)),
                               'timesteps': [.1, .3, .5, .7, .9]},
                  cost_policy='historical curve lookup is not deployment measurement',
                  bootstrap={'seed': 0, 'resamples': 20000})
    write_json(ROOT/'config.json', config, frozen=True)
    return config


def load_curves():
    config = freeze_config()
    for path, digest in config['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError(f'changed frozen source {path}')
    raw = json.loads((SOURCE/'capacity_curves_raw.json').read_text())
    data = summarize_curve_records(raw['projections'], GRID)
    entries = json.loads((SOURCE/'candidate_mask_manifest.json').read_text())['entries']
    if len(data['names']) != 224 or len(set(data['names'])) != 224:
        raise RuntimeError('expected 224 unique projections')
    if data['names'] != [e['name'] for e in entries]:
        raise RuntimeError('historical module ordering changed')
    if data['shapes'] != [tuple(e['shape']) for e in entries]:
        raise RuntimeError('module shapes changed')
    state_indices(data['state_keys'], range(8))
    for e in entries:
        rows, cols = e['shape']
        if e['weights'] != rows*cols:
            raise RuntimeError('incorrect weight count')
        for r, mask in zip(GRID, e['masks'], strict=True):
            if mask['nominal_sparsity'] != r or mask['pruned'] != rows*int(cols*r):
                raise RuntimeError('candidate grid/row-floor count mismatch')
    return data, entries


def evaluate_prediction(prediction, truth, shapes):
    allocation = allocate(prediction, shapes)
    non_anchor = [0, 1, 2, 4, 5]
    return dict(allocation=allocation,
                heldout_additive_damage=additive_damage(truth, allocation['levels']),
                non_anchor_correlation=correlation(prediction[:, non_anchor].ravel(),
                                                  truth[:, non_anchor].ravel()),
                marginal_correlation=correlation(np.diff(prediction, axis=1).ravel(),
                                                 np.diff(truth, axis=1).ravel()),
                marginal_by_increment={f'{GRID[i]}->{GRID[i+1]}':
                    correlation(np.diff(prediction, axis=1)[:, i], np.diff(truth, axis=1)[:, i])
                    for i in range(5)})


def freeze_allocation(method, curves, entries, extra):
    shapes = [tuple(e['shape']) for e in entries]
    a = allocate(curves, shapes)
    if a['pruned'] != TARGET or a['budget_error'] != 0:
        raise RuntimeError('candidate global budget mismatch')
    manifest = build_selected_manifest(method, entries, a['sparsities'], GRID)
    manifest['config_sha256'] = sha(ROOT/'config.json')
    path = ROOT/'allocations'/f'{method}_mask_manifest.json'
    write_json(path, manifest, frozen=True)
    uniform_counts = [e['masks'][3]['pruned'] for e in entries]
    # Candidate masks are nested row-wise prefixes of one ranking. This is the exact
    # XOR count under that invariant; actual payload XOR is verified before evaluation.
    xor_count = sum(abs(e['selected_mask']['pruned'] - u)
                    for e, u in zip(manifest['entries'], uniform_counts))
    document = dict(method=method, allocation=a, predictions=np.asarray(curves).tolist(),
                    mask_manifest=str(path), mask_manifest_sha256=sha(path),
                    changed_from_uniform=sum(level != 3 for level in a['levels']),
                    mask_xor_vs_uniform_from_nested_counts=xor_count,
                    mask_xor_fraction=xor_count/manifest['weights'], **extra)
    write_json(ROOT/'allocations'/f'{method}.json', document, frozen=True)
    return document


def freeze_candidate_index(path, final):
    """Permit the sole preregistered transition, then make the decision immutable."""
    path = Path(path)
    if not path.exists():
        write_json(path, final, frozen=True)
        return
    existing = json.loads(path.read_text())
    if existing == final:
        return
    allowed = (existing.get('status') == 'awaiting_dense_statistics'
               and final.get('status') == 'frozen'
               and existing.get('config_sha256') == final.get('config_sha256')
               and all(candidate in final.get('candidates', [])
                       for candidate in existing.get('candidates', [])))
    if not allowed:
        raise RuntimeError(f'attempt to change frozen candidate decision: {path}')
    write_json(path, final)


def probe_diagnostics(data, entries):
    start = time.monotonic()
    d, e = data['damage_states'], data['reconstruction_states']
    keys, shapes = data['state_keys'], data['shapes']
    folds = []
    for train_seq, test_seq in ((list(range(4)), list(range(4, 8))),
                                (list(range(4, 8)), list(range(4)))):
        train = state_indices(keys, train_seq)
        probe = state_indices(keys, train_seq, [.15, .85])
        test = state_indices(keys, test_seq)
        truth = d[:, test].mean(1)
        prediction, alpha = anchor_prediction(d, e, probe)
        all_state_prediction, _ = anchor_prediction(d, e, train)
        # Common shape is a supervised control, learned only on construction states.
        shape = common_shape(d, range(len(shapes)), train)
        anchor = d[:, probe, 3].mean(1)
        methods = {
            'reconstruction_probe_states': e[:, probe].mean(1),
            'reconstruction_all_construction_states': e[:, train].mean(1),
            'anchor_common_shape': anchor[:, None]*shape,
            'anchor_local_shape': prediction,
            'anchor_local_shape_all_construction_states': all_state_prediction,
            'measured_construction_curves': d[:, train].mean(1),
        }
        results = {name: evaluate_prediction(curve, truth, shapes)
                   for name, curve in methods.items()}
        folds.append(dict(construction_sequences=train_seq, evaluation_sequences=test_seq,
                          probe_state_indices=probe, common_shape=shape.tolist(),
                          alpha_distribution=distribution(alpha), methods=results))
    passed = all(f['methods']['anchor_local_shape']['heldout_additive_damage'] <
                 min(f['methods']['reconstruction_probe_states']['heldout_additive_damage'],
                     f['methods']['reconstruction_all_construction_states']['heldout_additive_damage'])
                 for f in folds)
    selected = state_indices(keys, range(8), [.15, .85])
    prediction, alpha = anchor_prediction(d, e, selected)
    ratio = d.mean(1)/e.mean(1)
    document = dict(status='complete', passed=passed, folds=folds,
                    config_sha256=sha(ROOT/'config.json'),
                    anchor_ratio_by_projection={name: row.tolist()
                        for name, row in zip(data['names'], ratio)},
                    ratio_cv_distribution=distribution(ratio.std(1)/ratio.mean(1)),
                    alpha16_vs_alpha80=correlation(alpha, ratio[:, 3]),
                    selected_state_indices=selected,
                    wall_seconds=time.monotonic()-start,
                    cost_measurement='CPU analysis of persisted curves only; '
                    'new 16-state functional probe wall time has NOT been measured',
                    training_cost='anchor_common_shape consumes offline six-level oracle labels',
                    decision='freeze probe candidate' if passed else 'stop probe path')
    write_json(ROOT/'probe_diagnostics.json', document)
    candidate = None
    if passed:
        candidate = freeze_allocation('probe16', prediction, entries,
                                      dict(alpha=alpha.tolist(), selected_state_indices=selected,
                                           construction='persisted 16-state anchor/local observations'))
    return document, candidate


def dense_features(payload, sequences):
    from experiments.dlm_capacity_predictor.collect_dense import aggregate_sequences
    return [aggregate_sequences([payload['per_sequence'][str(seq)][name] for seq in sequences])
            for name in payload['names']]


def dense_diagnostics(data, payload):
    names = data['names']
    all_features = dense_features(payload, range(8))
    split_features = [dense_features(payload, range(4)), dense_features(payload, range(4, 8))]
    alpha = data['damage'][:, 3]/data['reconstruction'][:, 3]
    types = [name.split('.')[1] for name in names]
    baselines = dict(activation_energy=[r['activation_energy'] for r in all_features],
                     activation_outlier_ratio=[r['activation_outlier_ratio'] for r in all_features],
                     reconstruction65=data['reconstruction'][:, 3], damage65=data['damage'][:, 3],
                     alpha65=alpha,
                     layer=[int(name.split('.')[0].split('_')[1]) for name in names])
    output = dict(status='complete', per_projection={n: row for n, row in zip(names, all_features)},
                  features={}, config_sha256=sha(ROOT/'config.json'),
                  caveat='q/k/v and up/ff share Linear inputs; input-only statistics cannot '
                  'distinguish members of each shared-input group by themselves')
    for feature in FEATURES:
        x = np.asarray([r[feature] if r[feature] is not None else np.nan for r in all_features])
        if not np.isfinite(x).all():
            output['features'][feature] = dict(eligible=False, reason='degenerate/missing raw statistic')
            continue
        associations = {}
        for baseline, y in baselines.items():
            associations[baseline] = dict(pooled=correlation(x, y),
                partial_rank_layer_type=residualized_rank_corr(x, y, names),
                per_type={p: correlation(x[np.array(types)==p], np.asarray(y)[np.array(types)==p])
                          for p in sorted(set(types))})
        split_x = [[r[feature] for r in rows] for rows in split_features]
        stable_finite = all(all(v is not None and np.isfinite(v) for v in row) for row in split_x)
        output['features'][feature] = dict(eligible=stable_finite, distribution=distribution(x),
            per_type_distribution={p: distribution(x[np.array(types)==p]) for p in sorted(set(types))},
            sequence_split_stability=correlation(*split_x) if stable_finite else None,
            associations=associations,
            marginal_associations={f'{GRID[i]}->{GRID[i+1]}':
                                   correlation(x, np.diff(data['damage'], axis=1)[:, i])
                                   for i in range(5)})
    # This artifact is persisted BEFORE any regression fitting/selection.
    write_json(ROOT/'dense_diagnostics.json', output)
    return output, all_features, split_features


def crossed_fold_predictions(data, feature_by_sequence_fold):
    """Eight crossed folds; neither test-layer labels nor test-sequence data train a fit."""
    layers = np.array([int(n.split('.')[0].split('_')[1]) for n in data['names']])
    d, e = data['damage_states'], data['reconstruction_states']
    features = np.asarray(feature_by_sequence_fold, dtype=float)
    if features.shape != (2, len(layers)) or not np.isfinite(features).all():
        raise ValueError('finite two-sequence-fold feature matrix required')
    folds = []
    for direction, (train_seq, test_seq) in enumerate(((list(range(4)), list(range(4, 8))),
                                                       (list(range(4, 8)), list(range(4))))):
        si, ti = state_indices(data['state_keys'], train_seq), state_indices(data['state_keys'], test_seq)
        train_e, train_d, truth = e[:, si].mean(1), d[:, si].mean(1), d[:, ti].mean(1)
        x = features[direction]
        for first in (0, 8, 16, 24):
            test_modules = np.flatnonzero((layers >= first) & (layers < first+8))
            train_modules = np.flatnonzero((layers < first) | (layers >= first+8))
            fit = fit_log_alpha(x[train_modules], train_d[train_modules, 3]/train_e[train_modules, 3])
            predicted_alpha = predict_alpha(fit, x[test_modules])
            predicted = predicted_alpha[:, None]*train_e[test_modules]
            shapes = [data['shapes'][i] for i in test_modules]
            result = evaluate_prediction(predicted, truth[test_modules], shapes)
            reference = evaluate_prediction(train_e[test_modules], truth[test_modules], shapes)
            folds.append(dict(construction_sequences=train_seq, evaluation_sequences=test_seq,
                train_modules=train_modules.tolist(), test_modules=test_modules.tolist(), fit=fit,
                predicted=result, reconstruction=reference,
                improved=result['heldout_additive_damage'] < reference['heldout_additive_damage'],
                projections_changed=sum(a != b for a, b in zip(
                    result['allocation']['levels'], reference['allocation']['levels']))))
    return folds


def validate_dense(data, entries, payload):
    raw, all_features, split_features = dense_diagnostics(data, payload)
    feature_results = {}
    for feature in FEATURES:
        if not raw['features'][feature]['eligible']:
            feature_results[feature] = dict(passed=False, reason='invalid raw diagnostic')
            continue
        folds = crossed_fold_predictions(data, [[r[feature] for r in split] for split in split_features])
        feature_results[feature] = dict(passed=all(f['improved'] for f in folds), folds=folds,
            mean_validation_damage=float(np.mean([f['predicted']['heldout_additive_damage'] for f in folds])))
    passing = [f for f in FEATURES if feature_results[f]['passed']]
    winner = min(passing, key=lambda f: feature_results[f]['mean_validation_damage']) if passing else None
    candidate = None
    if winner:
        x = np.array([r[winner] for r in all_features])
        fit = fit_log_alpha(x, data['damage'][:, 3]/data['reconstruction'][:, 3])
        alpha = predict_alpha(fit, x)
        candidate = freeze_allocation('dense_statistics', alpha[:, None]*data['reconstruction'], entries,
            dict(feature=winner, fit=fit, alpha=alpha.tolist(),
                 training='oracle-supervised; oracle labels required offline; not intervention-free training'))
    document = dict(status='complete', winner=winner, features=feature_results,
                    dense_statistics_sha256=sha(ROOT/'dense_statistics.pt'),
                    decision='freeze one dense-statistics candidate' if winner else 'stop dense-statistics path')
    write_json(ROOT/'predictor_validation.json', document)
    return document, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense', action='store_true')
    args = parser.parse_args()
    data, entries = load_curves()
    probe, probe_candidate = probe_diagnostics(data, entries)
    candidates = [probe_candidate] if probe_candidate else []
    dense = None
    if args.dense:
        from experiments.dlm_capacity_predictor.collect_dense import load_validated_statistics
        payload = load_validated_statistics(ROOT/'dense_statistics.pt')
        if payload['names'] != data['names'] or payload['config_sha256'] != sha(ROOT/'config.json'):
            raise RuntimeError('dense statistics provenance mismatch')
        dense, dense_candidate = validate_dense(data, entries, payload)
        if dense_candidate:
            candidates.append(dense_candidate)
    result = dict(status='frozen' if args.dense else 'awaiting_dense_statistics',
                  config_sha256=sha(ROOT/'config.json'),
                  candidates=[dict(method=c['method'], mask_manifest=c['mask_manifest'],
                                   mask_manifest_sha256=c['mask_manifest_sha256']) for c in candidates])
    if args.dense:
        freeze_candidate_index(ROOT/'candidates.json', result)
    else:
        write_json(ROOT/'candidates.json', result)
    lines = ['# DLM Capacity Predictor — 중간 연구 보고서', '',
             '## 현재 범위', '',
             '새 weight score가 아니라 Standard Wanda의 projection별 sparsity allocation을 검증한다.',
             '기존 6점 curve는 학습/진단 자료이며 새 방법의 무료 deployment 자료로 간주하지 않는다.', '',
             '## 16-state probe 진단', '',
             f"교차 검증 판정: {'후보 동결' if probe['passed'] else '중단'}.", '',
             '| Construction sequences | Reconstruction (40 states) | Probe (8 states) |',
             '|---|---:|---:|']
    for f in probe['folds']:
        lines.append(f"| {f['construction_sequences']} | "
                     f"{f['methods']['reconstruction_all_construction_states']['heldout_additive_damage']:.6f} | "
                     f"{f['methods']['anchor_local_shape']['heldout_additive_damage']:.6f} |")
    lines += ['', '표의 수치는 unseen sequence에서 평가한 single-projection additive damage이다.',
              'Full sparse model KL이나 GSM8K 성능이 아니며, 좋은 결과를 보장하지 않는다.', '',
              '## Dense 통계 경로', '',
              f"선택 feature: {dense['winner']}" if dense else '기존 GPU 작업 종료 후 raw statistics를 수집한다.',
              '', '## 비용과 남은 검증', '',
              '16-state probe 실제 실행시간, local reconstruction curve 수집시간은 아직 미측정이다.',
              '기존 oracle lookup으로 측정한 CPU 분석시간을 deployment speedup으로 해석하지 않는다.',
              '후보 동결 후 새 disjoint states에서 full-model gate, 통과 후보의 downstream 검증이 필요하다.',
              'EIS+type은 oracle에서 유도한 기술적 대조군이며 공식 EIS baseline이 아니다.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
