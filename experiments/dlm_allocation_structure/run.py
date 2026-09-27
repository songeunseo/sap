"""Run CPU-only DLM projection-allocation structure diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.dlm_allocation_structure.core import (
    allocation_from_marginals,
    allocation_from_per_parameter_costs,
    apply_temporal_models,
    bootstrap_mean_ci,
    compute_marginals,
    fit_reconstruction_residual,
    fit_static_models,
    fit_temporal_models,
    nested_mask_xor,
    prediction_metrics,
    selected_additive_damage,
)
from experiments.projection_capacity_allocation_65.core import GRID
from experiments.projection_capacity_followup_65.core import summarize_curve_records


REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
SOURCE = REPO / "experiments/projection_capacity_allocation_65"
FOLLOWUP = REPO / "experiments/projection_capacity_followup_65"
CURVES = SOURCE / "capacity_curves_raw.json"
MASKS = SOURCE / "candidate_mask_manifest.json"
ORACLE = SOURCE / "allocation.json"
RECONSTRUCTION = FOLLOWUP / "reconstruction65_mask_manifest.json"
EIS_TYPE = FOLLOWUP / "eis_type65_mask_manifest.json"
INCREMENTS = [f"{GRID[i]:.2f}->{GRID[i + 1]:.2f}" for i in range(5)]
STATIC_LEVELS = ("global", "depth", "type", "depth_type", "projection")
SEQUENCE_SPLITS = ((tuple(range(4)), tuple(range(4, 8))),
                   (tuple(range(4, 8)), tuple(range(4))))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, .10)),
        "p25": float(np.quantile(values, .25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, .75)),
        "p90": float(np.quantile(values, .90)),
        "max": float(values.max()),
        "negative_fraction": float(np.mean(values < 0)),
    }


def module_parts(names):
    layers, types = [], []
    for name in names:
        block, kind = name.split(".", 1)
        if not block.startswith("block_"):
            raise RuntimeError(f"bad module name: {name}")
        layers.append(int(block.removeprefix("block_")))
        types.append(kind)
    return np.asarray(layers), np.asarray(types)


def state_indices(state_keys, sequences):
    wanted = set(sequences)
    result = np.asarray([i for i, (sequence, _) in enumerate(state_keys)
                         if sequence in wanted], dtype=int)
    expected = 10 * len(wanted)
    if len(result) != expected:
        raise RuntimeError(f"expected {expected} states, found {len(result)}")
    return result


def load_inputs():
    raw = json.loads(CURVES.read_text())
    entries = json.loads(MASKS.read_text())["entries"]
    data = summarize_curve_records(raw["projections"], GRID)
    if len(data["names"]) != 224 or len(set(data["names"])) != 224:
        raise RuntimeError("expected 224 unique projections")
    if data["names"] != [row["name"] for row in entries]:
        raise RuntimeError("candidate manifest ordering differs from curves")
    if data["shapes"] != [tuple(row["shape"]) for row in entries]:
        raise RuntimeError("candidate manifest shapes differ from curves")
    expected_keys = [(sequence, timestep)
                     for timestep in np.arange(.05, 1.0, .10).round(2)
                     for sequence in range(8)]
    if data["state_keys"] != expected_keys:
        raise RuntimeError("historical state order/dimensions changed")
    for entry in entries:
        rows, cols = entry["shape"]
        if entry["weights"] != rows * cols or len(entry["masks"]) != len(GRID):
            raise RuntimeError("candidate weight or grid mismatch")
        for level, (sparsity, mask) in enumerate(zip(GRID, entry["masks"])):
            expected = rows * int(cols * sparsity)
            if mask["nominal_sparsity"] != sparsity or mask["pruned"] != expected:
                raise RuntimeError(f"row-floor count mismatch at {entry['name']} level {level}")
    references = {
        "uniform": [3] * 224,
        "oracle": json.loads(ORACLE.read_text())["levels"],
        "reconstruction": [row["level"] for row in json.loads(RECONSTRUCTION.read_text())["entries"]],
        "eis_type": [row["level"] for row in json.loads(EIS_TYPE.read_text())["entries"]],
    }
    if any(len(levels) != 224 for levels in references.values()):
        raise RuntimeError("reference allocation length mismatch")
    return data, entries, references


def allocation_summary(allocation, entries, references):
    levels = allocation["levels"]
    sparsities = allocation["sparsities"]
    total = allocation["weights"]
    summary = {
        "levels": levels,
        "sparsities": sparsities,
        "pruned": allocation["pruned"],
        "weights": total,
        "budget_error": allocation["budget_error"],
        "global_sparsity": allocation["pruned"] / total,
        "level_counts": {f"{100 * GRID[level]:.0f}": int(levels.count(level))
                         for level in range(len(GRID))},
        "changed_from_uniform": int(sum(level != 3 for level in levels)),
    }
    summary["xor_vs_reference"] = {
        name: nested_mask_xor(levels, reference, entries)
        for name, reference in references.items()
    }
    return summary


def raw_summaries(marginals, layers, types):
    functional = marginals["functional"]
    local = marginals["reconstruction"]
    mean_functional = functional.mean(axis=1)
    result = {
        "functional": {},
        "functional_per_parameter": {},
        "reconstruction": {},
        "mean_curve_monotonicity": {
            "projections_with_any_negative_increment": int(np.any(mean_functional < 0, axis=1).sum()),
            "negative_projection_increments": int((mean_functional < 0).sum()),
        },
        "state_curve_monotonicity": {
            "module_states_with_any_negative_increment": int(np.any(functional < 0, axis=2).sum()),
            "negative_module_state_increments": int((functional < 0).sum()),
            "total_module_state_increments": int(functional.size),
        },
        "by_type": {},
        "by_layer_quartile": {},
    }
    for i, label in enumerate(INCREMENTS):
        result["functional"][label] = distribution(functional[:, :, i])
        result["functional_per_parameter"][label] = distribution(
            marginals["functional_per_parameter"][:, :, i])
        result["reconstruction"][label] = distribution(local[:, :, i])
    for kind in sorted(set(types)):
        mask = types == kind
        result["by_type"][kind] = {
            label: distribution(functional[mask, :, i])
            for i, label in enumerate(INCREMENTS)
        }
    for start in (0, 8, 16, 24):
        mask = (layers >= start) & (layers < start + 8)
        result["by_layer_quartile"][f"B{start:02d}-B{start + 7:02d}"] = {
            label: distribution(functional[mask, :, i])
            for i, label in enumerate(INCREMENTS)
        }
    return result


def static_cross_validation(functional, functional_per_parameter, state_keys, layers, types,
                            shapes, entries, references):
    folds = []
    for construction, validation in SEQUENCE_SPLITS:
        train = state_indices(state_keys, construction)
        test = state_indices(state_keys, validation)
        truth = functional[:, test, :].mean(axis=1)
        truth_cost = functional_per_parameter[:, test, :].mean(axis=1)
        models = fit_static_models(functional_per_parameter[:, train, :], layers, types)
        rows = {}
        for name in STATIC_LEVELS:
            allocation = allocation_from_per_parameter_costs(models[name], shapes)
            per_sequence = {
                str(sequence): selected_additive_damage(
                    functional[:, [i for i in test if state_keys[i][0] == sequence], :].mean(axis=1),
                    allocation["levels"],
                ) for sequence in validation
            }
            rows[name] = {
                "cost_per_parameter_prediction": prediction_metrics(models[name], truth_cost),
                "validation_additive_damage": selected_additive_damage(truth, allocation["levels"]),
                "validation_additive_damage_by_sequence": per_sequence,
                "allocation": allocation_summary(allocation, entries, references),
            }
        folds.append({
            "construction_sequences": list(construction),
            "validation_sequences": list(validation),
            "models": rows,
        })
    mean_damage = {
        name: float(np.mean([fold["models"][name]["validation_additive_damage"] for fold in folds]))
        for name in STATIC_LEVELS
    }
    projection_better_both = all(
        fold["models"]["projection"]["validation_additive_damage"]
        < fold["models"]["depth_type"]["validation_additive_damage"]
        for fold in folds
    )
    sequence_summary = {}
    for name in STATIC_LEVELS:
        values, depth_type_values = [], []
        for fold in folds:
            values.extend(fold["models"][name]["validation_additive_damage_by_sequence"].values())
            depth_type_values.extend(
                fold["models"]["depth_type"]["validation_additive_damage_by_sequence"].values())
        values = np.asarray(values, dtype=float)
        depth_type_values = np.asarray(depth_type_values, dtype=float)
        sequence_summary[name] = {
            "absolute_damage": bootstrap_mean_ci(values, 20000, 0),
            "difference_vs_depth_type": bootstrap_mean_ci(values - depth_type_values, 20000, 0),
        }
    return {
        "folds": folds,
        "mean_validation_additive_damage": mean_damage,
        "sequence_bootstrap_summary": sequence_summary,
        "projection_beats_depth_type_both_folds": projection_better_both,
    }


def temporal_rank_and_variation(values, state_keys):
    """Full-calibration descriptive ranks and within/across-state variation."""
    values = np.asarray(values, dtype=float)
    sequences = sorted({key[0] for key in state_keys})
    timesteps = sorted({key[1] for key in state_keys})
    lookup = {key: i for i, key in enumerate(state_keys)}
    if len(lookup) != len(sequences) * len(timesteps):
        raise RuntimeError("state keys are not a complete sequence x timestep grid")
    means = np.stack([
        values[:, [lookup[(sequence, timestep)] for sequence in sequences], :].mean(axis=1)
        for timestep in timesteps
    ])
    rank_matrix = []
    for left in means:
        row = []
        for right in means:
            metric = prediction_metrics(left, right)["spearman"]
            row.append(metric)
        rank_matrix.append(row)
    within = []
    for timestep in timesteps:
        rows = [values[:, lookup[(sequence, timestep)], :] for sequence in sequences]
        within.extend(float(np.mean((rows[a] - rows[b]) ** 2))
                      for a in range(len(rows)) for b in range(a + 1, len(rows)))
    across = []
    for sequence in sequences:
        rows = [values[:, lookup[(sequence, timestep)], :] for timestep in timesteps]
        across.extend(float(np.mean((rows[a] - rows[b]) ** 2))
                      for a in range(len(rows)) for b in range(a + 1, len(rows)))
    return {
        "scope": "descriptive full calibration; not used for model selection",
        "timesteps": timesteps,
        "projection_increment_rank_spearman": rank_matrix,
        "within_timestep_across_sequence_mse": distribution(within),
        "within_sequence_across_timestep_mse": distribution(across),
    }


def model_mse_by_sequence(predictions, truth, keys, validation_sequences):
    result = {}
    for name, predicted in predictions.items():
        if name == "normalized_values":
            continue
        result[name] = {
            str(sequence): float(np.mean((predicted[:, [i for i, key in enumerate(keys)
                                                         if key[0] == sequence], :]
                                           - truth[:, [i for i, key in enumerate(keys)
                                                      if key[0] == sequence], :]) ** 2))
            for sequence in validation_sequences
        }
    return result


def temporal_fold(train_values, test_values, train_keys, test_keys, layers, types):
    train_times = np.asarray([key[1] for key in train_keys])
    test_times = np.asarray([key[1] for key in test_keys])
    fit = fit_temporal_models(train_values, train_times, layers, types)
    predictions = apply_temporal_models(fit, test_times, values=test_values)
    truth = predictions.pop("normalized_values")
    sequences = sorted({key[0] for key in test_keys})
    return {
        "fit": fit,
        "truth": truth,
        "predictions": predictions,
        "mse_by_sequence": model_mse_by_sequence(predictions, truth, test_keys, sequences),
        "overall": {name: prediction_metrics(predicted, truth)
                    for name, predicted in predictions.items()},
    }


def temporal_cross_validation(functional_per_parameter, reconstruction_per_parameter,
                              state_keys, layers, types):
    folds, raw_deltas, residual_deltas = [], [], []
    for construction, validation in SEQUENCE_SPLITS:
        train = state_indices(state_keys, construction)
        test = state_indices(state_keys, validation)
        train_keys = [state_keys[i] for i in train]
        test_keys = [state_keys[i] for i in test]
        f = temporal_fold(functional_per_parameter[:, train], functional_per_parameter[:, test],
                          train_keys, test_keys, layers, types)
        r_fit = fit_temporal_models(reconstruction_per_parameter[:, train],
                                    np.asarray([key[1] for key in train_keys]), layers, types)
        r_test = apply_temporal_models(r_fit, np.asarray([key[1] for key in test_keys]),
                                       values=reconstruction_per_parameter[:, test])["normalized_values"]
        residual = fit_reconstruction_residual(
            f["fit"]["normalized_train"], r_fit["normalized_train"], f["truth"], r_test)
        residual_fit = fit_temporal_models(
            residual["train_residual"], np.asarray([key[1] for key in train_keys]),
            layers, types, normalize=False)
        residual_predictions = apply_temporal_models(
            residual_fit, np.asarray([key[1] for key in test_keys]))
        residual_sequences = sorted({key[0] for key in test_keys})
        residual_mse = model_mse_by_sequence(
            residual_predictions, residual["test_residual"], test_keys, residual_sequences)

        for sequence in residual_sequences:
            key = str(sequence)
            raw_deltas.append(f["mse_by_sequence"]["structured_interactions"][key]
                              - f["mse_by_sequence"]["static_projection_time"][key])
            residual_deltas.append(residual_mse["structured_interactions"][key]
                                   - residual_mse["static_projection_time"][key])
        folds.append({
            "construction_sequences": list(construction),
            "validation_sequences": list(validation),
            "train_rms_by_timestep": f["fit"]["rms_by_timestep"].tolist(),
            "raw": {
                "overall": f["overall"],
                "mse_by_sequence": f["mse_by_sequence"],
            },
            "reconstruction_adjusted": {
                "intercepts": residual["intercepts"].tolist(),
                "slopes": residual["slopes"].tolist(),
                "overall": {name: prediction_metrics(predicted, residual["test_residual"])
                            for name, predicted in residual_predictions.items()},
                "mse_by_sequence": residual_mse,
            },
        })
    raw_ci = bootstrap_mean_ci(np.asarray(raw_deltas), 20000, 0)
    residual_ci = bootstrap_mean_ci(np.asarray(residual_deltas), 20000, 0)
    raw_both = all(
        np.mean(list(fold["raw"]["mse_by_sequence"]["structured_interactions"].values()))
        < np.mean(list(fold["raw"]["mse_by_sequence"]["static_projection_time"].values()))
        for fold in folds
    )
    residual_both = all(
        np.mean(list(fold["reconstruction_adjusted"]["mse_by_sequence"]["structured_interactions"].values()))
        < np.mean(list(fold["reconstruction_adjusted"]["mse_by_sequence"]["static_projection_time"].values()))
        for fold in folds
    )
    return {
        "descriptive_functional_cost_per_parameter": temporal_rank_and_variation(
            functional_per_parameter, state_keys),
        "descriptive_reconstruction_cost_per_parameter": temporal_rank_and_variation(
            reconstruction_per_parameter, state_keys),
        "folds": folds,
        "raw_structured_minus_static_sequence_mse": raw_ci,
        "reconstruction_adjusted_structured_minus_static_sequence_mse": residual_ci,
        "raw_improves_both_folds": raw_both,
        "reconstruction_adjusted_improves_both_folds": residual_both,
        "dlm_state_structure_supported": bool(
            raw_both and raw_ci["bootstrap_95_ci"][1] < 0
            and residual_both and residual_ci["bootstrap_95_ci"][1] < 0
        ),
    }


def make_report(static, temporal, raw):
    means = static["mean_validation_additive_damage"]
    raw_ci = temporal["raw_structured_minus_static_sequence_mse"]
    residual_ci = temporal["reconstruction_adjusted_structured_minus_static_sequence_mse"]
    projection_ci = static["sequence_bootstrap_summary"]["projection"]["difference_vs_depth_type"]
    verdict = ("지지" if temporal["dlm_state_structure_supported"] else "미지지")
    lines = [
        "# DLM Allocation 구조 분해 — CPU 진단",
        "",
        "## 질문",
        "",
        "Projection sparsity allocation의 유효한 설명 단위가 depth, type, depth+type, 개별 projection 중 무엇이며,",
        "DLM corruption timestep이 reconstruction으로 설명되지 않는 구조적 취약성을 추가하는가?",
        "",
        "## 고정 데이터와 범위",
        "",
        "기존 LLaDA-8B-Base 224×80×6 single-projection Standard-Wanda curve만 재분석했다.",
        "새 mask/model/GPU/downstream 평가는 만들지 않았다. 상태는 random-corruption DLM calibration states이며 실제 reverse trajectory가 아니다.",
        "",
        "## Raw marginal costs",
        "",
        f"평균 curve 기준 음의 projection-increment는 {raw['mean_curve_monotonicity']['negative_projection_increments']}개, ",
        f"하나 이상 음의 increment를 가진 projection은 {raw['mean_curve_monotonicity']['projections_with_any_negative_increment']}/224개다.",
        "원시 분포와 type/layer-quartile breakdown은 `raw_marginal_summary.json`에 있다. 값을 보정하거나 envelope로 바꾸지 않았다.",
        "",
        "## 정적 구조의 sequence 교차 검증",
        "",
        "| 설명 수준 | 평균 validation additive KL damage ↓ |",
        "|---|---:|",
    ]
    labels = {"global": "Global", "depth": "Depth", "type": "Type",
              "depth_type": "Depth+Type", "projection": "개별 projection"}
    lines += [f"| {labels[name]} | {means[name]:.6g} |" for name in STATIC_LEVELS]
    lines += [
        "",
        f"개별 projection 설명이 Depth+Type보다 양방향 fold 모두 낮은 damage를 냈는가: **{static['projection_beats_depth_type_both_folds']}**.",
        f"Sequence별 projection−Depth+Type damage 차이: mean {projection_ci['mean']:.6g}, 95% CI [{projection_ci['bootstrap_95_ci'][0]:.6g}, {projection_ci['bootstrap_95_ci'][1]:.6g}].",
        "각 fold의 exact-budget allocation, level counts, reference mask XOR은 `static_structure.json`에 있다.",
        "",
        "## Timestep 구조와 reconstruction 중복",
        "",
        "Primary 변수는 marginal KL / (0.05×projection parameter 수)다. Train-timestep RMS로 정규화한 projection+timestep model에",
        "depth×timestep/type×timestep 상호작용을 추가했을 때의",
        "validation sequence MSE 차이(structured − static)다. 음수면 상태별 구조가 추가 설명력을 갖는다.",
        "",
        f"- Raw marginal cost: mean {raw_ci['mean']:.6g}, 95% CI [{raw_ci['bootstrap_95_ci'][0]:.6g}, {raw_ci['bootstrap_95_ci'][1]:.6g}], 양방향 개선={temporal['raw_improves_both_folds']}.",
        f"- Reconstruction 보정 residual: mean {residual_ci['mean']:.6g}, 95% CI [{residual_ci['bootstrap_95_ci'][0]:.6g}, {residual_ci['bootstrap_95_ci'][1]:.6g}], 양방향 개선={temporal['reconstruction_adjusted_improves_both_folds']}.",
        "- Bootstrap 단위는 8개 held-out sequence이며 projection/state 수를 독립 표본으로 세지 않았다.",
        "",
        "## 판정",
        "",
        f"사전 기준에 따른 **DLM 상태 조건부 구조: {verdict}**.",
        "이는 후속 진단 우선순위를 정하는 결과이며 pruning 방법 또는 downstream 성능의 성공 판정이 아니다.",
        "기존 full GSM8K에서는 oracle Capacity 250/1319, Reconstruction 255/1319, oracle-derived EIS+type 263/1319로",
        "세 비균일 allocation의 차이가 유의하지 않았다. 따라서 이 additive 구조 순위를 jointly sparse downstream 순위로 해석하지 않는다.",
        "AR control이 없으므로 DLM 고유성은 주장하지 않는다. 기존 Probe16 FAIL 및 oracle 결과도 재분류하지 않는다.",
        "",
        "## 다음 단계 제한",
        "",
        "공식 EIS/OWL, actual trajectory, matched AR/DLM control 또는 full sparse downstream은 이 결과를 검토한 뒤 별도 사전 계획으로 진행한다.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ROOT.mkdir(exist_ok=True)
    data, entries, references = load_inputs()
    layers, types = module_parts(data["names"])
    marginals = compute_marginals(data["damage_states"], data["reconstruction_states"],
                                  data["shapes"], GRID)
    config = {
        "status": "frozen_cpu_diagnostic",
        "model": "GSAI-ML/LLaDA-8B-Base",
        "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
        "grid": list(GRID),
        "sequence_splits": [[list(a), list(b)] for a, b in SEQUENCE_SPLITS],
        "bootstrap": {"unit": "held-out sequence", "resamples": 20000, "seed": 0},
        "target_pruned": 4536008704,
        "primary_structure_variable": "functional marginal KL / (0.05 * projection parameter count)",
        "scope": "persisted curves only; no new masks, model execution, or downstream evaluation",
        "source_sha256": {str(path.relative_to(REPO)): sha256(path)
                          for path in (CURVES, MASKS, ORACLE, RECONSTRUCTION, EIS_TYPE)},
    }
    raw = raw_summaries(marginals, layers, types)
    static = static_cross_validation(
        marginals["functional"], marginals["functional_per_parameter"], data["state_keys"],
        layers, types, data["shapes"], entries, references)
    temporal = temporal_cross_validation(
        marginals["functional_per_parameter"], marginals["reconstruction_per_parameter"],
        data["state_keys"], layers, types)
    decision = {
        "status": "complete",
        "projection_beats_depth_type_both_folds": static["projection_beats_depth_type_both_folds"],
        "projection_minus_depth_type_sequence_damage": static[
            "sequence_bootstrap_summary"]["projection"]["difference_vs_depth_type"],
        "dlm_state_structure_supported": temporal["dlm_state_structure_supported"],
        "recommended_followup": (
            "freeze independent cheap static baselines before any new DLM-state predictor; "
            "full-model/downstream comparison requires a separate preregistered plan"
        ),
        "interpretation": "diagnostic priority only; not a pruning-method or downstream gate",
    }
    write_json(ROOT / "config.json", config)
    write_json(ROOT / "raw_marginal_summary.json", raw)
    write_json(ROOT / "static_structure.json", static)
    write_json(ROOT / "temporal_structure.json", temporal)
    write_json(ROOT / "decision.json", decision)
    (ROOT / "report.md").write_text(make_report(static, temporal, raw))
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
