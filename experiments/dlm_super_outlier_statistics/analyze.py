#!/usr/bin/env python3
"""Join dense super-outlier summaries to the frozen projection-capacity curves."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.dlm_super_outlier_statistics.collect import RAW, ROOT, sha
from experiments.dlm_super_outlier_statistics.core import correlation, partial_rank_correlation
from experiments.projection_capacity_followup_65.core import summarize_curve_records

CAPACITY = Path("experiments/projection_capacity_allocation_65/capacity_curves_raw.json")
GRID = (.50, .55, .60, .65, .70, .75)
FEATURES = (
    "input_dominant_share", "input_top1pct_share", "input_effective_fraction",
    "input_outlier7_ratio", "input_super_share", "input_super_rank",
    "input_super_token_argmax_fraction", "input_super_mean_sq_ratio",
    "input_excluded_dominant_share", "input_excluded_top1pct_share",
    "input_excluded_effective_fraction", "input_excluded_outlier7_ratio",
    "read_super_component_over_output", "output_super_share",
    "write_super_branch_over_after", "write_super_cross_over_after",
)
READ_TYPES = {"q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"}
WRITE_TYPES = {"attn_out", "ff_out"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _flatten(row, projection_type):
    result = {}
    for group in ("input", "output", "read", "write"):
        for key, value in (row.get(group) or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[f"{group}_{key}"] = float(value)
    super_keys = ("super_", "excluded_")
    if projection_type not in READ_TYPES:
        for key in list(result):
            if key.startswith("input_") and any(
                    key.startswith(f"input_{prefix}") for prefix in super_keys):
                del result[key]
    if projection_type not in WRITE_TYPES:
        for key in list(result):
            if key.startswith("output_super_") or key.startswith("output_excluded_"):
                del result[key]
    return result


def _mean_rows(rows):
    keys = sorted(set().union(*(row for row in rows)))
    return {key: float(np.mean([row[key] for row in rows if key in row and row[key] is not None]))
            for key in keys if any(key in row and row[key] is not None for row in rows)}


def _target_for_sequences(data, sequences):
    ids = [i for i, (sequence, _) in enumerate(data["state_keys"]) if sequence in sequences]
    damage = data["damage_states"][:, ids].mean(1)
    reconstruction = data["reconstruction_states"][:, ids].mean(1)
    sizes = np.asarray([math.prod(shape) for shape in data["shapes"]], float)
    return {
        "damage65": damage[:, 3],
        "marginal65_70_per_parameter": (damage[:, 4] - damage[:, 3]) / (.05 * sizes),
        "reconstruction65": reconstruction[:, 3],
        "reconstruction_marginal65_70": reconstruction[:, 4] - reconstruction[:, 3],
    }


def _association(feature, target, recon, layers, types):
    return {
        "raw": correlation(feature, target),
        "partial_layer_type": partial_rank_correlation(feature, target, layers, types),
        "partial_layer_type_reconstruction": partial_rank_correlation(
            feature, target, layers, types, recon),
        "by_type": {kind: correlation(feature[types == kind], target[types == kind])
                    for kind in sorted(set(types))},
        "leave_one_layer_out": {
            str(layer): correlation(feature[layers != layer], target[layers != layer])
            for layer in sorted(set(layers))
        },
    }


def _bootstrap(feature_states, data, target_name, layers, types,
               seed=20260910, resamples=2000):
    rng = np.random.default_rng(seed)
    values = []
    sequence_ids = np.arange(8)
    layer_ids = np.arange(32)
    feature_by_sequence = np.stack([
        np.mean(feature_states[:, [i for i, key in enumerate(data["state_keys"])
                                   if key[0] == sequence]], axis=1)
        for sequence in sequence_ids
    ])
    target_by_sequence = np.stack([
        _target_for_sequences(data, {int(sequence)})[target_name]
        for sequence in sequence_ids
    ])
    recon_name = ("reconstruction65" if target_name == "damage65"
                  else "reconstruction_marginal65_70")
    recon_by_sequence = np.stack([
        _target_for_sequences(data, {int(sequence)})[recon_name]
        for sequence in sequence_ids
    ])
    raw_values, partial_values, controlled_values = [], [], []
    for _ in range(resamples):
        sequences = rng.choice(sequence_ids, size=8, replace=True)
        selected_feature = np.mean(feature_by_sequence[sequences], axis=0)
        targets = np.mean(target_by_sequence[sequences], axis=0)
        reconstruction = np.mean(recon_by_sequence[sequences], axis=0)
        sampled_layers = rng.choice(layer_ids, size=32, replace=True)
        indices = np.concatenate([np.where(layers == layer)[0] for layer in sampled_layers])
        raw = correlation(selected_feature[indices], targets[indices])["spearman"]
        partial = partial_rank_correlation(selected_feature[indices], targets[indices],
                                           layers[indices], types[indices])["spearman"]
        controlled = partial_rank_correlation(
            selected_feature[indices], targets[indices], layers[indices], types[indices],
            reconstruction[indices])["spearman"]
        if raw is not None: raw_values.append(raw)
        if partial is not None: partial_values.append(partial)
        if controlled is not None: controlled_values.append(controlled)
    def summary(items):
        return {"valid": len(items),
                "95_ci": np.quantile(items, [.025, .975]).tolist() if items else None}
    return {
        "resamples": resamples, "seed": seed,
        "raw": summary(raw_values), "partial_layer_type": summary(partial_values),
        "partial_layer_type_reconstruction": summary(controlled_values),
    }


def _plots(rows, residual_rows, names, types, layers, damage65):
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    chosen = ["input_super_share", "input_excluded_dominant_share",
              "read_super_component_over_output", "write_super_branch_over_after"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feature in zip(axes.ravel(), chosen):
        values = np.asarray([row.get(feature, np.nan) for row in rows])
        ax.scatter(values, damage65, s=16, alpha=.7)
        ax.set_xlabel(feature); ax.set_ylabel("D_g(65)")
        rho = correlation(values, damage65)["spearman"]
        ax.set_title(f"Spearman={rho:.3f}" if rho is not None else "N/A")
    fig.tight_layout(); fig.savefig(figure_dir / "features_vs_damage65.png", dpi=180); plt.close(fig)

    type_order = sorted(set(types))
    heat = np.full((32, len(type_order)), np.nan)
    values = np.asarray([row.get("input_super_share", np.nan) for row in rows])
    for layer in range(32):
        for j, kind in enumerate(type_order):
            selected = (layers == layer) & (types == kind)
            if selected.any(): heat[layer, j] = np.nanmean(values[selected])
    fig, ax = plt.subplots(figsize=(10, 8)); image = ax.imshow(heat, aspect="auto")
    ax.set_xticks(range(len(type_order)), type_order, rotation=35, ha="right")
    ax.set_ylabel("Layer"); ax.set_title("Input channel 3848 energy share")
    fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(figure_dir / "super_share_layer_type.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for location in ("block_input", "attention_residual", "block_output"):
        values = [np.mean([row[location]["super_share"] for row in residual_rows
                           if row["layer"] == layer]) for layer in range(32)]
        ax.plot(range(32), values, label=location)
    ax.set_xlabel("Layer"); ax.set_ylabel("Channel 3848 energy share")
    ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir / "residual_super_share_by_layer.png", dpi=180); plt.close(fig)


def analyze():
    config = json.loads((ROOT / "config.json").read_text())
    if config["sources"][str(CAPACITY)] != sha(CAPACITY):
        raise RuntimeError("capacity curves changed")
    raw = torch_load(RAW)
    if raw["fingerprint"]["config_sha256"] != sha(ROOT / "config.json"):
        raise RuntimeError("raw/config identity mismatch")
    curve_records = json.loads(CAPACITY.read_text())["projections"]
    data = summarize_curve_records(curve_records, GRID)
    names = data["names"]
    if raw["fingerprint"]["module_names"] != names:
        raise RuntimeError("raw/capacity module order mismatch")
    raw_by_key = {(int(row["sequence_index"]), float(row["timestep"])): row
                  for row in raw["states"]}
    if len(raw_by_key) != 80 or set(raw_by_key) != set(data["state_keys"]):
        raise RuntimeError("raw/capacity state identities differ")
    ordered_states = [raw_by_key[key] for key in data["state_keys"]]

    by_name = {name: [] for name in names}
    residual_rows = []
    for state in ordered_states:
        for name in names:
            by_name[name].append(_flatten(state["linear"][name], name.split(".", 1)[1]))
        for layer, row in state["residual"].items():
            residual_rows.append({"sequence_index": state["sequence_index"],
                                  "timestep": state["timestep"], "layer": int(layer), **row})
    aggregate = [_mean_rows(by_name[name]) for name in names]
    layers = np.asarray([int(name.split(".")[0].removeprefix("block_")) for name in names])
    types = np.asarray([name.split(".")[1] for name in names])
    targets = _target_for_sequences(data, set(range(8)))

    results = {}
    for feature in FEATURES:
        matrix = np.full((len(names), 80), np.nan)
        for module, name in enumerate(names):
            for state, row in enumerate(by_name[name]):
                if feature in row: matrix[module, state] = row[feature]
        values = np.nanmean(matrix, axis=1)
        if np.isfinite(values).sum() < 3:
            results[feature] = {"available": False, "n": int(np.isfinite(values).sum())}
            continue
        feature_result = {"available": True, "n": int(np.isfinite(values).sum()),
                          "distribution": _distribution(values)}
        for target_name, target in (("damage65", targets["damage65"]),
                                    ("marginal65_70_per_parameter",
                                     targets["marginal65_70_per_parameter"])):
            recon = (targets["reconstruction65"] if target_name == "damage65"
                     else targets["reconstruction_marginal65_70"])
            feature_result[target_name] = _association(values, target, recon, layers, types)
            feature_result[target_name]["bootstrap_sequence_layer"] = _bootstrap(
                matrix, data, target_name, layers, types)
        feature_result["all_grid_raw"] = {
            str(level): correlation(values, data["damage"][:, level_index])
            for level_index, level in enumerate(GRID)
        }
        feature_result["all_marginals_raw"] = {
            f"{GRID[level]}->{GRID[level + 1]}": correlation(
                values, (data["damage"][:, level + 1] - data["damage"][:, level]) /
                (.05 * np.asarray([math.prod(shape) for shape in data["shapes"]], float)))
            for level in range(5)
        }
        split = {}
        for label, feature_sequences, target_sequences in (
            ("first_to_second", range(4), range(4, 8)),
            ("second_to_first", range(4, 8), range(4)),
        ):
            ids = [i for i, key in enumerate(data["state_keys"]) if key[0] in feature_sequences]
            split[label] = correlation(np.nanmean(matrix[:, ids], axis=1),
                                       _target_for_sequences(data, set(target_sequences))["damage65"])
        feature_result["cross_sequence_split_damage65"] = split
        quartiles = {}
        for kind in sorted(set(types)):
            ids = np.where((types == kind) & np.isfinite(values))[0]
            if len(ids) == 0:
                continue
            ordered = ids[np.argsort(targets["damage65"][ids])]
            count = max(1, len(ordered) // 4)
            robust, vulnerable = ordered[:count], ordered[-count:]
            quartiles[kind] = {
                "robust_count": int(len(robust)), "vulnerable_count": int(len(vulnerable)),
                "robust_mean": float(np.mean(values[robust])),
                "vulnerable_mean": float(np.mean(values[vulnerable])),
                "vulnerable_minus_robust": float(np.mean(values[vulnerable]) -
                                                  np.mean(values[robust])),
            }
        feature_result["within_type_damage_quartiles"] = quartiles
        results[feature] = feature_result

    per_projection = []
    for i, name in enumerate(names):
        row = {"module_index": i, "name": name, "layer": int(layers[i]),
               "projection_type": str(types[i]), "damage65": float(targets["damage65"][i]),
               "marginal65_70_per_parameter": float(targets["marginal65_70_per_parameter"][i]),
               "reconstruction65": float(targets["reconstruction65"][i]), **aggregate[i]}
        per_projection.append(row)
    write_json(ROOT / "analysis.json", {"status": "complete", "features": results,
               "residual_summary": _residual_summary(residual_rows),
               "per_projection": per_projection})
    with (ROOT / "projection_statistics.csv").open("w", newline="") as handle:
        keys = sorted(set().union(*(row.keys() for row in per_projection)))
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(per_projection)
    _plots(aggregate, residual_rows, names, types, layers, targets["damage65"])
    _write_report(results, per_projection, _residual_summary(residual_rows))
    return results


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _distribution(values):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "p25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "p75": float(np.quantile(values, .75)),
            "max": float(values.max())}


def _residual_summary(rows):
    result = {}
    for location in ("block_input", "attention_residual", "block_output"):
        result[location] = {str(layer): _distribution([
            row[location]["super_share"] for row in rows if row["layer"] == layer
        ]) for layer in range(32)}
    return result


def _ci(result, target="damage65", mode="partial_layer_type_reconstruction"):
    return result[target]["bootstrap_sequence_layer"][mode]["95_ci"]


def _write_report(results, rows, residual):
    ranked = []
    for feature, result in results.items():
        if result.get("available"):
            ranked.append((abs(result["damage65"]["partial_layer_type_reconstruction"]["spearman"] or 0),
                           feature, result))
    super_share = results["input_super_share"]["damage65"]
    read_share = results["read_super_component_over_output"]["damage65"]
    excluded_dom = results["input_excluded_dominant_share"]["damage65"]
    excluded_eff = results["input_excluded_effective_fraction"]["damage65"]
    marginal_dom = results["input_excluded_dominant_share"]["marginal65_70_per_parameter"]
    marginal_eff = results["input_excluded_effective_fraction"]["marginal65_70_per_parameter"]
    block_input = [residual["block_input"][str(layer)]["mean"] for layer in range(32)]
    attention_residual = [residual["attention_residual"][str(layer)]["mean"]
                          for layer in range(32)]
    lines = ["# LLaDA Super-Outlier Projection Statistics", "",
             "## Hypothesis", "",
             "Channel 3848 의존도가 projection별 pruning tolerance를 layer/type 및 local "
             "reconstruction 이상으로 설명하는지 검증한다. 동시에 3848을 제외한 나머지 "
             "channel 집중도가 더 일반적인 설명인지 비교한다.", "",
             "## Actual Setup", "",
             "Frozen LLaDA-8B-Base 80 corruption states에서 dense forward 통계를 수집하고, "
             "기존 224×6 single-projection capacity curve와 결합했다. 모델 channel이나 weight는 제거하지 않았다.",
             "분석 대상은 224 projections이며, residual channel과 직접 연결되는 read 통계는 "
             "q/k/v/ff_proj/up_proj 160개, write 통계는 attn_out/ff_out 64개다. "
             "Spearman 상관과 layer/type 고정효과 및 local reconstruction 통제 상관을 계산했고, "
             "8 sequences와 32 layers를 cluster 단위로 재표집하여 2,000회 bootstrap CI를 얻었다.",
             "", "## Results — observed statistics", "",
             f"Channel 3848 residual energy share는 layer 0 attention residual에서 "
             f"{attention_residual[0]:.4f}, block-input 기준 최대 {max(block_input):.4f} "
             f"(layer {int(np.argmax(block_input))}), 마지막 block output에서 "
             f"{residual['block_output']['31']['mean']:.4f}였다.", "",
             "단일 channel 지표는 raw D_g(65)와 강하게 연결됐지만 통제 후 사라졌다:", "",
             f"- input super share: raw rho={_fmt(super_share['raw']['spearman'])}, "
             f"controlled rho={_fmt(super_share['partial_layer_type_reconstruction']['spearman'])}, "
             f"95% CI={_fmt_ci(_ci(results['input_super_share']))}",
             f"- read contribution: raw rho={_fmt(read_share['raw']['spearman'])}, "
             f"controlled rho={_fmt(read_share['partial_layer_type_reconstruction']['spearman'])}, "
             f"95% CI={_fmt_ci(_ci(results['read_super_component_over_output']))}", "",
             "반면 channel 3848을 제외한 입력 집중도는 통제 후에도 관계가 남았다:", "",
             f"- excluded dominant share vs D_g(65): rho="
             f"{_fmt(excluded_dom['partial_layer_type_reconstruction']['spearman'])}, "
             f"95% CI={_fmt_ci(_ci(results['input_excluded_dominant_share']))}",
             f"- excluded effective fraction vs D_g(65): rho="
             f"{_fmt(excluded_eff['partial_layer_type_reconstruction']['spearman'])}, "
             f"95% CI={_fmt_ci(_ci(results['input_excluded_effective_fraction']))}",
             f"- excluded dominant share vs 65→70 marginal cost: rho="
             f"{_fmt(marginal_dom['partial_layer_type_reconstruction']['spearman'])}, "
             f"95% CI={_fmt_ci(_ci(results['input_excluded_dominant_share'], 'marginal65_70_per_parameter'))}",
             f"- excluded effective fraction vs 65→70 marginal cost: rho="
             f"{_fmt(marginal_eff['partial_layer_type_reconstruction']['spearman'])}, "
             f"95% CI={_fmt_ci(_ci(results['input_excluded_effective_fraction'], 'marginal65_70_per_parameter'))}",
             "", "## Main Associations", "",
             "| Feature | Raw ρ with D(65) | Partial layer/type ρ | + reconstruction ρ |",
             "|---|---:|---:|---:|"]
    for _, feature, result in sorted(ranked, reverse=True):
        d = result["damage65"]
        lines.append(f"| {feature} | {_fmt(d['raw']['spearman'])} | "
                     f"{_fmt(d['partial_layer_type']['spearman'])} | "
                     f"{_fmt(d['partial_layer_type_reconstruction']['spearman'])} |")
    lines += ["", "## Interpretation", "",
              "단일 super-outlier는 LLaDA의 depth/type 구조를 강하게 표시하지만, 그 자체가 "
              "projection tolerance를 독립적으로 설명한다는 증거는 없다. 더 흥미로운 관찰은 "
              "3848을 제거하고 계산한 나머지 channel 분포다. 집중도가 높은 read-projection일수록 "
              "D_g(65)와 65→70 marginal damage가 낮은 방향이어서, 이 데이터에서는 outlier가 "
              "많을수록 보호한다는 단순 OWL 직관보다 low-dimensional redundancy 해석과 더 잘 맞는다.",
              "", "이 관계는 layer/type 및 reconstruction 통제와 cross-sequence split에서 유지되지만, "
              "160개 read-projection에 한정된 다중 탐색 결과다. 인과관계나 downstream gain, "
              "새 pruning 방법을 아직 입증하지 않는다.", "", "## Decision", "",
              "- channel 3848 단독 기반 pruning score/allocation: NOT SUPPORTED.",
              "- channel 3848 제외 activation concentration: FOLLOW-UP DIAGNOSTIC WARRANTED.",
              "- 다음 단계는 기존 Wanda score와의 redundancy 및 실제 mask 변화량을 측정하는 작은 "
              "diagnostic이다. 이 검증 전에는 새 allocation이나 full downstream 평가를 만들지 않는다."]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def _fmt_ci(value):
    return "N/A" if value is None else f"[{value[0]:.4f}, {value[1]:.4f}]"


if __name__ == "__main__":
    analyze()
