#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from experiments.dlm_dual_role_allocation.io import atomic_write_json
from experiments.dlm_role_mechanism_analysis.core import (
    additive_group_r2, allocation_difference, marginal_cost, pair_order_summary,
    percentile_contrast, pooled_curves, quantiles, rank_summary, sign_agreement,
)
from experiments.dlm_role_validation.core import allocate_grid


ROOT = Path("experiments/dlm_role_mechanism_analysis")
SOURCE = Path("experiments/dlm_role_validation")
SUITES = ("random65", "target50", "target75")
BOOTSTRAPS = 2000
SEED = 1234


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_name(name: str) -> tuple[int, str]:
    block, projection = name.split(".")
    return int(block.removeprefix("block_")), projection


def load_suite(suite: str) -> dict:
    allocation_path = SOURCE / f"allocation_{suite}.json"
    allocation_doc = json.loads(allocation_path.read_text())
    names = allocation_doc["module_names"]
    grid = allocation_doc["grid"]
    payloads = []
    fields = None
    labels = None
    sequences = timesteps = None
    invariant = None
    source_paths = [allocation_path]
    shapes = []
    for module_index, name in enumerate(names):
        path = SOURCE / "runtime" / suite / f"{name}.json"
        payload = json.loads(path.read_text())
        source_paths.append(path)
        if int(payload["module_index"]) != module_index or payload["name"] != name:
            raise RuntimeError(f"module ordering mismatch: {name}")
        if list(map(float, payload["grid"])) != list(map(float, grid)):
            raise RuntimeError(f"grid mismatch: {name}")
        current = (payload["config_sha256"], payload["dense_model_sha256"], payload["state_digest"])
        if invariant is None:
            invariant = current
        elif invariant != current:
            raise RuntimeError("frozen source invariant changed across modules")
        shapes.append(tuple(map(int, payload["shape"])))
        states = payload["states"]
        if len(states) != 80:
            raise RuntimeError("expected 80 states")
        state_labels = list(states[0]["groups"])
        if labels is None:
            labels = state_labels
            fields = {
                label: {key: np.empty((len(names), 80, len(grid)), dtype=np.float64)
                        for key in ("num_masked", "den_masked", "num_unmasked", "den_unmasked")}
                for label in labels
            }
            sequences = np.asarray([int(row["sequence_index"]) for row in states])
            timesteps = np.asarray([float(row["timestep"]) for row in states])
        elif state_labels != labels:
            raise RuntimeError("partition labels changed across modules")
        if not np.array_equal(sequences, [int(row["sequence_index"]) for row in states]):
            raise RuntimeError("sequence ordering mismatch")
        if not np.allclose(timesteps, [float(row["timestep"]) for row in states], rtol=0, atol=0):
            raise RuntimeError("timestep ordering mismatch")
        for state_index, state in enumerate(states):
            if int(state["state_index"]) != state_index:
                raise RuntimeError("state ordering mismatch")
            for label in labels:
                rows = state["groups"][label]
                if len(rows) != len(grid):
                    raise RuntimeError("level count mismatch")
                for level, row in enumerate(rows):
                    if float(row["sparsity"]) != float(grid[level]):
                        raise RuntimeError("sparsity ordering mismatch")
                    fields[label]["num_masked"][module_index, state_index, level] = row["num_a"]
                    fields[label]["den_masked"][module_index, state_index, level] = row["den_a"]
                    fields[label]["num_unmasked"][module_index, state_index, level] = row["num_b"]
                    fields[label]["den_unmasked"][module_index, state_index, level] = row["den_b"]
        payloads.append(payload)
    for label in labels:
        for key, values in fields[label].items():
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite field: {suite}/{label}/{key}")
        if np.any(fields[label]["den_masked"] <= 0) or np.any(fields[label]["den_unmasked"] <= 0):
            raise RuntimeError("nonpositive denominator")
    return {
        "suite": suite, "names": names, "grid": grid, "shapes": shapes,
        "labels": labels, "fields": fields, "sequences": sequences, "timesteps": timesteps,
        "allocation": allocation_doc, "invariant": invariant,
        "source_files": [{"path": str(p), "sha256": file_sha256(p)} for p in source_paths],
    }


def group_rank_summaries(masked: np.ndarray, unmasked: np.ndarray,
                         layers: np.ndarray, types: np.ndarray) -> dict:
    per_increment = []
    for level in range(masked.shape[1]):
        by_type = {}
        for projection_type in sorted(set(types)):
            chosen = types == projection_type
            by_type[projection_type] = rank_summary(masked[chosen, level], unmasked[chosen, level])
        within_layer = []
        for layer in sorted(set(layers)):
            chosen = layers == layer
            within_layer.append({"layer": int(layer), **rank_summary(masked[chosen, level], unmasked[chosen, level])})
        per_increment.append({
            "increment_index": level,
            "overall": rank_summary(masked[:, level], unmasked[:, level]),
            "by_type": by_type,
            "within_layer": {
                "mean_spearman": float(np.mean([x["spearman"] for x in within_layer])),
                "mean_rank_reversal_fraction": float(np.mean([x["rank_reversal_fraction"] for x in within_layer])),
                "layers": within_layer,
            },
        })
    contrast = percentile_contrast(masked, unmasked)
    return {
        "per_increment": per_increment,
        "flattened": rank_summary(masked.ravel(), unmasked.ravel()),
        "percentile_contrast": quantiles(contrast),
        "mean_absolute_percentile_contrast": float(np.mean(np.abs(contrast))),
        "additive_layer_type_explanation": additive_group_r2(
            np.mean(np.abs(contrast), axis=1), layers, types
        ),
        "negative_marginals": {
            "masked": int(np.sum(masked < 0)), "unmasked": int(np.sum(unmasked < 0))
        },
    }


def pooled_marginals(data: dict, label: str, state_ids: list[int]) -> tuple[dict, dict]:
    pooled = pooled_curves(data["fields"][label], state_ids)
    costs = {key: marginal_cost(pooled[key], data["shapes"], data["grid"])
             for key in ("masked", "unmasked", "aggregate", "max")}
    return pooled, costs


def stability_analysis(data: dict, label: str) -> dict:
    sequences = data["sequences"]
    layers = np.asarray([parse_name(name)[0] for name in data["names"]])
    types = np.asarray([parse_name(name)[1] for name in data["names"]])
    folds = []
    for heldout in sorted(set(sequences)):
        train_ids = np.flatnonzero(sequences != heldout).tolist()
        test_ids = np.flatnonzero(sequences == heldout).tolist()
        _, train = pooled_marginals(data, label, train_ids)
        _, test = pooled_marginals(data, label, test_ids)
        train_contrast = percentile_contrast(train["masked"], train["unmasked"])
        test_contrast = percentile_contrast(test["masked"], test["unmasked"])
        by_type = {}
        for projection_type in sorted(set(types)):
            chosen = types == projection_type
            by_type[projection_type] = float(spearmanr(
                train_contrast[chosen].ravel(), test_contrast[chosen].ravel()).statistic)
        within_layer = []
        for layer in sorted(set(layers)):
            chosen = layers == layer
            within_layer.append(float(spearmanr(
                train_contrast[chosen].ravel(), test_contrast[chosen].ravel()).statistic))
        folds.append({
            "heldout_sequence": int(heldout),
            "spearman": float(spearmanr(train_contrast.ravel(), test_contrast.ravel()).statistic),
            "sign": sign_agreement(train_contrast.ravel(), test_contrast.ravel()),
            "by_type_spearman": by_type,
            "mean_within_layer_spearman": float(np.nanmean(within_layer)),
        })
    return {
        "folds": folds,
        "mean_spearman": float(np.mean([row["spearman"] for row in folds])),
        "mean_sign_agreement": float(np.mean([row["sign"]["agreement"] for row in folds])),
        "mean_within_layer_spearman": float(np.mean([row["mean_within_layer_spearman"] for row in folds])),
    }


def timestep_analysis(data: dict, label: str) -> list[dict]:
    results = []
    for timestep in sorted(set(data["timesteps"])):
        ids = np.flatnonzero(data["timesteps"] == timestep).tolist()
        _, costs = pooled_marginals(data, label, ids)
        summary = [rank_summary(costs["masked"][:, k], costs["unmasked"][:, k])
                   for k in range(costs["masked"].shape[1])]
        results.append({
            "timestep": float(timestep),
            "mean_spearman": float(np.mean([x["spearman"] for x in summary])),
            "mean_rank_reversal_fraction": float(np.mean([x["rank_reversal_fraction"] for x in summary])),
        })
    return results


def bootstrap_rank_stability(data: dict, label: str) -> dict:
    rng = np.random.default_rng(SEED)
    unique = np.asarray(sorted(set(data["sequences"])), dtype=int)
    correlations, reversals = [], []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([np.flatnonzero(data["sequences"] == seq) for seq in sampled]).tolist()
        _, costs = pooled_marginals(data, label, ids)
        summaries = [rank_summary(costs["masked"][:, k], costs["unmasked"][:, k])
                     for k in range(costs["masked"].shape[1])]
        correlations.append(np.mean([x["spearman"] for x in summaries]))
        reversals.append(np.mean([x["rank_reversal_fraction"] for x in summaries]))
    return {
        "resamples": BOOTSTRAPS, "seed": SEED,
        "mean_increment_spearman": {
            "estimate": float(np.mean(correlations)),
            "bootstrap_95_ci": np.quantile(correlations, [.025, .975]).tolist(),
        },
        "mean_increment_rank_reversal_fraction": {
            "estimate": float(np.mean(reversals)),
            "bootstrap_95_ci": np.quantile(reversals, [.025, .975]).tolist(),
        },
    }


def pooling_analysis(data: dict, pooled: dict, costs: dict) -> dict:
    alpha = pooled["alpha_masked"][:, 0]
    correlations, cases = [], []
    for level in range(costs["masked"].shape[1]):
        correlations.append({
            "increment_index": level,
            "aggregate_vs_masked": float(spearmanr(costs["aggregate"][:, level], costs["masked"][:, level]).statistic),
            "aggregate_vs_unmasked": float(spearmanr(costs["aggregate"][:, level], costs["unmasked"][:, level]).statistic),
        })
        pm = rankdata(costs["masked"][:, level]) / len(alpha)
        pu = rankdata(costs["unmasked"][:, level]) / len(alpha)
        for module, name in enumerate(data["names"]):
            if alpha[module] <= .5:
                role, weight, percentile = "masked", alpha[module], pm[module]
            else:
                role, weight, percentile = "unmasked", 1 - alpha[module], pu[module]
            cases.append({
                "name": name, "module_index": module, "increment_index": level,
                "underweighted_role": role, "role_energy_weight": float(weight),
                "role_cost_percentile": float(percentile),
                "underweight_cost_score": float((.5 - weight) * percentile),
                "masked_cost": float(costs["masked"][module, level]),
                "unmasked_cost": float(costs["unmasked"][module, level]),
                "aggregate_cost": float(costs["aggregate"][module, level]),
            })
    return {
        "masked_energy_weight": quantiles(alpha),
        "aggregate_rank_alignment": correlations,
        "top20_underweighted_high_cost_role": sorted(
            cases, key=lambda row: row["underweight_cost_score"], reverse=True)[:20],
    }


def allocation_analysis(data: dict, pooled: dict) -> dict:
    allocations = {
        name: allocate_grid(pooled[field], data["shapes"], data["grid"])
        for name, field in (("aggregate", "aggregate"), ("masked_only", "masked"),
                            ("unmasked_only", "unmasked"), ("max", "max"))
    }
    for name, allocation in allocations.items():
        if allocation["budget_error"] != 0:
            raise RuntimeError(f"budget error for {name}")
    stored = data["allocation"]["methods"]
    if allocations["aggregate"]["levels"] != stored["aggregate"]["levels"]:
        raise RuntimeError("Aggregate allocation reproduction failed")
    if allocations["max"]["levels"] != stored.get("actual", stored.get("role"))["levels"]:
        raise RuntimeError("Max allocation reproduction failed")
    pairs = {}
    for candidate in ("masked_only", "unmasked_only", "max"):
        pairs[f"{candidate}_vs_aggregate"] = allocation_difference(
            allocations["aggregate"], allocations[candidate], data["shapes"], data["grid"]
        )
    loo = []
    for heldout in sorted(set(data["sequences"])):
        ids = np.flatnonzero(data["sequences"] != heldout).tolist()
        fold_pooled, _ = pooled_marginals(data, "actual", ids)
        row = {"heldout_sequence": int(heldout)}
        for name, field in (("aggregate", "aggregate"), ("masked_only", "masked"),
                            ("unmasked_only", "unmasked"), ("max", "max")):
            fold_allocation = allocate_grid(fold_pooled[field], data["shapes"], data["grid"])
            row[name] = allocation_difference(allocations[name], fold_allocation,
                                              data["shapes"], data["grid"])
        loo.append(row)
    summaries = {
        name: {"counts": {str(level): allocation["sparsities"].count(level) for level in data["grid"]},
               "pruned": allocation["pruned"], "budget_error": allocation["budget_error"],
               "levels": allocation["levels"], "sparsities": allocation["sparsities"]}
        for name, allocation in allocations.items()
    }
    return {"allocations": summaries, "pairwise_vs_aggregate": pairs, "leave_one_sequence_out": loo}


def write_projection_csv(data: dict, pooled: dict, costs: dict, allocation: dict) -> None:
    destination = ROOT / f"projection_table_{data['suite']}.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module_index", "name", "layer", "type", "masked_energy_weight",
                         "mean_masked_marginal", "mean_unmasked_marginal",
                         "mean_abs_percentile_contrast", "aggregate_sparsity", "max_sparsity"])
        contrast = percentile_contrast(costs["masked"], costs["unmasked"])
        for index, name in enumerate(data["names"]):
            layer, projection_type = parse_name(name)
            writer.writerow([
                index, name, layer, projection_type, pooled["alpha_masked"][index, 0],
                np.mean(costs["masked"][index]), np.mean(costs["unmasked"][index]),
                np.mean(np.abs(contrast[index])),
                allocation["allocations"]["aggregate"]["sparsities"][index],
                allocation["allocations"]["max"]["sparsities"][index],
            ])


def make_plots(rank_doc: dict, stability_doc: dict, pooling_doc: dict) -> None:
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = rank_doc["random65"]["actual"]["per_increment"]
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, [row["overall"]["spearman"] for row in primary], marker="o")
    axes[0].set(xlabel="5pp increment", ylabel="Spearman", title="Masked vs unmasked marginal rank")
    labels = list(stability_doc["random65"])
    axes[1].bar(labels, [stability_doc["random65"][key]["mean_spearman"] for key in labels])
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set(ylabel="LOO mean Spearman", title="Role-contrast stability")
    fig.tight_layout()
    fig.savefig(figure_dir / "rank_and_stability.png", dpi=180)
    plt.close(fig)

    rows = pooling_doc["random65"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [r["aggregate_vs_masked"] for r in rows["aggregate_rank_alignment"]], marker="o", label="masked")
    ax.plot(x, [r["aggregate_vs_unmasked"] for r in rows["aggregate_rank_alignment"]], marker="o", label="unmasked")
    ax.set(xlabel="5pp increment", ylabel="Spearman", title="Aggregate marginal alignment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "aggregate_alignment.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    config = {"status": "frozen_before_analysis", "primary_suite": "random65",
              "secondary_suites": ["target50", "target75"], "bootstrap_resamples": BOOTSTRAPS,
              "bootstrap_seed": SEED, "unit": "8 sequences with 10 timestep repeated states",
              "no_curve_smoothing": True, "no_gpu_or_downstream": True}
    atomic_write_json(ROOT / "config.json", config)
    datasets = {suite: load_suite(suite) for suite in SUITES}
    invariant = datasets["random65"]["invariant"]
    if any(data["invariant"] != invariant for data in datasets.values()):
        raise RuntimeError("suite frozen invariants differ")
    manifest = {"status": "verified", "invariant": invariant,
                "suites": {suite: {"module_count": len(data["names"]), "state_count": 80,
                                    "grid": data["grid"], "labels": data["labels"],
                                    "sources": data["source_files"]}
                           for suite, data in datasets.items()}}
    atomic_write_json(ROOT / "input_manifest.json", manifest)

    rank_doc, stability_doc, pooling_doc, allocation_doc = {}, {}, {}, {}
    projection_doc = {}
    for suite, data in datasets.items():
        layers = np.asarray([parse_name(name)[0] for name in data["names"]])
        types = np.asarray([parse_name(name)[1] for name in data["names"]])
        rank_doc[suite], stability_doc[suite] = {}, {}
        all_ids = list(range(80))
        for label in data["labels"]:
            pooled, costs = pooled_marginals(data, label, all_ids)
            rank_doc[suite][label] = group_rank_summaries(
                costs["masked"], costs["unmasked"], layers, types
            )
            stability_doc[suite][label] = stability_analysis(data, label)
            stability_doc[suite][label]["by_timestep"] = timestep_analysis(data, label)
            if suite == "random65":
                stability_doc[suite][label]["cluster_bootstrap"] = bootstrap_rank_stability(data, label)
        pooled, costs = pooled_marginals(data, "actual", all_ids)
        pooling_doc[suite] = pooling_analysis(data, pooled, costs)
        allocation_doc[suite] = allocation_analysis(data, pooled)
        write_projection_csv(data, pooled, costs, allocation_doc[suite])
        projection_doc[suite] = {
            "mean_absolute_percentile_contrast_by_projection": [
                {"module_index": i, "name": name, "value": float(value)}
                for i, (name, value) in enumerate(zip(
                    data["names"], np.mean(np.abs(percentile_contrast(costs["masked"], costs["unmasked"])), axis=1)
                ))
            ]
        }
    atomic_write_json(ROOT / "rank_analysis.json", rank_doc)
    atomic_write_json(ROOT / "stability_analysis.json", stability_doc)
    atomic_write_json(ROOT / "pooling_analysis.json", pooling_doc)
    atomic_write_json(ROOT / "allocation_analysis.json", allocation_doc)
    atomic_write_json(ROOT / "projection_analysis.json", projection_doc)
    make_plots(rank_doc, stability_doc, pooling_doc)

    primary_rank = rank_doc["random65"]["actual"]
    primary_stability = stability_doc["random65"]["actual"]
    comparisons = allocation_doc["random65"]["pairwise_vs_aggregate"]
    report = f"""# Masked/Unmasked Allocation Signal Analysis

## Setup

- Frozen 224 projections × 80 states × 6 sparsity levels.
- Primary target: 65%; secondary targets: 50%, 75%.
- Sequence is the repeated-measure unit; bootstrap resamples sequences.
- No smoothing, GPU collection, or downstream selection.

## Primary descriptive results

- Mean absolute masked/unmasked marginal percentile contrast: {primary_rank['mean_absolute_percentile_contrast']:.6f}
- Flattened marginal Spearman: {primary_rank['flattened']['spearman']:.6f}
- Flattened rank-reversal fraction: {primary_rank['flattened']['rank_reversal_fraction']:.6f}
- Leave-one-sequence-out contrast Spearman: {primary_stability['mean_spearman']:.6f}
- Leave-one-sequence-out contrast sign agreement: {primary_stability['mean_sign_agreement']:.6f}
- Aggregate→Max changed projections: {comparisons['max_vs_aggregate']['changed_projections']}
- Aggregate→Max nested-mask XOR: {comparisons['max_vs_aggregate']['xor_fraction']:.6%}

## Interpretation boundary

These statistics test whether role separation carries stable allocation information. They do not establish that Max is the correct aggregator or that local reconstruction predicts jointly sparse downstream capability.
"""
    (ROOT / "report.md").write_text(report)
    print(json.dumps({"status": "complete", "report": str(ROOT / "report.md")}, sort_keys=True))


if __name__ == "__main__":
    main()
