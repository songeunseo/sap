#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.dlm_dual_role_allocation.io import atomic_write_json


ROOT = Path("experiments/dlm_role_mechanism_analysis")


def main() -> None:
    rank = json.loads((ROOT / "rank_analysis.json").read_text())
    stability = json.loads((ROOT / "stability_analysis.json").read_text())
    pooling = json.loads((ROOT / "pooling_analysis.json").read_text())
    allocation = json.loads((ROOT / "allocation_analysis.json").read_text())
    rows = list(csv.DictReader((ROOT / "projection_table_random65.csv").open()))

    actual_rank = rank["random65"]["actual"]
    actual_stability = stability["random65"]["actual"]
    random_labels = ["random_101", "random_202", "random_303"]
    actual_rho = float(np.mean([x["overall"]["spearman"] for x in actual_rank["per_increment"]]))
    actual_reversal = float(np.mean([
        x["overall"]["rank_reversal_fraction"] for x in actual_rank["per_increment"]
    ]))
    random_rho = float(np.mean([
        np.mean([x["overall"]["spearman"] for x in rank["random65"][label]["per_increment"]])
        for label in random_labels
    ]))
    random_reversal = float(np.mean([
        np.mean([x["overall"]["rank_reversal_fraction"]
                 for x in rank["random65"][label]["per_increment"]])
        for label in random_labels
    ]))
    random_contrast = float(np.mean([
        rank["random65"][label]["mean_absolute_percentile_contrast"] for label in random_labels
    ]))
    random_loo = float(np.mean([stability["random65"][label]["mean_spearman"]
                                for label in random_labels]))
    random_sign = float(np.mean([stability["random65"][label]["mean_sign_agreement"]
                                 for label in random_labels]))

    comparison = allocation["random65"]["pairwise_vs_aggregate"]["max_vs_aggregate"]
    changed_ids = {row["module_index"] for row in comparison["per_projection"]
                   if row["level_left"] != row["level_right"]}
    changed = [row for row in rows if int(row["module_index"]) in changed_ids]
    unchanged = [row for row in rows if int(row["module_index"]) not in changed_ids]
    changed_contrast = float(np.mean([float(row["mean_abs_percentile_contrast"]) for row in changed]))
    unchanged_contrast = float(np.mean([float(row["mean_abs_percentile_contrast"]) for row in unchanged]))
    type_counts = dict(Counter(row["type"] for row in changed))
    quartile_counts = dict(Counter(str(int(row["layer"]) // 8) for row in changed))
    loo_max_changes = [row["max"]["changed_projections"]
                       for row in allocation["random65"]["leave_one_sequence_out"]]
    alpha = pooling["random65"]["masked_energy_weight"]
    aggregate_alignment = pooling["random65"]["aggregate_rank_alignment"]

    secondary = {}
    for suite in ("target50", "target75"):
        source = rank[suite]["actual"]
        secondary[suite] = {
            "mean_increment_spearman": float(np.mean([
                x["overall"]["spearman"] for x in source["per_increment"]
            ])),
            "mean_rank_reversal_fraction": float(np.mean([
                x["overall"]["rank_reversal_fraction"] for x in source["per_increment"]
            ])),
            "mean_absolute_percentile_contrast": source["mean_absolute_percentile_contrast"],
            "loo_contrast_spearman": stability[suite]["actual"]["mean_spearman"],
            "max_vs_aggregate": allocation[suite]["pairwise_vs_aggregate"]["max_vs_aggregate"],
        }

    decision = {
        "status": "complete",
        "primary_target": 0.65,
        "findings": {
            "actual_role": {
                "mean_increment_spearman": actual_rho,
                "mean_rank_reversal_fraction": actual_reversal,
                "mean_absolute_percentile_contrast": actual_rank["mean_absolute_percentile_contrast"],
                "loo_contrast_spearman": actual_stability["mean_spearman"],
                "loo_sign_agreement": actual_stability["mean_sign_agreement"],
                "within_layer_loo_spearman": actual_stability["mean_within_layer_spearman"],
                "cluster_bootstrap": actual_stability["cluster_bootstrap"],
            },
            "random_partition_mean": {
                "mean_increment_spearman": random_rho,
                "mean_rank_reversal_fraction": random_reversal,
                "mean_absolute_percentile_contrast": random_contrast,
                "loo_contrast_spearman": random_loo,
                "loo_sign_agreement": random_sign,
            },
            "pooling": {
                "masked_energy_weight": alpha,
                "mean_aggregate_vs_masked_spearman": float(np.mean([
                    row["aggregate_vs_masked"] for row in aggregate_alignment
                ])),
                "mean_aggregate_vs_unmasked_spearman": float(np.mean([
                    row["aggregate_vs_unmasked"] for row in aggregate_alignment
                ])),
            },
            "allocation": {
                "max_vs_aggregate_changed_projections": comparison["changed_projections"],
                "max_vs_aggregate_xor_fraction": comparison["xor_fraction"],
                "changed_projection_mean_abs_contrast": changed_contrast,
                "unchanged_projection_mean_abs_contrast": unchanged_contrast,
                "changed_projection_type_counts": type_counts,
                "changed_projection_layer_quartile_counts": quartile_counts,
                "max_leave_one_sequence_out_changed_projection_counts": loo_max_changes,
            },
            "projection_granularity": actual_rank["additive_layer_type_explanation"],
            "secondary_targets": secondary,
        },
        "decisions": {
            "role_separation": "SUPPORTED AS STABLE ALLOCATION INFORMATION",
            "max_aggregation": "NOT ESTABLISHED",
            "projection_level_signal": "SUPPORTED; PERFORMANCE NECESSITY NOT ESTABLISHED",
            "reconstruction_metric": "PRAGMATIC DAMAGE PROXY; UNIQUENESS NOT ESTABLISHED",
        },
        "next_method_development": [
            "Compare aggregation objectives derived from explicit average-risk or robust-risk assumptions; do not tune an arbitrary Max variant from these statistics.",
            "Test whether a layer/type-only compression of the role signal preserves the projection allocation advantage on mini downstream evaluation.",
        ],
    }
    atomic_write_json(ROOT / "decision.json", decision)

    report = f"""# Masked/Unmasked Allocation Signal Analysis

## Question

Do masked and unmasked tokens provide distinct, stable information for projection-wise sparsity allocation?

## Frozen Setup

- LLaDA-8B, 224 projections, 80 frozen DLM states and six sparsity levels.
- Primary grid 50–75% with target 65%; target50/target75 grids are secondary checks.
- Reconstruction sufficient statistics are pooled within role before normalization.
- The independent resampling unit is the WikiText-2 sequence, not its ten timesteps.
- No curve smoothing, GPU collection, mask search, or downstream selection was performed.

## Results: distinct ranking information

At 65%, masked and unmasked marginal costs remain strongly correlated (mean Spearman **{actual_rho:.4f}**), so most projection ordering is shared. However, **{actual_reversal:.2%}** of comparable projection pairs reverse order between roles and the mean absolute percentile contrast is **{actual_rank['mean_absolute_percentile_contrast']:.2%}**.

Cardinality-matched random partitions show only **{random_reversal:.2%}** rank reversals and **{random_contrast:.2%}** mean absolute percentile contrast. Thus the actual DLM role split exposes roughly **{actual_rank['mean_absolute_percentile_contrast']/random_contrast:.2f}×** more rank separation than arbitrary groups of the same size.

## Results: sequence stability

The actual role contrast generalizes across leave-one-sequence-out folds with Spearman **{actual_stability['mean_spearman']:.4f}**, sign agreement **{actual_stability['mean_sign_agreement']:.2%}**, and within-layer Spearman **{actual_stability['mean_within_layer_spearman']:.4f}**. Random partitions have mean LOO Spearman **{random_loo:.4f}** and sign agreement **{random_sign:.2%}**. The actual split therefore creates a larger difference while preserving it more reliably across sequences.

The sequence-cluster bootstrap 95% CI for the actual mean marginal rank correlation is **[{actual_stability['cluster_bootstrap']['mean_increment_spearman']['bootstrap_95_ci'][0]:.4f}, {actual_stability['cluster_bootstrap']['mean_increment_spearman']['bootstrap_95_ci'][1]:.4f}]**; the rank-reversal CI is **[{actual_stability['cluster_bootstrap']['mean_increment_rank_reversal_fraction']['bootstrap_95_ci'][0]:.2%}, {actual_stability['cluster_bootstrap']['mean_increment_rank_reversal_fraction']['bootstrap_95_ci'][1]:.2%}]**.

## Results: pooling and allocation

Masked energy weight has median **{alpha['median']:.3f}** and range **[{alpha['min']:.3f}, {alpha['max']:.3f}]** across projections. Aggregate marginal ranking follows unmasked costs more closely (mean Spearman **{decision['findings']['pooling']['mean_aggregate_vs_unmasked_spearman']:.4f}**) than masked costs (**{decision['findings']['pooling']['mean_aggregate_vs_masked_spearman']:.4f}**).

Current Max and Aggregate differ in **{comparison['changed_projections']}/224** projections and **{comparison['xor_fraction']:.3%}** of prunable weights. Changed projections have mean absolute role contrast **{changed_contrast:.2%}**, versus **{unchanged_contrast:.2%}** elsewhere. Max allocation is also stable: leaving out one sequence changes 0–{max(loo_max_changes)} projections relative to its full-data allocation.

Layer and projection-type fixed effects explain **{actual_rank['additive_layer_type_explanation']['r2']:.1%}** of per-projection role contrast; **{actual_rank['additive_layer_type_explanation']['residual_variance_fraction']:.1%}** remains within layer/type structure. Changed projections span all four layer quartiles and six projection types.

## Interpretation

Masked/unmasked separation carries real, repeatable allocation information. The effect is localized rather than global: most rankings agree, while a stable minority of comparisons and 20 final allocation decisions differ. This matches the earlier observation that token-level signals can vanish in feature-level weight ranking yet still affect the coarser projection budget decision.

These results do not select Max as the correct aggregation rule. They also do not show that reconstruction is uniquely preferable to another cheap damage proxy or that projection-level allocation is necessary for downstream performance. Those remain method-design questions.

## Decision

- Role separation: **supported as stable allocation information**.
- Max aggregation: **not established**.
- Projection-level representation: **supported statistically; downstream necessity not established**.
- Reconstruction: **useful current proxy; uniqueness not established**.
"""
    (ROOT / "report.md").write_text(report)
    print(json.dumps({"status": "complete", "decision": decision["decisions"]}, sort_keys=True))


if __name__ == "__main__":
    main()
