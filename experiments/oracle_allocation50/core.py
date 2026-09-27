"""Exact projection-level budget accounting for the frozen 50% re-validation."""

from __future__ import annotations

import math

from experiments.oracle_allocation75.core import _assign_exact_donor_counts, stratified_random_groups


GLOBAL_SPARSITY = 0.50
MIDDLE_SPARSITY = 0.50
MILD_RELATIVE_PROTECTION = 0.72 / 0.75
PROTECTED_SPARSITY = GLOBAL_SPARSITY * MILD_RELATIVE_PROTECTION  # 0.48
RANDOM_SEEDS = (20260907, 20260908, 20260909)


def solve_exact_plan(records, protected, donors, variant):
    lookup = {row["module"]: row for row in records}
    protected, donors = set(protected), set(donors)
    if protected & donors:
        raise ValueError("protected/donor overlap")
    total = sum(row["nweight"] for row in records)
    target = int(round(GLOBAL_SPARSITY * total))
    if variant == "V0-50":
        count_map = {
            row["module"]: int(GLOBAL_SPARSITY * row["in_features"])
            for row in records
        }
        donor_requested = None
    else:
        count_map = {}
        fixed = 0
        for row in records:
            if row["module"] in donors:
                continue
            ratio = PROTECTED_SPARSITY if row["module"] in protected else MIDDLE_SPARSITY
            count = math.floor(ratio * row["in_features"])
            count_map[row["module"]] = count
            fixed += count * row["out_features"]
        donor_rows = [lookup[name] for name in sorted(donors)]
        remaining = target - fixed
        donor_weight = sum(row["nweight"] for row in donor_rows)
        donor_requested = remaining / donor_weight
        if not 0 <= donor_requested <= 1:
            raise ValueError(f"infeasible donor sparsity {donor_requested} for {variant}")
        count_map.update(_assign_exact_donor_counts(donor_rows, donor_requested, remaining))

    entries, total_pruned = [], 0
    for row in sorted(records, key=lambda x: x["module"]):
        name = row["module"]
        if variant == "V0-50":
            status, requested = "uniform", GLOBAL_SPARSITY
        elif name in protected:
            status, requested = "protected", PROTECTED_SPARSITY
        elif name in donors:
            status, requested = "donor", donor_requested
        else:
            status, requested = "middle", MIDDLE_SPARSITY
        count = count_map[name]
        pruned = count * row["out_features"]
        total_pruned += pruned
        entries.append({
            **row,
            "status": status,
            "requested_sparsity": requested,
            "prune_per_row": count,
            "actual_sparsity": count / row["in_features"],
            "sparsity_deviation_from_uniform": count / row["in_features"] - GLOBAL_SPARSITY,
            "pruned_parameters": pruned,
            "retained_parameters": row["nweight"] - pruned,
        })
    if total_pruned != target:
        raise RuntimeError(f"exact 50% budget failed: {total_pruned} != {target}")
    return {
        "variant": variant,
        "schedule": "uniform" if variant == "V0-50" else "mild-relative",
        "global_target_sparsity": GLOBAL_SPARSITY,
        "protected_sparsity": None if variant == "V0-50" else PROTECTED_SPARSITY,
        "middle_sparsity": MIDDLE_SPARSITY,
        "donor_requested_sparsity": donor_requested,
        "protected_groups": sorted(protected),
        "donor_groups": sorted(donors),
        "total_parameters": total,
        "total_pruned": total_pruned,
        "global_actual_sparsity": total_pruned / total,
        "entries": entries,
    }


def build_plans(records, selections_75):
    functional = selections_75["functional_kl"]
    plans = {
        "V0_50_uniform": solve_exact_plan(records, [], [], "V0-50"),
        "V3_50_functional_kl_mild": solve_exact_plan(
            records, functional["protected"], functional["donors"], "V3-50"
        ),
    }
    for seed in RANDOM_SEEDS:
        protected, donors = stratified_random_groups(
            records, functional["protected"], functional["donors"], seed
        )
        plans[f"V1_50_random_{seed}"] = solve_exact_plan(
            records, protected, donors, f"V1-50-{seed}"
        )
    return plans
