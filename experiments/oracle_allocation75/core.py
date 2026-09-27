"""Pure allocation and accounting helpers for the 75% oracle experiment.

The module deliberately contains no pruning score.  It only assigns an integer
row-wise prune count to each already-defined Wanda projection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


GLOBAL_SPARSITY = 0.75
MIDDLE_SPARSITY = 0.75
MODULE_COUNT = 224
PROTECTED_FRACTION = 0.10
PROTECTED_COUNT = 22

# The requested .65/.55/.45 schedules are infeasible for the observed KL
# top/bottom sets.  These are the preregistered, monotonically stronger fallback
# schedules allowed by the prompt's "reduce protection strength" rule.
FALLBACK_SCHEDULES = {
    "mild": 0.72,
    "medium": 0.69,
    "aggressive": 0.66,
}
REQUESTED_SCHEDULES = {
    "mild": 0.65,
    "medium": 0.55,
    "aggressive": 0.45,
}
RANDOM_SEEDS = (20260907, 20260908, 20260909)

MODULE_SHAPES = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 4096),
    "v_proj": (4096, 4096),
    "attn_out": (4096, 4096),
    "ff_proj": (12288, 4096),
    "up_proj": (12288, 4096),
    "ff_out": (4096, 12288),
}

PUBLIC_ALIASES = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "attn_out",
    "gate_proj": "ff_proj",
    "up_proj": "up_proj",
    "down_proj": "ff_out",
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module_records(path):
    records = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            module_type = row["module_type"]
            if module_type not in MODULE_SHAPES:
                raise ValueError(f"unknown projection type: {module_type}")
            out_features, in_features = MODULE_SHAPES[module_type]
            nweight = int(row["nweight"])
            if nweight != out_features * in_features:
                raise ValueError(f"shape/count mismatch for {row['module']}")
            records.append(
                {
                    "module": row["module"],
                    "layer": int(row["layer"]),
                    "module_type": module_type,
                    "out_features": out_features,
                    "in_features": in_features,
                    "nweight": nweight,
                    "functional_kl": float(row["Y50_KL"]),
                    "reconstruction_error": float(row["Erec50"]),
                }
            )
    if len(records) != MODULE_COUNT or len({x["module"] for x in records}) != MODULE_COUNT:
        raise ValueError("expected exactly 224 unique modules")
    return records


def ranked_extremes(records, field, count=PROTECTED_COUNT):
    ordered = sorted(records, key=lambda row: (row[field], row["module"]))
    donors = [row["module"] for row in ordered[:count]]
    protected = [row["module"] for row in reversed(ordered[-count:])]
    if set(protected) & set(donors):
        raise RuntimeError("protected and donor groups overlap")
    return protected, donors


def stratified_random_groups(records, template_protected, template_donors, seed):
    by_type = defaultdict(list)
    lookup = {row["module"]: row for row in records}
    for row in records:
        by_type[row["module_type"]].append(row["module"])
    pcounts = Counter(lookup[name]["module_type"] for name in template_protected)
    dcounts = Counter(lookup[name]["module_type"] for name in template_donors)
    rng = random.Random(seed)
    protected, donors = [], []
    for module_type in sorted(by_type):
        names = sorted(by_type[module_type])
        rng.shuffle(names)
        pc, dc = pcounts[module_type], dcounts[module_type]
        if pc + dc > len(names):
            raise RuntimeError(f"stratified request exceeds {module_type} population")
        protected.extend(names[:pc])
        donors.extend(names[pc : pc + dc])
    if len(protected) != len(template_protected) or len(donors) != len(template_donors):
        raise RuntimeError("stratified random group count mismatch")
    if set(protected) & set(donors):
        raise RuntimeError("random protected/donor overlap")
    return sorted(protected), sorted(donors)


def continuous_donor_sparsity(records, protected, donors, protected_sparsity):
    lookup = {row["module"]: row for row in records}
    pweight = sum(lookup[name]["nweight"] for name in protected)
    dweight = sum(lookup[name]["nweight"] for name in donors)
    return MIDDLE_SPARSITY + (MIDDLE_SPARSITY - protected_sparsity) * pweight / dweight


def feasibility_audit(records, field, q_values=(0.10, 0.15, 0.20)):
    rows = []
    for q in q_values:
        count = round(len(records) * q)
        protected, donors = ranked_extremes(records, field, count)
        lookup = {row["module"]: row for row in records}
        pweight = sum(lookup[name]["nweight"] for name in protected)
        dweight = sum(lookup[name]["nweight"] for name in donors)
        for schedule, protected_sparsity in REQUESTED_SCHEDULES.items():
            donor = continuous_donor_sparsity(records, protected, donors, protected_sparsity)
            rows.append(
                {
                    "ranking": field,
                    "q": q,
                    "group_count": count,
                    "schedule": schedule,
                    "requested_protected_sparsity": protected_sparsity,
                    "protected_parameters": pweight,
                    "donor_parameters": dweight,
                    "protected_to_donor_parameter_ratio": pweight / dweight,
                    "solved_donor_sparsity": donor,
                    "feasible": donor <= 1.0,
                }
            )
    return rows


def _assign_exact_donor_counts(donor_rows, desired, remaining_pruned):
    counts = {
        row["module"]: math.floor(desired * row["in_features"])
        for row in donor_rows
    }
    base = sum(counts[row["module"]] * row["out_features"] for row in donor_rows)
    gap = remaining_pruned - base
    if gap < 0 or gap % 4096:
        raise RuntimeError(f"unexpected donor integer gap: {gap}")
    units = gap // 4096
    by_coefficient = defaultdict(list)
    for row in sorted(donor_rows, key=lambda x: x["module"]):
        coefficient = row["out_features"] // 4096
        by_coefficient[coefficient].append(row)
    # Fractional largest-remainder preference, with an exact 1/3-unit solution.
    for coefficient in by_coefficient:
        by_coefficient[coefficient].sort(
            key=lambda row: (
                -(desired * row["in_features"] - counts[row["module"]]),
                row["module"],
            )
        )
    threes = by_coefficient.get(3, [])
    ones = by_coefficient.get(1, [])
    chosen_three = None
    for nthree in range(min(len(threes), units // 3), -1, -1):
        none = units - 3 * nthree
        if 0 <= none <= len(ones):
            chosen_three = nthree, none
            break
    if chosen_three is None:
        raise RuntimeError("cannot satisfy exact global budget with row-wise donor counts")
    for row in threes[: chosen_three[0]] + ones[: chosen_three[1]]:
        counts[row["module"]] += 1
    achieved = sum(counts[row["module"]] * row["out_features"] for row in donor_rows)
    if achieved != remaining_pruned:
        raise RuntimeError("exact donor accounting failed")
    return counts


def solve_exact_plan(records, protected, donors, protected_sparsity, variant, schedule):
    lookup = {row["module"]: row for row in records}
    protected, donors = set(protected), set(donors)
    if protected & donors:
        raise ValueError("protected/donor overlap")
    if not protected and not donors:
        if variant != "V0":
            raise ValueError("only V0 may be uniform")
    target = int(round(GLOBAL_SPARSITY * sum(row["nweight"] for row in records)))
    if variant == "V0":
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
            ratio = protected_sparsity if row["module"] in protected else MIDDLE_SPARSITY
            count = math.floor(ratio * row["in_features"])
            count_map[row["module"]] = count
            fixed += count * row["out_features"]
        donor_rows = [lookup[name] for name in sorted(donors)]
        donor_weight = sum(row["nweight"] for row in donor_rows)
        remaining = target - fixed
        donor_requested = remaining / donor_weight
        if not 0 <= donor_requested <= 1:
            raise ValueError(f"infeasible donor sparsity {donor_requested} for {variant}/{schedule}")
        count_map.update(_assign_exact_donor_counts(donor_rows, donor_requested, remaining))
    entries = []
    total_pruned = 0
    for row in sorted(records, key=lambda x: x["module"]):
        name = row["module"]
        status = "protected" if name in protected else "donor" if name in donors else "middle"
        if variant == "V0":
            status = "uniform"
            requested = GLOBAL_SPARSITY
        elif status == "protected":
            requested = protected_sparsity
        elif status == "donor":
            requested = donor_requested
        else:
            requested = MIDDLE_SPARSITY
        count = count_map[name]
        pruned = count * row["out_features"]
        total_pruned += pruned
        entries.append(
            {
                **row,
                "status": status,
                "requested_sparsity": requested,
                "prune_per_row": count,
                "actual_sparsity": count / row["in_features"],
                "pruned_parameters": pruned,
                "retained_parameters": row["nweight"] - pruned,
            }
        )
    total = sum(row["nweight"] for row in records)
    if total_pruned != target or total_pruned / total != GLOBAL_SPARSITY:
        raise RuntimeError("global 75% budget is not exact")
    return {
        "variant": variant,
        "schedule": schedule,
        "protected_sparsity": None if variant == "V0" else protected_sparsity,
        "middle_sparsity": GLOBAL_SPARSITY,
        "donor_requested_sparsity": donor_requested,
        "protected_groups": sorted(protected),
        "donor_groups": sorted(donors),
        "total_parameters": total,
        "total_pruned": total_pruned,
        "global_actual_sparsity": total_pruned / total,
        "entries": entries,
    }


def build_all_plans(records):
    functional_p, functional_d = ranked_extremes(records, "functional_kl")
    reconstruction_p, reconstruction_d = ranked_extremes(records, "reconstruction_error")
    selections = {
        "functional_kl": {"protected": functional_p, "donors": functional_d},
        "reconstruction": {"protected": reconstruction_p, "donors": reconstruction_d},
        "random": {},
    }
    for seed in RANDOM_SEEDS:
        p, d = stratified_random_groups(records, reconstruction_p, reconstruction_d, seed)
        selections["random"][str(seed)] = {"protected": p, "donors": d}
    plans = {"V0_uniform": solve_exact_plan(records, [], [], GLOBAL_SPARSITY, "V0", "uniform")}
    for schedule, protected_sparsity in FALLBACK_SCHEDULES.items():
        plans[f"V2_reconstruction_{schedule}"] = solve_exact_plan(
            records, reconstruction_p, reconstruction_d, protected_sparsity, "V2", schedule
        )
        plans[f"V3_functional_kl_{schedule}"] = solve_exact_plan(
            records, functional_p, functional_d, protected_sparsity, "V3", schedule
        )
        for seed in RANDOM_SEEDS:
            groups = selections["random"][str(seed)]
            plans[f"V1_random_{seed}_{schedule}"] = solve_exact_plan(
                records, groups["protected"], groups["donors"], protected_sparsity,
                "V1", schedule,
            )
    return selections, plans


def module_rank(records, field, module):
    ordered = sorted(records, key=lambda row: (-row[field], row["module"]))
    return next(index + 1 for index, row in enumerate(ordered) if row["module"] == module)
