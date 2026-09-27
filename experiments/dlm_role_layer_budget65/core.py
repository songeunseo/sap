"""Pure exact-budget helpers for layer-controlled projection allocation."""
from __future__ import annotations

import math
from functools import reduce

import numpy as np

from experiments.projection_capacity_allocation_65.core import GRID


def _reachable(counts: list[list[int]], levels: list[int], remaining: int) -> bool:
    if remaining < 0:
        return False
    reachable = 1
    limit = (1 << (remaining + 1)) - 1
    for row, level in zip(counts, levels):
        base = row[level]
        nxt = 0
        for value in row[level:]:
            delta = value - base
            if delta <= remaining:
                nxt |= reachable << delta
        reachable = nxt & limit
    return bool((reachable >> remaining) & 1)


def allocate_exact_target(curves, shapes, target_pruned: int) -> dict:
    """Apply the historical marginal-greedy rule under one exact group budget."""
    damage = np.asarray(curves, dtype=float)
    if damage.shape != (len(shapes), len(GRID)) or not np.isfinite(damage).all():
        raise ValueError("complete finite group curves required")
    counts = [[int(rows) * int(int(cols) * rate) for rate in GRID]
              for rows, cols in shapes]
    unit = reduce(math.gcd, (value for row in counts for value in row))
    units = [[value // unit for value in row] for row in counts]
    if int(target_pruned) % unit:
        raise ValueError("target is not aligned to the group's integer budget unit")
    target = int(target_pruned) // unit
    levels = [0] * len(shapes)
    remaining = target - sum(row[0] for row in units)
    if not _reachable(units, levels, remaining):
        raise ValueError("exact group budget is not reachable on the frozen grid")

    trace, skipped = [], []
    while remaining:
        candidates = []
        for local_index, level in enumerate(levels):
            if level < len(GRID) - 1:
                marginal = float(damage[local_index, level + 1] - damage[local_index, level])
                nominal_gain = 0.05 * math.prod(shapes[local_index])
                candidates.append((marginal / nominal_gain, local_index, marginal))
        for cost_per_parameter, local_index, marginal in sorted(candidates):
            old = levels[local_index]
            gain = units[local_index][old + 1] - units[local_index][old]
            if gain > remaining:
                continue
            levels[local_index] += 1
            if _reachable(units, levels, remaining - gain):
                remaining -= gain
                trace.append({
                    "local_index": local_index,
                    "from": GRID[old],
                    "to": GRID[old + 1],
                    "marginal_cost": marginal,
                    "cost_per_nominal_parameter": cost_per_parameter,
                    "actual_parameter_gain": gain * unit,
                    "remaining": remaining * unit,
                })
                break
            levels[local_index] -= 1
            skipped.append({"step": len(trace), "local_index": local_index,
                            "from": GRID[old], "reason": "exact budget reachability"})
        else:
            raise RuntimeError("no feasible marginal increment")

    pruned = sum(row[level] for row, level in zip(counts, levels))
    if pruned != int(target_pruned):
        raise RuntimeError("exact group budget invariant failed")
    return {
        "levels": levels,
        "sparsities": [GRID[level] for level in levels],
        "pruned": pruned,
        "integer_budget_unit": unit,
        "trace": trace,
        "feasibility_skips": skipped,
    }


def layer_indices(names: list[str]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for index, name in enumerate(names):
        prefix, _ = name.split(".", 1)
        if not prefix.startswith("block_"):
            raise ValueError(f"invalid projection name: {name}")
        layer = int(prefix.removeprefix("block_"))
        groups.setdefault(layer, []).append(index)
    if sorted(groups) != list(range(32)) or any(len(rows) != 7 for rows in groups.values()):
        raise ValueError("expected exactly seven projections in each of 32 layers")
    return groups


def allocate_by_layer(curves, shapes, names: list[str], layer_targets: dict[int, int]) -> dict:
    curves = np.asarray(curves, dtype=float)
    if curves.shape != (len(names), len(GRID)) or len(shapes) != len(names):
        raise ValueError("global inputs do not align")
    groups = layer_indices(names)
    if set(layer_targets) != set(groups):
        raise ValueError("layer target keys mismatch")
    levels = [None] * len(names)
    details = []
    for layer, indices in groups.items():
        result = allocate_exact_target(
            curves[indices], [shapes[index] for index in indices], layer_targets[layer]
        )
        for index, level in zip(indices, result["levels"]):
            levels[index] = level
        details.append({"layer": layer, "module_indices": indices, **result})
    if any(level is None for level in levels):
        raise RuntimeError("incomplete layer allocation")
    pruned = sum(row["pruned"] for row in details)
    return {
        "levels": levels,
        "sparsities": [GRID[level] for level in levels],
        "pruned": pruned,
        "layer_targets": {str(key): int(value) for key, value in sorted(layer_targets.items())},
        "layers": details,
    }


def layer_pruned_counts(manifest: dict) -> dict[int, int]:
    names = [row["name"] for row in manifest["entries"]]
    groups = layer_indices(names)
    return {
        layer: sum(int(manifest["entries"][index]["selected_mask"]["pruned"])
                   for index in indices)
        for layer, indices in groups.items()
    }
