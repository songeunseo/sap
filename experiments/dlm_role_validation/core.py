"""Pure helpers for random-role controls and shifted-sparsity allocation."""
from __future__ import annotations

import hashlib
import math
from functools import reduce
from typing import Sequence

import numpy as np
import torch


GRIDS = {
    "random65": (.50, .55, .60, .65, .70, .75),
    "target50": (.35, .40, .45, .50, .55, .60),
    "target75": (.60, .65, .70, .75, .80, .85),
}
TARGET_INDEX = 3
RANDOM_SEEDS = (101, 202, 303)


def deterministic_random_role(masked_positions: torch.Tensor, seed: int,
                              state_index: int) -> torch.Tensor:
    """Return a deterministic random group with the exact real-mask cardinality."""
    real = torch.as_tensor(masked_positions, dtype=torch.bool).cpu()
    if real.ndim != 1 or not bool(real.any()) or bool(real.all()):
        raise ValueError("one-dimensional nontrivial real token mask required")
    generator = np.random.default_rng(np.random.SeedSequence([int(seed), int(state_index)]))
    chosen = generator.choice(real.numel(), size=int(real.sum()), replace=False)
    result = torch.zeros(real.numel(), dtype=torch.bool)
    result[torch.as_tensor(chosen, dtype=torch.long)] = True
    return result


def partition_digest(partitions: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for mask in partitions:
        value = torch.as_tensor(mask, dtype=torch.uint8).cpu().contiguous().numpy()
        digest.update(value.tobytes())
    return digest.hexdigest()


def partition_sums(outputs: torch.Tensor, group_a: torch.Tensor) -> dict:
    """Squared reconstruction sufficient statistics for one binary partition."""
    outputs = outputs.float()
    selected = torch.as_tensor(group_a, dtype=torch.bool, device=outputs.device)
    if outputs.ndim != 3 or outputs.shape[0] != 7:
        raise ValueError("seven Linear variants are required")
    if selected.shape != (outputs.shape[1],) or not bool(selected.any()) or not bool((~selected).any()):
        raise ValueError("nonempty aligned binary token partition required")
    dense = outputs[0]
    delta = (outputs[1:] - dense.unsqueeze(0)).square()
    energy = dense.square()
    return {
        "num_a": [float(x.double().cpu()) for x in delta[:, selected].sum((1, 2))],
        "den_a": float(energy[selected].sum().double().cpu()),
        "num_b": [float(x.double().cpu()) for x in delta[:, ~selected].sum((1, 2))],
        "den_b": float(energy[~selected].sum().double().cpu()),
    }


def pool_partition(records: Sequence[dict], label: str) -> dict:
    """Pool numerator/denominator sums over states, then form Aggregate and Role."""
    if not records:
        raise ValueError("records cannot be empty")
    levels = len(records[0]["groups"][label])
    totals = {key: np.zeros(levels, dtype=float) for key in ("num_a", "den_a", "num_b", "den_b")}
    for record in records:
        rows = record["groups"][label]
        if len(rows) != levels:
            raise ValueError("inconsistent level count")
        for level, row in enumerate(rows):
            for key in totals:
                totals[key][level] += float(row[key])
    if np.any(totals["den_a"] <= 0) or np.any(totals["den_b"] <= 0):
        raise ValueError("zero partition denominator")
    a = totals["num_a"] / totals["den_a"]
    b = totals["num_b"] / totals["den_b"]
    aggregate = (totals["num_a"] + totals["num_b"]) / (totals["den_a"] + totals["den_b"])
    return {"a": a, "b": b, "aggregate": aggregate, "role": np.maximum(a, b)}


def _reachable(counts, levels, remaining):
    if remaining < 0:
        return False
    if remaining == 0:
        return True
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
        if (reachable >> remaining) & 1:
            return True
    return False


def allocate_grid(curves, shapes, grid, target_index: int = TARGET_INDEX) -> dict:
    """Exact row-floor greedy marginal allocation on any ordered six-point grid."""
    grid = tuple(float(x) for x in grid)
    damage = np.asarray(curves, dtype=float)
    if len(grid) != 6 or damage.shape != (len(shapes), len(grid)) or not np.isfinite(damage).all():
        raise ValueError("complete finite module x six-level curves required")
    if not all(grid[i + 1] > grid[i] for i in range(5)) or not 0 <= target_index < 6:
        raise ValueError("strictly increasing grid and valid target index required")
    counts = [[int(rows) * int(int(cols) * r) for r in grid] for rows, cols in shapes]
    unit = reduce(math.gcd, (value for row in counts for value in row))
    units = [[value // unit for value in row] for row in counts]
    target = sum(row[target_index] for row in units)
    levels = [0] * len(shapes)
    remaining = target - sum(row[0] for row in units)
    if not _reachable(units, levels, remaining):
        raise RuntimeError("target budget is not reachable")
    trace, skipped = [], []
    while remaining:
        candidates = []
        for module, level in enumerate(levels):
            if level < len(grid) - 1:
                cost = float(damage[module, level + 1] - damage[module, level])
                nominal_gain = (grid[level + 1] - grid[level]) * math.prod(shapes[module])
                candidates.append((cost / nominal_gain, module, cost))
        for per_parameter, module, cost in sorted(candidates):
            old = levels[module]
            gain = units[module][old + 1] - units[module][old]
            if gain > remaining:
                continue
            levels[module] += 1
            if _reachable(units, levels, remaining - gain):
                remaining -= gain
                trace.append({"module_index": module, "from": grid[old], "to": grid[old + 1],
                              "marginal_cost": cost, "cost_per_parameter": per_parameter,
                              "actual_parameter_gain": gain * unit, "remaining": remaining * unit})
                break
            levels[module] -= 1
            skipped.append({"step": len(trace), "module_index": module, "from": grid[old]})
        else:
            raise RuntimeError("no feasible increment despite reachability invariant")
    pruned = sum(row[level] for row, level in zip(counts, levels))
    return {
        "grid": list(grid), "target_index": target_index, "levels": levels,
        "sparsities": [grid[level] for level in levels], "pruned": pruned,
        "uniform_pruned": target * unit, "budget_error": pruned - target * unit,
        "weights": sum(math.prod(shape) for shape in shapes), "trace": trace,
        "feasibility_skips": skipped, "integer_budget_unit": unit,
    }


def random_mini_gate(actual_correct: int, random_correct: Sequence[int]) -> dict:
    values = [int(x) for x in random_correct]
    if len(values) != 3:
        raise ValueError("exactly three random controls required")
    wins = sum(int(actual_correct) > value for value in values)
    passed = float(actual_correct) > float(np.mean(values)) and wins >= 2
    return {"passed": bool(passed), "actual_role_correct": int(actual_correct),
            "random_correct": values, "random_mean": float(np.mean(values)),
            "strict_seed_wins": wins,
            "decision": "RUN RANDOM FULL" if passed else "STOP RANDOM CONTROL"}


def sparsity_mini_gate(uniform: int, aggregate: int, role: int) -> dict:
    passed = int(role) > int(uniform) and int(role) > int(aggregate)
    return {"passed": bool(passed), "uniform_correct": int(uniform),
            "aggregate_correct": int(aggregate), "role_correct": int(role),
            "decision": "RUN TARGET FULL" if passed else "STOP TARGET"}
