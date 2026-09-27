from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


def pooled_curves(fields: dict[str, np.ndarray], state_ids: Sequence[int]) -> dict[str, np.ndarray]:
    """Pool sufficient statistics over states without treating states as IID."""
    selected = np.asarray(state_ids, dtype=int)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("nonempty one-dimensional state selection required")
    nm = fields["num_masked"][:, selected].sum(axis=1)
    nu = fields["num_unmasked"][:, selected].sum(axis=1)
    dm = fields["den_masked"][:, selected].sum(axis=1)
    du = fields["den_unmasked"][:, selected].sum(axis=1)
    if np.any(dm <= 0) or np.any(du <= 0):
        raise ValueError("role denominators must be positive")
    masked, unmasked = nm / dm, nu / du
    return {
        "masked": masked,
        "unmasked": unmasked,
        "aggregate": (nm + nu) / (dm + du),
        "max": np.maximum(masked, unmasked),
        "alpha_masked": dm / (dm + du),
        "num_masked": nm,
        "num_unmasked": nu,
        "den_masked": dm,
        "den_unmasked": du,
    }


def marginal_cost(curves: np.ndarray, shapes: Sequence[Sequence[int]], grid: Sequence[float]) -> np.ndarray:
    curves = np.asarray(curves, dtype=float)
    sizes = np.asarray([int(a) * int(b) for a, b in shapes], dtype=float)
    steps = np.diff(np.asarray(grid, dtype=float))
    if curves.shape != (len(sizes), len(grid)) or np.any(steps <= 0):
        raise ValueError("curve/shape/grid mismatch")
    return np.diff(curves, axis=1) / (sizes[:, None] * steps[None, :])


def percentile_contrast(masked_cost: np.ndarray, unmasked_cost: np.ndarray) -> np.ndarray:
    """Masked minus unmasked percentile; invariant to role-specific scale."""
    m, u = np.asarray(masked_cost, float), np.asarray(unmasked_cost, float)
    if m.shape != u.shape or m.ndim != 2:
        raise ValueError("aligned module x increment arrays required")
    denominator = max(m.shape[0] - 1, 1)
    return np.column_stack([
        (rankdata(m[:, k], method="average") - rankdata(u[:, k], method="average")) / denominator
        for k in range(m.shape[1])
    ])


def pair_order_summary(x: np.ndarray, y: np.ndarray) -> dict:
    """Summarize whether two costs order projection pairs alike."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("aligned vectors with at least two entries required")
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    upper = np.triu(np.ones(dx.shape, dtype=bool), 1)
    product = dx[upper] * dy[upper]
    concordant = int(np.sum(product > 0))
    discordant = int(np.sum(product < 0))
    ties = int(np.sum(product == 0))
    comparable = concordant + discordant
    return {
        "pairs": int(product.size),
        "concordant": concordant,
        "discordant": discordant,
        "ties": ties,
        "rank_reversal_fraction": float(discordant / comparable) if comparable else None,
        "pareto_comparable_fraction": float(comparable / product.size),
        "tradeoff_fraction": float(discordant / product.size),
    }


def rank_summary(x: np.ndarray, y: np.ndarray) -> dict:
    order = pair_order_summary(x, y)
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "kendall_tau_b": float(kendalltau(x, y, variant="b").statistic),
        **order,
    }


def sign_agreement(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = (x != 0) & (y != 0) & np.isfinite(x) & np.isfinite(y)
    return {
        "comparable": int(valid.sum()),
        "agreement": float(np.mean(np.sign(x[valid]) == np.sign(y[valid]))) if valid.any() else None,
    }


def allocation_difference(left: dict, right: dict, shapes: Sequence[Sequence[int]],
                          grid: Sequence[float]) -> dict:
    levels_left = np.asarray(left["levels"], int)
    levels_right = np.asarray(right["levels"], int)
    if levels_left.shape != levels_right.shape or len(levels_left) != len(shapes):
        raise ValueError("allocation length mismatch")
    per_projection, xor = [], 0
    for index, ((rows, cols), a, b) in enumerate(zip(shapes, levels_left, levels_right)):
        pa = int(rows) * int(int(cols) * float(grid[a]))
        pb = int(rows) * int(int(cols) * float(grid[b]))
        difference = abs(pa - pb)
        xor += difference
        per_projection.append({"module_index": index, "level_left": int(a), "level_right": int(b),
                               "xor_pruned_weights": int(difference)})
    weights = sum(int(a) * int(b) for a, b in shapes)
    return {
        "changed_projections": int(np.sum(levels_left != levels_right)),
        "xor_pruned_weights": int(xor),
        "xor_fraction": float(xor / weights),
        "mean_absolute_sparsity_shift": float(np.mean([
            abs(float(grid[a]) - float(grid[b])) for a, b in zip(levels_left, levels_right)
        ])),
        "per_projection": per_projection,
    }


def additive_group_r2(values: np.ndarray, layers: np.ndarray, types: np.ndarray) -> dict:
    """R2 of additive layer and type fixed effects for a flat projection statistic."""
    y = np.asarray(values, float)
    layer_values = sorted(set(int(x) for x in layers))
    type_values = sorted(set(str(x) for x in types))
    columns = [np.ones(len(y))]
    columns += [(layers == value).astype(float) for value in layer_values[1:]]
    columns += [(types == value).astype(float) for value in type_values[1:]]
    design = np.column_stack(columns)
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    return {"r2": 1.0 - residual / total if total else 1.0,
            "residual_variance_fraction": residual / total if total else 0.0}


def quantiles(values: np.ndarray) -> dict:
    x = np.asarray(values, float)
    return {"min": float(np.min(x)), "q10": float(np.quantile(x, .1)),
            "q25": float(np.quantile(x, .25)), "median": float(np.median(x)),
            "q75": float(np.quantile(x, .75)), "q90": float(np.quantile(x, .9)),
            "max": float(np.max(x)), "mean": float(np.mean(x)), "std": float(np.std(x))}
