"""Pure numerical helpers for the dual-role allocation diagnostic."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch


def partition_role_sums(outputs: torch.Tensor, masked_positions: torch.Tensor) -> dict:
    """Partition seven Linear variants into masked/unmasked squared-energy sums."""
    outputs = outputs.float()
    mask = torch.as_tensor(masked_positions, dtype=torch.bool, device=outputs.device)
    if outputs.ndim != 3 or outputs.shape[0] != 7 or mask.shape != (outputs.shape[1],):
        raise ValueError("expected seven variants and one position mask")
    if not bool(mask.any()) or not bool((~mask).any()):
        raise ValueError("empty token role")
    if not bool(torch.isfinite(outputs).all()):
        raise ValueError("nonfinite Linear output")

    dense = outputs[0]
    delta_squared = (outputs[1:] - dense.unsqueeze(0)).square()
    dense_squared = dense.square()
    den_masked = float(dense_squared[mask].sum().double().cpu())
    den_unmasked = float(dense_squared[~mask].sum().double().cpu())
    if den_masked == 0.0 or den_unmasked == 0.0:
        raise ValueError("zero role denominator")
    return {
        "num_masked": [float(value.double().cpu())
                       for value in delta_squared[:, mask].sum(dim=(1, 2))],
        "den_masked": den_masked,
        "num_unmasked": [float(value.double().cpu())
                         for value in delta_squared[:, ~mask].sum(dim=(1, 2))],
        "den_unmasked": den_unmasked,
    }


def role_contrast(masked, unmasked) -> np.ndarray:
    """Return bounded role contrast; define the exact zero/zero case as zero."""
    masked = np.asarray(masked, dtype=float)
    unmasked = np.asarray(unmasked, dtype=float)
    if masked.shape != unmasked.shape or not np.isfinite(masked).all() or not np.isfinite(unmasked).all():
        raise ValueError("finite aligned role arrays required")
    denominator = masked + unmasked
    if np.any(denominator < 0):
        raise ValueError("role errors must be nonnegative")
    result = np.zeros_like(denominator)
    nonzero = denominator != 0
    result[nonzero] = np.abs(masked[nonzero] - unmasked[nonzero]) / denominator[nonzero]
    return result


def pool_role_curves(raw: Sequence[dict], state_ids: Iterable[int]) -> dict:
    """Pool state sufficient statistics as ratios of sums for one projection."""
    requested = [int(value) for value in state_ids]
    by_state = {}
    for row in raw:
        state_index = int(row["state_index"])
        if state_index in by_state:
            raise ValueError("duplicate state index")
        by_state[state_index] = row
    if len(set(requested)) != len(requested) or any(state not in by_state for state in requested):
        raise ValueError("requested states are missing or duplicated")
    selected = [by_state[state] for state in requested]
    if not selected:
        raise ValueError("requested states are empty")
    levels = len(selected[0]["levels"])
    if levels == 0 or any(len(row["levels"]) != levels for row in selected):
        raise ValueError("state level counts differ")

    num_masked = np.zeros(levels, dtype=float)
    num_unmasked = np.zeros(levels, dtype=float)
    den_masked = np.zeros(levels, dtype=float)
    den_unmasked = np.zeros(levels, dtype=float)
    for row in selected:
        for level, values in enumerate(row["levels"]):
            num_masked[level] += float(values["num_masked"])
            num_unmasked[level] += float(values["num_unmasked"])
            den_masked[level] += float(values["den_masked"])
            den_unmasked[level] += float(values["den_unmasked"])
    if (den_masked <= 0).any() or (den_unmasked <= 0).any():
        raise ValueError("zero role denominator")
    masked = num_masked / den_masked
    unmasked = num_unmasked / den_unmasked
    aggregate = (num_masked + num_unmasked) / (den_masked + den_unmasked)
    return {
        "masked": masked,
        "unmasked": unmasked,
        "aggregate": aggregate,
        "role": np.maximum(masked, unmasked),
        "contrast": role_contrast(masked, unmasked),
    }


def allocation_mask_xor(levels_a: Sequence[int], levels_b: Sequence[int],
                        manifest: Sequence[dict]) -> dict:
    """Compute exact XOR for nested masks selected from one Wanda ranking."""
    if len(levels_a) != len(levels_b) or len(levels_a) != len(manifest):
        raise ValueError("allocation and manifest lengths differ")
    xor = 0
    changed = 0
    weights = 0
    per_projection = []
    for index, (left, right, row) in enumerate(zip(levels_a, levels_b, manifest)):
        left, right = int(left), int(right)
        masks = row["masks"]
        if not 0 <= left < len(masks) or not 0 <= right < len(masks):
            raise ValueError("allocation level is outside manifest")
        difference = abs(int(masks[left]["pruned"]) - int(masks[right]["pruned"]))
        xor += difference
        weights += int(row["weights"])
        changed += left != right
        per_projection.append({"module_index": index, "level_a": left, "level_b": right,
                               "xor_pruned_weights": difference})
    if weights <= 0:
        raise ValueError("manifest contains no prunable weights")
    return {
        "xor_pruned_weights": xor,
        "xor_fraction_of_prunable_weights": xor / weights,
        "changed_projection_count": changed,
        "total_prunable_weights": weights,
        "per_projection": per_projection,
    }


def bootstrap_mean_ci(values, resamples: int = 20000, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("finite nonempty one-dimensional values required")
    if resamples <= 0:
        raise ValueError("positive bootstrap resample count required")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "bootstrap_95_ci": np.quantile(means, [.025, .975]).tolist(),
        "resamples": int(resamples),
        "seed": int(seed),
    }
