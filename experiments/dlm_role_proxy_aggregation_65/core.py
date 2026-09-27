"""Pure helpers for role-conditioned allocation candidates."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch


AGGREGATIONS = ("balanced", "max_level", "max_marginal")


def aggregate_role_curves(masked, unmasked, rule: str) -> np.ndarray:
    """Turn two module x level role curves into an allocation cost curve.

    ``max_level`` induces Delta max(E_M,E_U), whereas ``max_marginal``
    induces max(Delta E_M, Delta E_U).  The latter is represented as a
    cumulative curve because the allocator consumes adjacent differences.
    """
    masked = np.asarray(masked, dtype=float)
    unmasked = np.asarray(unmasked, dtype=float)
    if masked.shape != unmasked.shape or masked.ndim != 2 or masked.shape[1] != 6:
        raise ValueError("aligned module x six-level role curves required")
    if not np.isfinite(masked).all() or not np.isfinite(unmasked).all():
        raise ValueError("role curves must be finite")
    if rule == "balanced":
        return (masked + unmasked) / 2.0
    if rule == "max_level":
        return np.maximum(masked, unmasked)
    if rule == "max_marginal":
        marginal = np.maximum(np.diff(masked, axis=1), np.diff(unmasked, axis=1))
        return np.concatenate([np.zeros((len(masked), 1)), np.cumsum(marginal, axis=1)], axis=1)
    raise ValueError(f"unknown aggregation: {rule}")


def diagonal_reconstruction_curve(weight: torch.Tensor, masks: list[torch.Tensor],
                                  role_input_square_sum: torch.Tensor,
                                  role_dense_output_energy: float) -> list[float]:
    """Diagonal-covariance approximation of normalized output reconstruction."""
    weight = torch.as_tensor(weight).float()
    x2 = torch.as_tensor(role_input_square_sum).float().to(weight.device)
    if x2.shape != (weight.shape[1],) or not torch.isfinite(x2).all() or bool((x2 < 0).any()):
        raise ValueError("invalid role input second moment")
    if not np.isfinite(role_dense_output_energy) or role_dense_output_energy <= 0:
        raise ValueError("positive dense-output energy required")
    result = []
    square = weight.square()
    for mask in masks:
        mask = torch.as_tensor(mask, dtype=torch.bool, device=weight.device)
        if mask.shape != weight.shape:
            raise ValueError("mask/weight shape mismatch")
        removed_column_energy = (square * mask).sum(dim=0)
        result.append(float(torch.dot(removed_column_energy, x2).double().cpu()) /
                      float(role_dense_output_energy))
    return result


def selected_signature(manifest: dict) -> str:
    rows = [(row["name"], row["selected_mask"]["mask_sha256"],
             int(row["selected_mask"]["pruned"])) for row in manifest["entries"]]
    encoded = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def allocation_difference(left: dict, right: dict) -> dict:
    a = [int(row["level"]) for row in left["entries"]]
    b = [int(row["level"]) for row in right["entries"]]
    weights = [int(row["weights"]) for row in left["entries"]]
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    # Candidate masks are nested row-wise.  Use persisted row-floor counts rather
    # than nominal 5pp arithmetic (e.g. 5% of 11008 is not an integer per row).
    xor = sum(abs(int(left["entries"][i]["selected_mask"]["pruned"])
                  - int(right["entries"][i]["selected_mask"]["pruned"]))
              for i in changed)
    return {
        "changed_projections": len(changed),
        "changed_indices": changed,
        "sum_absolute_level_difference": int(sum(abs(x - y) for x, y in zip(a, b))),
        "mask_xor_weights": xor,
        "mask_xor_fraction_of_prunable_weights": xor / sum(weights),
    }
