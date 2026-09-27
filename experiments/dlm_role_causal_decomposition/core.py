"""Pure helpers for role-error causal interventions and analysis."""
from __future__ import annotations

import numpy as np
import torch


CONDITIONS = (
    "sham",
    "masked_only",
    "unmasked_only",
    "both_mixed",
    "both_direct",
    "masked_energy_matched",
    "unmasked_energy_matched",
    "random_101_a",
    "random_101_b",
    "random_202_a",
    "random_202_b",
    "random_303_a",
    "random_303_b",
)


def role_intervention_outputs(dense: torch.Tensor, sparse: torch.Tensor,
                              actual_mask: torch.Tensor,
                              random_masks: dict[int, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    """Compose diagnostic output rows from aligned dense/sparse Linear outputs."""
    if dense.shape != sparse.shape or dense.ndim != 3 or dense.shape[0] != len(CONDITIONS):
        raise ValueError("dense/sparse outputs must be condition x token x feature")
    actual = torch.as_tensor(actual_mask, dtype=torch.bool, device=dense.device).reshape(-1)
    if actual.shape != (dense.shape[1],) or not bool(actual.any()) or bool(actual.all()):
        raise ValueError("nontrivial aligned actual token mask required")
    if set(random_masks) != {101, 202, 303}:
        raise ValueError("exact random control seeds required")
    delta = sparse - dense
    reference_delta = delta[0].float()
    norm_m = reference_delta[actual].norm()
    norm_u = reference_delta[~actual].norm()
    target = torch.minimum(norm_m, norm_u)
    scale_m = target / norm_m if float(norm_m) > 0 else torch.zeros_like(target)
    scale_u = target / norm_u if float(norm_u) > 0 else torch.zeros_like(target)

    result = dense.clone()
    actual_rows = {
        1: (actual, 1.0), 2: (~actual, 1.0),
        5: (actual, scale_m), 6: (~actual, scale_u),
    }
    for row, (positions, scale) in actual_rows.items():
        result[row, positions] = dense[row, positions] + delta[row, positions] * scale
    result[3] = torch.where(actual[:, None], sparse[3], dense[3])
    result[3] = torch.where((~actual)[:, None], sparse[3], result[3])
    result[4] = sparse[4]
    row = 7
    for seed in (101, 202, 303):
        group = torch.as_tensor(random_masks[seed], dtype=torch.bool,
                                device=dense.device).reshape(-1)
        if group.shape != actual.shape or int(group.sum()) != int(actual.sum()):
            raise ValueError("random group must match actual-mask cardinality")
        result[row, group] = sparse[row, group]
        result[row + 1, ~group] = sparse[row + 1, ~group]
        row += 2
    audit = {
        "masked_delta_norm": float(norm_m.cpu()),
        "unmasked_delta_norm": float(norm_u.cpu()),
        "matched_target_norm": float(target.cpu()),
        "masked_scale": float(scale_m.cpu()),
        "unmasked_scale": float(scale_u.cpu()),
    }
    return result, audit


def normalized_abs_contrast(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = np.abs(left) + np.abs(right)
    result = np.zeros_like(denominator)
    np.divide(np.abs(left - right), denominator, out=result, where=denominator > 0)
    return result


def logit_additivity_metrics(logits: torch.Tensor, token_mask: torch.Tensor) -> dict:
    """Measure whether actual M/U interventions add in logit space."""
    selected = logits[:, torch.as_tensor(token_mask, dtype=torch.bool,
                                         device=logits.device).reshape(-1)].float()
    sham = selected[0]
    dm = selected[1] - sham
    du = selected[2] - sham
    dboth = selected[3] - sham
    residual = dboth - dm - du
    denominator = dboth.norm()
    return {
        "masked_logit_delta_norm": float(dm.norm().cpu()),
        "unmasked_logit_delta_norm": float(du.norm().cpu()),
        "both_logit_delta_norm": float(dboth.norm().cpu()),
        "additivity_residual_norm": float(residual.norm().cpu()),
        "relative_additivity_residual": float((residual.norm() / denominator.clamp_min(1e-30)).cpu()),
    }


def sequence_cluster_interval(values, sequence_ids, *, resamples=20000, seed=20260912):
    """Percentile interval for a mean, resampling the eight source sequences."""
    values = np.asarray(values, dtype=float)
    sequence_ids = np.asarray(sequence_ids, dtype=int)
    sequences = np.unique(sequence_ids)
    means = np.asarray([values[sequence_ids == sequence].mean() for sequence in sequences])
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(0, len(means), size=(resamples, len(means)))].mean(1)
    return {
        "mean": float(means.mean()),
        "sequence_means": {str(int(k)): float(v) for k, v in zip(sequences, means)},
        "bootstrap_95_ci": np.quantile(sampled, [.025, .975]).tolist(),
        "resamples": int(resamples), "seed": int(seed),
    }
