#!/usr/bin/env python3
"""Analyze role-specific causal intervention results without downstream labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.dlm_dual_role_allocation.io import load_frozen_inputs
from experiments.dlm_dual_role_allocation.io import atomic_write_json, file_sha256
from experiments.dlm_role_causal_decomposition.core import (
    CONDITIONS, normalized_abs_contrast, sequence_cluster_interval,
)


ROOT = Path("experiments/dlm_role_causal_decomposition")
RESULTS = Path("/DATA/tmluser1/sap-dlm-role-causal-decomposition/results.pt")


def safe_spearman(left, right) -> float:
    value = spearmanr(np.asarray(left), np.asarray(right)).statistic
    return float(value) if np.isfinite(value) else float("nan")


def summarize(values) -> dict:
    values = np.asarray(values, dtype=float).reshape(-1)
    q = np.quantile(values, [.1, .25, .5, .75, .9])
    return {"mean": float(values.mean()), "std": float(values.std()),
            "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]),
            "min": float(values.min()), "max": float(values.max())}


def loo_stability(projection_by_sequence: np.ndarray) -> dict:
    full = projection_by_sequence.mean(1)
    correlations, signs = [], []
    for held in range(projection_by_sequence.shape[1]):
        train = np.delete(projection_by_sequence, held, axis=1).mean(1)
        correlations.append(safe_spearman(full, train))
        signs.append(float(np.mean(np.sign(full) == np.sign(train))))
    return {"spearman": summarize(correlations), "sign_agreement": summarize(signs),
            "per_heldout_sequence_spearman": correlations,
            "per_heldout_sequence_sign_agreement": signs}


def main() -> None:
    manifest = json.loads((ROOT / "collection_manifest.json").read_text())
    if manifest["status"] != "complete" or file_sha256(RESULTS) != manifest["result_sha256"]:
        raise RuntimeError("complete verified collection required")
    payload = torch.load(RESULTS, map_location="cpu", weights_only=False)
    if int(payload["completed_states"]) != 80 or payload["condition_names"] != list(CONDITIONS):
        raise RuntimeError("incomplete or mismatched causal payload")
    names = payload["module_names"]
    sequence = payload["sequence_index"].numpy()
    unique_sequences = np.unique(sequence)
    fields = {name: value.numpy() for name, value in payload["fields"].items()}
    kl = fields["kl"]
    index = {name: CONDITIONS.index(name) for name in CONDITIONS}

    actual_contrast = normalized_abs_contrast(kl[:, :, index["masked_only"]],
                                               kl[:, :, index["unmasked_only"]])
    random_contrasts = np.stack([
        normalized_abs_contrast(kl[:, :, index[f"random_{seed}_a"]],
                                kl[:, :, index[f"random_{seed}_b"]])
        for seed in (101, 202, 303)], axis=-1)
    random_mean = random_contrasts.mean(-1)
    matched_contrast = normalized_abs_contrast(
        kl[:, :, index["masked_energy_matched"]],
        kl[:, :, index["unmasked_energy_matched"]])
    primary_difference = actual_contrast - random_mean

    local = payload["local"].numpy()
    num_m, den_m, num_u, den_u = (local[:, :, 0, i] for i in range(4))
    error_m, error_u = num_m / den_m, num_u / den_u
    local_signed = np.divide(error_m - error_u, error_m + error_u,
                             out=np.zeros_like(error_m), where=(error_m + error_u) > 0)
    causal_m = kl[:, :, index["masked_only"]]
    causal_u = kl[:, :, index["unmasked_only"]]
    causal_signed = np.divide(causal_m - causal_u, causal_m + causal_u,
                              out=np.zeros_like(causal_m), where=(causal_m + causal_u) > 0)
    numerator_signed = np.sign(num_m - num_u)
    normalized_signed = np.sign(error_m - error_u)
    causal_sign = np.sign(causal_m - causal_u)

    frozen_states = load_frozen_inputs().states["states"]
    count_m = np.asarray([np.asarray(state["mask"], dtype=bool).sum()
                          for state in frozen_states], dtype=float)[:, None]
    count_u = 256.0 - count_m

    def role_agreement(local_left, local_right, causal_left, causal_right):
        left_sign = np.sign(local_left - local_right)
        right_sign = np.sign(causal_left - causal_right)
        valid = (left_sign != 0) & (right_sign != 0)
        values = (left_sign[valid] == right_sign[valid]).astype(float)
        ids = np.repeat(sequence[:, None], len(names), axis=1)[valid]
        return values, ids

    actual_agreements = {}
    random_agreements = {key: [] for key in ("raw_sum", "per_token", "dense_energy_normalized")}
    local_variants = {
        "raw_sum": (num_m, num_u),
        "per_token": (num_m / count_m, num_u / count_u),
        "dense_energy_normalized": (error_m, error_u),
    }
    for label, (left, right) in local_variants.items():
        values, ids = role_agreement(left, right, causal_m, causal_u)
        actual_agreements[label] = (values, ids)
    for partition_index, seed in enumerate((101, 202, 303), start=1):
        r_num_a, r_den_a, r_num_b, r_den_b = (
            local[:, :, partition_index, field] for field in range(4))
        r_causal_a = kl[:, :, index[f"random_{seed}_a"]]
        r_causal_b = kl[:, :, index[f"random_{seed}_b"]]
        variants = {
            "raw_sum": (r_num_a, r_num_b),
            "per_token": (r_num_a / count_m, r_num_b / count_u),
            "dense_energy_normalized": (r_num_a / r_den_a, r_num_b / r_den_b),
        }
        for label, (left, right) in variants.items():
            values, _ = role_agreement(left, right, r_causal_a, r_causal_b)
            random_agreements[label].append(values.reshape(80, 28))

    agreement_reports = {}
    repeated_sequence = np.repeat(sequence[:, None], len(names), axis=1).reshape(-1)
    for label, (actual_values, actual_ids) in actual_agreements.items():
        random_values = np.stack(random_agreements[label], axis=-1).mean(-1).reshape(-1)
        actual_grid = actual_values.reshape(80, 28).reshape(-1)
        difference = actual_grid - random_values
        agreement_reports[label] = {
            "actual": sequence_cluster_interval(actual_grid, repeated_sequence,
                                                resamples=20000, seed=20260920),
            "random_mean": sequence_cluster_interval(random_values, repeated_sequence,
                                                     resamples=20000, seed=20260921),
            "actual_minus_random": sequence_cluster_interval(difference, repeated_sequence,
                                                             resamples=20000, seed=20260922),
        }

    sequence_primary = np.asarray([
        primary_difference[sequence == seq].mean() for seq in unique_sequences])
    primary_cluster = sequence_cluster_interval(
        primary_difference.mean(1), sequence, resamples=20000, seed=20260912)
    alignment = (normalized_signed == causal_sign).astype(float)
    non_tied = (normalized_signed != 0) & (causal_sign != 0)
    aligned_values = alignment[non_tied]
    aligned_sequences = np.repeat(sequence[:, None], len(names), axis=1)[non_tied]
    alignment_cluster = sequence_cluster_interval(
        aligned_values, aligned_sequences, resamples=20000, seed=20260913)

    projection_sequence = np.empty((len(names), len(unique_sequences)))
    for module in range(len(names)):
        for column, seq in enumerate(unique_sequences):
            projection_sequence[module, column] = causal_signed[sequence == seq, module].mean()

    random_signed = []
    random_loo = {}
    for seed in (101, 202, 303):
        left = kl[:, :, index[f"random_{seed}_a"]]
        right = kl[:, :, index[f"random_{seed}_b"]]
        signed = np.divide(left - right, left + right, out=np.zeros_like(left),
                           where=(left + right) > 0)
        random_signed.append(signed)
        by_sequence = np.empty((len(names), len(unique_sequences)))
        for module in range(len(names)):
            for column, seq in enumerate(unique_sequences):
                by_sequence[module, column] = signed[sequence == seq, module].mean()
        random_loo[str(seed)] = loo_stability(by_sequence)
    random_signed_mean = np.stack(random_signed, axis=-1).mean(-1)
    signed_actual_minus_random = causal_signed - random_signed_mean

    both = kl[:, :, index["both_mixed"]]
    kl_interaction = both - causal_m - causal_u
    mechanism = {name: value.numpy() for name, value in payload["logit_mechanism"].items()}
    scale = payload["matched_scales"].numpy()
    report = {
        "status": "complete",
        "question": "Does the actual masked/unmasked split expose stable causal differences in how projection pruning damages masked-token predictions?",
        "sample": {"states": 80, "sequences": 8, "projections": 28,
                   "projection_names": names, "conditions": list(CONDITIONS)},
        "execution_gates": {
            "mixed_direct_max_abs": manifest["mixed_direct_max_abs"],
            "external_dense_vs_same_path_sham_max_abs": manifest["external_sham_max_abs"],
            "primary_reference": "same-path sham because external batch-1 dense logits exhibit BF16 batch-shape drift",
        },
        "role_vs_random": {
            "actual_normalized_abs_kl_contrast": summarize(actual_contrast),
            "random_normalized_abs_kl_contrast": summarize(random_contrasts),
            "actual_minus_random_mean": primary_cluster,
            "per_sequence_primary_difference": sequence_primary.tolist(),
            "sequences_positive": int((sequence_primary > 0).sum()),
            "signed_causal_contrast_actual": sequence_cluster_interval(
                causal_signed.mean(1), sequence, resamples=20000, seed=20260914),
            "signed_causal_contrast_random": sequence_cluster_interval(
                random_signed_mean.mean(1), sequence, resamples=20000, seed=20260915),
            "signed_actual_minus_random": sequence_cluster_interval(
                signed_actual_minus_random.mean(1), sequence,
                resamples=20000, seed=20260916),
        },
        "energy_matched": {
            "normalized_abs_kl_contrast": summarize(matched_contrast),
            "unmatched_contrast": summarize(actual_contrast),
            "matched_to_unmatched_mean_ratio": float(matched_contrast.mean() / actual_contrast.mean()),
            "masked_scale": summarize(scale[:, :, 0]), "unmasked_scale": summarize(scale[:, :, 1]),
        },
        "local_to_causal": {
            "signed_projection_mean_spearman": safe_spearman(local_signed.mean(0), causal_signed.mean(0)),
            "absolute_projection_mean_spearman": safe_spearman(np.abs(local_signed).mean(0), np.abs(causal_signed).mean(0)),
            "pooled_signed_spearman": safe_spearman(local_signed.reshape(-1), causal_signed.reshape(-1)),
            "vulnerable_role_agreement": alignment_cluster,
            "vulnerable_role_agreement_count": int(len(aligned_values)),
            "agreement_by_local_normalization": agreement_reports,
            "numerator_vs_normalized_role_agreement": float(np.mean(numerator_signed == normalized_signed)),
            "normalization_flips": int(np.sum((numerator_signed != normalized_signed) &
                                               (numerator_signed != 0) & (normalized_signed != 0))),
            "loo_projection_causal_contrast": loo_stability(projection_sequence),
            "random_loo_projection_contrast": random_loo,
        },
        "interaction": {
            "kl_both_minus_m_minus_u": summarize(kl_interaction),
            "kl_superadditive_fraction": float(np.mean(kl_interaction > 0)),
            "relative_logit_additivity_residual": summarize(mechanism["relative_additivity_residual"]),
        },
        "condition_mean_kl": {name: float(kl[:, :, i].mean()) for i, name in enumerate(CONDITIONS)},
        "interpretation_limits": [
            "The 28 projections are fixed coverage samples, not a random sample of all projections.",
            "Normalized absolute contrast measures separation magnitude, not which role ought to be protected.",
            "Mechanistic KL effects do not establish downstream task superiority.",
        ],
    }
    atomic_write_json(ROOT / "analysis.json", report)
    print(json.dumps({"status": "complete",
                      "actual_minus_random": report["role_vs_random"]["actual_minus_random_mean"],
                      "local_to_causal": report["local_to_causal"],
                      "energy_matched_ratio": report["energy_matched"]["matched_to_unmatched_mean_ratio"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
