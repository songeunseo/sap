"""CPU analysis pipeline for dual-role reconstruction allocation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.dlm_dual_role_allocation.core import (
    allocation_mask_xor,
    bootstrap_mean_ci,
    role_contrast,
)
from experiments.dlm_dual_role_allocation.io import (
    atomic_write_json,
    json_sha256,
    load_frozen_inputs,
    validate_checkpoint,
)
from experiments.dlm_dual_role_allocation.collect import checkpoint_expected
from experiments.dlm_dual_role_allocation.models import (
    apply_diagnostic_gate,
    fit_nested_ols,
    predict_nested_ols,
    rmse,
)
from experiments.projection_capacity_allocation_65.core import (
    GRID,
    allocate,
    correlation,
    distribution,
)

ROOT = Path("experiments/dlm_dual_role_allocation")
WANDA_STATS = Path("experiments/cgq_wanda_structured_diagnostic/sufficient_statistics.pt")


def validate_reconstruction(reconstructed, historical, *, absolute_limit=1e-7,
                            relative_limit=1e-5) -> dict:
    """Require the role partition to reconstruct every historical aggregate error."""
    reconstructed = np.asarray(reconstructed, dtype=float)
    historical = np.asarray(historical, dtype=float)
    if reconstructed.shape != historical.shape or reconstructed.ndim != 3:
        raise ValueError("reconstruction arrays must share module x state x level shape")
    if not np.isfinite(reconstructed).all() or not np.isfinite(historical).all():
        raise ValueError("reconstruction arrays must be finite")
    absolute = np.abs(reconstructed - historical)
    relative = np.zeros_like(absolute)
    nonzero = historical != 0
    relative[nonzero] = absolute[nonzero] / np.abs(historical[nonzero])
    relative[~nonzero & (absolute != 0)] = np.inf
    absolute_index = tuple(int(value) for value in np.unravel_index(np.argmax(absolute), absolute.shape))
    relative_index = tuple(int(value) for value in np.unravel_index(np.argmax(relative), relative.shape))
    result = {
        "passed": bool(absolute.max() <= absolute_limit and relative.max() <= relative_limit),
        "max_absolute_error": float(absolute.max()),
        "max_absolute_error_index": list(absolute_index),
        "max_relative_error": float(relative.max()),
        "max_relative_error_index": list(relative_index),
        "absolute_limit": float(absolute_limit),
        "relative_limit": float(relative_limit),
        "comparison_count": int(absolute.size),
    }
    if result["max_absolute_error"] > absolute_limit:
        raise RuntimeError(f"absolute reconstruction sanity failed: {result}")
    if result["max_relative_error"] > relative_limit:
        raise RuntimeError(f"relative reconstruction sanity failed: {result}")
    return result


def _safe_distribution(values) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0}
    result = distribution(values)
    result["count"] = int(values.size)
    return result


def describe_roles(masked, unmasked, names: list[str], aggregate=None) -> dict:
    """Describe full-calibration role curves without repairing non-monotonicity."""
    masked = np.asarray(masked, dtype=float)
    unmasked = np.asarray(unmasked, dtype=float)
    if (masked.shape != unmasked.shape or masked.ndim != 2 or
            masked.shape[0] != len(names) or not np.isfinite(masked).all() or
            not np.isfinite(unmasked).all() or (masked < 0).any() or (unmasked < 0).any()):
        raise ValueError("finite nonnegative module x level role curves required")
    role = np.maximum(masked, unmasked)
    contrast = role_contrast(masked, unmasked)
    if aggregate is None:
        aggregate = (masked + unmasked) / 2
    aggregate = np.asarray(aggregate, dtype=float)
    if aggregate.shape != masked.shape or not np.isfinite(aggregate).all():
        raise ValueError("aggregate curves must align with role curves")
    dominance = np.sign(masked - unmasked)
    layers = np.asarray([int(name.split(".")[0].removeprefix("block_")) for name in names])
    types = np.asarray([name.split(".")[1] for name in names])
    role_diff = np.diff(role, axis=1)
    ratio = np.full_like(masked, np.nan)
    valid_ratio = unmasked != 0
    ratio[valid_ratio] = masked[valid_ratio] / unmasked[valid_ratio]
    contrast_marginal = np.diff(contrast, axis=1)
    contrast_increments = [
        {"name": names[module], "module_index": module, "from_level": level,
         "to_level": level + 1, "destination_contrast": float(contrast[module, level + 1]),
         "contrast_change": float(contrast_marginal[module, level])}
        for module in range(masked.shape[0]) for level in range(masked.shape[1] - 1)
    ]
    result = {
        "dominance_counts": {
            "masked": int((dominance > 0).sum()),
            "unmasked": int((dominance < 0).sum()),
            "tie": int((dominance == 0).sum()),
        },
        "role_negative_marginal_count": int((role_diff < 0).sum()),
        "masked_negative_marginal_count": int((np.diff(masked, axis=1) < 0).sum()),
        "unmasked_negative_marginal_count": int((np.diff(unmasked, axis=1) < 0).sum()),
        "masked_over_unmasked_zero_denominator_count": int((~valid_ratio).sum()),
        "role_monotonicity_violations": [
            {"name": names[module], "from_level": level, "to_level": level + 1,
             "marginal": float(role_diff[module, level])}
            for module, level in zip(*np.where(role_diff < 0))
        ],
        "by_level": [],
        "by_type": {},
        "by_layer_quartile": {},
        "top20_destination_contrast_increments": sorted(
            contrast_increments, key=lambda row: -row["destination_contrast"]
        )[:20],
        "bottom20_destination_contrast_increments": sorted(
            contrast_increments, key=lambda row: row["destination_contrast"]
        )[:20],
        "marginal_by_increment": [],
    }
    for level in range(masked.shape[1]):
        result["by_level"].append({
            "level_index": level,
            "masked": _safe_distribution(masked[:, level]),
            "unmasked": _safe_distribution(unmasked[:, level]),
            "aggregate": _safe_distribution(aggregate[:, level]),
            "role": _safe_distribution(role[:, level]),
            "contrast": _safe_distribution(contrast[:, level]),
            "masked_over_unmasked": _safe_distribution(ratio[:, level][np.isfinite(ratio[:, level])]),
        })
    for level in range(masked.shape[1] - 1):
        result["marginal_by_increment"].append({
            "from_level": level,
            "to_level": level + 1,
            "masked": _safe_distribution(np.diff(masked, axis=1)[:, level]),
            "unmasked": _safe_distribution(np.diff(unmasked, axis=1)[:, level]),
            "aggregate": _safe_distribution(np.diff(aggregate, axis=1)[:, level]),
            "role": _safe_distribution(role_diff[:, level]),
            "contrast_change": _safe_distribution(contrast_marginal[:, level]),
        })
    for kind in sorted(set(types)):
        selected = types == kind
        result["by_type"][kind] = {
            "masked": _safe_distribution(masked[selected]),
            "unmasked": _safe_distribution(unmasked[selected]),
            "role": _safe_distribution(role[selected]),
            "contrast": _safe_distribution(contrast[selected]),
        }
    for quartile in range(4):
        selected = layers // 8 == quartile
        if selected.any():
            result["by_layer_quartile"][str(quartile)] = {
                "masked": _safe_distribution(masked[selected]),
                "unmasked": _safe_distribution(unmasked[selected]),
                "role": _safe_distribution(role[selected]),
                "contrast": _safe_distribution(contrast[selected]),
            }
    for layer in sorted(set(layers)):
        selected = layers == layer
        result.setdefault("by_layer", {})[str(layer)] = {
            "masked": _safe_distribution(masked[selected]),
            "unmasked": _safe_distribution(unmasked[selected]),
            "role": _safe_distribution(role[selected]),
            "contrast": _safe_distribution(contrast[selected]),
        }
    return result


def _checkpoint_arrays(inputs, destination: Path) -> dict:
    modules = len(inputs.candidate["entries"])
    states = inputs.metadata["state_count"]
    levels = len(GRID)
    fields = {
        name: np.empty((modules, states, levels), dtype=np.float64)
        for name in ("num_masked", "den_masked", "num_unmasked", "den_unmasked",
                     "reconstructed")
    }
    payloads = []
    checkpoint_root = destination / "runtime" / "role_stats"
    for module_index, row in enumerate(inputs.candidate["entries"]):
        path = checkpoint_root / f"{row['name']}.json"
        if not path.exists():
            raise RuntimeError(f"missing role checkpoint: {row['name']}")
        payload = json.loads(path.read_text())
        validate_checkpoint(payload, checkpoint_expected(inputs, row))
        stored_digest = payload.get("payload_sha256")
        body = {key: value for key, value in payload.items() if key != "payload_sha256"}
        if stored_digest != json_sha256(body):
            raise RuntimeError(f"role checkpoint payload hash mismatch: {row['name']}")
        for state_index, state in enumerate(payload["states"]):
            if int(state["state_index"]) != state_index:
                raise RuntimeError("checkpoint state ordering mismatch")
            for level, record in enumerate(state["levels"]):
                if float(record["sparsity"]) != GRID[level]:
                    raise RuntimeError("checkpoint level ordering mismatch")
                for field in ("num_masked", "den_masked", "num_unmasked", "den_unmasked"):
                    fields[field][module_index, state_index, level] = float(record[field])
                fields["reconstructed"][module_index, state_index, level] = float(
                    record["reconstructed_aggregate_error"]
                )
        payloads.append(payload)
    return {"fields": fields, "payloads": payloads}


def _historical_arrays(inputs) -> dict:
    modules = len(inputs.curves["projections"])
    states = inputs.metadata["state_count"]
    levels = len(GRID)
    functional = np.empty((modules, states, levels), dtype=np.float64)
    reconstruction = np.empty_like(functional)
    for module, projection in enumerate(inputs.curves["projections"]):
        for level, curve in enumerate(projection["curves"]):
            per_state = curve["per_state"]
            if len(per_state) != states:
                raise RuntimeError("historical curve state count mismatch")
            for state, record in enumerate(per_state):
                if int(record["state_index"]) != state:
                    raise RuntimeError("historical curve state ordering mismatch")
                functional[module, state, level] = float(record["mean_kl"])
                reconstruction[module, state, level] = float(record["local_reconstruction_error"])
    return {"functional": functional, "reconstruction": reconstruction}


def _role_arrays(fields: dict) -> dict:
    masked = fields["num_masked"] / fields["den_masked"]
    unmasked = fields["num_unmasked"] / fields["den_unmasked"]
    aggregate = ((fields["num_masked"] + fields["num_unmasked"])
                 / (fields["den_masked"] + fields["den_unmasked"]))
    return {
        "masked": masked,
        "unmasked": unmasked,
        "aggregate": aggregate,
        "role": np.maximum(masked, unmasked),
        "contrast": role_contrast(masked, unmasked),
    }


def _pool_fields(fields: dict, state_ids: list[int]) -> dict:
    selected = np.asarray(state_ids, dtype=int)
    num_masked = fields["num_masked"][:, selected].sum(axis=1)
    den_masked = fields["den_masked"][:, selected].sum(axis=1)
    num_unmasked = fields["num_unmasked"][:, selected].sum(axis=1)
    den_unmasked = fields["den_unmasked"][:, selected].sum(axis=1)
    masked = num_masked / den_masked
    unmasked = num_unmasked / den_unmasked
    return {
        "masked": masked,
        "unmasked": unmasked,
        "aggregate": (num_masked + num_unmasked) / (den_masked + den_unmasked),
        "role": np.maximum(masked, unmasked),
        "contrast": role_contrast(masked, unmasked),
    }


def selected_sequence_damage(functional, levels, sequence_indices) -> dict[int, float]:
    """Evaluate additive single-projection KL damage for each sequence."""
    functional = np.asarray(functional, dtype=float)
    levels = np.asarray(levels, dtype=int)
    sequence_indices = np.asarray(sequence_indices, dtype=int)
    if functional.ndim != 3 or levels.shape != (functional.shape[0],):
        raise ValueError("functional curves and allocation do not align")
    selected = functional[np.arange(functional.shape[0])[:, None],
                          np.arange(functional.shape[1])[None, :], levels[:, None]]
    state_damage = selected.sum(axis=0)
    return {
        int(sequence): float(state_damage[sequence_indices == sequence].mean())
        for sequence in sorted(set(sequence_indices))
    }


def _sequence_rmse(predicted, truth, sequence_indices) -> dict[int, float]:
    return {
        int(sequence): rmse(predicted[:, sequence_indices == sequence],
                            truth[:, sequence_indices == sequence])
        for sequence in sorted(set(sequence_indices))
    }


def evaluate_crossfit(functional, fields, shapes, candidate_entries, sequence_indices,
                      layers, types) -> dict:
    """Run frozen 4/4 sequence cross-fit allocations and nested OLS validation."""
    functional = np.asarray(functional, dtype=float)
    sequence_indices = np.asarray(sequence_indices, dtype=int)
    gains = np.asarray([.05 * int(row) * int(column) for row, column in shapes], dtype=float)
    per_state = _role_arrays(fields)
    folds = [
        {"construction_sequences": [0, 1, 2, 3], "validation_sequences": [4, 5, 6, 7]},
        {"construction_sequences": [4, 5, 6, 7], "validation_sequences": [0, 1, 2, 3]},
    ]
    fold_results = []
    damage_differences = []
    rmse_differences = []
    xor_fractions = []
    for fold_index, fold in enumerate(folds):
        construction_ids = np.flatnonzero(np.isin(sequence_indices, fold["construction_sequences"])).tolist()
        validation_mask = np.isin(sequence_indices, fold["validation_sequences"])
        pooled = _pool_fields(fields, construction_ids)
        aggregate_allocation = allocate(pooled["aggregate"], shapes)
        role_allocation = allocate(pooled["role"], shapes)
        if aggregate_allocation["budget_error"] != 0 or role_allocation["budget_error"] != 0:
            raise RuntimeError("cross-fit allocation missed exact uniform row-floor budget")
        xor = allocation_mask_xor(aggregate_allocation["levels"], role_allocation["levels"],
                                  candidate_entries)
        xor_fractions.append(xor["xor_fraction_of_prunable_weights"])

        aggregate_damage = selected_sequence_damage(
            functional[:, validation_mask], aggregate_allocation["levels"],
            sequence_indices[validation_mask]
        )
        role_damage = selected_sequence_damage(
            functional[:, validation_mask], role_allocation["levels"],
            sequence_indices[validation_mask]
        )
        sequence_damage = []
        for sequence in fold["validation_sequences"]:
            difference = role_damage[sequence] - aggregate_damage[sequence]
            damage_differences.append(difference)
            sequence_damage.append({
                "sequence_index": sequence,
                "aggregate_additive_damage": aggregate_damage[sequence],
                "role_additive_damage": role_damage[sequence],
                "role_minus_aggregate": difference,
            })

        train_mask = ~validation_mask
        functional_marginal = np.diff(functional, axis=2) / gains[:, None, None]
        aggregate_marginal = np.diff(per_state["aggregate"], axis=2) / gains[:, None, None]
        role_marginal = np.diff(per_state["role"], axis=2) / gains[:, None, None]
        fit = fit_nested_ols(functional_marginal[:, train_mask],
                             aggregate_marginal[:, train_mask],
                             role_marginal[:, train_mask], layers, types)
        predictions = predict_nested_ols(fit, aggregate_marginal[:, validation_mask],
                                         role_marginal[:, validation_mask], layers, types)
        truth = functional_marginal[:, validation_mask]
        baseline_by_sequence = _sequence_rmse(predictions["baseline"], truth,
                                              sequence_indices[validation_mask])
        extended_by_sequence = _sequence_rmse(predictions["extended"], truth,
                                              sequence_indices[validation_mask])
        sequence_ols = []
        for sequence in fold["validation_sequences"]:
            difference = extended_by_sequence[sequence] - baseline_by_sequence[sequence]
            rmse_differences.append(difference)
            sequence_ols.append({
                "sequence_index": sequence,
                "baseline_rmse": baseline_by_sequence[sequence],
                "extended_rmse": extended_by_sequence[sequence],
                "extended_minus_baseline": difference,
            })
        fold_results.append({
            **fold,
            "fold_index": fold_index,
            "aggregate_allocation": {
                "levels": aggregate_allocation["levels"],
                "sparsities": aggregate_allocation["sparsities"],
                "pruned": aggregate_allocation["pruned"],
            },
            "role_allocation": {
                "levels": role_allocation["levels"],
                "sparsities": role_allocation["sparsities"],
                "pruned": role_allocation["pruned"],
            },
            "allocation_xor": xor,
            "sequence_damage": sequence_damage,
            "mean_role_minus_aggregate_damage": float(np.mean([
                row["role_minus_aggregate"] for row in sequence_damage
            ])),
            "nested_ols": {
                "fit": fit,
                "baseline_validation_rmse": rmse(predictions["baseline"], truth),
                "extended_validation_rmse": rmse(predictions["extended"], truth),
                "per_sequence": sequence_ols,
            },
        })
    damage_ci = bootstrap_mean_ci(damage_differences, 20000, 0)
    rmse_ci = bootstrap_mean_ci(rmse_differences, 20000, 0)
    return {
        "folds": fold_results,
        "sequence_damage_difference": damage_ci,
        "sequence_rmse_difference": rmse_ci,
        "role_better_both_folds": all(
            row["mean_role_minus_aggregate_damage"] < 0 for row in fold_results
        ),
        "extended_better_both_folds": all(
            row["nested_ols"]["extended_validation_rmse"]
            < row["nested_ols"]["baseline_validation_rmse"] for row in fold_results
        ),
        "minimum_mask_xor_fraction": float(min(xor_fractions)),
    }


def _pairwise_rank_stability(curves: dict[object, np.ndarray]) -> dict:
    keys = sorted(curves)
    pairs = []
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            x, y = curves[left].reshape(-1), curves[right].reshape(-1)
            rho = None if np.ptp(x) == 0 or np.ptp(y) == 0 else float(spearmanr(x, y).statistic)
            pairs.append({"left": left, "right": right, "spearman": rho})
    finite = [row["spearman"] for row in pairs if row["spearman"] is not None]
    return {"pairs": pairs, "distribution": _safe_distribution(finite)}


def _group_pooled_distributions(fields, labels) -> dict:
    labels = np.asarray(labels)
    result = {}
    for label in sorted(set(labels)):
        pooled = _pool_fields(fields, np.flatnonzero(labels == label).tolist())
        result[str(label)] = {
            name: [_safe_distribution(values[:, level]) for level in range(values.shape[1])]
            for name, values in pooled.items()
        }
    return result


def _redundancy(inputs, role_arrays, historical_reconstruction, names) -> dict:
    role_marginal = np.diff(role_arrays["role"], axis=2)
    aggregate_marginal = np.diff(role_arrays["aggregate"], axis=2)
    historical_marginal = np.diff(historical_reconstruction, axis=2)
    result = {
        "per_state_level": {
            "role_vs_aggregate": correlation(role_arrays["role"].reshape(-1),
                                             role_arrays["aggregate"].reshape(-1)),
            "contrast_vs_aggregate": correlation(role_arrays["contrast"].reshape(-1),
                                                 role_arrays["aggregate"].reshape(-1)),
        },
        "per_state_marginal": {
            "role_vs_aggregate": correlation(role_marginal.reshape(-1),
                                             aggregate_marginal.reshape(-1)),
            "role_vs_historical_reconstruction": correlation(role_marginal.reshape(-1),
                                                              historical_marginal.reshape(-1)),
        },
    }
    stats = torch.load(WANDA_STATS, map_location="cpu", weights_only=False)
    activation = stats["statistics"]
    if set(activation) != set(names):
        raise RuntimeError("Wanda activation module names differ from frozen modules")
    activation_mean = np.asarray([
        float(activation[name]["overall_uniform"].float().mean()) for name in names
    ])
    module_contrast = role_arrays["contrast"].mean(axis=(1, 2))
    module_excess = (role_arrays["role"] - role_arrays["aggregate"]).mean(axis=(1, 2))
    result["module_level_wanda_activation"] = {
        "contrast_vs_mean_activation": correlation(module_contrast, activation_mean),
        "role_excess_vs_mean_activation": correlation(module_excess, activation_mean),
    }
    layers = np.asarray([int(name.split(".")[0].removeprefix("block_")) for name in names])
    result["module_level_structure"] = {
        "mean_contrast_vs_layer": correlation(module_contrast, layers),
        "mean_role_excess_vs_layer": correlation(module_excess, layers),
    }
    return result


def run_analysis(destination: str | Path = ROOT) -> dict:
    destination = Path(destination)
    inputs = load_frozen_inputs()
    collected = _checkpoint_arrays(inputs, destination)
    historical = _historical_arrays(inputs)
    sanity = validate_reconstruction(collected["fields"]["reconstructed"],
                                     historical["reconstruction"])
    role_arrays = _role_arrays(collected["fields"])
    names = inputs.metadata["module_names"]
    all_states = list(range(inputs.metadata["state_count"]))
    pooled = _pool_fields(collected["fields"], all_states)
    descriptive = describe_roles(pooled["masked"], pooled["unmasked"], names,
                                 pooled["aggregate"])

    sequence_indices = np.asarray([int(row["sequence_index"]) for row in inputs.states["states"]])
    timesteps = np.asarray([float(row["timestep"]) for row in inputs.states["states"]])
    sequence_curves = {
        int(sequence): _pool_fields(collected["fields"],
                                    np.flatnonzero(sequence_indices == sequence).tolist())["contrast"]
        for sequence in sorted(set(sequence_indices))
    }
    timestep_curves = {
        float(timestep): _pool_fields(collected["fields"],
                                     np.flatnonzero(timesteps == timestep).tolist())["contrast"]
        for timestep in sorted(set(timesteps))
    }
    descriptive["contrast_sequence_rank_stability"] = _pairwise_rank_stability(sequence_curves)
    descriptive["contrast_timestep_rank_stability"] = _pairwise_rank_stability(timestep_curves)
    descriptive["by_sequence"] = _group_pooled_distributions(collected["fields"], sequence_indices)
    descriptive["by_timestep"] = _group_pooled_distributions(collected["fields"], timesteps)
    construction_a = _pool_fields(collected["fields"],
                                  np.flatnonzero(np.isin(sequence_indices, [0, 1, 2, 3])).tolist())
    construction_b = _pool_fields(collected["fields"],
                                  np.flatnonzero(np.isin(sequence_indices, [4, 5, 6, 7])).tolist())
    descriptive["bidirectional_fold_reproducibility"] = {
        key: correlation(construction_a[key].reshape(-1), construction_b[key].reshape(-1))
        for key in ("masked", "unmasked", "aggregate", "role", "contrast")
    }

    shapes = [row["shape"] for row in inputs.candidate["entries"]]
    layers = np.asarray([int(name.split(".")[0].removeprefix("block_")) for name in names])
    types = np.asarray([name.split(".")[1] for name in names])
    crossfit = evaluate_crossfit(historical["functional"], collected["fields"], shapes,
                                 inputs.candidate["entries"], sequence_indices, layers, types)
    redundancy = _redundancy(inputs, role_arrays, historical["reconstruction"], names)
    collection_manifest_path = destination / "collection_manifest.json"
    if not collection_manifest_path.exists():
        raise RuntimeError("missing collection manifest with before/after dense hashes")
    collection_manifest = json.loads(collection_manifest_path.read_text())
    dense_hash_passed = (
        collection_manifest.get("dense_sha_before") == inputs.metadata["dense_model_sha256"]
        == collection_manifest.get("dense_sha_after")
    )
    evidence = {
        "reconstruction_sanity_passed": sanity["passed"],
        "dense_hash_passed": dense_hash_passed,
        "role_better_both_folds": crossfit["role_better_both_folds"],
        "damage_difference_ci": crossfit["sequence_damage_difference"]["bootstrap_95_ci"],
        "mask_xor_fraction": crossfit["minimum_mask_xor_fraction"],
        "extended_better_both_folds": crossfit["extended_better_both_folds"],
        "rmse_difference_ci": crossfit["sequence_rmse_difference"]["bootstrap_95_ci"],
    }
    decision = {**apply_diagnostic_gate(evidence), "evidence": evidence,
                "reconstruction_sanity": sanity}

    raw_document = {
        "source_receipt": inputs.receipt,
        "state_digest": inputs.metadata["state_digest"],
        "grid": list(GRID),
        "projections": collected["payloads"],
    }
    atomic_write_json(destination / "role_reconstruction_raw.json", raw_document)
    atomic_write_json(destination / "role_distributions.json", descriptive)
    atomic_write_json(destination / "redundancy_analysis.json", redundancy)
    atomic_write_json(destination / "crossfit_allocations.json", crossfit)
    atomic_write_json(destination / "decision.json", decision)
    result = {"decision": decision["decision"], "passed": decision["passed"],
              "destination": str(destination)}
    print(json.dumps({"event": "analysis_complete", **result}, sort_keys=True), flush=True)
    return result
