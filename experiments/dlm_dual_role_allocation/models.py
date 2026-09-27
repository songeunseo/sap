"""Frozen nested OLS and decision gate for dual-role allocation."""
from __future__ import annotations

import numpy as np


def _arrays(functional, aggregate, role, layers, types):
    functional = np.asarray(functional, dtype=float)
    aggregate = np.asarray(aggregate, dtype=float)
    role = np.asarray(role, dtype=float)
    layers = np.asarray(layers, dtype=int)
    types = np.asarray(types, dtype=str)
    if (functional.ndim != 3 or aggregate.shape != functional.shape or
            role.shape != functional.shape or layers.shape != (functional.shape[0],) or
            types.shape != (functional.shape[0],)):
        raise ValueError("nested OLS inputs do not align")
    if not all(np.isfinite(value).all() for value in (functional, aggregate, role)):
        raise ValueError("nested OLS inputs must be finite")
    if 0 not in set(layers) or "attn_out" not in set(types):
        raise ValueError("fixed layer/type reference categories are absent")
    return functional, aggregate, role, layers, types


def _metadata_design(shape, layers, types):
    modules, states, increments = shape
    layer_columns = [int(value) for value in sorted(set(layers)) if int(value) != 0]
    type_columns = [str(value) for value in sorted(set(types)) if str(value) != "attn_out"]
    repeated_layers = np.broadcast_to(layers[:, None, None], shape).reshape(-1)
    repeated_types = np.broadcast_to(types[:, None, None], shape).reshape(-1)
    columns = [np.ones(modules * states * increments)]
    names = ["intercept"]
    for layer in layer_columns:
        columns.append((repeated_layers == layer).astype(float))
        names.append(f"layer_{layer}")
    for kind in type_columns:
        columns.append((repeated_types == kind).astype(float))
        names.append(f"type_{kind}")
    return columns, names


def fit_nested_ols(functional, aggregate, role, layers, types) -> dict:
    """Fit baseline and one-feature extension with construction-only scaling."""
    functional, aggregate, role, layers, types = _arrays(
        functional, aggregate, role, layers, types
    )
    columns, names = _metadata_design(functional.shape, layers, types)
    aggregate_flat = aggregate.reshape(-1)
    delta_flat = (role - aggregate).reshape(-1)
    aggregate_mean, aggregate_scale = float(aggregate_flat.mean()), float(aggregate_flat.std())
    delta_mean, delta_scale = float(delta_flat.mean()), float(delta_flat.std())
    if aggregate_scale == 0 or delta_scale == 0:
        raise ValueError("zero-variance reconstruction feature")
    aggregate_z = (aggregate_flat - aggregate_mean) / aggregate_scale
    delta_z = (delta_flat - delta_mean) / delta_scale
    baseline_columns = names + ["aggregate_marginal_per_parameter"]
    extended_columns = baseline_columns + ["role_minus_aggregate_marginal_per_parameter"]
    baseline_design = np.column_stack(columns + [aggregate_z])
    extended_design = np.column_stack(columns + [aggregate_z, delta_z])
    target = functional.reshape(-1)
    baseline_coefficients = np.linalg.lstsq(baseline_design, target, rcond=None)[0]
    extended_coefficients = np.linalg.lstsq(extended_design, target, rcond=None)[0]
    return {
        "baseline_columns": baseline_columns,
        "extended_columns": extended_columns,
        "baseline_coefficients": baseline_coefficients.tolist(),
        "extended_coefficients": extended_coefficients.tolist(),
        "scaling": {
            "aggregate_mean": aggregate_mean,
            "aggregate_scale": aggregate_scale,
            "delta_mean": delta_mean,
            "delta_scale": delta_scale,
        },
        "layers": sorted(int(value) for value in set(layers)),
        "types": sorted(str(value) for value in set(types)),
    }


def predict_nested_ols(fit: dict, aggregate, role, layers, types) -> dict:
    aggregate = np.asarray(aggregate, dtype=float)
    role = np.asarray(role, dtype=float)
    layers = np.asarray(layers, dtype=int)
    types = np.asarray(types, dtype=str)
    if aggregate.ndim != 3 or role.shape != aggregate.shape:
        raise ValueError("prediction role arrays do not align")
    if sorted(int(value) for value in set(layers)) != fit["layers"] or sorted(
            str(value) for value in set(types)) != fit["types"]:
        raise ValueError("prediction metadata levels differ from construction")
    columns, names = _metadata_design(aggregate.shape, layers, types)
    if names != fit["baseline_columns"][:-1]:
        raise ValueError("prediction design columns differ from construction")
    scaling = fit["scaling"]
    aggregate_z = ((aggregate.reshape(-1) - scaling["aggregate_mean"])
                   / scaling["aggregate_scale"])
    delta_z = (((role - aggregate).reshape(-1) - scaling["delta_mean"])
               / scaling["delta_scale"])
    baseline_design = np.column_stack(columns + [aggregate_z])
    extended_design = np.column_stack(columns + [aggregate_z, delta_z])
    return {
        "baseline": (baseline_design @ np.asarray(fit["baseline_coefficients"])).reshape(aggregate.shape),
        "extended": (extended_design @ np.asarray(fit["extended_coefficients"])).reshape(aggregate.shape),
    }


def rmse(predicted, target) -> float:
    predicted, target = np.asarray(predicted, float), np.asarray(target, float)
    if predicted.shape != target.shape or not predicted.size:
        raise ValueError("RMSE arrays do not align")
    return float(np.sqrt(np.mean((predicted - target) ** 2)))


def apply_diagnostic_gate(evidence: dict) -> dict:
    """Apply exactly the five pre-registered conjunctive criteria."""
    criteria = {
        "numerical_and_dense_hash_sanity": bool(
            evidence["reconstruction_sanity_passed"] and evidence["dense_hash_passed"]
        ),
        "role_better_both_sequence_folds": bool(evidence["role_better_both_folds"]),
        "sequence_damage_ci_entirely_below_zero": bool(
            evidence["damage_difference_ci"][1] < 0
        ),
        "mask_xor_at_least_one_percent": bool(evidence["mask_xor_fraction"] >= .01),
        "extended_ols_better_both_folds_and_ci_below_zero": bool(
            evidence["extended_better_both_folds"] and evidence["rmse_difference_ci"][1] < 0
        ),
    }
    passed = all(criteria.values())
    return {
        "decision": "DIAGNOSTIC SUPPORTED" if passed else "NOT SUPPORTED",
        "passed": passed,
        "criteria": criteria,
        "failed_criteria": [name for name, value in criteria.items() if not value],
    }
