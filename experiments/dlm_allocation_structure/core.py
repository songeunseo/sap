"""Pure numerical helpers for projection-allocation structure analysis."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from experiments.projection_capacity_allocation_65.core import GRID, allocate


def _values(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("values must be a finite module x state x increment array")
    return array


def _metadata(module_count, layers, types):
    layers = np.asarray(layers, dtype=int)
    types = np.asarray(types, dtype=str)
    if layers.shape != (module_count,) or types.shape != (module_count,):
        raise ValueError("module metadata must align with values")
    return layers, types


def compute_marginals(damage, reconstruction, shapes, grid=GRID):
    """Return raw and nominal-parameter-normalized consecutive curve differences."""
    damage = np.asarray(damage, dtype=float)
    reconstruction = np.asarray(reconstruction, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if damage.ndim != 3 or damage.shape != reconstruction.shape:
        raise ValueError("damage and reconstruction must share module x state x level shape")
    if damage.shape[0] != len(shapes) or damage.shape[2] != len(grid):
        raise ValueError("curve shape does not match module shapes or grid")
    if len(grid) < 2 or not np.allclose(np.diff(grid), .05):
        raise ValueError("the frozen grid must use consecutive 5 percentage-point increments")
    if not np.isfinite(damage).all() or not np.isfinite(reconstruction).all():
        raise ValueError("curves must be finite")
    functional = np.diff(damage, axis=2)
    local = np.diff(reconstruction, axis=2)
    gains = np.asarray([.05 * int(rows) * int(cols) for rows, cols in shapes], dtype=float)
    return {
        "functional": functional,
        "reconstruction": local,
        "functional_per_parameter": functional / gains[:, None, None],
        "reconstruction_per_parameter": local / gains[:, None, None],
        "nominal_parameter_gain": gains,
    }


def fit_static_models(values, layers, types):
    """Fit frozen hierarchy levels using construction states only."""
    values = _values(values)
    layers, types = _metadata(values.shape[0], layers, types)
    mean = values.mean(axis=1)
    grand = mean.mean(axis=0)
    depth_means = {int(layer): mean[layers == layer].mean(axis=0)
                   for layer in np.unique(layers)}
    type_means = {str(kind): mean[types == kind].mean(axis=0)
                  for kind in np.unique(types)}
    depth = np.stack([depth_means[int(layer)] for layer in layers])
    kind = np.stack([type_means[str(name)] for name in types])
    return {
        "global": np.broadcast_to(grand, mean.shape).copy(),
        "depth": depth,
        "type": kind,
        "depth_type": depth + kind - grand,
        "projection": mean,
    }


def fit_temporal_models(values, timesteps, layers, types, normalize=True):
    """Fit train-RMS-normalized temporal hierarchy models on construction states."""
    values = _values(values)
    layers, types = _metadata(values.shape[0], layers, types)
    timesteps = np.asarray(timesteps, dtype=float)
    if timesteps.shape != (values.shape[1],) or not np.isfinite(timesteps).all():
        raise ValueError("timesteps must align with construction states")
    unique_times = np.unique(timesteps)
    rms = (np.stack([
        np.sqrt(np.mean(values[:, timesteps == timestep, :] ** 2, axis=(0, 1)))
        for timestep in unique_times
    ]) if normalize else np.ones((len(unique_times), values.shape[2]), dtype=float))
    if normalize and np.any(rms == 0):
        raise ValueError("zero construction RMS cannot be normalized")
    time_lookup = {float(t): i for i, t in enumerate(unique_times)}
    normalized = np.empty_like(values)
    for state, timestep in enumerate(timesteps):
        normalized[:, state, :] = values[:, state, :] / rms[time_lookup[float(timestep)]]

    grand = normalized.mean(axis=(0, 1))
    projection = normalized.mean(axis=1)
    time = np.stack([normalized[:, timesteps == t, :].mean(axis=(0, 1))
                     for t in unique_times])
    depth = {int(layer): normalized[layers == layer].mean(axis=(0, 1))
             for layer in np.unique(layers)}
    kind = {str(name): normalized[types == name].mean(axis=(0, 1))
            for name in np.unique(types)}
    depth_time = {(int(layer), float(t)): normalized[
        layers == layer][:, timesteps == t, :].mean(axis=(0, 1))
        for layer in np.unique(layers) for t in unique_times}
    type_time = {(str(name), float(t)): normalized[
        types == name][:, timesteps == t, :].mean(axis=(0, 1))
        for name in np.unique(types) for t in unique_times}
    projection_time = np.stack([
        np.stack([normalized[module, timesteps == t, :].mean(axis=0)
                  for t in unique_times])
        for module in range(values.shape[0])
    ])
    return {
        "unique_timesteps": unique_times,
        "rms_by_timestep": rms,
        "normalized_train": normalized,
        "layers": layers,
        "types": types,
        "grand": grand,
        "projection": projection,
        "time": time,
        "depth": depth,
        "type": kind,
        "depth_time": depth_time,
        "type_time": type_time,
        "projection_time": projection_time,
    }


def apply_temporal_models(fit, timesteps, values=None):
    """Apply frozen temporal models to states at already-observed timestep levels."""
    timesteps = np.asarray(timesteps, dtype=float)
    time_lookup = {float(t): i for i, t in enumerate(fit["unique_timesteps"])}
    try:
        time_indices = [time_lookup[float(t)] for t in timesteps]
    except KeyError as error:
        raise ValueError(f"unseen timestep: {error.args[0]}") from error
    module_count = len(fit["layers"])
    increment_count = len(fit["grand"])
    static = np.empty((module_count, len(timesteps), increment_count))
    structured = np.empty_like(static)
    full = np.empty_like(static)
    for module, (layer, kind) in enumerate(zip(fit["layers"], fit["types"])):
        for state, (timestep, time_index) in enumerate(zip(timesteps, time_indices)):
            common = (fit["projection"][module] + fit["time"][time_index] - fit["grand"])
            depth_interaction = (fit["depth_time"][(int(layer), float(timestep))]
                                 - fit["depth"][int(layer)] - fit["time"][time_index]
                                 + fit["grand"])
            type_interaction = (fit["type_time"][(str(kind), float(timestep))]
                                - fit["type"][str(kind)] - fit["time"][time_index]
                                + fit["grand"])
            static[module, state] = common
            structured[module, state] = common + depth_interaction + type_interaction
            full[module, state] = fit["projection_time"][module, time_index]
    result = {
        "static_projection_time": static,
        "structured_interactions": structured,
        "projection_time": full,
    }
    if values is not None:
        values = _values(values)
        if values.shape[:2] != (module_count, len(timesteps)):
            raise ValueError("evaluation values do not align with fitted temporal model")
        normalized = np.empty_like(values)
        for state, index in enumerate(time_indices):
            normalized[:, state, :] = values[:, state, :] / fit["rms_by_timestep"][index]
        result["normalized_values"] = normalized
    return result


def fit_reconstruction_residual(functional_train, reconstruction_train,
                                functional_test, reconstruction_test):
    """Remove a per-increment construction-fitted affine reconstruction signal."""
    functional_train = _values(functional_train)
    reconstruction_train = _values(reconstruction_train)
    functional_test = _values(functional_test)
    reconstruction_test = _values(reconstruction_test)
    if functional_train.shape != reconstruction_train.shape:
        raise ValueError("construction functional/reconstruction shapes differ")
    if functional_test.shape != reconstruction_test.shape:
        raise ValueError("evaluation functional/reconstruction shapes differ")
    if functional_train.shape[0::2] != functional_test.shape[0::2]:
        raise ValueError("construction and evaluation module/increment axes differ")
    increments = functional_train.shape[2]
    slopes, intercepts = [], []
    train_residual = np.empty_like(functional_train)
    test_residual = np.empty_like(functional_test)
    for increment in range(increments):
        x = reconstruction_train[:, :, increment].reshape(-1)
        y = functional_train[:, :, increment].reshape(-1)
        design = np.column_stack([np.ones_like(x), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        intercepts.append(float(intercept))
        slopes.append(float(slope))
        train_residual[:, :, increment] = (
            functional_train[:, :, increment]
            - intercept - slope * reconstruction_train[:, :, increment]
        )
        test_residual[:, :, increment] = (
            functional_test[:, :, increment]
            - intercept - slope * reconstruction_test[:, :, increment]
        )
    return {
        "intercepts": np.asarray(intercepts),
        "slopes": np.asarray(slopes),
        "train_residual": train_residual,
        "test_residual": test_residual,
    }


def allocation_from_marginals(marginals, shapes):
    """Convert five raw marginal KL costs to six-point curves and allocate exactly."""
    marginals = np.asarray(marginals, dtype=float)
    if marginals.shape != (len(shapes), len(GRID) - 1) or not np.isfinite(marginals).all():
        raise ValueError("complete finite module x five-increment costs required")
    curves = np.concatenate([np.zeros((len(shapes), 1)),
                             np.cumsum(marginals, axis=1)], axis=1)
    return allocate(curves, shapes)


def allocation_from_per_parameter_costs(costs, shapes):
    """Allocate from the oracle rule's actual cost-per-nominal-parameter variable."""
    costs = np.asarray(costs, dtype=float)
    if costs.shape != (len(shapes), len(GRID) - 1) or not np.isfinite(costs).all():
        raise ValueError("complete finite module x five-increment costs required")
    gains = np.asarray([.05 * int(rows) * int(cols) for rows, cols in shapes], dtype=float)
    return allocation_from_marginals(costs * gains[:, None], shapes)


def bootstrap_mean_ci(values, resamples=20000, seed=0):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty one-dimensional values required")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "bootstrap_95_ci": np.quantile(means, [.025, .975]).tolist(),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def prediction_metrics(predicted, truth):
    predicted = np.asarray(predicted, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if predicted.shape != truth.shape or not predicted.size:
        raise ValueError("prediction and truth must have equal nonempty shape")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("prediction metrics require finite values")
    delta = predicted.ravel() - truth.ravel()
    x, y = predicted.ravel(), truth.ravel()
    pearson = None if np.ptp(x) == 0 or np.ptp(y) == 0 else float(np.corrcoef(x, y)[0, 1])
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        spearman = None
    else:
        spearman = float(spearmanr(x, y).statistic)
    return {
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "mae": float(np.mean(np.abs(delta))),
        "pearson": pearson,
        "spearman": spearman,
    }


def selected_additive_damage(marginals, levels):
    marginals = np.asarray(marginals, dtype=float)
    if marginals.ndim != 2 or marginals.shape[0] != len(levels):
        raise ValueError("marginals and selected levels must align")
    total = 0.0
    for module, level in enumerate(levels):
        level = int(level)
        if not 0 <= level <= marginals.shape[1]:
            raise ValueError("selected level is outside the marginal grid")
        total += float(marginals[module, :level].sum())
    return total


def nested_mask_xor(levels_a, levels_b, entries):
    """Compute exact XOR count for verified row-prefix nested mask families."""
    if not (len(levels_a) == len(levels_b) == len(entries)):
        raise ValueError("levels and candidate entries must align")
    xor = 0
    weights = 0
    for a, b, entry in zip(levels_a, levels_b, entries):
        masks = entry.get("masks", [])
        if not (0 <= int(a) < len(masks) and 0 <= int(b) < len(masks)):
            raise ValueError("selected level is outside candidate masks")
        xor += abs(int(masks[int(a)]["pruned"]) - int(masks[int(b)]["pruned"]))
        weights += int(entry["weights"])
    return {
        "xor_weights": int(xor),
        "total_weights": int(weights),
        "xor_fraction": float(xor / weights),
    }
