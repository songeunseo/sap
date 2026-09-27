"""Leakage-safe, CPU-only primitives for the preregistered two-path diagnostic."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('experiments/dlm_capacity_predictor')
SOURCE = Path('experiments/projection_capacity_allocation_65')
FOLLOWUP = Path('experiments/projection_capacity_followup_65')
CALIBRATION = Path('experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json')
FEATURES = ('variation_ratio', 'feature_use_variability', 'log_pr_ratio')
GRID = (.50, .55, .60, .65, .70, .75)
TARGET = 4_536_008_704


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if frozen and path.exists():
        if path.read_text() != text:
            raise RuntimeError(f'attempt to change frozen artifact: {path}')
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def state_indices(keys, sequences, timesteps=None):
    normalized = [(int(s), round(float(t), 8)) for s, t in keys]
    if len(normalized) != len(set(normalized)):
        raise ValueError('duplicate state identity')
    times = set(round(float(t), 8) for t in timesteps) if timesteps is not None else None
    result = [i for i, (s, t) in enumerate(normalized)
              if s in sequences and (times is None or t in times)]
    expected_times = times if times is not None else {t for _, t in normalized}
    if len(result) != len(set(sequences)) * len(expected_times):
        raise ValueError('incomplete state grid')
    return result


def fit_log_alpha(feature, alpha):
    x, a = np.asarray(feature, float), np.asarray(alpha, float)
    if x.ndim != 1 or x.shape != a.shape or len(x) < 2:
        raise ValueError('aligned nonempty training vectors required')
    if not np.isfinite(x).all() or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('finite features and positive alpha required')
    mean, std = float(x.mean()), float(x.std())
    z = (x - mean) / std if std else np.zeros_like(x)
    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(z)), z]),
                                     np.log(a), rcond=None)[0]
    return dict(feature_mean=mean, feature_std=std,
                intercept=float(intercept), slope=float(slope), training_count=len(x))


def predict_alpha(fit, feature):
    x = np.asarray(feature, float)
    z = (x - fit['feature_mean']) / fit['feature_std'] if fit['feature_std'] else np.zeros_like(x)
    a = np.exp(fit['intercept'] + fit['slope'] * z)
    if not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError('invalid predicted alpha; no clipping policy permitted')
    return a


def common_shape(damage, modules, states, anchor=3):
    d = np.asarray(damage)[np.asarray(modules)][:, np.asarray(states)].mean(1)
    if np.any(d[:, anchor] <= 0):
        raise ValueError('positive anchor KL required')
    return (d / d[:, anchor, None]).mean(0)


def anchor_prediction(damage, local, states, anchor=3):
    d = np.asarray(damage)[:, states].mean(1)
    e = np.asarray(local)[:, states].mean(1)
    if np.any(e[:, anchor] <= 0) or np.any(d[:, anchor] <= 0):
        raise ValueError('positive anchor KL and reconstruction required')
    alpha = d[:, anchor] / e[:, anchor]
    return alpha[:, None] * e, alpha


def sketch_matrix(width, size=64, seed=20260910):
    # Same input width -> same projection, independent of module name/order.
    return np.random.default_rng(np.random.SeedSequence([seed, width])).normal(
        size=(width, size)).astype(np.float32) / np.sqrt(size)


def participation_ratio(samples):
    x = np.asarray(samples, dtype=np.float64)
    centered = x - x.mean(0)
    covariance = centered.T @ centered / len(x)
    trace = np.trace(covariance)
    return float(trace * trace / np.sum(covariance * covariance)) if trace > 0 else None


def temporal_statistics(inputs, sketch):
    """Reference implementation: [timestep, token, feature], one sequence only."""
    x = np.asarray(inputs, dtype=np.float64)
    mean = x.mean(0)
    energy = float(np.mean(np.sum(x * x, axis=-1)))
    if energy == 0:
        raise ValueError('zero input energy')
    e = np.mean(x * x, axis=1)
    normalized = e / e.sum(1, keepdims=True)
    average = normalized.mean(0)
    projected = x @ sketch
    projected_mean = projected.mean(0)
    pr_mean = participation_ratio(projected_mean)
    pr_var = participation_ratio((projected - projected_mean).reshape(-1, sketch.shape[1]))
    return dict(variation_ratio=float(np.mean(np.sum((x - mean) ** 2, axis=-1)) / energy),
                feature_use_variability=float(np.mean(np.sum((normalized-average)**2, axis=1))
                                              / np.sum(average**2)),
                pr_mean=pr_mean, pr_variation=pr_var,
                log_pr_ratio=float(np.log(pr_var/pr_mean)) if pr_var and pr_mean else None,
                activation_energy=energy)


def passes_gate(comparison):
    return (comparison['mean_difference'] < 0
            and comparison['state_bootstrap_95_ci'][1] < 0
            and comparison['sequence_means_improved'] > 4
            and comparison['timestep_means_improved'] > 2)


def require_disjoint_intervals(groups):
    intervals = sorted((int(r['start']), int(r['end_exclusive'])) for group in groups for r in group)
    if any(b <= a for a, b in intervals):
        raise ValueError('invalid half-open corpus interval')
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError('overlapping corpus spans, including within-split overlaps')


def dependency_complete(document):
    return (document.get('status') == 'complete'
            and {r.get('limit') for r in document.get('evaluations', [])} == {100, 1319})
