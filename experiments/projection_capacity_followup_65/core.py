"""Pure analysis helpers for projection-capacity follow-up experiments."""
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def summarize_curve_records(records, grid):
    """Validate persisted curve records and return aligned dense arrays."""
    grid = tuple(float(x) for x in grid)
    if not records:
        raise ValueError("at least one projection record is required")
    state_keys = None
    names, shapes, damage_states, reconstruction_states = [], [], [], []
    for record in records:
        curves = record.get("curves", [])
        if tuple(float(row["sparsity"]) for row in curves) != grid:
            raise ValueError(f"grid mismatch for {record.get('name')}")
        module_damage, module_reconstruction = [], []
        module_keys = None
        for curve in curves:
            rows = curve.get("per_state", [])
            keys = [(int(row["sequence_index"]), float(row["timestep"])) for row in rows]
            if module_keys is None:
                module_keys = keys
            elif keys != module_keys:
                raise ValueError(f"state order mismatch within {record['name']}")
            module_damage.append([float(row["mean_kl"]) for row in rows])
            module_reconstruction.append(
                [float(row["local_reconstruction_error"]) for row in rows]
            )
        if state_keys is None:
            state_keys = module_keys
        elif module_keys != state_keys:
            raise ValueError(f"state order mismatch across modules at {record['name']}")
        names.append(record["name"])
        shapes.append(tuple(int(x) for x in record["shape"]))
        # Persisted layout is level x state; analysis layout is state x level.
        damage_states.append(np.asarray(module_damage, dtype=float).T)
        reconstruction_states.append(np.asarray(module_reconstruction, dtype=float).T)
    damage_states = np.asarray(damage_states, dtype=float)
    reconstruction_states = np.asarray(reconstruction_states, dtype=float)
    if not np.isfinite(damage_states).all() or not np.isfinite(reconstruction_states).all():
        raise ValueError("curve data contain non-finite values")
    return {
        "names": names,
        "shapes": shapes,
        "state_keys": state_keys,
        "damage_states": damage_states,
        "reconstruction_states": reconstruction_states,
        "damage": damage_states.mean(axis=1),
        "reconstruction": reconstruction_states.mean(axis=1),
    }


def fit_anchor_curve(damage_states, reconstruction_states,
                     construction_state_indices, anchor_index=3):
    """Fit one projection-wise functional/local ratio without test-state leakage."""
    damage = np.asarray(damage_states, dtype=float)
    reconstruction = np.asarray(reconstruction_states, dtype=float)
    if damage.shape != reconstruction.shape or damage.ndim != 3:
        raise ValueError("damage and reconstruction must be module x state x level")
    indices = np.asarray(construction_state_indices, dtype=int)
    if indices.size == 0:
        raise ValueError("construction states cannot be empty")
    local_anchor = reconstruction[:, indices, anchor_index].mean(axis=1)
    if np.any(local_anchor == 0):
        raise ValueError("zero anchor reconstruction; no epsilon policy is preregistered")
    functional_anchor = damage[:, indices, anchor_index].mean(axis=1)
    alpha = functional_anchor / local_anchor
    predicted = alpha[:, None] * reconstruction[:, indices, :].mean(axis=1)
    return predicted, alpha


def _module_parts(name):
    block, projection = name.split(".", 1)
    if not block.startswith("block_"):
        raise ValueError(f"unrecognized module name: {name}")
    return int(block.removeprefix("block_")), projection


def eis_type_control(names, shapes, oracle_sparsities):
    """Permute each projection type's oracle level multiset earlier-is-sparser."""
    if not (len(names) == len(shapes) == len(oracle_sparsities)):
        raise ValueError("names, shapes and sparsities must have equal length")
    groups = defaultdict(list)
    for index, (name, shape, sparsity) in enumerate(zip(names, shapes, oracle_sparsities)):
        layer, projection = _module_parts(name)
        groups[projection].append((index, layer, tuple(shape), float(sparsity)))
    result = [None] * len(names)
    for projection, rows in groups.items():
        if len({row[2] for row in rows}) != 1:
            raise ValueError(f"shape mismatch within projection type {projection}")
        destinations = sorted(rows, key=lambda row: (row[1], row[0]))
        levels = sorted((row[3] for row in rows), reverse=True)
        for destination, level in zip(destinations, levels):
            result[destination[0]] = level
    return result


def additive_damage(curves, levels):
    """Sum single-projection curve increases relative to the 50% baseline."""
    values = np.asarray(curves, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(levels):
        raise ValueError("curves must be module x level and align with levels")
    return float(sum(values[i, int(level)] - values[i, 0] for i, level in enumerate(levels)))


def build_selected_manifest(method, entries, sparsities, grid):
    """Select persisted candidate masks without changing their payload metadata."""
    if len(entries) != len(sparsities):
        raise ValueError("entries and sparsities must align")
    grid = tuple(float(x) for x in grid)
    selected = []
    for row, sparsity in zip(entries, sparsities):
        try:
            level = grid.index(float(sparsity))
        except ValueError as error:
            raise ValueError(f"sparsity {sparsity} is not on grid") from error
        masks = row.get("masks", [])
        if len(masks) != len(grid):
            raise ValueError(f"candidate mask count mismatch for {row.get('name')}")
        mask = dict(masks[level])
        if float(mask.get("sparsity", grid[level])) != grid[level]:
            raise ValueError(f"candidate mask grid mismatch for {row.get('name')}")
        item = {key: value for key, value in row.items() if key != "masks"}
        item.update(assigned_sparsity=grid[level], level=level, selected_mask=mask)
        selected.append(item)
    return {
        "method": method,
        "pruned": sum(int(row["selected_mask"]["pruned"]) for row in selected),
        "weights": sum(int(row["weights"]) for row in selected),
        "entries": selected,
    }


def interval_overlaps(existing, candidates):
    """Return all pairs of half-open corpus intervals with nonempty overlap."""
    return [
        {"existing": old, "candidate": new}
        for old in existing for new in candidates
        if max(int(old["start"]), int(new["start"]))
        < min(int(old["end_exclusive"]), int(new["end_exclusive"]))
    ]


def paired_comparison(reference, candidate, resamples=20000, seed=0):
    """Paired KL difference with state and sequence-cluster percentile intervals."""
    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired rows must have equal nonzero length")
    keys = [(int(row["sequence_index"]), float(row["timestep"])) for row in reference]
    candidate_keys = [(int(row["sequence_index"]), float(row["timestep"]))
                      for row in candidate]
    if keys != candidate_keys:
        raise ValueError("state pairing mismatch")
    delta = np.asarray([float(b["mean_kl"]) - float(a["mean_kl"])
                        for a, b in zip(reference, candidate)], dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("non-finite paired differences")
    rng = np.random.default_rng(seed)
    state_means = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(1)
    sequences = sorted(set(sequence for sequence, _ in keys))
    sequence_means = np.asarray([
        delta[[i for i, key in enumerate(keys) if key[0] == sequence]].mean()
        for sequence in sequences
    ])
    cluster_means = sequence_means[
        rng.integers(0, len(sequences), size=(resamples, len(sequences)))
    ].mean(1)
    timesteps = sorted(set(timestep for _, timestep in keys))
    timestep_means = np.asarray([
        delta[[i for i, key in enumerate(keys) if key[1] == timestep]].mean()
        for timestep in timesteps
    ])
    return {
        "mean_difference": float(delta.mean()),
        "median_difference": float(np.median(delta)),
        "state_bootstrap_95_ci": np.quantile(state_means, [0.025, 0.975]).tolist(),
        "sequence_cluster_bootstrap_95_ci": np.quantile(
            cluster_means, [0.025, 0.975]
        ).tolist(),
        "states_improved": int(np.sum(delta < 0)),
        "states_worsened": int(np.sum(delta > 0)),
        "states_tied": int(np.sum(delta == 0)),
        "sequence_means_improved": int(np.sum(sequence_means < 0)),
        "timestep_means_improved": int(np.sum(timestep_means < 0)),
        "per_sequence_mean_difference": {
            str(sequence): float(value) for sequence, value in zip(sequences, sequence_means)
        },
        "per_timestep_mean_difference": {
            str(timestep): float(value) for timestep, value in zip(timesteps, timestep_means)
        },
        "resamples": int(resamples),
        "seed": int(seed),
    }


def paired_binary_comparison(reference_correct, candidate_correct):
    """Paired exact-match transition table with exact two-sided McNemar p-value."""
    reference = np.asarray(reference_correct, dtype=bool)
    candidate = np.asarray(candidate_correct, dtype=bool)
    if reference.shape != candidate.shape or reference.ndim != 1 or reference.size == 0:
        raise ValueError("paired correctness arrays must have equal nonzero length")
    gains = int(np.sum(~reference & candidate))
    losses = int(np.sum(reference & ~candidate))
    discordant = gains + losses
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(gains, losses), discordant, 0.5, alternative="two-sided").pvalue
    )
    return {
        "reference_wrong_candidate_correct": gains,
        "reference_correct_candidate_wrong": losses,
        "both_correct": int(np.sum(reference & candidate)),
        "both_wrong": int(np.sum(~reference & ~candidate)),
        "net_correct": gains - losses,
        "discordant": discordant,
        "exact_mcnemar_p": p_value,
    }
