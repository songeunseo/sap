#!/usr/bin/env python3
"""Analyze persisted oracle curves and freeze diagnostic comparator allocations."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from experiments.projection_capacity_allocation_65.core import allocate
from experiments.projection_capacity_followup_65.core import (
    additive_damage,
    build_selected_manifest,
    eis_type_control,
    fit_anchor_curve,
    summarize_curve_records,
)


ROOT = Path("experiments/projection_capacity_followup_65")
SOURCE = Path("experiments/projection_capacity_allocation_65")
GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size != y.size or x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"pearson": None, "spearman": None, "n": int(x.size)}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(spearmanr(x, y).statistic),
        "n": int(x.size),
    }


def residualized_rank_corr(x, y, names):
    """Partial rank correlation after categorical layer and projection-type effects."""
    layers = [int(name.split(".")[0].removeprefix("block_")) for name in names]
    types = [name.split(".")[1] for name in names]
    columns = [np.ones(len(names))]
    for value in sorted(set(layers))[1:]:
        columns.append(np.asarray([item == value for item in layers], float))
    for value in sorted(set(types))[1:]:
        columns.append(np.asarray([item == value for item in types], float))
    design = np.column_stack(columns)
    rx, ry = rankdata(x), rankdata(y)
    rx -= design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry -= design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return corr(rx, ry)


def level_counts(sparsities):
    return {str(level): int(sum(float(x) == level for x in sparsities)) for level in GRID}


def assignment_comparison(left, right, shapes):
    left, right = np.asarray(left), np.asarray(right)
    weights = np.asarray([math.prod(shape) for shape in shapes], dtype=float)
    return {
        "projections_changed": int(np.sum(left != right)),
        "mean_absolute_sparsity_difference": float(np.mean(np.abs(left - right))),
        "parameter_weighted_absolute_budget_moved": float(
            np.sum(np.abs(left - right) * weights) / np.sum(weights)
        ),
        "spearman": corr(left, right)["spearman"],
    }


def main():
    source_curves = json.loads((SOURCE / "capacity_curves_raw.json").read_text())
    records = source_curves["projections"]
    aligned = summarize_curve_records(records, GRID)
    names, shapes = aligned["names"], aligned["shapes"]
    if len(names) != 224 or len(set(names)) != 224:
        raise RuntimeError("expected 224 unique projection records")
    candidate = json.loads((SOURCE / "candidate_mask_manifest.json").read_text())
    allocation = json.loads((SOURCE / "allocation.json").read_text())
    candidate_names = [row["name"] for row in candidate["entries"]]
    oracle_names = [row["name"] for row in allocation["assignments"]]
    if names != candidate_names or names != oracle_names:
        raise RuntimeError("source module ordering mismatch")
    if shapes != [tuple(row["shape"]) for row in candidate["entries"]]:
        raise RuntimeError("source module shape mismatch")

    damage = aligned["damage"]
    reconstruction = aligned["reconstruction"]
    damage_marginal = np.diff(damage, axis=1)
    reconstruction_marginal = np.diff(reconstruction, axis=1)
    oracle_sparsities = [float(row["assigned_sparsity"]) for row in allocation["assignments"]]
    oracle_levels = [int(row["level"]) for row in allocation["assignments"]]

    reconstruction_result = allocate(reconstruction, shapes)
    reconstruction_sparsities = reconstruction_result["sparsities"]
    eis_sparsities = eis_type_control(names, shapes, oracle_sparsities)
    eis_levels = [GRID.index(float(value)) for value in eis_sparsities]

    target = int(allocation["pruned"])
    reconstruction_manifest = build_selected_manifest(
        "reconstruction", candidate["entries"], reconstruction_sparsities, GRID
    )
    eis_manifest = build_selected_manifest("eis_type", candidate["entries"], eis_sparsities, GRID)
    for manifest in (reconstruction_manifest, eis_manifest):
        if manifest["pruned"] != target:
            raise RuntimeError(f"{manifest['method']} does not match exact target budget")

    level_correlations = {}
    for level_index, level in enumerate(GRID):
        level_correlations[str(level)] = {
            "raw": corr(reconstruction[:, level_index], damage[:, level_index]),
            "layer_type_adjusted_rank": residualized_rank_corr(
                reconstruction[:, level_index], damage[:, level_index], names
            ),
        }
    marginal_correlations = {
        "all_increments": corr(reconstruction_marginal.ravel(), damage_marginal.ravel()),
        "by_increment": {
            f"{GRID[k]}->{GRID[k + 1]}": corr(
                reconstruction_marginal[:, k], damage_marginal[:, k]
            ) for k in range(5)
        },
    }

    state_keys = aligned["state_keys"]
    sequence_ids = sorted(set(sequence for sequence, _ in state_keys))
    if sequence_ids != list(range(8)):
        raise RuntimeError(f"unexpected calibration sequence IDs: {sequence_ids}")
    folds = []
    for construction_sequences, evaluation_sequences in ((set(range(4)), set(range(4, 8))),
                                                           (set(range(4, 8)), set(range(4)))):
        construction = [i for i, key in enumerate(state_keys) if key[0] in construction_sequences]
        evaluation = [i for i, key in enumerate(state_keys) if key[0] in evaluation_sequences]
        predicted, alpha = fit_anchor_curve(
            aligned["damage_states"], aligned["reconstruction_states"], construction, 3
        )
        local_construction = aligned["reconstruction_states"][:, construction, :].mean(axis=1)
        test_damage = aligned["damage_states"][:, evaluation, :].mean(axis=1)
        predicted_result = allocate(predicted, shapes)
        local_result = allocate(local_construction, shapes)
        test_oracle_result = allocate(test_damage, shapes)
        non_anchor = [0, 1, 2, 4, 5]
        folds.append({
            "construction_sequences": sorted(construction_sequences),
            "evaluation_sequences": sorted(evaluation_sequences),
            "alpha_distribution": {
                "min": float(alpha.min()), "median": float(np.median(alpha)),
                "mean": float(alpha.mean()), "max": float(alpha.max()),
            },
            "prediction_correlation_excluding_anchor": corr(
                predicted[:, non_anchor].ravel(), test_damage[:, non_anchor].ravel()
            ),
            "marginal_prediction_correlation": corr(
                np.diff(predicted, axis=1).ravel(), np.diff(test_damage, axis=1).ravel()
            ),
            "heldout_additive_damage": {
                "single_anchor": additive_damage(test_damage, predicted_result["levels"]),
                "reconstruction_only": additive_damage(test_damage, local_result["levels"]),
                "fold_oracle_reference": additive_damage(test_damage, test_oracle_result["levels"]),
            },
            "single_anchor_vs_reconstruction": assignment_comparison(
                predicted_result["sparsities"], local_result["sparsities"], shapes
            ),
        })

    diagnostics = {
        "source": {
            "capacity_curves_raw_sha256": sha(SOURCE / "capacity_curves_raw.json"),
            "candidate_mask_manifest_sha256": sha(SOURCE / "candidate_mask_manifest.json"),
            "allocation_sha256": sha(SOURCE / "allocation.json"),
        },
        "projection_count": len(names),
        "state_count": len(state_keys),
        "grid": GRID,
        "local_reconstruction_definition": (
            "all-token Linear output squared error divided by dense Linear output squared energy"
        ),
        "level_correlations": level_correlations,
        "marginal_correlations": marginal_correlations,
        "allocations": {
            "oracle": {
                "counts": level_counts(oracle_sparsities),
                "additive_calibration_damage": additive_damage(damage, oracle_levels),
            },
            "reconstruction": {
                "counts": level_counts(reconstruction_sparsities),
                "additive_calibration_damage": additive_damage(
                    damage, reconstruction_result["levels"]
                ),
                "vs_oracle": assignment_comparison(
                    reconstruction_sparsities, oracle_sparsities, shapes
                ),
            },
            "eis_type": {
                "counts": level_counts(eis_sparsities),
                "additive_calibration_damage": additive_damage(damage, eis_levels),
                "vs_oracle": assignment_comparison(eis_sparsities, oracle_sparsities, shapes),
            },
        },
        "crossfit_single_anchor_65": folds,
    }

    config = {
        "status": "frozen_before_new_model_evaluation",
        "model": json.loads((SOURCE / "config.json").read_text())["model"],
        "grid": GRID,
        "target_pruned": target,
        "selector": "unchanged Standard Wanda candidate masks from source experiment",
        "reconstruction_allocation": (
            "raw next-increment all-token normalized Linear reconstruction error / (0.05*N)"
        ),
        "eis_type_control": (
            "preserve each projection type's oracle sparsity multiset; assign larger levels "
            "to earlier layers"
        ),
        "source_hashes": diagnostics["source"],
    }
    write_json(ROOT / "config.json", config)
    for manifest in (reconstruction_manifest, eis_manifest):
        manifest.update({
            "config_sha256": sha(ROOT / "config.json"),
            "source_candidate_manifest_sha256": diagnostics["source"]["candidate_mask_manifest_sha256"],
            "exact_target_match": manifest["pruned"] == target,
        })
        write_json(ROOT / f"{manifest['method']}65_mask_manifest.json", manifest)
    write_json(ROOT / "existing_curve_diagnostics.json", diagnostics)

    anchor_better = all(
        fold["heldout_additive_damage"]["single_anchor"]
        < fold["heldout_additive_damage"]["reconstruction_only"] for fold in folds
    )
    report = [
        "# Projection Capacity Follow-up @ 65%",
        "",
        "## Existing-data diagnostic",
        "",
        f"- Projections/states: {len(names)}/{len(state_keys)}",
        f"- Reconstruction allocation exact pruned count: {reconstruction_manifest['pruned']:,}",
        f"- EIS+type control exact pruned count: {eis_manifest['pruned']:,}",
        f"- Single-anchor beats reconstruction in both cross-fit directions: {anchor_better}",
        "",
        "## Allocation comparison",
        "",
        "| Allocation | Additive calibration damage | Projections changed vs oracle |",
        "|---|---:|---:|",
        f"| Oracle | {diagnostics['allocations']['oracle']['additive_calibration_damage']:.8g} | 0 |",
        f"| Reconstruction | {diagnostics['allocations']['reconstruction']['additive_calibration_damage']:.8g} | "
        f"{diagnostics['allocations']['reconstruction']['vs_oracle']['projections_changed']} |",
        f"| EIS+type | {diagnostics['allocations']['eis_type']['additive_calibration_damage']:.8g} | "
        f"{diagnostics['allocations']['eis_type']['vs_oracle']['projections_changed']} |",
        "",
        "These additive single-projection costs are diagnostics, not jointly sparse model results.",
    ]
    (ROOT / "report.md").write_text("\n".join(report) + "\n")
    write_json(ROOT / "analysis_status.json", {
        "status": "existing_data_analysis_complete",
        "single_anchor_crossfit_passed": anchor_better,
        "gpu_evaluation_started": False,
    })
    print(json.dumps({
        "status": "existing_data_analysis_complete",
        "single_anchor_crossfit_passed": anchor_better,
        "reconstruction_pruned": reconstruction_manifest["pruned"],
        "eis_type_pruned": eis_manifest["pruned"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
