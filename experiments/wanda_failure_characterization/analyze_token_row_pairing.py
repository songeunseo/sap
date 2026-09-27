import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from experiments.wanda_failure_characterization.propagation_core import token_row_pairing_contrasts


ROOT = Path(__file__).parent
SHARDS = ROOT / "token_row_pairing_shards"
SEED = 20260905
BOOTSTRAPS = 10_000
CELLS = ("NN", "NS", "SN", "SS")


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.quantile(x, .1)), "p90": float(np.quantile(x, .9)), "fraction_positive": float((x > 0).mean())}


def clustered_ci(values, sequence):
    rng = np.random.default_rng(SEED)
    groups = np.unique(sequence)
    out = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        selected = rng.choice(groups, len(groups), replace=True)
        out[index] = np.concatenate([values[sequence == group] for group in selected]).mean()
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def safe_spearman(x, y):
    value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)


def family_receiver_row(state_index, sequence, timestep, family, results):
    kl = results["kl"].numpy()
    effects = {key: value.numpy() for key, value in token_row_pairing_contrasts(torch.from_numpy(kl)).items()}
    nn = kl[:, 0]
    row = {"state_index": state_index, "sequence_index": int(sequence), "timestep_index": int(timestep), "family": family}
    for ci, name in enumerate(CELLS):
        row[f"{name}_mean"] = float(kl[:, ci].mean())
        row[f"{name}_median"] = float(np.median(kl[:, ci]))
        row[f"{name}_p10"] = float(np.quantile(kl[:, ci], .1))
        row[f"{name}_p90"] = float(np.quantile(kl[:, ci], .9))
    for key, values in effects.items():
        row[f"{key}_mean"] = float(values.mean())
        row[f"{key}_median"] = float(np.median(values))
    for ci, name in enumerate(("NS", "SN", "SS"), start=1):
        row[f"fraction_NN_gt_{name}"] = float((nn > kl[:, ci]).mean())
        row[f"R_{name}"] = float(kl[:, ci].mean() / nn.mean())
    return row, effects


def main():
    run_manifest = json.loads((ROOT / "token_row_pairing_run_manifest.json").read_text())
    frozen = json.loads((ROOT / "token_row_pairing_manifest.json").read_text())
    permutation_artifact = torch.load(ROOT / "token_row_pairing_permutations.pt", map_location="cpu", weights_only=True)
    prior = torch.load(ROOT / "perturbation_transplant.pt", map_location="cpu", weights_only=True)
    sequence, timestep = prior["sequence_index"].numpy(), prior["timestep_index"].numpy()
    rows, receiver_effects = [], {family: {key: np.empty(40) for key in ("pairing", "position", "feature_placement")} for family in ("unrestricted", "class_preserving")}
    all_results = {family: [] for family in receiver_effects}
    pre_norm = {family: [] for family in receiver_effects}
    norm_errors = {family: [] for family in receiver_effects}

    for state_index in range(40):
        shard = torch.load(SHARDS / f"state_{state_index:02d}.pt", map_location="cpu", weights_only=True)
        for family in receiver_effects:
            result = shard[family]
            row, effects = family_receiver_row(state_index, sequence[state_index], timestep[state_index], family, result)
            rows.append(row)
            for key in receiver_effects[family]:
                receiver_effects[family][key][state_index] = effects[key].mean()
            all_results[family].append(result)
            spread_key = "unrestricted_pre_norm_spread" if family == "unrestricted" else "class_pre_norm_spread"
            pre_norm[family].append(shard[spread_key])
            norm_errors[family].append(result["norm_deviation"].flatten())
    receiver_df = pd.DataFrame(rows)
    receiver_df.to_csv(ROOT / "token_row_pairing_per_receiver.csv", index=False)

    analysis = {"families": {}, "gates": {"numerical": json.loads((ROOT / "token_row_pairing_numerical_gate.json").read_text()), "run": run_manifest}}
    for family in receiver_effects:
        family_rows = receiver_df[receiver_df.family == family]
        cell_global = {}
        for name in CELLS:
            cell_global[name] = {"mean_of_receiver_means": float(family_rows[f"{name}_mean"].mean()), "median_of_receiver_means": float(family_rows[f"{name}_mean"].median())}
        effect_output = {}
        for key, values in receiver_effects[family].items():
            effect_output[key] = summarize(values)
            effect_output[key]["cluster_bootstrap_95ci"] = clustered_ci(values, sequence)
        errors = torch.cat(norm_errors[family]).numpy()
        spreads = torch.cat(pre_norm[family]).numpy()
        analysis["families"][family] = {
            "cells": cell_global,
            "effects": effect_output,
            "retention_ratio_of_global_means": {name: float(family_rows[f"{name}_mean"].mean() / family_rows["NN_mean"].mean()) for name in ("NS", "SN", "SS")},
            "norm_matching": {"mean": float(errors.mean()), "p99": float(np.quantile(errors, .99)), "max": float(errors.max())},
            "pre_normalization_norm_spread": {"mean": float(spreads.mean()), "p99": float(np.quantile(spreads, .99)), "max": float(spreads.max())},
        }

    grouped_rows = []
    for family in receiver_effects:
        subset = receiver_df[receiver_df.family == family]
        for field in ("timestep_index", "sequence_index"):
            for group, frame in subset.groupby(field):
                row = {"family": family, "group_type": field, "group": int(group)}
                for name in CELLS:
                    row[name] = float(frame[f"{name}_mean"].mean())
                for key in receiver_effects[family]:
                    row[key] = float(frame[f"{key}_mean"].mean())
                grouped_rows.append(row)
    pd.DataFrame(grouped_rows).to_csv(ROOT / "token_row_pairing_by_timestep_sequence.csv", index=False)

    # Unrestricted circular-distance characterization.
    distance_rows = []
    for distance in range(1, 129):
        indices = [offset - 1 for offset in range(1, 256) if min(offset, 256 - offset) == distance]
        cells = np.concatenate([result["kl"].numpy()[indices] for result in all_results["unrestricted"]], axis=0)
        effects = token_row_pairing_contrasts(torch.from_numpy(cells))
        distance_rows.append({"distance": distance, **{name: float(cells[:, ci].mean()) for ci, name in enumerate(CELLS)}, **{key: float(value.float().mean()) for key, value in effects.items()}})
    pd.DataFrame(distance_rows).to_csv(ROOT / "token_row_pairing_by_distance.csv", index=False)

    # Magnitude controls and KL-difference associations across every intervention.
    magnitude = {}
    for family in receiver_effects:
        family_output = {}
        for metric in ("final_hidden_relative_energy", "masked_logit_relative_energy", "logit_rms"):
            native, comparisons = [], {name: {"metric_difference": [], "kl_difference": []} for name in ("NS", "SN", "SS")}
            for result in all_results[family]:
                values, kl = result[metric].numpy(), result["kl"].numpy()
                native.extend(values[:, 0])
                for ci, name in enumerate(("NS", "SN", "SS"), start=1):
                    comparisons[name]["metric_difference"].extend(values[:, 0] - values[:, ci])
                    comparisons[name]["kl_difference"].extend(kl[:, 0] - kl[:, ci])
            family_output[metric] = {"NN_mean": float(np.mean(native)), "comparisons": {name: {"NN_minus_cell_metric_mean": float(np.mean(data["metric_difference"])), "spearman_with_KL_difference": safe_spearman(data["metric_difference"], data["kl_difference"])} for name, data in comparisons.items()}}
        magnitude[family] = family_output
    analysis["downstream_magnitude_controls"] = magnitude

    (ROOT / "token_row_pairing_analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    files = {}
    for name in ("token_row_pairing_per_receiver.csv", "token_row_pairing_by_timestep_sequence.csv", "token_row_pairing_by_distance.csv", "token_row_pairing_analysis.json"):
        files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (ROOT / "token_row_pairing_analysis_manifest.json").write_text(json.dumps({"status": "complete", "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "source_manifest_sha256": frozen["manifest_sha256"], "files": files}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"families": analysis["families"], "downstream_magnitude_controls": magnitude}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
