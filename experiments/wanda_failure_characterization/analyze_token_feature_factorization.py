import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from experiments.wanda_failure_characterization.propagation_core import token_feature_factorial_contrasts


ROOT = Path(__file__).parent
SEED = 20260905
BOOTSTRAPS = 10_000
CELLS = ("NN", "NF", "FN", "FF")


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "p10": float(np.quantile(values, .1)), "p90": float(np.quantile(values, .9)), "fraction_positive": float((values > 0).mean())}


def cluster_ci(values, sequence):
    rng = np.random.default_rng(SEED)
    unique = np.unique(sequence)
    means = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sampled = rng.choice(unique, len(unique), replace=True)
        means[index] = np.concatenate([values[sequence == item] for item in sampled]).mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def corr(x, y):
    value = spearmanr(x, y).statistic
    return None if not np.isfinite(value) else float(value)


def main():
    artifact = torch.load(ROOT / "token_feature_factorization.pt", map_location="cpu", weights_only=True)
    mapping = json.loads((ROOT / "token_feature_donor_manifest.json").read_text())
    manifest = json.loads((ROOT / "token_feature_factorization_manifest.json").read_text())
    kl = artifact["results"]["kl"].numpy()
    sequence = artifact["sequence_index"].numpy()
    timestep = artifact["timestep_index"].numpy()
    pair_receiver = np.array([item["receiver_index"] for item in mapping["pairs"]])
    pair_donor = np.array([item["donor_index"] for item in mapping["pairs"]])

    pair_rows = []
    effects_pair = token_feature_factorial_contrasts(torch.from_numpy(kl))
    for index, item in enumerate(mapping["pairs"]):
        row = dict(item)
        row.update({f"KL_{name}": float(kl[index, ci]) for ci, name in enumerate(CELLS)})
        row.update({key: float(value[index]) for key, value in effects_pair.items()})
        row.update({key: float(value[index]) for key, value in artifact["similarities"].items()})
        for metric in ("final_hidden_relative_energy", "masked_logit_relative_energy", "logit_rms"):
            values = artifact["results"][metric].numpy()[index]
            row.update({f"{metric}_{name}": float(values[ci]) for ci, name in enumerate(CELLS)})
        pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(ROOT / "token_feature_per_pair.csv", index=False)

    receiver_rows = []
    receiver_cells = np.empty((40, 4))
    donor_robustness = []
    for receiver in range(40):
        indices = np.flatnonzero(pair_receiver == receiver)
        cells = kl[indices].mean(axis=0)
        receiver_cells[receiver] = cells
        effects = token_feature_factorial_contrasts(torch.from_numpy(cells))
        native_excess = cells[0] - max(cells[1], cells[2])
        row = {"receiver_index": receiver, "sequence_index": int(sequence[receiver]), "timestep_index": int(timestep[receiver])}
        row.update({name: float(cells[ci]) for ci, name in enumerate(CELLS)})
        row.update({key: float(value) for key, value in effects.items()})
        row["native_excess"] = float(native_excess)
        for ci, name in enumerate(("NF", "FN", "FF"), start=1):
            values = kl[indices, ci]
            row[f"R_{name}"] = float(values.mean() / cells[0])
            donor_robustness.append({"receiver_index": receiver, "condition": name, "mean": float(values.mean()), "std": float(values.std()), "cv": float(values.std() / values.mean()), "min": float(values.min()), "max": float(values.max()), "fraction_ge_native": float((values >= cells[0]).mean())})
        receiver_rows.append(row)
    receiver_df = pd.DataFrame(receiver_rows)
    receiver_df.to_csv(ROOT / "token_feature_per_receiver.csv", index=False)
    pd.DataFrame(donor_robustness).to_csv(ROOT / "token_feature_donor_robustness.csv", index=False)

    effects_receiver = token_feature_factorial_contrasts(torch.from_numpy(receiver_cells))
    analysis = {
        "primary_cells": {name: summary(receiver_cells[:, ci]) for ci, name in enumerate(CELLS)},
        "effects": {},
        "retained_damage_ratio_of_means": {name: float(receiver_cells[:, ci].mean() / receiver_cells[:, 0].mean()) for ci, name in enumerate(("NF", "FN", "FF"), start=1)},
        "nn_donor_spread_max": float(np.max([np.ptp(kl[pair_receiver == receiver, 0]) for receiver in range(40)])),
        "norm_matching": {},
        "nn_gate": {key: {"mean": float(value.mean()), "max": float(value.max())} for key, value in artifact["nn_gate"].items()},
        "batch_validation": manifest["batch_validation"],
        "repeatability": manifest["repeatability"],
    }
    for key, tensor in effects_receiver.items():
        values = tensor.numpy()
        analysis["effects"][key] = summary(values)
        analysis["effects"][key]["cluster_bootstrap_95ci"] = cluster_ci(values, sequence)
    norms = artifact["results"]["norm_deviation"].numpy()
    analysis["norm_matching"] = {"mean": float(norms.mean()), "p99": float(np.quantile(norms, .99)), "max": float(norms.max())}

    timestep_rows, sequence_rows = [], []
    for field, values in (("timestep_index", timestep), ("sequence_index", sequence)):
        for group in np.unique(values):
            indices = values == group
            cells = receiver_cells[indices].mean(axis=0)
            effects = token_feature_factorial_contrasts(torch.from_numpy(cells))
            row = {field: int(group), **{name: float(cells[ci]) for ci, name in enumerate(CELLS)}, **{key: float(value) for key, value in effects.items()}}
            (timestep_rows if field == "timestep_index" else sequence_rows).append(row)
    pd.DataFrame(timestep_rows).to_csv(ROOT / "token_feature_by_timestep.csv", index=False)
    pd.DataFrame(sequence_rows).to_csv(ROOT / "token_feature_by_sequence.csv", index=False)

    magnitude = {}
    for metric in ("final_hidden_relative_energy", "masked_logit_relative_energy", "logit_rms"):
        values = artifact["results"][metric].numpy()
        diffs = {}
        for ci, name in enumerate(("NF", "FN", "FF"), start=1):
            kl_diff = kl[:, 0] - kl[:, ci]
            metric_diff = values[:, 0] - values[:, ci]
            diffs[name] = {"mean_cell": float(values[:, ci].mean()), "native_minus_foreign_mean": float(metric_diff.mean()), "spearman_with_kl_difference": corr(metric_diff, kl_diff)}
        magnitude[metric] = {"native_mean": float(values[:, 0].mean()), "comparisons": diffs}
    analysis["downstream_magnitude_controls"] = magnitude

    similarity = {key: value.numpy() for key, value in artifact["similarities"].items()}
    analysis["similarity_associations"] = {
        "allocation_cosine_vs_K_FN": corr(similarity["allocation_cosine"], kl[:, 2]),
        "allocation_cosine_vs_K_FF": corr(similarity["allocation_cosine"], kl[:, 3]),
        "allocation_cosine_vs_token_effect": corr(similarity["allocation_cosine"], effects_pair["token_main"].numpy()),
        "allocation_spearman_vs_K_FN": corr(similarity["allocation_spearman"], kl[:, 2]),
        "feature_weighted_cosine_vs_K_NF": corr(similarity["feature_weighted_cosine"], kl[:, 1]),
        "feature_weighted_cosine_vs_K_FF": corr(similarity["feature_weighted_cosine"], kl[:, 3]),
        "feature_weighted_cosine_vs_feature_effect": corr(similarity["feature_weighted_cosine"], effects_pair["feature_main"].numpy()),
        "feature_mean_cosine_vs_K_NF": corr(similarity["feature_mean_cosine"], kl[:, 1]),
    }
    (ROOT / "token_feature_analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    files = {}
    for name in ("token_feature_per_pair.csv", "token_feature_per_receiver.csv", "token_feature_donor_robustness.csv", "token_feature_by_timestep.csv", "token_feature_by_sequence.csv", "token_feature_analysis.json"):
        files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (ROOT / "token_feature_analysis_manifest.json").write_text(json.dumps({"status": "complete", "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "files": files}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
