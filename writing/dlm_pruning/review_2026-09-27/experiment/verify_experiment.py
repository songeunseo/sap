"""Independent, read-only audit of the completed cross-chain control.

This script reads the frozen per-question checkpoints and diagnostic JSON files,
recomputes the primary paired statistics, calls the existing CPU closeout
verification routines, and writes only artifacts in this directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
WRITING = HERE.parents[1]
REPO = HERE.parents[3]
RUN = REPO / "experiments/dlm_crosschain_control50/output"
MINI = REPO / "experiments/dlm_multi_validation10050/output"
OLD = REPO / "experiments/dlm_multiscale_ac50/output"
CLOSEOUT = WRITING / "closeout.json"
REPORT = RUN / "report.json"
CONFIG = RUN / "config.json"
REQUESTS = RUN / "requests.json"
ARMS = ("A", "Multi", "Cross", "CrossMatched")
PRIMARY_FAMILY = ("Multi-A", "Multi-Cross", "Multi-CrossMatched")
SELECTED_DIAGNOSTICS = (
    "A", "C_natural", "C_cross", "query_CE",
    "response_sign_flip_rate", "near_zero_dense_response_fraction",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path: Path):
    path = Path(path)
    with path.open() as stream:
        return json.load(stream)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: {actual!r} != {expected!r}")


def assert_close(actual: float, expected: float, message: str, tolerance: float = 1e-10) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError(f"{message}: {actual!r} != {expected!r}")


def exact_mcnemar(gain: int, loss: int) -> float:
    discordant = gain + loss
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gain, loss) + 1))
    return min(1.0, 2.0 * tail / (2.0 ** discordant))


def paired_statistics(reference, candidate, seed: int = 20260927, draws: int = 10000):
    """Recompute the fixed outcome estimand from one boolean per question."""
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if reference.ndim != 1 or candidate.shape != reference.shape:
        raise AssertionError("Unpaired primary outcomes")
    gain = int(np.sum(candidate & ~reference))
    loss = int(np.sum(reference & ~candidate))
    delta = candidate.astype(float) - reference.astype(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for start in range(0, draws, 500):
        take = min(500, draws - start)
        indices = rng.integers(0, len(delta), size=(take, len(delta)))
        samples[start:start + take] = delta[indices].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "gain": gain,
        "loss": loss,
        "net": gain - loss,
        "difference_pp": float(100.0 * delta.mean()),
        "exact_mcnemar_p": exact_mcnemar(gain, loss),
        "paired_bootstrap_95pp_unadjusted": [float(100.0 * lo), float(100.0 * hi)],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def holm(family: dict[str, dict]) -> None:
    previous = 0.0
    ordered = sorted(family, key=lambda name: family[name]["exact_mcnemar_p"])
    for index, name in enumerate(ordered):
        previous = max(previous, min(1.0, (len(family) - index) * family[name]["exact_mcnemar_p"]))
        family[name]["holm_p"] = previous


def span_statistics(reference, candidate, seed: int = 20260927, draws: int = 10000):
    """Recompute saved eight-span paired bootstrap uncertainty."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != (8,) or candidate.shape != (8,):
        raise AssertionError(f"Expected eight paired spans, got {reference.shape}/{candidate.shape}")
    differences = candidate - reference
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 8, size=(draws, 8))
    lo, hi = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
    return {
        "mean_difference": float(differences.mean()),
        "paired_bootstrap_95_unadjusted": [float(lo), float(hi)],
        "span_count": 8,
        "bootstrap_draws": draws,
        "seed": seed,
    }


def load_full_rows():
    config = read(CONFIG)
    requests = read(REQUESTS)["development"]
    ids = [request["example_id"] for request in requests]
    assert_equal(ids, list(range(1319)), "full request order")
    rows = {}
    checkpoint_counts = {}
    row_hashes = {}
    for arm in ARMS:
        arm_rows = {}
        hashes = []
        for example_id in ids:
            shard = example_id // 128
            path = RUN / "gsm8k" / arm / f"shard{shard:02d}" / "examples" / f"{example_id:04d}.json"
            saved = read(path)
            row = saved["row"]
            assert_equal(row["example_id"], example_id, f"{arm} checkpoint example id")
            assert_equal(row["method"], arm, f"{arm} checkpoint method")
            assert_equal(saved["row_sha256"], digest(row), f"{arm} checkpoint row hash")
            arm_rows[example_id] = row
            hashes.append(saved["row_sha256"])
        rows[arm] = arm_rows
        checkpoint_counts[arm] = len(arm_rows)
        row_hashes[arm] = digest(hashes)
    exposed = config["crosschain_control"]["exposed_ids"]
    primary = config["crosschain_control"]["primary_ids"]
    assert_equal(len(exposed), 200, "exposed sample size")
    assert_equal(len(primary), 1119, "primary sample size")
    assert_equal(len(set(exposed)), 200, "exposed sample uniqueness")
    assert_equal(len(set(primary)), 1119, "primary sample uniqueness")
    assert_equal(set(exposed) | set(primary), set(ids), "sample partition")
    assert_equal(set(exposed) & set(primary), set(), "sample overlap")
    return config, requests, rows, exposed, primary, checkpoint_counts, row_hashes


def compare_saved(name: str, observed: dict, saved: dict) -> None:
    for key, value in observed.items():
        if isinstance(value, list):
            if len(value) != len(saved[key]) or any(abs(float(a) - float(b)) > 1e-10 for a, b in zip(value, saved[key])):
                raise AssertionError(f"saved {name}/{key} differs")
        elif isinstance(value, float):
            assert_close(value, saved[key], f"saved {name}/{key}")
        else:
            assert_equal(value, saved[key], f"saved {name}/{key}")


def independent_primary(config, rows, primary):
    outcomes = {arm: [bool(rows[arm][i]["correct"]) for i in primary] for arm in ARMS}
    scores = {arm: {"correct": int(sum(outcomes[arm])), "total": len(primary)} for arm in ARMS}
    comparisons = {
        "Multi-A": paired_statistics(outcomes["A"], outcomes["Multi"]),
        "Multi-Cross": paired_statistics(outcomes["Cross"], outcomes["Multi"]),
        "Multi-CrossMatched": paired_statistics(outcomes["CrossMatched"], outcomes["Multi"]),
    }
    holm(comparisons)
    return outcomes, scores, comparisons


def diagnostic_metrics(split: str):
    from experiments.dlm_crosschain_control50.core import response_metrics

    config = read(CONFIG)
    bank = read(config["banks"][split]["path"])
    dense = read(RUN / "readouts" / "dense" / f"{split}.json")["values"]
    diagnostics = {}
    for arm in ARMS:
        readout = read(RUN / "readouts" / arm / f"{split}.json")["values"]
        diagnostics[arm] = response_metrics(readout, dense, bank)
    return diagnostics


def independent_spans():
    diagnostics = diagnostic_metrics("fresh")
    saved_report = read(REPORT)
    comparisons = {}
    for control in ("A", "Cross", "CrossMatched"):
        name = f"Multi-{control}"
        comparisons[name] = {}
        for metric in saved_report["fresh_diagnostic_span_comparisons"][name]:
            reference = [row[metric] for row in diagnostics[control]["sequences"]]
            candidate = [row[metric] for row in diagnostics["Multi"]["sequences"]]
            comparisons[name][metric] = span_statistics(reference, candidate)
    return diagnostics, comparisons


def verify_existing_closeout():
    sys.path.insert(0, str(WRITING))
    os.chdir(WRITING)
    import closeout
    closeout.verify_frozen()
    evaluator = closeout.task()
    mini = closeout.verify_mini(evaluator)
    full = closeout.verify_full(evaluator)
    return closeout, mini, full


def plot_primary(points):
    import matplotlib.pyplot as plt

    labels = [point["label"] for point in points]
    effects = np.asarray([point["effect_pp"] for point in points], dtype=float)
    lower = np.asarray([point["ci_low_pp"] for point in points], dtype=float)
    upper = np.asarray([point["ci_high_pp"] for point in points], dtype=float)
    y = np.arange(len(points))
    xerr = np.vstack((effects - lower, upper - effects))
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.86, bottom=0.27)
    ax.errorbar(effects, y, xerr=xerr, fmt="o", color="#174a7e", ecolor="#174a7e",
                elinewidth=2.2, capsize=5, markersize=7, linewidth=0)
    ax.axvline(0.0, color="#444444", linewidth=1.2, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Multi minus control effect (percentage points)")
    ax.set_title("Primary GSM8K contrasts: remaining frozen sample")
    fig.text(0.5, 0.105,
             "Primary remaining frozen sample: n=1,119 GSM8K questions; units=percentage points",
             ha="center", va="center", fontsize=8.5)
    fig.text(0.5, 0.052,
             "Intervals: paired 95% percentile bootstrap, unadjusted; seed=20260927; zero line shown",
             ha="center", va="center", fontsize=8.5)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "primary-effects.png", dpi=220)
    fig.savefig(HERE / "primary-effects.pdf")
    plt.close(fig)


def main() -> None:
    started = time.time()
    closeout, mini_verified, full_verified = verify_existing_closeout()
    config, requests, rows, exposed, primary, checkpoint_counts, row_hashes = load_full_rows()
    outcomes, scores, comparisons = independent_primary(config, rows, primary)
    saved_closeout = read(CLOSEOUT)
    saved_report = read(REPORT)
    for name, values in comparisons.items():
        closeout_saved = saved_closeout["full"]["comparisons"]["primary_remaining_1119"][name]
        compare_saved(f"closeout/full/comparisons/primary_remaining_1119/{name}",
                      {key: value for key, value in values.items() if key in closeout_saved},
                      closeout_saved)
        compare_saved(f"report/comparisons/primary_remaining_1119/{name}", values,
                      saved_report["comparisons"]["primary_remaining_1119"][name])
    for arm in ARMS:
        assert_equal(scores[arm], saved_closeout["full"]["scores"]["primary_remaining_1119"][arm],
                     f"closeout primary score {arm}")
        assert_equal(scores[arm], saved_report["scores"]["primary_remaining_1119"][arm],
                     f"report primary score {arm}")

    diagnostics, span_comparisons = independent_spans()
    for name, values in span_comparisons.items():
        for metric, result in values.items():
            compare_saved(f"report/fresh_diagnostic_span_comparisons/{name}/{metric}", result,
                          saved_report["fresh_diagnostic_span_comparisons"][name][metric])

    fresh_summary = {}
    for arm in ARMS:
        fresh = diagnostics[arm]["mean"]
        fresh_saved = read(RUN / "diagnostics" / arm / "fresh.json")["mean"]
        calibration = diagnostic_metrics("calibration")[arm]["mean"]
        calibration_saved = read(RUN / "diagnostics" / arm / "calibration.json")["mean"]
        fresh_values = {key: fresh[key] if key in fresh else fresh_saved[key] for key in SELECTED_DIAGNOSTICS}
        calibration_values = {key: calibration[key] if key in calibration else calibration_saved[key]
                              for key in SELECTED_DIAGNOSTICS}
        fresh_summary[arm] = {
            "fresh_mean": fresh_values,
            "calibration_mean": calibration_values,
            "fresh_minus_calibration": {key: fresh_values[key] - calibration_values[key]
                                         for key in SELECTED_DIAGNOSTICS},
            "fresh_sequence_count": len(diagnostics[arm]["sequences"]),
            "fresh_chain_count": len(read(RUN / "diagnostics" / arm / "fresh.json")["chains"]),
        }

    points = []
    for name in PRIMARY_FAMILY:
        result = comparisons[name]
        points.append({
            "label": name.replace("-", " − ", 1),
            "reference": name.split("-", 1)[1],
            "candidate": "Multi",
            "n": len(primary),
            "effect_pp": result["difference_pp"],
            "ci_low_pp": result["paired_bootstrap_95pp_unadjusted"][0],
            "ci_high_pp": result["paired_bootstrap_95pp_unadjusted"][1],
            "units": "percentage points",
            "interval": "paired percentile bootstrap 95%, unadjusted",
            "bootstrap_seed": result["bootstrap_seed"],
            "bootstrap_draws": result["bootstrap_draws"],
        })
    plot_primary(points)

    source_files = {
        "closeout_json": {"path": str(CLOSEOUT), "sha256": sha(CLOSEOUT)},
        "experiment_report_json": {"path": str(REPORT), "sha256": sha(REPORT)},
        "config_json": {"path": str(CONFIG), "sha256": sha(CONFIG)},
        "requests_json": {"path": str(REQUESTS), "sha256": sha(REQUESTS)},
        "allocation_json": {"path": str(RUN / "allocation.json"), "sha256": sha(RUN / "allocation.json")},
    }
    role_paths = {
        "original_experiment_role": Path("/home/tmluser1/.codex/attachments/07806206-ed65-4894-8b99-f0eb2a0d114d/experiment.md"),
        "project_experiment_role": WRITING / "roles/experiment.md",
        "contract_v1": WRITING / "contract-v1.json",
    }
    role_documents = {name: {"path": str(path), "sha256": sha(path)} for name, path in role_paths.items()}
    outcome_digests = {
        arm: {"primary_boolean_digest": digest(outcomes[arm]), "all_row_sha256_digest": row_hashes[arm]}
        for arm in ARMS
    }
    plotted_data = {
        "schema": "dlm-pruning-primary-effects-v1",
        "source": source_files,
        "sample": {
            "name": "primary_remaining_1119",
            "n": len(primary),
            "ids": primary,
            "ids_sha256": digest(primary),
            "definition": "config.json crosschain_control.primary_ids; exposed_ids removed",
        },
        "checkpoint_source": {
            "root": str(RUN / "gsm8k"),
            "path_pattern": "{arm}/shard{example_id//128:02d}/examples/{example_id:04d}.json",
            "counts": checkpoint_counts,
            "all_rows_verified_against_row_sha256": True,
            "outcome_digests": outcome_digests,
        },
        "statistic": {
            "estimand": "paired per-question correct indicator; candidate Multi minus control",
            "units": "percentage points",
            "interval": "paired percentile bootstrap 95%, unadjusted",
            "bootstrap_seed": 20260927,
            "bootstrap_draws": 10000,
            "zero_reference": 0.0,
        },
        "points": points,
    }
    (HERE / "plotted-data.json").write_text(json.dumps(plotted_data, indent=2, ensure_ascii=False) + "\n")

    verification = {
        "schema": "dlm-pruning-experiment-review-v1",
        "status": "passed",
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "role_documents_read": role_documents,
        "authority": "create-new under review_2026-09-27/experiment only; all audited sources read-only",
        "verification_routines": {
            "closeout.verify_frozen": "passed",
            "closeout.verify_mini": "passed",
            "closeout.verify_full": "passed",
            "official_cpu_grader": "invoked by verify_mini and verify_full",
        },
        "source_files": source_files,
        "primary_sample": {"n": len(primary), "exposed_n": len(exposed), "ids_sha256": digest(primary)},
        "independent_primary_scores": scores,
        "independent_primary_comparisons": comparisons,
        "saved_values_match": {
            "closeout_json": True,
            "experiment_report_json": True,
            "span_bootstrap_diagnostics": True,
        },
        "mini_verified": {
            "scores": mini_verified["scores"],
            "development": mini_verified["development"],
            "paired": mini_verified["paired"],
        },
        "historical_reproduction": full_verified["historical_reproduction"],
        "costs": full_verified["costs"],
        "fresh_fidelity": {
            "selected_metrics": list(SELECTED_DIAGNOSTICS),
            "arms": fresh_summary,
            "saved_span_uncertainty": span_comparisons,
            "interpretation": "Descriptive fresh-bank diagnostic fidelity only; span effects and calibration-to-fresh changes are not causal evidence.",
        },
        "limitations": full_verified["limitations"],
        "artifacts": {
            "verification_script": str(HERE / "verify_experiment.py"),
            "verification_json": str(HERE / "verification.json"),
            "report_md": str(HERE / "report.md"),
            "plotted_data_json": str(HERE / "plotted-data.json"),
            "primary_png": str(HERE / "primary-effects.png"),
            "primary_pdf": str(HERE / "primary-effects.pdf"),
        },
        "elapsed_seconds": time.time() - started,
    }
    (HERE / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": verification["status"],
        "primary_scores": scores,
        "primary_comparisons": comparisons,
        "artifacts": verification["artifacts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
