"""CPU allocation, paired comparisons, confirmation selection, and reports."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import ARMS, checked, digest, freeze, read, sha, write
from .core import aliases, holm, marginal_cost, metrics, paired, rank_rates
from .evaluation import read_predictions
from .prepare import validate


def allocate(root):
    from experiments.dlm_owl65.core import exact_row_counts
    root = Path(root)
    config = validate(root)
    bank = read(config["banks"]["calibration"]["path"])
    teacher = read(root / "readouts/dense/calibration.json")["values"]
    refs = read(config["legacy_manifests"]["uniform"]["path"])["entries"]
    costs, per_sequence, sources = [], [], {}
    for block in range(32):
        path = root / "probes" / f"block{block:02d}.json"
        row = read(path)
        if row["config_sha256"] != sha(root / "config.json") or row["block"] != block:
            raise RuntimeError("Wrong probe identity")
        low, high = [row["conditions"][str(rate)] for rate in config["pruning"]["probe_rates"]]
        for condition in (low, high):
            checked(condition["readout_path"], condition["readout_sha256"])
            recomputed = metrics(read(condition["readout_path"])["values"], teacher, bank)
            if recomputed != condition["metrics"]:
                raise RuntimeError("Probe metrics differ from actual saved outputs")
        cost = marginal_cost(low["metrics"], high["metrics"], low["pruned"], high["pruned"])
        if cost != row["costs"]:
            raise RuntimeError("Marginal cost mismatch")
        costs.append(cost)
        per_sequence.append({arm:[(h["metrics"][arm]-l["metrics"][arm])/(high["pruned"]-low["pruned"])
                                  for l, h in zip(low["metrics"]["sequences"], high["metrics"]["sequences"], strict=True)]
                             for arm in ARMS})
        sources[str(path)] = sha(path)
    allocations = {}
    for arm in ARMS:
        scores = [c[arm] for c in costs]
        rates = rank_rates(scores)
        counts, budget = exact_row_counts(refs, rates, config["pruning"]["pruned"])
        allocations[arm] = dict(scores=scores, rates=rates.tolist(), row_counts=counts, budget=budget)
    result = dict(config_sha256=sha(root / "config.json"), allocations=allocations,
                  aliases=aliases(allocations), probe_sources=sources)
    freeze(root / "allocation.json", result)
    # Uncertainty is over eight spans, not over all correlated edges/nodes.
    rng = np.random.default_rng(20260924)
    draws = rng.integers(0, 8, size=(200, 8))
    stability = {}
    for arm in ARMS:
        values = np.asarray([r[arm] for r in per_sequence])
        draws_rates = np.asarray([rank_rates(values[:, draw].mean(axis=1)) for draw in draws])
        stability[arm] = dict(ideal_rate_sd=draws_rates.std(axis=0).tolist(),
                             ideal_rate_q05=np.quantile(draws_rates, .05, axis=0).tolist(),
                             ideal_rate_q95=np.quantile(draws_rates, .95, axis=0).tolist())
    freeze(root / "allocation_stability.json", dict(seed=20260924, draws=200, units=8,
           resampling_unit="clean_span", stage="ideal rates before exact rounding", arms=stability))
    return result


def load_result(root, split, method, config, requests):
    folder = root / "gsm8k" / split / method
    result = read(folder / "results.json")
    identity = read(root / "candidates" / method / "model_identity.json")
    expected = dict(config_sha256=sha(root / "config.json"), requests_sha256=sha(root / "requests.json"),
                    mask_identity=identity["mask_identity"], sparse_model_sha256=identity["sparse_model_sha256"],
                    split=split, protocol_hash=config["protocol_hash"])
    if identity["config_sha256"] != expected["config_sha256"]:
        raise RuntimeError("Candidate identity config mismatch")
    if split == "confirmation":
        expected["selection_sha256"] = sha(root / "confirmation_selection.json")
    if result["fingerprint"] != expected:
        raise RuntimeError("Result fingerprint mismatch")
    checked(folder / "predictions.json", result["predictions_sha256"])
    rows = read_predictions(folder, requests, expected, config["protocol_hash"])
    if rows != read(folder / "predictions.json") or result["correct"] != sum(r["correct"] for r in rows):
        raise RuntimeError("Aggregate predictions differ from individual checkpoints")
    return rows


def summarize(root, split="development"):
    root = Path(root)
    config = validate(root)
    req = read(root / "requests.json")[split]
    if split == "development":
        allocation = read(root / "allocation.json")
        if allocation["config_sha256"] != sha(root / "config.json"):
            raise RuntimeError("Allocation config changed")
        for path, expected in allocation["probe_sources"].items():
            checked(path, expected)
        rows = {arm:load_result(root, split, allocation["aliases"][arm], config, req) for arm in ARMS}
        cached = read(root / "cached_development.json")
        rows.update({"legacy_"+name:values for name, values in cached.items()})
        scores = {arm:sum(r["correct"] for r in values) for arm, values in rows.items()}
        controls = config["plan"]["gsm8k"]["best_simple_tie_break"]
        best = max(controls, key=lambda arm:scores[arm])
        comparisons = holm({arm:paired([r["correct"] for r in rows[arm]], [r["correct"] for r in rows["Multi"]])
                            for arm in controls})
        gate = scores["Multi"] > scores[best] and scores["Multi"] >= scores["legacy_AC"]
        result = dict(status="development_complete", config_sha256=sha(root / "config.json"),
            scores=scores, primary="Multi_vs_Short", multi_vs_simple=comparisons,
            multi_vs_legacy_AC=paired([r["correct"] for r in rows["legacy_AC"]], [r["correct"] for r in rows["Multi"]]),
            best_simple=best, confirmation_recommended=gate, aliases=allocation["aliases"],
            weak_gap=0 < scores["Multi"]-scores[best] <= 2,
            interpretation="Reused development set; confirmation is a separate explicit phase, never automatic.")
    else:
        selection = read(root / "confirmation_selection.json")
        rows = {arm:load_result(root, split, target, config, req) for arm, target in selection["aliases"].items()}
        result = dict(status="confirmation_complete", config_sha256=sha(root / "config.json"),
            selection_sha256=sha(root / "confirmation_selection.json"),
            scores={arm:sum(r["correct"] for r in values) for arm, values in rows.items()},
            comparisons=holm({arm:paired([r["correct"] for r in values], [r["correct"] for r in rows["Multi"]])
                              for arm, values in rows.items() if arm != "Multi"}),
            interpretation="Held out from current method selection; prior project exposure not ruled out.")
    freeze(root / f"{split}_summary.json", result)
    lines = [f"# Multiscale A+C — {split}", "", "| Method | Correct / 100 |", "|---|---:|"]
    lines.extend(f"| {arm} | {score} |" for arm, score in result["scores"].items())
    lines.extend(["", result["interpretation"], "", "```json", __import__("json").dumps(result, indent=2), "```", ""])
    (root / f"{split}_report.md").write_text("\n".join(lines))
    diagnostics_report(root, config)
    return result


def diagnostics_report(root, config):
    paths = sorted((root / "diagnostics").glob("*/*.json"))
    table = ["# Response diagnostics", "", "Query CE is not NELBO/PPL. Sign flips exclude |dense response| ≤ 1e-6.", "",
             "| Candidate | Bank | A | C1 | C2 | C4 | Path C | All C | Query CE | Sign flip |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for path in paths:
        if path.parent.name.startswith("probe_"):
            continue
        row = read(path)
        if row["config_sha256"] != sha(root / "config.json"):
            raise RuntimeError("Diagnostic config mismatch")
        checked(root / "readouts" / path.parent.name / path.name, row["readout_sha256"])
        checked(root / "readouts/dense" / path.name, row["teacher_sha256"])
        keys = ("A", "C1", "C2", "C4", "C_path", "C_all", "query_CE", "response_sign_flip_rate")
        table.append(f"| {path.parent.name} | {path.stem} | " + " | ".join(f"{row['mean'][k]:.6g}" for k in keys) + " |")
    (root / "diagnostics_report.md").write_text("\n".join(table)+"\n")


def select_confirmation(root):
    root = Path(root)
    summary = summarize(root, "development")
    if not summary["confirmation_recommended"]:
        raise RuntimeError("Predeclared confirmation gate was not met; inspect development_report.md")
    labels = ["Multi", summary["best_simple"], "legacy_AC"]
    selected, mapping, identities = [], {}, {}
    for label in labels:
        canonical = summary["aliases"].get(label, label)
        identity = read(root / "candidates" / canonical / "model_identity.json")
        key = (identity["mask_identity"], identity["sparse_model_sha256"])
        if key not in identities:
            identities[key] = canonical
            selected.append(canonical)
        mapping[label] = identities[key]
    result = dict(config_sha256=sha(root / "config.json"), development_summary_sha256=sha(root / "development_summary.json"),
                  aliases=mapping, canonical_methods=selected, fixed_before_confirmation=True)
    freeze(root / "confirmation_selection.json", result)
    return result
