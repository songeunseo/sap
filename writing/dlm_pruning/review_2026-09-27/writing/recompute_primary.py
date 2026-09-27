#!/usr/bin/env python3
"""Recompute the frozen primary GSM8K contrasts from per-question JSON.

This audit reads prediction files and the frozen ID split directly. It does not
load a model, rerun grading, or modify experiment artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "experiments/dlm_crosschain_control50/output"
REPORT_PATH = RUN / "report.json"
CLOSEOUT_PATH = ROOT / "writing/dlm_pruning/closeout.json"
SPLIT_PATH = RUN / "request_split.json"
ARMS = ("A", "Multi", "Cross", "CrossMatched")
CONTRASTS = (("Multi-A", "A", "Multi"),
             ("Multi-Cross", "Cross", "Multi"),
             ("Multi-CrossMatched", "CrossMatched", "Multi"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(gain: int, loss: int) -> float:
    n = gain + loss
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(gain, loss) + 1)) / (2**n))


def paired_stats(reference: list[bool], candidate: list[bool]) -> dict:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("unpaired or empty outcomes")
    gain = sum(c and not r for r, c in zip(reference, candidate))
    loss = sum(r and not c for r, c in zip(reference, candidate))
    delta = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    rng = np.random.default_rng(20260927)
    samples = []
    for _ in range(20):
        indices = rng.integers(0, len(delta), size=(500, len(delta)))
        samples.extend(delta[indices].mean(axis=1))
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "n": len(reference),
        "gain": int(gain),
        "loss": int(loss),
        "net": int(gain - loss),
        "difference_pp": float(100 * delta.mean()),
        "exact_mcnemar_p": float(exact_mcnemar(gain, loss)),
        "holm_p": None,
        "paired_bootstrap_95pp_unadjusted": [float(100 * lo), float(100 * hi)],
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260927,
    }


def holm(results: dict[str, dict]) -> None:
    previous = 0.0
    ordered = sorted(results, key=lambda key: (results[key]["exact_mcnemar_p"], key))
    for i, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(results) - i) * results[key]["exact_mcnemar_p"]))
        results[key]["holm_p"] = float(previous)


def load_rows(arm: str) -> list[dict]:
    path = RUN / "gsm8k" / arm / "predictions.json"
    rows = json.loads(path.read_text())
    if [row["example_id"] for row in rows] != list(range(1319)):
        raise ValueError(f"{arm}: IDs are not the canonical 0..1318 sequence")
    if any(row.get("method") != arm for row in rows):
        raise ValueError(f"{arm}: method labels do not match the arm")
    return rows


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    closeout = json.loads(CLOSEOUT_PATH.read_text())
    split = json.loads(SPLIT_PATH.read_text())
    primary = split["primary_ids"]
    exposed = split["exposed_ids"]
    if len(primary) != 1119 or len(exposed) != 200 or set(primary) & set(exposed):
        raise ValueError("frozen sample split is invalid")

    rows = {arm: load_rows(arm) for arm in ARMS}
    outcomes = {arm: [bool(rows[arm][i]["correct"]) for i in primary] for arm in ARMS}
    scores = {arm: {"correct": int(sum(outcomes[arm])), "total": len(primary)} for arm in ARMS}
    comparisons = {name: paired_stats(outcomes[ref], outcomes[cand])
                   for name, ref, cand in CONTRASTS}
    holm(comparisons)

    # Keep a direct comparison against both persisted summaries. The result is
    # an audit record; failures are raised rather than silently accepted.
    saved_report = report["comparisons"]["primary_remaining_1119"]
    saved_closeout = closeout["full"]["comparisons"]["primary_remaining_1119"]
    checks = {}
    for name, result in comparisons.items():
        expected = saved_report[name]
        expected_closeout = saved_closeout[name]
        fields = ("gain", "loss", "net", "difference_pp", "exact_mcnemar_p",
                  "holm_p", "paired_bootstrap_95pp_unadjusted")
        checks[name] = {}
        for field in fields:
            got, a, b = result[field], expected[field], expected_closeout[field]
            if isinstance(got, list):
                ok = all(abs(x - y) <= 1e-10 for x, y in zip(got, a)) and all(abs(x - y) <= 1e-10 for x, y in zip(got, b))
            else:
                ok = abs(got - a) <= 1e-10 and abs(got - b) <= 1e-10
            if not ok:
                raise AssertionError(f"{name}/{field}: recomputed={got!r}, report={a!r}, closeout={b!r}")
            checks[name][field] = True

    payload = {
        "status": "passed",
        "purpose": "Independent primary-score and paired-effect audit from per-question JSON",
        "source_files": {
            "predictions": {arm: str(RUN / "gsm8k" / arm / "predictions.json") for arm in ARMS},
            "request_split": str(SPLIT_PATH),
            "source_report": str(REPORT_PATH),
            "closeout": str(CLOSEOUT_PATH),
        },
        "source_sha256": {
            "report": sha256(REPORT_PATH),
            "request_split": sha256(SPLIT_PATH),
            **{f"predictions_{arm}": sha256(RUN / "gsm8k" / arm / "predictions.json") for arm in ARMS},
        },
        "sample": {"primary_key": "request_split.json#/primary_ids", "n": len(primary),
                   "exposed_n": len(exposed), "primary_ids_sha256": hashlib.sha256(json.dumps(primary, separators=(",", ":")).encode()).hexdigest()},
        "scores_primary_remaining_1119": scores,
        "comparisons_primary_remaining_1119": comparisons,
        "matches_persisted_report_and_closeout": checks,
        "method": {
            "gain": "candidate correct and reference incorrect",
            "loss": "reference correct and candidate incorrect",
            "effect_pp": "100 * (gain - loss) / n",
            "exact_mcnemar": "two-sided exact binomial tail over gain+loss discordant pairs",
            "holm": "step-down adjustment over the three fixed contrasts",
            "bootstrap": "20 batches x 500 draws, np.random.default_rng(20260927), paired question resampling, unadjusted percentile 95% interval",
        },
    }
    out = Path(__file__).with_name("numeric-audit.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"scores": scores, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
