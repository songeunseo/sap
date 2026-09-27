"""CPU-only final paired analysis of six frozen Dense/A decoding conditions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np

from experiments.dlm_multiscale_ac50.artifacts import digest, sha, write
from experiments.dlm_multiscale_ac50.evaluation import grade, task_and_protocol, validate_prediction
from .audit import ROOT, source_audit, read

CELLS = ("Dense_256", "A_256", "Dense_64", "A_64", "Dense_32", "A_32")


def exact_mcnemar(dense, sparse):
    """Two-sided exact binomial test of matched discordances."""
    if len(dense) != len(sparse):
        raise ValueError("Pair lengths differ")
    dense_only = sum(bool(d) and not bool(a) for d, a in zip(dense, sparse))
    sparse_only = sum(bool(a) and not bool(d) for d, a in zip(dense, sparse))
    n = dense_only + sparse_only
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(dense_only, sparse_only) + 1)) / 2**n) if n else 1.0
    return {"dense_only": dense_only, "A_only": sparse_only,
            "both_correct": sum(bool(d) and bool(a) for d, a in zip(dense, sparse)),
            "both_wrong": sum(not bool(d) and not bool(a) for d, a in zip(dense, sparse)),
            "exact_p": p}


def interaction_rows(ids, rows):
    """Pair by ID and retain every problem's contribution in accuracy units."""
    expected = set(ids)
    if len(ids) != len(expected):
        raise ValueError("Duplicate requested IDs")
    for cell in CELLS:
        if set(rows[cell]) != expected:
            raise ValueError(f"{cell}: missing or extra paired IDs")
    result = []
    for i in ids:
        correct = {cell: rows[cell][i]["correct"] for cell in CELLS}
        gaps = {str(step): int(correct[f"Dense_{step}"]) - int(correct[f"A_{step}"])
                for step in (256, 64, 32)}
        result.append({"example_id": i, "correct": correct, "gap": gaps,
                       "interaction_32": gaps["32"] - gaps["256"],
                       "interaction_64": gaps["64"] - gaps["256"]})
    return result


def paired_bootstrap(values, draws=10000, seed=20260927):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Expected nonempty question-level vector")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for j in range(draws):
        samples[j] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(v) for v in np.quantile(samples, (0.025, 0.975))]


def summarize(ids, rows, draws=10000, seed=20260927):
    paired = interaction_rows(ids, rows)
    n = len(ids)
    absolute = {cell: {"correct": sum(rows[cell][i]["correct"] for i in ids),
                       "accuracy": sum(rows[cell][i]["correct"] for i in ids) / n}
                for cell in CELLS}
    comparisons = {}
    for step in (256, 64, 32):
        test = exact_mcnemar([rows[f"Dense_{step}"][i]["correct"] for i in ids],
                             [rows[f"A_{step}"][i]["correct"] for i in ids])
        test["gap"] = (absolute[f"Dense_{step}"]["correct"] - absolute[f"A_{step}"]["correct"]) / n
        comparisons[str(step)] = test
    interactions = {}
    for step in (32, 64):
        key = f"interaction_{step}"
        vals = [p[key] for p in paired]
        interactions[str(step)] = {"estimate": statistics.mean(vals),
                                   "bootstrap_95": paired_bootstrap(vals, draws, seed),
                                   "sum": sum(vals), "distribution": {str(k): vals.count(k) for k in range(-2, 3)}}
    return {"n": n, "absolute": absolute, "paired_dense_minus_A": comparisons,
            "interaction_primary_32": interactions["32"], "interaction_secondary_64": interactions["64"],
            "bootstrap": {"unit": "GSM8K question ID", "draws": draws, "seed": seed},
            "per_question": paired}


def read_cell(root, cfg, requests, cell, task):
    from .prepare import cell_info
    arm, step_text = cell.split("_")
    step = int(step_text)
    info = cell_info(cfg, arm, step)
    folder = root / "gsm8k" / cell
    ids = [r["example_id"] for r in requests]
    identity = read(folder / "identity.json")
    if identity != {"fingerprint": info["fingerprint"], "document_ids": ids}:
        raise RuntimeError(f"{cell}: cell identity differs")
    checkpoint_files = list((folder / "examples").glob("*.json"))
    if len(checkpoint_files) != len(ids) or {p.name for p in checkpoint_files} != {f"{i:04d}.json" for i in ids}:
        raise RuntimeError(f"{cell}: missing/extra/ambiguous checkpoint filenames")
    rows = {}
    for request in requests:
        i = request["example_id"]
        path = folder / "examples" / f"{i:04d}.json"
        saved = read(path)
        if saved["fingerprint"] != info["fingerprint"] or saved["row_sha256"] != digest(saved["row"]):
            raise RuntimeError(f"{cell}: checkpoint hash/fingerprint differs at {i}")
        row = saved["row"]
        validate_prediction(row, request, info["protocol_hash"])
        if row["method"] != arm or (row.get("denoising_steps") != step if step != 256 else
                                      row.get("denoising_steps", 256) != 256):
            raise RuntimeError(f"{cell}: method or NFE differs at {i}")
        official = grade(task, request, row["generated_text"])
        if official != {"correct": row["correct"], "extracted_answer": row["extracted_answer"]}:
            raise RuntimeError(f"{cell}: official regrade differs at {i}")
        rows[i] = row
    predictions_path = folder / "predictions.json"
    if read(predictions_path) != [rows[i] for i in ids]:
        raise RuntimeError(f"{cell}: prediction aggregate differs from sealed checkpoints")
    result = read(folder / "results.json")
    if result["status"] != "complete" or result["fingerprint"] != info["fingerprint"] or result["total"] != len(ids) or result["correct"] != sum(rows[i]["correct"] for i in ids) or result["predictions_sha256"] != sha(predictions_path):
        raise RuntimeError(f"{cell}: result receipt differs")
    return rows


def collect_costs(root, ids):
    attempts = root / "attempts"
    if not attempts.exists():
        return {"records": 0, "forwards": 0, "forward_tokens": 0, "by_status": {}}
    counts = {}
    records = []
    for path in sorted(attempts.glob("*.json")):
        row = read(path)
        if row.get("example_id") not in ids:
            continue
        records.append(row)
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return {"records": len(records), "forwards": sum(x.get("forwards", 0) for x in records),
            "forward_tokens": sum(x.get("forward_tokens", 0) for x in records),
            "seconds_attempt_sum": sum(x.get("seconds", 0.0) for x in records),
            "by_status": counts,
            "by_cell": {cell: {"forwards": sum(x.get("forwards", 0) for x in records if f"{x.get('arm')}_{x.get('step')}" == cell),
                               "forward_tokens": sum(x.get("forward_tokens", 0) for x in records if f"{x.get('arm')}_{x.get('step')}" == cell)} for cell in CELLS}}



def validate_completion(root, cfg, rows):
    expected = {"Dense": cfg["identity"]["model_sha256_dense"],
                "A": cfg["identity"]["model_sha256_A"]}
    receipts = {}
    for arm in ("Dense", "A"):
        receipt = read(root / f"complete_{arm}.json")
        model = read(root / "models" / f"{arm}.json")
        if (receipt["status"] != "complete" or receipt["config_identity"] != cfg["config_identity"]
                or receipt["model_sha256"] != expected[arm]
                or model["config_identity"] != cfg["config_identity"]
                or model["model_sha256"] != expected[arm]):
            raise RuntimeError(f"{arm}: physical model or worker completion differs")
        if arm == "A" and (model["mask_hash"] != cfg["identity"]["mask_hash"]
                            or model["pruned_count"] != cfg["identity"]["pruned_count"]):
            raise RuntimeError("A physical prune identity differs")
        receipts[arm] = receipt
    parity = read(root / "instrumentation/Dense256_cache_parity.json")
    first = cfg["exposed_ids"][0]
    if (parity["example_id"] != first or parity["row_sha256"] != digest(rows["Dense_256"][first])
            or parity["generated_text_sha256"] != digest(rows["Dense_256"][first]["generated_text"])
            or parity["protocol_hash"] != cfg["cells"]["Dense_256"]["protocol_hash"]
            or parity["model_sha256"] != expected["Dense"] or parity["validation_forwards"] != 256):
        raise RuntimeError("Dense256 fresh cache parity receipt differs")
    hooked = read(root / "instrumentation/Dense32_parity.json")
    if (hooked["example_id"] not in cfg["exposed_ids"]
            or hooked["reference_text_sha256"] != hooked["hooked_text_sha256"]
            or hooked["hooked_text_sha256"] != digest(rows["Dense_32"][hooked["example_id"]]["generated_text"])
            or hooked["model_sha256"] != expected["Dense"]
            or hooked["protocol_hash"] != cfg["cells"]["Dense_32"]["protocol_hash"]
            or hooked["reference_forwards"] != 32):
        raise RuntimeError("Dense32 instrumentation parity receipt differs")
    return {"worker_receipts": receipts, "Dense256_cache_parity": parity,
            "Dense32_instrumentation_parity": hooked}


def diagnostic_summary(root, cfg, requests, draws=10000, seed=20260927):
    from .diagnostic import check_trace, teacher_paths, trace_path, sparse_path
    keys = ("dense_response_l1", "response_error_l1", "endpoint_kl_before",
            "endpoint_kl_after", "dense_confidence_before", "dense_mask_fraction",
            "response_mask_fraction")
    metric_keys = keys[:4]
    by_question = []
    all_steps = {step: [] for step in (0, 8, 16, 24)}
    bins = ((0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0))
    all_bins = {f"[{lo},{hi}{']' if hi == 1.0 else ')'}": [] for lo, hi in bins}
    for request in requests:
        i = request["example_id"]
        trace_file = trace_path(root, i)
        trace = check_trace(root, request, read(trace_file), cfg["config_identity"])
        tensor_file, receipt_file = teacher_paths(root, i)
        receipt = read(receipt_file)
        if (receipt["example_id"] != i or receipt["config_identity"] != cfg["config_identity"]
                or receipt["trace_sha256"] != sha(trace_file) or receipt["tensor_sha256"] != sha(tensor_file)
                or receipt["shape"][:2] != [4, 7]):
            raise RuntimeError(f"Dense diagnostic receipt differs at {i}")
        sparse = read(sparse_path(root, i))
        if (sparse["example_id"] != i or sparse["config_identity"] != cfg["config_identity"]
                or sparse["trace_sha256"] != receipt["trace_sha256"]
                or sparse["teacher_sha256"] != receipt["tensor_sha256"]
                or len(sparse["states"]) != 4):
            raise RuntimeError(f"A diagnostic identity or coverage differs at {i}")
        question_targets = []
        per_step = {}
        for state, source in zip(sparse["states"], trace["states"]):
            step = source["step"]
            if (state["step"] != step or state["source"] != source["source"]
                    or len(state["targets"]) != 7
                    or [t["position"] for t in state["targets"]] != [t["position"] for t in source["targets"]]):
                raise RuntimeError(f"Diagnostic target identity differs at {i}/{step}")
            for target in state["targets"]:
                for key in keys:
                    value = target[key]
                    if isinstance(value, bool) or not math.isfinite(value):
                        raise RuntimeError(f"Nonfinite diagnostic {key} at {i}/{step}")
                if (target["dense_response_l1"] < -1e-6 or target["response_error_l1"] < -1e-6
                        or not 0 <= target["dense_confidence_before"] <= 1
                        or not 0 <= target["dense_mask_fraction"] <= 1
                        or not 0 <= target["response_mask_fraction"] <= 1):
                    raise RuntimeError(f"Invalid diagnostic range at {i}/{step}")
            per_step[step] = state["targets"]
            question_targets.extend(state["targets"])
        recomputed = {key: statistics.mean(t[key] for t in question_targets) for key in keys}
        for key in keys:
            if not math.isclose(recomputed[key], sparse["question_means"][key], rel_tol=1e-8, abs_tol=1e-8):
                raise RuntimeError(f"Question diagnostic mean differs at {i}: {key}")
        by_question.append({"example_id": i, **recomputed})
        for step, targets in per_step.items():
            all_steps[step].append({"example_id": i, "target_count": len(targets),
                                    **{key: statistics.mean(t[key] for t in targets) for key in keys}})
        for lo, hi in bins:
            label = f"[{lo},{hi}{']' if hi == 1.0 else ')'}"
            group = [t for t in question_targets if
                     lo <= t["dense_confidence_before"] <= hi] if hi == 1.0 else [
                     t for t in question_targets if lo <= t["dense_confidence_before"] < hi]
            if group:
                all_bins[label].append({"example_id": i, "target_count": len(group),
                                        **{key: statistics.mean(t[key] for t in group) for key in keys}})
    def aggregate(items, with_ci=False):
        if not items:
            return {"questions": 0, "targets": 0, "mean": None}
        answer = {"questions": len(items), "targets": sum(x.get("target_count", 28) for x in items),
                  "mean": {key: statistics.mean(x[key] for x in items) for key in keys}}
        if with_ci:
            answer["question_bootstrap_95"] = {key: paired_bootstrap([x[key] for x in items], draws, seed)
                                               for key in metric_keys}
        return answer
    if len(by_question) != 200 or sum(len(x) for x in all_steps.values()) != 800:
        raise RuntimeError("Diagnostic has fewer than 200 questions or 800 states")
    return {"status": "complete", "question_count": 200, "state_count": 800,
            "target_count": 5600, "bootstrap": {"unit": "GSM8K question", "draws": draws, "seed": seed},
            "overall": aggregate(by_question, with_ci=True),
            "by_step": {str(step): aggregate(items) for step, items in all_steps.items()},
            "by_dense_confidence": {label: aggregate(items) for label, items in all_bins.items()},
            "per_question": by_question,
            "interpretation": "Shared fixed-state full-distribution response and endpoint KL; not NELBO or causal downstream damage."}

def analyze(root=ROOT):
    root = Path(root)
    historical = source_audit()
    from .prepare import validate, requests_for
    cfg = validate(root / "config.json")
    requests = requests_for(cfg)
    ids = [r["example_id"] for r in requests]
    if ids != historical["ids"]:
        raise RuntimeError("Selected requests differ from frozen exposed 200")
    source_cfg = read(cfg["source"]["config_path"])
    task, _, protocol = task_and_protocol(source_cfg)
    if protocol != historical["protocol_hash"]:
        raise RuntimeError("Grading protocol changed")
    missing = [cell for cell in CELLS if not (root / "gsm8k" / cell / "results.json").exists()]
    pending = [str(path.relative_to(root)) for path in
               (root / "complete_Dense.json", root / "complete_A.json",
                root / "instrumentation/Dense256_cache_parity.json",
                root / "instrumentation/Dense32_parity.json") if not path.exists()]
    for i in ids:
        for relative in (f"traces/{i:04d}.json", f"diagnostic/dense/{i:04d}.json",
                         f"diagnostic/dense/{i:04d}.pt", f"diagnostic/A/{i:04d}.json"):
            if not (root / relative).exists():
                pending.append(relative)
    if missing or pending:
        return {"status": "incomplete", "sample": "exploratory exposed 200", "missing_cells": missing,
                "pending_artifacts_count": len(pending), "pending_artifacts_first": pending[:12],
                "source_audit": {k: v for k, v in historical.items() if k != "ids"}}
    rows = {cell: read_cell(root, cfg, requests, cell, task) for cell in CELLS}
    completion = validate_completion(root, cfg, rows)
    diagnostic = diagnostic_summary(root, cfg, requests)
    result = summarize(ids, rows)
    result.update(status="complete", sample="exploratory exposed 200", source_audit={k: v for k, v in historical.items() if k != "ids"},
                  completion=completion, diagnostic=diagnostic,
                  costs=collect_costs(root, set(ids)),
                  interpretation_limits=["The exposed 200 questions are development data.",
                    "A bootstrap interval crossing zero is inconclusive, not evidence of equivalent gaps.",
                    "Low absolute accuracies at 32 steps can compress the Dense-A gap (floor effect).",
                    "Fixed-state response KL is functional fidelity, not NELBO or a causal accuracy effect."])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = analyze(args.root)
    output = args.root / "review" / "analysis.json"
    write(output, report)
    if report["status"] == "complete":
        write(args.root / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_question", "diagnostic")}, indent=2))
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
