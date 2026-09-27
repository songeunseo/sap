"""CPU audit and paired analysis of the frozen exposed-200 Multi32 follow-up."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.dlm_multiscale_ac50.artifacts import digest, read, sha, write
from experiments.dlm_multiscale_ac50.evaluation import grade, task_and_protocol, validate_prediction
from .prepare import ROOT, validate, requests_for

CELLS = ("Multi_32", "A_32", "Multi_256", "A_256")
SEED = 20260927
DRAWS = 10000


def exact_mcnemar(first, second):
    if len(first) != len(second):
        raise ValueError("Paired arrays have different lengths")
    gain = sum(bool(a) and not bool(b) for a, b in zip(first, second))
    loss = sum(bool(b) and not bool(a) for a, b in zip(first, second))
    discordant = gain + loss
    tail = sum(math.comb(discordant, k) for k in range(min(gain, loss) + 1))
    return {"gain": gain, "loss": loss, "discordant": discordant,
            "both_correct": sum(bool(a) and bool(b) for a, b in zip(first, second)),
            "both_wrong": sum(not bool(a) and not bool(b) for a, b in zip(first, second)),
            "exact_p_two_sided": min(1.0, 2 * tail / 2**discordant) if discordant else 1.0}


def paired_bootstrap(values, draws=DRAWS, seed=SEED):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or draws < 1:
        raise ValueError("Expected a nonempty question vector and positive draws")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for j in range(draws):
        means[j] = values[rng.integers(0, len(values), size=len(values))].mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(ids, rows, draws=DRAWS, seed=SEED):
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("Expected exactly 200 distinct exposed question IDs")
    for cell in CELLS:
        if set(rows[cell]) != set(ids):
            raise ValueError(f"{cell}: paired ID coverage differs")
    n = len(ids)
    absolute = {cell: {"correct": sum(bool(rows[cell][i]["correct"]) for i in ids),
                       "accuracy": sum(bool(rows[cell][i]["correct"]) for i in ids) / n}
                for cell in CELLS}
    per_question = []
    for i in ids:
        correct = {cell: bool(rows[cell][i]["correct"]) for cell in CELLS}
        gap32 = int(correct["Multi_32"]) - int(correct["A_32"])
        gap256 = int(correct["Multi_256"]) - int(correct["A_256"])
        per_question.append({"example_id": i, "correct": correct,
                             "multi_minus_A_32": gap32, "multi_minus_A_256": gap256,
                             "interaction": gap32 - gap256})
    primary_values = [r["multi_minus_A_32"] for r in per_question]
    secondary_values = [r["interaction"] for r in per_question]
    primary = exact_mcnemar([rows["Multi_32"][i]["correct"] for i in ids],
                            [rows["A_32"][i]["correct"] for i in ids])
    primary.update(estimate=sum(primary_values) / n,
                   bootstrap_95=paired_bootstrap(primary_values, draws, seed))
    secondary = {"estimate": sum(secondary_values) / n,
                 "sum": sum(secondary_values),
                 "bootstrap_95": paired_bootstrap(secondary_values, draws, seed),
                 "distribution": {str(k): secondary_values.count(k) for k in range(-2, 3)}}
    return {"n": n, "absolute": absolute, "primary_multi32_minus_A32": primary,
            "secondary_interaction_32_minus_256": secondary,
            "bootstrap": {"unit": "GSM8K question ID", "draws": draws, "seed": seed},
            "per_question": per_question}


def read_cell(root, cfg, requests, cell, task):
    info = cfg["cells"][cell]
    folder = root / "gsm8k" / cell
    ids = [r["example_id"] for r in requests]
    identity = read(folder / "identity.json")
    if identity != {"fingerprint": info["fingerprint"], "document_ids": ids}:
        raise RuntimeError(f"{cell}: identity differs")
    filenames = {p.name for p in (folder / "examples").glob("*.json")}
    expected = {f"{i:04d}.json" for i in ids}
    if len(list((folder / "examples").glob("*.json"))) != len(ids) or filenames != expected:
        raise RuntimeError(f"{cell}: missing or noncanonical checkpoint filenames")
    arm, steps_text = cell.split("_")
    steps = int(steps_text)
    rows = {}
    for request in requests:
        i = request["example_id"]
        saved = read(folder / "examples" / f"{i:04d}.json")
        row = saved["row"]
        if saved["fingerprint"] != info["fingerprint"] or saved["row_sha256"] != digest(row):
            raise RuntimeError(f"{cell}/{i}: sealed checkpoint differs")
        validate_prediction(row, request, info["protocol_hash"])
        if row["method"] != arm or row.get("denoising_steps", 256) != steps:
            raise RuntimeError(f"{cell}/{i}: method or denoising steps differ")
        if grade(task, request, row["generated_text"]) != {
                "correct": row["correct"], "extracted_answer": row["extracted_answer"]}:
            raise RuntimeError(f"{cell}/{i}: official regrade differs")
        rows[i] = row
    predictions = folder / "predictions.json"
    if read(predictions) != [rows[i] for i in ids]:
        raise RuntimeError(f"{cell}: aggregate differs from checkpoints")
    result = read(folder / "results.json")
    if (result["status"] != "complete" or result["fingerprint"] != info["fingerprint"]
            or result["total"] != len(ids) or result["correct"] != sum(rows[i]["correct"] for i in ids)
            or result["predictions_sha256"] != sha(predictions)):
        raise RuntimeError(f"{cell}: result receipt differs")
    return rows


def validate_completion(root, cfg):
    receipt = read(root / "complete_Multi.json")
    model = read(root / "models/Multi.json")
    expected = cfg["identity"]
    for name, item in (("completion", receipt), ("physical model", model)):
        if (item["config_identity"] != cfg["config_identity"]
                or item["model_sha256"] != expected["model_sha256_Multi"]
                or item["mask_hash"] != expected["mask_hash"]):
            raise RuntimeError(f"Multi {name} identity differs")
    if (receipt["status"] != "complete" or model["pruned_count"] != expected["pruned_count"]
            or receipt["arm"] != "Multi" or model["arm"] != "Multi"):
        raise RuntimeError("Multi completion or physical prune differs")
    return {"worker_receipt": receipt, "physical_model": model}


def collect_costs(root):
    records = [read(p) for p in sorted((root / "attempts").glob("*.json"))]
    status = {}
    for row in records:
        status[row["status"]] = status.get(row["status"], 0) + 1
    return {"records": len(records), "forwards": sum(r.get("forwards", 0) for r in records),
            "forward_tokens": sum(r.get("forward_tokens", 0) for r in records),
            "seconds_attempt_sum": sum(r.get("seconds", 0) for r in records),
            "by_status": status, "scope": "all recorded attempts, including failed and retried"}


def validate_attempts(root, ids):
    """Require one recorded successful 32-forward run per selected question."""
    records = [read(p) for p in sorted((root / "attempts").glob("*.json"))]
    successful = [r for r in records if r.get("status") == "complete"]
    if (len(successful) != len(ids) or {r.get("example_id") for r in successful} != set(ids)
            or any(r.get("arm") != "Multi" or r.get("step") != 32
                   or r.get("forwards") != 32 for r in successful)):
        raise RuntimeError("Multi32 successful attempt coverage or forward count differs")
    if any(r.get("arm") != "Multi" or r.get("step") != 32 for r in records):
        raise RuntimeError("Attempt ledger contains another cell")
    return {"successful_questions": len(successful), "forwards_per_success": 32}


def analyze(root=ROOT):
    root = Path(root)
    cfg = validate(root / "config.json")
    requests = requests_for(cfg)
    ids = [r["example_id"] for r in requests]
    if ids != cfg["exposed_ids"]:
        raise RuntimeError("Selected requests differ from frozen exposed 200")
    source_cfg = read(cfg["source"]["config_path"])
    task, _, source_protocol_hash = task_and_protocol(source_cfg)
    for cell in CELLS:
        if int(cell.split("_")[1]) == 256 and cfg["cells"][cell]["protocol_hash"] != source_protocol_hash:
            raise RuntimeError(f"{cell}: source grading protocol differs")
    missing = [str(p.relative_to(root)) for p in
               [*(root / "gsm8k" / cell / "results.json" for cell in CELLS),
                root / "complete_Multi.json", root / "models/Multi.json"] if not p.exists()]
    if missing:
        return {"status": "incomplete", "sample": cfg["selection"], "missing": missing,
                "costs": collect_costs(root)}
    rows = {cell: read_cell(root, cfg, requests, cell, task) for cell in CELLS}
    completion = validate_completion(root, cfg)
    result = summarize(ids, rows, cfg["statistics"]["bootstrap_draws"],
                       cfg["statistics"]["bootstrap_seed"])
    result.update(status="complete", sample=cfg["selection"],
                  comparison_plan=cfg["statistics"], completion=completion,
                  costs=collect_costs(root), attempt_coverage=validate_attempts(root, ids),
                  provenance={"config_sha256": sha(root / "config.json"),
                              "code_receipt_sha256": sha(root / "code_receipt.json")},
                  interpretation_limits=[
                      "32 steps was chosen after observing the preceding Dense/A run; this is exploratory.",
                      "These 200 questions were previously exposed development data.",
                      "The interaction is descriptive; no multiplicity-adjusted discovery claim.",
                      "A confidence interval crossing zero does not establish equivalence."])
    if result["costs"]["records"] != completion["worker_receipt"]["attempts"]:
        raise RuntimeError("Completion attempt count differs from full attempt ledger")
    if result["costs"]["forwards"] != completion["worker_receipt"]["forwards"]:
        raise RuntimeError("Completion forward count differs from full attempt ledger")
    if result["costs"]["forward_tokens"] != completion["worker_receipt"]["forward_tokens"]:
        raise RuntimeError("Completion token count differs from full attempt ledger")
    return result


def markdown_report(report):
    a = report["absolute"]
    p = report["primary_multi32_minus_A32"]
    s = report["secondary_interaction_32_minus_256"]
    lines = ["# Multi32 exposed-200 follow-up", "", "Exploratory: 32 steps was selected after the preceding Dense/A results. The same previously exposed 200 questions were used.",
             "", "| Cell | Correct / 200 | Accuracy |", "|---|---:|---:|"]
    for cell in CELLS:
        lines.append(f"| {cell} | {a[cell]['correct']} | {100*a[cell]['accuracy']:.1f}% |")
    lines += ["", f"Primary Multi32 − A32: {100*p['estimate']:+.2f} percentage points; gains/losses {p['gain']}/{p['loss']}; two-sided exact McNemar p={p['exact_p_two_sided']:.6g}; question bootstrap 95% [{100*p['bootstrap_95'][0]:+.2f}, {100*p['bootstrap_95'][1]:+.2f}] points.",
              "", f"Descriptive interaction (Multi32 − A32) − (Multi256 − A256): {100*s['estimate']:+.2f} points; question bootstrap 95% [{100*s['bootstrap_95'][0]:+.2f}, {100*s['bootstrap_95'][1]:+.2f}] points.",
              "", f"Cost: {report['costs']['forwards']} recorded generation forwards across {report['costs']['records']} attempts, including retries and failures.",
              "", "The secondary comparison is descriptive. These exposed questions and the post-result choice of 32 steps do not support a confirmatory claim.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = analyze(args.root)
    except Exception as exc:
        result = {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}",
                  "costs": collect_costs(args.root)}
    write(args.root / "review/analysis.json", result)
    if result["status"] == "complete":
        write(args.root / "report.json", result)
        (args.root / "report.md").write_text(markdown_report(result))
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, indent=2))
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
