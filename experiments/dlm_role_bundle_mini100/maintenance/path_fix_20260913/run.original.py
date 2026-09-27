"""Frozen 12-hybrid mini100 counterfactual audit; no search or allocation fitting."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.dlm_capacity_predictor.audit_existing import audit_prediction_file, sha256, IDENTITY_FIELDS, strict_exact_match
from experiments.dlm_role_bundle_mini100.core import METHODS, make_hybrids, signature, summarize

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
OLD = REPO / "experiments/dlm_dual_role_mini100"
AUDIT = REPO / "experiments/dlm_role_gain_attribution"
EVAL = REPO / "experiments/dlm_loss_aggregation/exp002/config.yaml"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if frozen and path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"frozen file changed: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def event(method, stage, **kw):
    row = dict(method=method, stage=stage, timestamp=time.time(), pid=os.getpid(), **kw)
    write(ROOT / method / "progress.json", row)
    print(json.dumps(row), flush=True)


def validate_rows(rows, ref, protocol):
    if len(rows) != 100:
        raise RuntimeError("exact mini100 required")
    for i, (x, y) in enumerate(zip(rows, ref, strict=True)):
        if x["example_id"] != i or x["evaluation_config_hash"] != protocol:
            raise RuntimeError("example order/protocol mismatch")
        if any(x[k] != y[k] for k in IDENTITY_FIELDS):
            raise RuntimeError("historical mini100 identity mismatch")
        if strict_exact_match(x["extracted_answer"], x["reference_answer"]) != x["correct"]:
            raise RuntimeError("strict EM mismatch")


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def freeze():
    from experiments.dlm_loss_aggregation.exp002.run import load_config, _evaluation_config_hash
    from experiments.dlm_dual_role_mini100.core import validate_manifest
    if (ROOT / "config.json").exists():
        return validate()
    sources = {}
    def source(path):
        path = Path(path)
        sources[str(path)] = sha256(path)
        return read(path)
    prior = source(AUDIT / "allocation_audit.json")
    receipt = source(AUDIT / "receipt.json")
    if receipt["status"] != "complete" or receipt["outputs"]["allocation_audit.json"] != sha256(AUDIT / "allocation_audit.json"):
        raise RuntimeError("offline bundle audit invalid")
    oldcfg = source(OLD / "config.json")
    fullcfg = source(REPO / "experiments/dlm_dual_role_full1319/config.json")
    cfg = load_config(EVAL)
    ph, protocol = _evaluation_config_hash(cfg)
    if ph != fullcfg["protocol_hash"]:
        raise RuntimeError("historical eval protocol changed")
    manifests, baselines, estimates = {}, {}, []
    for method in ("aggregate", "role"):
        path = OLD / f"{method}65_mask_manifest.json"
        m = source(path)
        if sha256(path) != fullcfg["manifests"][method]["sha256"]:
            raise RuntimeError("original manifest changed")
        validate_manifest(m, [e["name"] for e in m["entries"]], oldcfg["target_pruned"], oldcfg["weights"])
        manifests[method] = m
        pred = OLD / f"gsm8k/{method}_100_predictions.jsonl"
        audit = audit_prediction_file(pred, 100)
        rec = source(pred.with_suffix(".receipt.json"))
        fp = rec["fingerprint"]
        if (rec["status"] != "complete" or rec["predictions_sha256"] != audit["sha256"] or
                fp["config_sha256"] != sha256(OLD / "config.json") or fp["manifest_sha256"] != sha256(path) or
                fp["protocol_sha256"] != ph or audit["evaluation_config_hash"] != ph):
            raise RuntimeError("baseline provenance mismatch")
        sources[str(pred)] = audit["sha256"]
        baselines[method] = str(pred)
        estimates.append(rec["metrics"]["eval_seconds"])
    validate_rows(rows(baselines["role"]), rows(baselines["aggregate"]), ph)
    hybrids = make_hybrids(manifests["aggregate"], manifests["role"], prior["bundles"])
    if set(hybrids) != set(METHODS):
        raise RuntimeError("exactly six frozen bundles / twelve hybrids required")
    known = [signature(m) for m in manifests.values()]
    hybrid_paths = {}
    for method in METHODS:
        m = hybrids[method]
        sig = signature(m)
        if sig in known:
            raise RuntimeError("duplicate/baseline hybrid")
        known.append(sig)
        path = ROOT / method / "mask_manifest.json"
        write(path, m, frozen=True)
        sources[str(path)] = sha256(path)
        hybrid_paths[method] = str(path)
    for path in (EVAL, REPO / "experiments/dlm_loss_aggregation/config.yaml", REPO / "eval_llada.py", REPO / "generate.py",
                 REPO / "experiments/dlm_loss_aggregation/exp002/run.py", REPO / "experiments/dlm_loss_aggregation/run.py",
                 REPO / "experiments/projection_capacity_allocation_65/run.py",
                 REPO / "experiments/projection_capacity_followup_65/run_heldout.py",
                 REPO / "experiments/wanda_failure_characterization/run_failure_map.py",
                 REPO / "experiments/dlm_loss_aggregation/core.py",
                 REPO / "experiments/dlm_capacity_predictor/audit_existing.py",
                 ROOT / "run.py", ROOT / "core.py", ROOT / "status.py", ROOT / "run.sh"):
        sources[str(path)] = sha256(path)
    config = dict(status="frozen_before_hybrid_generation", created=time.time(), source_sha256=sources,
        model=oldcfg["model"], target_pruned=oldcfg["target_pruned"], weights=oldcfg["weights"],
        protocol=protocol, protocol_hash=ph, limit=100, methods=list(METHODS), manifests=hybrid_paths,
        baselines=baselines, bundles=prior["bundles"],
        queues={"0": [m for m in METHODS if m.startswith("add_")], "1": [m for m in METHODS if m.startswith("revert_")]},
        historical_eval_seconds_per100=sum(estimates)/len(estimates), preparation_allowance_seconds=120,
        statistics="12 paired exact McNemar tests with Holm family12; paired example bootstrap20000 seed20260913; six interaction CIs unadjusted exploratory",
        scope="attribution only; no candidate selection, full evaluation, role-path causal claim, or retuning")
    write(ROOT / "config.json", config, frozen=True)
    print(json.dumps({"stage": "frozen", "hybrids": 12, "config_sha256": sha256(ROOT / "config.json")}), flush=True)
    return config


def validate():
    c = read(ROOT / "config.json")
    for p, expected in c["source_sha256"].items():
        if sha256(Path(p)) != expected:
            raise RuntimeError(f"frozen input/code changed: {p}")
    return c


def completed(method, c):
    dest = ROOT / method
    if not (dest / "results.json").exists():
        if (dest / "predictions.jsonl").exists():
            raise RuntimeError("partial predictions without receipt; preserve and investigate")
        return None
    result = read(dest / "results.json")
    if (result["status"] != "complete" or result["config_sha256"] != sha256(ROOT / "config.json") or
            result["predictions_sha256"] != sha256(dest / "predictions.jsonl") or
            result["manifest_sha256"] != sha256(c["manifests"][method]) or result["protocol_hash"] != c["protocol_hash"]):
        raise RuntimeError("completed receipt mismatch")
    data = rows(dest / "predictions.jsonl")
    validate_rows(data, rows(c["baselines"]["aggregate"]), c["protocol_hash"])
    if result["correct"] != sum(r["correct"] for r in data):
        raise RuntimeError("receipt score mismatch")
    return data


def evaluate(method):
    import torch
    from transformers import AutoTokenizer
    from experiments.dlm_loss_aggregation.exp002.run import _evaluate_gsm8k, _evaluation_config_hash, _write_jsonl, load_config
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    dest = ROOT / method
    with (dest / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            event(method, "preflight")
            c = validate()
            if completed(method, c) is not None:
                event(method, "complete", reused_completed=True, correct=read(dest / "results.json")["correct"])
                return
            event(method, "loading_dense")
            cfg = load_config(EVAL)
            ph, _ = _evaluation_config_hash(cfg)
            if ph != c["protocol_hash"]:
                raise RuntimeError("runtime protocol mismatch")
            with torch.inference_mode():
                model, mapping = load_dense()
                event(method, "applying_masks")
                manifest = read(c["manifests"][method])
                if apply_manifest(model, mapping, manifest) != c["target_pruned"]:
                    raise RuntimeError("applied exact budget mismatch")
                sparse_hash = model_sha(model)
                tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"], trust_remote_code=True)
                event(method, "gsm8k", completed=0, total=100, gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
                metrics, data = _evaluate_gsm8k(model, tokenizer, cfg, method, 100, ph)
                event(method, "verifying", completed=100, total=100)
                if model_sha(model) != sparse_hash:
                    raise RuntimeError("evaluation mutated sparse weights")
            validate_rows(data, rows(c["baselines"]["aggregate"]), ph)
            validate()
            _write_jsonl(dest / "predictions.jsonl", data)
            result = dict(status="complete", method=method, correct=sum(r["correct"] for r in data), total=100,
                config_sha256=sha256(ROOT / "config.json"), manifest_sha256=sha256(c["manifests"][method]), protocol_hash=ph,
                predictions_sha256=sha256(dest / "predictions.jsonl"), sparse_model_sha256=sparse_hash, metrics=metrics,
                global_pruned=c["target_pruned"], gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
            write(dest / "results.json", result, frozen=True)
            event(method, "complete", correct=result["correct"], total=100)
        except BaseException as e:
            event(method, "failed", error=f"{type(e).__name__}: {e}")
            raise


def finalize():
    with (ROOT / "finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        c = validate()
        candidates = {m: completed(m, c) for m in METHODS}
        if any(v is None for v in candidates.values()):
            return
        result = summarize({m: rows(p) for m, p in c["baselines"].items()}, candidates)
        result.update(status="complete", config_sha256=sha256(ROOT / "config.json"),
            prediction_sha256={m: sha256(ROOT / m / "predictions.jsonl") for m in METHODS})
        write(ROOT / "results.json", result, frozen=True)
        lines = ["# Role–Aggregate exact-budget bundle mini100", "", "## Setup", "",
            "Frozen six bundles, two backgrounds, 12 hybrids. Same mini100/5shot/256 steps/strict EM, original Wanda masks, equal exact global budget. Baselines Aggregate19, Role24. No downstream-driven allocation selection.", "",
            "## Results", "", "Benefit sign: add=hybrid−Aggregate; revert=Role−hybrid. Positive means Role-side bundle helps in that background.", "",
            "| Hybrid | Correct /100 | Role-side benefit (questions) | raw p | Holm p (12) |", "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            p = result["pairs"][method]
            lines.append(f"| {method} | {p['hybrid_correct']} | {p['net_correct_a_minus_b']:+d} | {p['exact_mcnemar_p']:.5f} | {p['holm_p_family12']:.5f} |")
        lines += ["", "## Background interaction", "", "Role-side benefit in Role background minus benefit in Aggregate background. CI is unadjusted and exploratory, not a multiplicity-controlled claim.", ""]
        for key, value in result["background_interactions"].items():
            lines.append(f"- B{key}: {value}")
        lines += ["", "## Interpretation / Decision", "", "Conditional bundle effects are measured, not individual-module/semantic-role mechanisms. Mini100 cannot establish full1319 attribution. No automatic best-hybrid selection or full launch. Inspect paired records before interpreting; retain role-separation premise.", ""]
        path = ROOT / "report.md"
        text = "\n".join(lines)
        if path.exists() and path.read_text() != text:
            raise RuntimeError("completed report changed")
        path.write_text(text)


def worker(gpu):
    with (ROOT / f"worker{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = validate()
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
            raise RuntimeError("worker/GPU binding mismatch")
        for method in c["queues"][str(gpu)]:
            event(f"worker{gpu}", "running", current=method)
            log = ROOT / "logs" / f"{method}.log"
            log.parent.mkdir(exist_ok=True)
            with log.open("a") as f:
                proc = subprocess.run([sys.executable, "-u", "-m", "experiments.dlm_role_bundle_mini100.run", "evaluate", "--method", method], stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
            if proc.returncode:
                event(f"worker{gpu}", "failed", current=method, returncode=proc.returncode)
                raise RuntimeError(f"{method} failed; queue stopped, log preserved")
            finalize()
        event(f"worker{gpu}", "complete")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("freeze", "evaluate", "worker", "finalize"))
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--gpu", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.phase == "freeze": freeze()
    elif args.phase == "finalize": finalize()
    elif args.phase == "worker":
        if args.gpu is None: p.error("--gpu required")
        worker(args.gpu)
    else:
        if args.method is None: p.error("--method required")
        evaluate(args.method)


if __name__ == "__main__":
    main()
