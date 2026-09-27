#!/usr/bin/env python3
"""Evaluate frozen Aggregate/Role allocations on the full 1,319 GSM8K set."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.dlm_capacity_predictor.audit_existing import strict_exact_match
from experiments.dlm_dual_role_mini100.core import validate_manifest
from experiments.dlm_dual_role_mini100.run import prepare as prepare_mini
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _read_jsonl,
    _write_jsonl,
    load_config as eval_config,
)
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, load_dense
from experiments.projection_capacity_followup_65.core import paired_binary_comparison
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest, sha, write_json
from experiments.wanda_failure_characterization.run_failure_map import model_sha

ROOT = Path("experiments/dlm_dual_role_full1319")
MINI_ROOT = Path("experiments/dlm_dual_role_mini100")
AUDIT = Path("experiments/dlm_capacity_predictor/existing_results_audit.json")
REFERENCE = Path("experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl")
METHODS = ("aggregate", "role")
LIMIT = 1319
TARGET = 4_536_008_704
WEIGHTS = 6_979_321_856


def event(kind, **values):
    print(json.dumps({"event": kind, **values}, sort_keys=True), flush=True)


def _write_frozen(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"frozen artifact changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def _historical():
    audit = json.loads(AUDIT.read_text())
    rows = {}
    for method in ("uniform", "eis_type"):
        row = audit["predictions"][method][str(LIMIT)]
        if not row["verified"] or row["status"] != "complete":
            raise RuntimeError(f"historical {method} is not verified")
        path = Path(row["path"])
        if sha(path) != row["sha256"]:
            raise RuntimeError(f"historical {method} predictions changed")
        rows[method] = row
    return rows


def prepare():
    mini = json.loads((MINI_ROOT / "mini100_results.json").read_text())
    if mini["status"] != "complete" or not mini["decision"]["primary_role_better_than_aggregate"]:
        raise RuntimeError("mini gate did not authorize full confirmation")
    preparation = prepare_mini()
    config_eval = eval_config("experiments/dlm_loss_aggregation/exp002/config.yaml")
    protocol_hash, protocol = _evaluation_config_hash(config_eval)
    historical = _historical()
    if any(row["evaluation_config_hash"] != protocol_hash for row in historical.values()):
        raise RuntimeError("historical protocol mismatch")
    manifests = {}
    for method in METHODS:
        path = MINI_ROOT / f"{method}65_mask_manifest.json"
        if sha(path) != preparation["allocations"][method]["manifest_sha256"]:
            raise RuntimeError(f"frozen {method} manifest changed")
        manifest = json.loads(path.read_text())
        validate_manifest(manifest, [row["name"] for row in manifest["entries"]], TARGET, WEIGHTS)
        manifests[method] = {"path": str(path), "sha256": sha(path)}
    config = {
        "status": "frozen_before_full_evaluation",
        "question": "Does frozen Role allocation beat Aggregate on full GSM8K?",
        "model": {"id": "GSAI-ML/LLaDA-8B-Base",
                  "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                  "dtype": "bfloat16"},
        "limit": LIMIT,
        "target_pruned": TARGET,
        "weights": WEIGHTS,
        "manifests": manifests,
        "mini_results_sha256": sha(MINI_ROOT / "mini100_results.json"),
        "audit_sha256": sha(AUDIT),
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "historical": historical,
        "primary": "paired strict exact match: role vs aggregate",
        "descriptive": ["role vs eis_type", "role vs uniform"],
        "no_retuning": True,
    }
    _write_frozen(ROOT / "config.json", config)
    event("full_prepared", config_sha256=sha(ROOT / "config.json"))
    return config


def _validate_rows(rows, reference, protocol_hash):
    if len(rows) != LIMIT:
        raise RuntimeError(f"full evaluation requires {LIMIT} rows")
    _assert_same_examples(reference[:LIMIT], rows)
    for index, row in enumerate(rows):
        if row.get("example_id") != index or row.get("evaluation_config_hash") != protocol_hash:
            raise RuntimeError("prediction identity or protocol mismatch")
        expected = strict_exact_match(row.get("extracted_answer"), row.get("reference_answer"))
        if expected != row.get("correct"):
            raise RuntimeError("stored strict exact match is inconsistent")


def _load_completed(method, config, reference):
    prediction = ROOT / "gsm8k" / f"{method}_1319_predictions.jsonl"
    receipt_path = ROOT / "gsm8k" / f"{method}_1319_predictions.receipt.json"
    if not prediction.exists() and not receipt_path.exists():
        return None
    if not prediction.exists() or not receipt_path.exists():
        raise RuntimeError(f"partial {method} output exists")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "config_sha256": sha(ROOT / "config.json"),
        "manifest_sha256": config["manifests"][method]["sha256"],
        "protocol_sha256": config["protocol_hash"],
        "limit": LIMIT,
    }
    if receipt.get("fingerprint") != expected or receipt.get("predictions_sha256") != sha(prediction):
        raise RuntimeError(f"{method} receipt mismatch")
    rows = _read_jsonl(prediction)
    _validate_rows(rows, reference, config["protocol_hash"])
    return rows


@torch.inference_mode()
def evaluate(method):
    if method not in METHODS:
        raise ValueError(method)
    config = prepare()
    reference = _read_jsonl(REFERENCE)
    rows = _load_completed(method, config, reference)
    if rows is None:
        manifest = json.loads(Path(config["manifests"][method]["path"]).read_text())
        model, mapping = load_dense()
        if model_sha(model) != DENSE_SHA:
            raise RuntimeError("wrong dense model")
        if apply_manifest(model, mapping, manifest) != TARGET:
            raise RuntimeError("applied mask budget mismatch")
        sparse_sha = model_sha(model)
        eval_cfg = eval_config("experiments/dlm_loss_aggregation/exp002/config.yaml")
        metrics, rows = _evaluate_gsm8k(
            model,
            AutoTokenizer.from_pretrained(eval_cfg["model"]["id"],
                                          revision=eval_cfg["model"]["revision"],
                                          trust_remote_code=True),
            eval_cfg, f"dual_role_{method}_65", LIMIT, config["protocol_hash"])
        if model_sha(model) != sparse_sha:
            raise RuntimeError("sparse model changed during evaluation")
        _validate_rows(rows, reference, config["protocol_hash"])
        prediction = ROOT / "gsm8k" / f"{method}_1319_predictions.jsonl"
        receipt_path = ROOT / "gsm8k" / f"{method}_1319_predictions.receipt.json"
        _write_jsonl(prediction, rows)
        fingerprint = {"config_sha256": sha(ROOT / "config.json"),
                       "manifest_sha256": config["manifests"][method]["sha256"],
                       "protocol_sha256": config["protocol_hash"], "limit": LIMIT}
        write_json(receipt_path, {"status": "complete", "fingerprint": fingerprint,
                                  "predictions_sha256": sha(prediction),
                                  "sparse_model_sha256": sparse_sha, "metrics": metrics})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    event("full_method_complete", method=method,
          correct=sum(bool(row["correct"]) for row in rows), total=LIMIT)
    maybe_finalize()


def maybe_finalize():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "finalize.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = prepare()
        reference = _read_jsonl(REFERENCE)
        methods = {method: _load_completed(method, config, reference) for method in METHODS}
        if any(rows is None for rows in methods.values()):
            event("finalize_pending")
            return None
        historical = {method: _read_jsonl(Path(config["historical"][method]["path"]))
                      for method in ("uniform", "eis_type")}
        for rows in historical.values():
            _validate_rows(rows, reference, config["protocol_hash"])
        correct = {method: sum(bool(row["correct"]) for row in rows)
                   for method, rows in {**methods, **historical}.items()}
        result = {
            "status": "complete", "limit": LIMIT,
            "config_sha256": sha(ROOT / "config.json"), "correct": correct,
            "accuracy": {method: value / LIMIT for method, value in correct.items()},
            "paired": {
                "role_vs_aggregate": paired_binary_comparison(
                    [r["correct"] for r in methods["aggregate"]],
                    [r["correct"] for r in methods["role"]]),
                "role_vs_eis_type": paired_binary_comparison(
                    [r["correct"] for r in historical["eis_type"]],
                    [r["correct"] for r in methods["role"]]),
                "role_vs_uniform": paired_binary_comparison(
                    [r["correct"] for r in historical["uniform"]],
                    [r["correct"] for r in methods["role"]]),
            },
            "decision": {
                "role_better_than_aggregate": correct["role"] > correct["aggregate"],
                "role_better_than_eis_type": correct["role"] > correct["eis_type"],
            },
            "interpretation_limit": "EIS+type is an oracle-derived descriptive control, not official cheap EIS",
        }
        write_json(ROOT / "full1319_results.json", result)
        event("full_complete", **correct)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate", "finalize"))
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "evaluate":
        if args.method is None:
            parser.error("evaluate requires --method")
        evaluate(args.method)
    else:
        maybe_finalize()


if __name__ == "__main__":
    main()
