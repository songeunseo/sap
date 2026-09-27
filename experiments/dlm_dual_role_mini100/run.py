#!/usr/bin/env python3
"""Prepare and evaluate full-pooled Aggregate/Role allocations on GSM8K mini-100."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.dlm_dual_role_allocation.analyze import _checkpoint_arrays, _pool_fields
from experiments.dlm_dual_role_allocation.core import allocation_mask_xor
from experiments.dlm_dual_role_allocation.io import load_frozen_inputs
from experiments.dlm_dual_role_mini100.core import (
    mini_decision,
    same_selected_masks,
    validate_manifest,
)
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _read_jsonl,
    _write_jsonl,
    load_config as eval_config,
)
from experiments.dlm_capacity_predictor.audit_existing import strict_exact_match
from experiments.projection_capacity_allocation_65.core import GRID, allocate
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, load_dense
from experiments.projection_capacity_followup_65.core import (
    build_selected_manifest,
    paired_binary_comparison,
)
from experiments.projection_capacity_followup_65.run_heldout import (
    apply_manifest,
    selected_mask,
    sha,
    write_json,
)
from experiments.wanda_failure_characterization.run_failure_map import model_sha

ROOT = Path("experiments/dlm_dual_role_mini100")
ROLE_ROOT = Path("experiments/dlm_dual_role_allocation")
CAPACITY_ROOT = Path("experiments/projection_capacity_allocation_65")
FOLLOWUP_ROOT = Path("experiments/projection_capacity_followup_65")
METHODS = ("aggregate", "role")
TARGET = 4_536_008_704
WEIGHTS = 6_979_321_856


def event(kind, **values):
    print(json.dumps({"event": kind, **values}, sort_keys=True), flush=True)


def _write_frozen(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"frozen artifact changed: {path}")
        return
    write_json(path, value)


def prepare() -> dict:
    inputs = load_frozen_inputs()
    collected = _checkpoint_arrays(inputs, ROLE_ROOT)
    pooled = _pool_fields(collected["fields"], list(range(inputs.metadata["state_count"])))
    shapes = [row["shape"] for row in inputs.candidate["entries"]]
    allocations = {method: allocate(pooled[method], shapes) for method in METHODS}
    for method, allocation in allocations.items():
        if allocation["pruned"] != TARGET or allocation["budget_error"] != 0:
            raise RuntimeError(f"{method} allocation missed exact target")

    config = {
        "status": "frozen_before_gsm8k_mini",
        "question": "Does full-pooled max(masked,unmasked) allocation beat identically pooled aggregate reconstruction on frozen GSM8K mini-100?",
        "model": inputs.config["model"],
        "source_receipt_sha256": inputs.receipt["receipt_sha256"],
        "source_artifacts": {
            str(ROLE_ROOT / "role_reconstruction_raw.json"): sha(ROLE_ROOT / "role_reconstruction_raw.json"),
            str(ROLE_ROOT / "collection_manifest.json"): sha(ROLE_ROOT / "collection_manifest.json"),
            str(CAPACITY_ROOT / "candidate_mask_manifest.json"): sha(CAPACITY_ROOT / "candidate_mask_manifest.json"),
        },
        "pooling": "ratio of numerator sums to denominator sums over all frozen 80 states",
        "methods": {
            "aggregate": "(sum num_M + sum num_U)/(sum den_M + sum den_U)",
            "role": "max(sum num_M/sum den_M, sum num_U/sum den_U)",
        },
        "allocation": "raw 5pp marginal cost per nominal additional parameter; repository-order ties; exact row-floor budget",
        "grid": list(GRID),
        "target_pruned": TARGET,
        "weights": WEIGHTS,
        "evaluation": "historical frozen GSM8K mini-100, 5-shot, 256 steps, strict exact match",
        "primary": "role correct > aggregate correct",
        "secondary": "role correct > historical uniform correct",
        "full_policy": "keep for separate full-GSM8K plan only if primary is true",
        "no_retuning": True,
    }
    _write_frozen(ROOT / "config.json", config)
    config_sha = sha(ROOT / "config.json")

    manifests = {}
    for method in METHODS:
        manifest = build_selected_manifest(
            f"dual_role_{method}", inputs.candidate["entries"],
            allocations[method]["sparsities"], GRID
        )
        manifest.update(
            config_sha256=config_sha,
            source_receipt_sha256=inputs.receipt["receipt_sha256"],
            allocation_levels=allocations[method]["levels"],
        )
        validate_manifest(manifest, inputs.metadata["module_names"], TARGET, WEIGHTS)
        path = ROOT / f"{method}65_mask_manifest.json"
        _write_frozen(path, manifest)
        manifests[method] = manifest

    xor = allocation_mask_xor(
        allocations["aggregate"]["levels"], allocations["role"]["levels"],
        inputs.candidate["entries"]
    )
    old_reconstruction = json.loads(
        (FOLLOWUP_ROOT / "reconstruction65_mask_manifest.json").read_text()
    )
    preparation = {
        "status": "frozen",
        "config_sha256": config_sha,
        "allocations": {
            method: {
                "pruned": allocations[method]["pruned"],
                "sparsities": allocations[method]["sparsities"],
                "level_counts": {
                    str(level): allocations[method]["sparsities"].count(level) for level in GRID
                },
                "manifest": str(ROOT / f"{method}65_mask_manifest.json"),
                "manifest_sha256": sha(ROOT / f"{method}65_mask_manifest.json"),
            } for method in METHODS
        },
        "role_vs_aggregate": xor,
        "aggregate_matches_historical_reconstruction": same_selected_masks(
            manifests["aggregate"], old_reconstruction
        ),
        "aggregate_changed_projections_vs_historical_reconstruction": sum(
            left != right for left, right in zip(
                [row["selected_mask"]["mask_sha256"] for row in manifests["aggregate"]["entries"]],
                [row["selected_mask"]["mask_sha256"] for row in old_reconstruction["entries"]],
            )
        ),
    }
    _write_frozen(ROOT / "preparation.json", preparation)
    event("preparation_complete", xor_fraction=xor["xor_fraction_of_prunable_weights"],
          changed_projections=xor["changed_projection_count"],
          aggregate_reusable=preparation["aggregate_matches_historical_reconstruction"])
    return preparation


def _validated_manifests() -> tuple[dict, dict]:
    preparation = prepare()
    inputs = load_frozen_inputs()
    manifests = {}
    for method in METHODS:
        path = ROOT / f"{method}65_mask_manifest.json"
        if sha(path) != preparation["allocations"][method]["manifest_sha256"]:
            raise RuntimeError(f"{method} manifest changed after freeze")
        manifest = json.loads(path.read_text())
        validate_manifest(manifest, inputs.metadata["module_names"], TARGET, WEIGHTS)
        manifests[method] = manifest
    return preparation, manifests


def _validate_rows(rows: list[dict], reference: list[dict], protocol_hash: str) -> None:
    if len(rows) != 100:
        raise RuntimeError("mini evaluation must contain exactly 100 rows")
    _assert_same_examples(reference[:100], rows)
    for index, row in enumerate(rows):
        if row.get("example_id") != index or row.get("evaluation_config_hash") != protocol_hash:
            raise RuntimeError("prediction identity or protocol mismatch")
        if strict_exact_match(row.get("extracted_answer"), row.get("reference_answer")) != row.get("correct"):
            raise RuntimeError("stored strict exact match is inconsistent")


def _load_completed_predictions(path: Path, receipt_path: Path, expected: dict,
                                reference: list[dict]) -> list[dict] | None:
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.exists() or not receipt_path.exists():
        raise RuntimeError("partial prediction artifact exists")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("fingerprint") != expected or receipt.get("predictions_sha256") != sha(path):
        raise RuntimeError("prediction receipt mismatch")
    rows = _read_jsonl(path)
    _validate_rows(rows, reference, expected["protocol_sha256"])
    return rows


@torch.inference_mode()
def evaluate() -> dict:
    preparation, manifests = _validated_manifests()
    config = eval_config("experiments/dlm_loss_aggregation/exp002/config.yaml")
    protocol_hash, protocol = _evaluation_config_hash(config)
    historical = json.loads((CAPACITY_ROOT / "downstream.json").read_text())
    if historical["protocol_hash"] != protocol_hash:
        raise RuntimeError("historical GSM8K protocol mismatch")
    historical_mini = next(row for row in historical["evaluations"]
                           if int(row["uniform"]["limit"]) == 100)
    uniform_rows = _read_jsonl(Path(historical_mini["uniform"]["predictions"]))
    reference = _read_jsonl(
        Path("experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl")
    )
    _assert_same_examples(reference[:100], uniform_rows)

    outcomes = {}
    for method in METHODS:
        manifest_path = ROOT / f"{method}65_mask_manifest.json"
        prediction_path = ROOT / "gsm8k" / f"{method}_100_predictions.jsonl"
        receipt_path = ROOT / "gsm8k" / f"{method}_100_predictions.receipt.json"
        fingerprint = {
            "config_sha256": sha(ROOT / "config.json"),
            "manifest_sha256": sha(manifest_path),
            "protocol_sha256": protocol_hash,
            "limit": 100,
        }
        rows = _load_completed_predictions(prediction_path, receipt_path,
                                           fingerprint, reference)
        if rows is None:
            model, mapping = load_dense()
            if model_sha(model) != DENSE_SHA:
                raise RuntimeError("wrong dense model before mask application")
            applied = apply_manifest(model, mapping, manifests[method])
            if applied != TARGET:
                raise RuntimeError("applied mask budget mismatch")
            sparse_sha = model_sha(model)
            metrics, rows = _evaluate_gsm8k(
                model, AutoTokenizer.from_pretrained(
                    config["model"]["id"], revision=config["model"]["revision"],
                    trust_remote_code=True
                ), config, f"dual_role_{method}_65", 100, protocol_hash
            )
            if model_sha(model) != sparse_sha:
                raise RuntimeError("sparse model changed during mini evaluation")
            _validate_rows(rows, reference, protocol_hash)
            _write_jsonl(prediction_path, rows)
            write_json(receipt_path, {
                "status": "complete", "fingerprint": fingerprint,
                "predictions_sha256": sha(prediction_path), "sparse_model_sha256": sparse_sha,
                "metrics": metrics,
            })
            del model
            gc.collect()
            torch.cuda.empty_cache()
        else:
            metrics = json.loads(receipt_path.read_text()).get("metrics", {"reused": True})
        outcomes[method] = {
            "correct": int(sum(bool(row["correct"]) for row in rows)),
            "predictions": str(prediction_path),
            "predictions_sha256": sha(prediction_path),
            "metrics": metrics,
            "rows": rows,
        }
        event("mini_method_complete", method=method,
              correct=outcomes[method]["correct"], total=100)

    decision = mini_decision(outcomes["aggregate"]["correct"],
                             outcomes["role"]["correct"],
                             int(historical_mini["uniform"]["correct"]))
    result = {
        "status": "complete",
        "config_sha256": sha(ROOT / "config.json"),
        "preparation_sha256": sha(ROOT / "preparation.json"),
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "uniform": {
            "correct": int(historical_mini["uniform"]["correct"]),
            "predictions": historical_mini["uniform"]["predictions"],
            "predictions_sha256": historical_mini["uniform"]["sha256"],
        },
        "methods": {
            method: {key: value for key, value in outcomes[method].items() if key != "rows"}
            for method in METHODS
        },
        "paired": {
            "role_vs_aggregate": paired_binary_comparison(
                [row["correct"] for row in outcomes["aggregate"]["rows"]],
                [row["correct"] for row in outcomes["role"]["rows"]],
            ),
            "role_vs_uniform": paired_binary_comparison(
                [row["correct"] for row in uniform_rows],
                [row["correct"] for row in outcomes["role"]["rows"]],
            ),
        },
        "decision": decision,
        "interpretation_limit": "exploratory mini-100 screen; full superiority is not established",
    }
    result_path = ROOT / "mini100_results.json"
    if result_path.exists() and json.loads(result_path.read_text()) != result:
        raise RuntimeError("completed mini result differs on rerun")
    write_json(result_path, result)
    event("mini_complete", decision=decision["decision"],
          aggregate=decision["aggregate_correct"], role=decision["role_correct"],
          uniform=decision["uniform_correct"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate", "all"))
    args = parser.parse_args()
    if args.phase in ("prepare", "all"):
        prepare()
    if args.phase in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
