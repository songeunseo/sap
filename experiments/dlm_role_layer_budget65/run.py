#!/usr/bin/env python3
"""Prepare, evaluate, and finalize the layer-budget-controlled mini-100 test."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.dlm_capacity_predictor.audit_existing import strict_exact_match
from experiments.dlm_dual_role_allocation.analyze import _checkpoint_arrays, _pool_fields
from experiments.dlm_dual_role_allocation.io import atomic_write_json, load_frozen_inputs
from experiments.dlm_dual_role_mini100.core import validate_manifest
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _read_jsonl,
    _write_jsonl,
    load_config as eval_config,
)
from experiments.dlm_role_layer_budget65.core import (
    allocate_by_layer,
    layer_pruned_counts,
)
from experiments.projection_capacity_allocation_65.core import GRID
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, load_dense
from experiments.projection_capacity_followup_65.core import (
    build_selected_manifest,
    paired_binary_comparison,
)
from experiments.projection_capacity_followup_65.run_heldout import (
    apply_manifest,
    sha,
)
from experiments.wanda_failure_characterization.run_failure_map import model_sha

ROOT = Path("experiments/dlm_role_layer_budget65")
ROLE_ROOT = Path("experiments/dlm_dual_role_allocation")
CAPACITY_ROOT = Path("experiments/projection_capacity_allocation_65")
FOLLOWUP_ROOT = Path("experiments/projection_capacity_followup_65")
EIS_MANIFEST = FOLLOWUP_ROOT / "eis_type65_mask_manifest.json"
EIS_PREDICTIONS = FOLLOWUP_ROOT / "gsm8k/eis_type_100_predictions.jsonl"
METHODS = ("layer_aggregate", "layer_role")
TARGET = 4_536_008_704
WEIGHTS = 6_979_321_856


def event(kind: str, **values) -> None:
    print(json.dumps({"event": kind, **values}, sort_keys=True), flush=True)


def write_frozen(path: Path, payload: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"frozen artifact changed: {path}")
        return
    atomic_write_json(path, payload)


def validate_rows(rows: list[dict], reference: list[dict], protocol_hash: str) -> None:
    if len(rows) != 100:
        raise RuntimeError("expected exactly 100 GSM8K rows")
    _assert_same_examples(reference[:100], rows)
    for index, row in enumerate(rows):
        if row.get("example_id") != index or row.get("evaluation_config_hash") != protocol_hash:
            raise RuntimeError("prediction identity or protocol mismatch")
        if strict_exact_match(row.get("extracted_answer"), row.get("reference_answer")) != row.get("correct"):
            raise RuntimeError("stored strict exact match mismatch")


def prepare() -> dict:
    inputs = load_frozen_inputs()
    fields = _checkpoint_arrays(inputs, ROLE_ROOT)["fields"]
    pooled = _pool_fields(fields, list(range(inputs.metadata["state_count"])))
    entries = inputs.candidate["entries"]
    names = inputs.metadata["module_names"]
    shapes = [tuple(row["shape"]) for row in entries]

    eis = json.loads(EIS_MANIFEST.read_text())
    validate_manifest(eis, names, TARGET, WEIGHTS)
    targets = layer_pruned_counts(eis)
    allocations = {
        "layer_aggregate": allocate_by_layer(pooled["aggregate"], shapes, names, targets),
        "layer_role": allocate_by_layer(pooled["role"], shapes, names, targets),
    }
    for method, allocation in allocations.items():
        if allocation["pruned"] != TARGET:
            raise RuntimeError(f"{method} global budget mismatch")

    config = {
        "status": "frozen_before_downstream",
        "question": "Does masked/unmasked Role information improve within-layer projection allocation after freezing every EIS+type layer budget?",
        "model": inputs.config["model"],
        "grid": list(GRID),
        "projection_count": 224,
        "target_pruned": TARGET,
        "weights": WEIGHTS,
        "baseline": "oracle-derived multiset-sorted EIS+type descriptive control; not official cheap EIS",
        "constraint": "all 32 selected-mask pruned counts exactly equal the frozen EIS+type layer counts",
        "methods": {
            "layer_aggregate": "pooled reconstruction level curves; historical raw marginal-cost greedy independently within each layer",
            "layer_role": "max(masked,unmasked) reconstruction level curves; same greedy independently within each layer",
        },
        "local_selector": "frozen Standard Wanda candidate masks; ranking unchanged",
        "evaluation": "historical fixed GSM8K mini-100, 5-shot, 256 steps, strict EM",
        "primary": "layer_role vs layer_aggregate paired comparison",
        "secondary": "each candidate vs frozen EIS+type",
        "full_policy": "no automatic full evaluation or tuning",
        "source_hashes": {
            str(EIS_MANIFEST): sha(EIS_MANIFEST),
            str(CAPACITY_ROOT / "candidate_mask_manifest.json"): sha(CAPACITY_ROOT / "candidate_mask_manifest.json"),
            str(ROLE_ROOT / "role_reconstruction_raw.json"): sha(ROLE_ROOT / "role_reconstruction_raw.json"),
            str(ROLE_ROOT / "collection_manifest.json"): sha(ROLE_ROOT / "collection_manifest.json"),
        },
    }
    write_frozen(ROOT / "config.json", config)
    config_hash = sha(ROOT / "config.json")

    manifests = {}
    for method, allocation in allocations.items():
        manifest = build_selected_manifest(method, entries, allocation["sparsities"], GRID)
        manifest.update(config_sha256=config_hash,
                        allocation_levels=allocation["levels"],
                        layer_pruned=allocation["layer_targets"])
        validate_manifest(manifest, names, TARGET, WEIGHTS)
        if layer_pruned_counts(manifest) != targets:
            raise RuntimeError(f"{method} did not preserve every layer budget")
        path = ROOT / f"{method}_mask_manifest.json"
        write_frozen(path, manifest)
        manifests[method] = manifest

    def comparison(left: dict, right: dict) -> dict:
        left_levels = [int(row["level"]) for row in left["entries"]]
        right_levels = [int(row["level"]) for row in right["entries"]]
        moved = sum(abs(int(a["selected_mask"]["pruned"])-int(b["selected_mask"]["pruned"]))
                    for a, b in zip(left["entries"], right["entries"])) // 2
        return {
            "changed_projections": sum(a != b for a, b in zip(left_levels, right_levels)),
            "parameter_count_moved_each_direction": moved,
        }

    preparation = {
        "status": "frozen",
        "config_sha256": config_hash,
        "eis_manifest_sha256": sha(EIS_MANIFEST),
        "layer_targets": {str(k): v for k, v in sorted(targets.items())},
        "allocations": {
            method: {
                "manifest": str(ROOT / f"{method}_mask_manifest.json"),
                "manifest_sha256": sha(ROOT / f"{method}_mask_manifest.json"),
                "pruned": allocation["pruned"],
                "level_counts": {str(rate): allocation["sparsities"].count(rate) for rate in GRID},
                "layers": allocation["layers"],
            }
            for method, allocation in allocations.items()
        },
        "comparisons": {
            "layer_role_vs_layer_aggregate": comparison(manifests["layer_aggregate"], manifests["layer_role"]),
            "layer_aggregate_vs_eis": comparison(eis, manifests["layer_aggregate"]),
            "layer_role_vs_eis": comparison(eis, manifests["layer_role"]),
        },
    }
    write_frozen(ROOT / "preparation.json", preparation)
    event("preparation_complete", comparisons=preparation["comparisons"])
    return preparation


def load_manifest(method: str) -> tuple[dict, dict]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    preparation = prepare()
    inputs = load_frozen_inputs()
    path = ROOT / f"{method}_mask_manifest.json"
    if sha(path) != preparation["allocations"][method]["manifest_sha256"]:
        raise RuntimeError("manifest changed after freeze")
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, inputs.metadata["module_names"], TARGET, WEIGHTS)
    return preparation, manifest


@torch.inference_mode()
def evaluate(method: str) -> dict:
    preparation, manifest = load_manifest(method)
    config = eval_config("experiments/dlm_loss_aggregation/exp002/config.yaml")
    protocol_hash, _ = _evaluation_config_hash(config)
    reference = _read_jsonl(Path("experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl"))
    prediction_path = ROOT / "gsm8k" / f"{method}_100_predictions.jsonl"
    receipt_path = ROOT / "gsm8k" / f"{method}_100_predictions.receipt.json"
    fingerprint = {
        "config_sha256": preparation["config_sha256"],
        "manifest_sha256": sha(ROOT / f"{method}_mask_manifest.json"),
        "protocol_sha256": protocol_hash,
        "limit": 100,
    }
    if prediction_path.exists() or receipt_path.exists():
        if not prediction_path.exists() or not receipt_path.exists():
            raise RuntimeError("partial evaluation artifacts exist")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("fingerprint") != fingerprint or receipt.get("predictions_sha256") != sha(prediction_path):
            raise RuntimeError("completed receipt mismatch")
        rows = _read_jsonl(prediction_path)
        validate_rows(rows, reference, protocol_hash)
        event("evaluation_reused", method=method, correct=sum(bool(row["correct"]) for row in rows))
        return receipt

    model, mapping = load_dense()
    if model_sha(model) != DENSE_SHA:
        raise RuntimeError("wrong dense model before mask application")
    if apply_manifest(model, mapping, manifest) != TARGET:
        raise RuntimeError("applied mask count mismatch")
    sparse_hash = model_sha(model)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True
    )
    metrics, rows = _evaluate_gsm8k(
        model, tokenizer, config, f"{method}_65", 100, protocol_hash
    )
    if model_sha(model) != sparse_hash:
        raise RuntimeError("sparse model changed during evaluation")
    validate_rows(rows, reference, protocol_hash)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(prediction_path, rows)
    receipt = {
        "status": "complete",
        "method": method,
        "fingerprint": fingerprint,
        "predictions_sha256": sha(prediction_path),
        "sparse_model_sha256": sparse_hash,
        "correct": int(sum(bool(row["correct"]) for row in rows)),
        "metrics": metrics,
    }
    atomic_write_json(receipt_path, receipt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    event("evaluation_complete", method=method, correct=receipt["correct"], total=100)
    return receipt


def finalize() -> dict:
    preparation = prepare()
    config = eval_config("experiments/dlm_loss_aggregation/exp002/config.yaml")
    protocol_hash, protocol = _evaluation_config_hash(config)
    reference = _read_jsonl(Path("experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl"))
    eis_rows = _read_jsonl(EIS_PREDICTIONS)
    validate_rows(eis_rows, reference, protocol_hash)
    rows = {}
    methods = {}
    for method in METHODS:
        path = ROOT / "gsm8k" / f"{method}_100_predictions.jsonl"
        receipt_path = ROOT / "gsm8k" / f"{method}_100_predictions.receipt.json"
        if not path.exists() or not receipt_path.exists():
            raise RuntimeError(f"{method} evaluation is incomplete")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("predictions_sha256") != sha(path):
            raise RuntimeError(f"{method} receipt mismatch")
        rows[method] = _read_jsonl(path)
        validate_rows(rows[method], reference, protocol_hash)
        methods[method] = {
            "correct": int(sum(bool(row["correct"]) for row in rows[method])),
            "predictions": str(path),
            "predictions_sha256": sha(path),
            "manifest_sha256": preparation["allocations"][method]["manifest_sha256"],
        }
    eis_correct = int(sum(bool(row["correct"]) for row in eis_rows))
    comparisons = {
        "layer_role_vs_layer_aggregate": paired_binary_comparison(
            [row["correct"] for row in rows["layer_aggregate"]],
            [row["correct"] for row in rows["layer_role"]],
        ),
        "layer_aggregate_vs_eis": paired_binary_comparison(
            [row["correct"] for row in eis_rows],
            [row["correct"] for row in rows["layer_aggregate"]],
        ),
        "layer_role_vs_eis": paired_binary_comparison(
            [row["correct"] for row in eis_rows],
            [row["correct"] for row in rows["layer_role"]],
        ),
    }
    result = {
        "status": "complete",
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "eis_type": {"correct": eis_correct, "predictions": str(EIS_PREDICTIONS),
                     "predictions_sha256": sha(EIS_PREDICTIONS),
                     "manifest_sha256": sha(EIS_MANIFEST)},
        "methods": methods,
        "paired": comparisons,
        "decision": {
            "primary_role_strictly_above_aggregate": methods["layer_role"]["correct"] > methods["layer_aggregate"]["correct"],
            "within_layer_nonuniform_above_eis": max(methods[m]["correct"] for m in METHODS) > eis_correct,
            "automatic_full": False,
        },
        "interpretation_limit": "repeated-use development mini-100; no generalization or causal-role claim",
    }
    write_frozen(ROOT / "mini100_results.json", result)
    event("finalized", eis=eis_correct,
          layer_aggregate=methods["layer_aggregate"]["correct"],
          layer_role=methods["layer_role"]["correct"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "evaluate", "finalize"))
    parser.add_argument("method", nargs="?", choices=METHODS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "evaluate":
        if args.method is None:
            parser.error("evaluate requires a method")
        evaluate(args.method)
    else:
        finalize()


if __name__ == "__main__":
    main()
