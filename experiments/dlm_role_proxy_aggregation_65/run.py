#!/usr/bin/env python3
"""Build and mini-screen role damage proxy x aggregation allocations at 65%."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.dlm_capacity_predictor.audit_existing import strict_exact_match
from experiments.dlm_dual_role_allocation.io import atomic_write_json, file_sha256, load_frozen_inputs
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples, _evaluate_gsm8k, _evaluation_config_hash,
    _read_jsonl, _write_jsonl, load_config as eval_config,
)
from experiments.dlm_role_proxy_aggregation_65.core import (
    AGGREGATIONS, aggregate_role_curves, allocation_difference,
    diagonal_reconstruction_curve, selected_signature,
)
from experiments.dlm_role_validation.core import GRIDS, TARGET_INDEX, allocate_grid
from experiments.dlm_role_validation.run import load_dense
from experiments.dlm_dual_role_mini100.core import validate_manifest
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, read_mask
from experiments.projection_capacity_followup_65.core import build_selected_manifest, paired_binary_comparison
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
from experiments.wanda_failure_characterization.run_failure_map import model_sha


ROOT = Path("experiments/dlm_role_proxy_aggregation_65")
DATA_ROOT = Path("/DATA/tmluser1/sap-dlm-role-proxy-aggregation-65")
ROLE_RAW = Path("experiments/dlm_dual_role_allocation/role_reconstruction_raw.json")
ROLE_CURVES = Path("experiments/dlm_role_validation/curves_random65.json")
OLD_MINI = Path("experiments/dlm_dual_role_mini100/mini100_results.json")
OLD_PREP = Path("experiments/dlm_dual_role_mini100/preparation.json")
OLD_ROLE_MANIFEST = Path("experiments/dlm_dual_role_mini100/role65_mask_manifest.json")
OLD_AGG_MANIFEST = Path("experiments/dlm_dual_role_mini100/aggregate65_mask_manifest.json")
EVAL_CONFIG = Path("experiments/dlm_loss_aggregation/exp002/config.yaml")
GRID = GRIDS["random65"]
TARGET = 4_536_008_704
WEIGHTS = 6_979_321_856
PROXIES = ("reconstruction", "diagonal")


def event(kind: str, **values) -> None:
    print(json.dumps({"event": kind, "time": time.time(), **values}, sort_keys=True), flush=True)


def sha(path: str | Path) -> str:
    return file_sha256(path)


def save_tensor(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def freeze() -> dict:
    inputs = load_frozen_inputs()
    protocol_hash, protocol = _evaluation_config_hash(eval_config(EVAL_CONFIG))
    config = {
        "status": "frozen_before_new_collection_or_downstream_evaluation",
        "question": "Which role aggregation and damage proxy yields the best fixed-budget projection allocation?",
        "model": inputs.config["model"],
        "dense_model_sha256": DENSE_SHA,
        "projection_count": 224,
        "state_count": 80,
        "state_digest": inputs.metadata["state_digest"],
        "selector": "unchanged exact historical Standard Wanda candidate masks",
        "target_sparsity": .65,
        "grid": list(GRID),
        "target_pruned": TARGET,
        "weights": WEIGHTS,
        "damage_proxies": {
            "reconstruction": "measured role-normalized local squared output reconstruction error",
            "diagonal": "sum_j(sum_i removed_W_ij^2 * sum_role x_j^2) / role dense-output energy",
        },
        "aggregations": {
            "balanced": "Delta[(E_M+E_U)/2]",
            "max_level": "Delta[max(E_M,E_U)]",
            "max_marginal": "max(Delta E_M, Delta E_U)",
        },
        "allocation": "raw adjacent 5pp cost per nominal additional parameter; exact row-floor 65% budget",
        "evaluation": {"dataset": "historical fixed GSM8K mini-100", "protocol_hash": protocol_hash,
                       "protocol": protocol, "primary": "strict exact match"},
        "screening": "deduplicate exact selected-mask signatures; evaluate only novel representatives",
        "references": {"uniform": 12, "aggregate": 19, "reconstruction_max_level": 24},
        "source_hashes": {
            str(ROLE_RAW): sha(ROLE_RAW), str(ROLE_CURVES): sha(ROLE_CURVES),
            str(OLD_MINI): sha(OLD_MINI), str(OLD_PREP): sha(OLD_PREP),
            str(OLD_ROLE_MANIFEST): sha(OLD_ROLE_MANIFEST),
            str(OLD_AGG_MANIFEST): sha(OLD_AGG_MANIFEST), str(EVAL_CONFIG): sha(EVAL_CONFIG),
            "experiments/projection_capacity_allocation_65/candidate_mask_manifest.json":
                sha("experiments/projection_capacity_allocation_65/candidate_mask_manifest.json"),
        },
        "no_clipping_smoothing_or_retuning": True,
    }
    path = ROOT / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("frozen config differs")
    if not path.exists():
        atomic_write_json(path, config)
    event("freeze_complete", config_sha256=sha(path))
    return config


def require_frozen() -> dict:
    config = freeze()
    for path, digest in config["source_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"frozen source changed: {path}")
    return config


class RoleMomentAccumulator:
    def __init__(self):
        self.masked = None
        self.unmasked = None
        self.masked_tokens = 0
        self.unmasked_tokens = 0

    def add(self, value: torch.Tensor, masked: torch.Tensor) -> None:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("expected a single [1,token,feature] activation")
        role = masked.to(value.device).reshape(-1)
        flat = value.reshape(-1, value.shape[-1]).float()
        if role.numel() != flat.shape[0] or not bool(role.any()) or bool(role.all()):
            raise ValueError("invalid aligned masked/unmasked partition")
        masked_sum = flat[role].square().sum(0)
        unmasked_sum = flat[~role].square().sum(0)
        if self.masked is None:
            self.masked = masked_sum
            self.unmasked = unmasked_sum
        else:
            self.masked.add_(masked_sum)
            self.unmasked.add_(unmasked_sum)
        self.masked_tokens += int(role.sum())
        self.unmasked_tokens += int((~role).sum())


def _role_denominators(raw: dict, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = raw["projections"]
    if [row["name"] for row in rows] != names:
        raise RuntimeError("role raw module ordering mismatch")
    masked, unmasked = [], []
    for row in rows:
        masked.append(sum(float(state["levels"][0]["den_masked"]) for state in row["states"]))
        unmasked.append(sum(float(state["levels"][0]["den_unmasked"]) for state in row["states"]))
    return np.asarray(masked), np.asarray(unmasked)


@torch.inference_mode()
def collect() -> dict:
    config = require_frozen()
    inputs = load_frozen_inputs()
    destination = DATA_ROOT / "role_input_second_moments.pt"
    curves_path = ROOT / "diagonal_role_curves.json"
    receipt_path = ROOT / "collection_receipt.json"
    if destination.exists() and curves_path.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("tensor_sha256") != sha(destination) or receipt.get("curves_sha256") != sha(curves_path):
            raise RuntimeError("completed collection receipt mismatch")
        event("collection_reused", tensor=str(destination))
        return receipt

    model, mapping = load_dense()
    names = list(mapping)
    if names != inputs.metadata["module_names"]:
        raise RuntimeError("model module ordering mismatch")
    accumulators = {name: RoleMomentAccumulator() for name in names}
    current = {"mask": None}
    handles = []
    for name, module in mapping.items():
        def hook(_module, inp, _out, module_name=name):
            accumulators[module_name].add(inp[0].detach(), current["mask"])
        handles.append(module.register_forward_hook(hook))
    device = next(model.parameters()).device
    started = time.monotonic()
    try:
        for index, state in enumerate(inputs.states["states"]):
            current["mask"] = torch.as_tensor(state["mask"], dtype=torch.bool, device=device)
            ids = torch.as_tensor(state["noisy_ids"], dtype=torch.long, device=device)
            model(ids)
            event("moment_state", state=index + 1, total=80,
                  elapsed_seconds=time.monotonic() - started)
    finally:
        for handle in handles:
            handle.remove()
    if model_sha(model) != DENSE_SHA:
        raise RuntimeError("dense model changed during role-moment collection")

    payload = {
        "config_sha256": sha(ROOT / "config.json"), "state_digest": config["state_digest"],
        "dense_model_sha256": DENSE_SHA, "names": names,
        "statistics": {name: {
            "masked_input_square_sum": accumulator.masked.cpu(),
            "unmasked_input_square_sum": accumulator.unmasked.cpu(),
            "masked_tokens": accumulator.masked_tokens,
            "unmasked_tokens": accumulator.unmasked_tokens,
        } for name, accumulator in accumulators.items()},
    }
    save_tensor(destination, payload)

    raw = json.loads(ROLE_RAW.read_text())
    den_m, den_u = _role_denominators(raw, names)
    diagonal_m, diagonal_u = [], []
    for index, (entry, name) in enumerate(zip(inputs.candidate["entries"], names)):
        module = mapping[name]
        masks = [read_mask(entry, level, module.weight.device) for level in range(6)]
        stats = accumulators[name]
        diagonal_m.append(diagonal_reconstruction_curve(
            module.weight, masks, stats.masked, den_m[index]))
        diagonal_u.append(diagonal_reconstruction_curve(
            module.weight, masks, stats.unmasked, den_u[index]))
        del masks
        event("diagonal_module", module=index + 1, total=224, name=name)
    curves = {"grid": list(GRID), "names": names, "masked": diagonal_m,
              "unmasked": diagonal_u, "normalization": "exact pooled role dense-output energy",
              "tensor_sha256": sha(destination)}
    atomic_write_json(curves_path, curves)
    receipt = {"status": "complete", "tensor": str(destination), "tensor_sha256": sha(destination),
               "curves": str(curves_path), "curves_sha256": sha(curves_path),
               "elapsed_seconds": time.monotonic() - started, "states": 80, "modules": 224}
    atomic_write_json(receipt_path, receipt)
    event("collection_complete", elapsed_seconds=receipt["elapsed_seconds"])
    return receipt


def _reconstruction_curves(inputs) -> tuple[np.ndarray, np.ndarray]:
    doc = json.loads(ROLE_CURVES.read_text())
    if doc["names"] != inputs.metadata["module_names"] or tuple(doc["grid"]) != GRID:
        raise RuntimeError("reconstruction curves mismatch")
    return np.asarray(doc["groups"]["actual"]["a"]), np.asarray(doc["groups"]["actual"]["b"])


def _diagonal_curves(inputs) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "diagonal_role_curves.json"
    if not path.exists():
        collect()
    doc = json.loads(path.read_text())
    if doc["names"] != inputs.metadata["module_names"] or tuple(doc["grid"]) != GRID:
        raise RuntimeError("diagonal curves mismatch")
    return np.asarray(doc["masked"]), np.asarray(doc["unmasked"])


def prepare() -> dict:
    require_frozen()
    inputs = load_frozen_inputs()
    shapes = [row["shape"] for row in inputs.candidate["entries"]]
    role_sources = {"reconstruction": _reconstruction_curves(inputs),
                    "diagonal": _diagonal_curves(inputs)}
    methods, manifests = {}, {}
    for proxy in PROXIES:
        masked, unmasked = role_sources[proxy]
        for aggregation in AGGREGATIONS:
            method = f"{proxy}_{aggregation}"
            values = aggregate_role_curves(masked, unmasked, aggregation)
            allocation = allocate_grid(values, shapes, GRID, TARGET_INDEX)
            if allocation["pruned"] != TARGET or allocation["budget_error"] != 0:
                raise RuntimeError(f"{method} missed exact budget")
            manifest = build_selected_manifest(method, inputs.candidate["entries"],
                                               allocation["sparsities"], GRID)
            manifest.update(config_sha256=sha(ROOT / "config.json"),
                            allocation_levels=allocation["levels"],
                            proxy=proxy, aggregation=aggregation)
            validate_manifest(manifest, inputs.metadata["module_names"], TARGET, WEIGHTS)
            path = ROOT / "manifests" / f"{method}.json"
            atomic_write_json(path, manifest)
            manifests[method] = manifest
            methods[method] = {
                "proxy": proxy, "aggregation": aggregation,
                "manifest": str(path), "manifest_sha256": sha(path),
                "signature": selected_signature(manifest),
                "pruned": allocation["pruned"], "weights": allocation["weights"],
                "global_sparsity": allocation["pruned"] / allocation["weights"],
                "levels": allocation["levels"], "sparsities": allocation["sparsities"],
                "level_counts": {str(x): allocation["sparsities"].count(x) for x in GRID},
            }

    old_role = json.loads(OLD_ROLE_MANIFEST.read_text())
    old_aggregate = json.loads(OLD_AGG_MANIFEST.read_text())
    references = {"historical_role": old_role, "historical_aggregate": old_aggregate}
    reference_signatures = {name: selected_signature(value) for name, value in references.items()}
    if methods["reconstruction_max_level"]["signature"] != reference_signatures["historical_role"]:
        raise RuntimeError("reconstruction max-level failed to reproduce historical Role-Max")

    aliases, representatives, seen = {}, [], dict(reference_signatures)
    for method in methods:
        signature = methods[method]["signature"]
        match = next((name for name, old in seen.items() if old == signature), None)
        if match is None:
            representatives.append(method)
            seen[method] = signature
            aliases[method] = method
        else:
            aliases[method] = match
    pairwise = {}
    all_manifests = {**references, **manifests}
    for method, manifest in manifests.items():
        pairwise[method] = {
            reference: allocation_difference(all_manifests[reference], manifest)
            for reference in ("historical_role", "historical_aggregate")
        }
    document = {"status": "frozen_before_mini", "methods": methods,
                "reference_signatures": reference_signatures, "aliases": aliases,
                "novel_representatives": representatives, "pairwise": pairwise}
    atomic_write_json(ROOT / "preparation.json", document)
    event("preparation_complete", novel_representatives=representatives, aliases=aliases)
    return document


def _validate_rows(rows, reference, protocol_hash):
    if len(rows) != 100:
        raise RuntimeError("mini evaluation requires 100 rows")
    _assert_same_examples(reference[:100], rows)
    for index, row in enumerate(rows):
        if row.get("example_id") != index or row.get("evaluation_config_hash") != protocol_hash:
            raise RuntimeError("prediction identity mismatch")
        if strict_exact_match(row.get("extracted_answer"), row.get("reference_answer")) != row.get("correct"):
            raise RuntimeError("stored exact match mismatch")


@torch.inference_mode()
def mini() -> dict:
    preparation = prepare()
    config = eval_config(EVAL_CONFIG)
    protocol_hash, protocol = _evaluation_config_hash(config)
    if protocol_hash != json.loads((ROOT / "config.json").read_text())["evaluation"]["protocol_hash"]:
        raise RuntimeError("evaluation protocol changed")
    reference = _read_jsonl(Path("experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl"))
    historical = json.loads(OLD_MINI.read_text())
    historical_rows = {
        "historical_role": _read_jsonl(Path(historical["methods"]["role"]["predictions"])),
        "historical_aggregate": _read_jsonl(Path(historical["methods"]["aggregate"]["predictions"])),
    }
    for rows in historical_rows.values():
        _validate_rows(rows, reference, protocol_hash)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"],
                                               trust_remote_code=True)
    outcomes = {}
    representatives = preparation["novel_representatives"]
    for position, method in enumerate(representatives):
        meta = preparation["methods"][method]
        manifest_path = Path(meta["manifest"])
        manifest = json.loads(manifest_path.read_text())
        prediction_path = ROOT / "gsm8k" / f"{method}_100_predictions.jsonl"
        receipt_path = prediction_path.with_suffix(".receipt.json")
        fingerprint = {"config_sha256": sha(ROOT / "config.json"),
                       "manifest_sha256": sha(manifest_path), "protocol_sha256": protocol_hash,
                       "method": method, "limit": 100}
        rows = None
        if prediction_path.exists() or receipt_path.exists():
            if not prediction_path.exists() or not receipt_path.exists():
                raise RuntimeError(f"partial prediction artifact for {method}")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("fingerprint") != fingerprint or receipt.get("predictions_sha256") != sha(prediction_path):
                raise RuntimeError(f"prediction receipt mismatch for {method}")
            rows = _read_jsonl(prediction_path)
            _validate_rows(rows, reference, protocol_hash)
            metrics = receipt.get("metrics", {"reused": True})
        if rows is None:
            model, mapping = load_dense()
            if apply_manifest(model, mapping, manifest) != TARGET:
                raise RuntimeError("applied budget mismatch")
            sparse_sha = model_sha(model)
            metrics, rows = _evaluate_gsm8k(model, tokenizer, config, method, 100, protocol_hash)
            if model_sha(model) != sparse_sha:
                raise RuntimeError("sparse model changed during evaluation")
            _validate_rows(rows, reference, protocol_hash)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            _write_jsonl(prediction_path, rows)
            atomic_write_json(receipt_path, {"status": "complete", "fingerprint": fingerprint,
                                             "predictions_sha256": sha(prediction_path),
                                             "sparse_model_sha256": sparse_sha, "metrics": metrics})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        outcomes[method] = {"correct": sum(bool(row["correct"]) for row in rows),
                            "predictions": str(prediction_path), "predictions_sha256": sha(prediction_path),
                            "metrics": metrics, "paired_vs_role": paired_binary_comparison(
                                [row["correct"] for row in historical_rows["historical_role"]],
                                [row["correct"] for row in rows])}
        event("mini_method_complete", method=method, method_index=position + 1,
              method_total=len(representatives), correct=outcomes[method]["correct"])

    resolved = {}
    reference_correct = {"historical_role": int(historical["methods"]["role"]["correct"]),
                         "historical_aggregate": int(historical["methods"]["aggregate"]["correct"])}
    for method, alias in preparation["aliases"].items():
        if alias in outcomes:
            resolved[method] = {"source": alias, "correct": outcomes[alias]["correct"]}
        else:
            resolved[method] = {"source": alias, "correct": reference_correct[alias]}
    result = {"status": "complete", "protocol_hash": protocol_hash, "protocol": protocol,
              "historical": {"uniform": 12, "aggregate": reference_correct["historical_aggregate"],
                             "reconstruction_max_level": reference_correct["historical_role"]},
              "evaluated": outcomes, "resolved_candidates": resolved,
              "ranking": sorted(({"method": name, **row} for name, row in resolved.items()),
                                key=lambda row: (-row["correct"], row["method"])),
              "decision_rule": "retain candidates strictly above historical Role-Max=24; ties only as lower-cost efficiency candidates",
              "interpretation_limit": "exploratory fixed mini-100 method-development screen"}
    atomic_write_json(ROOT / "mini100_results.json", result)
    event("mini_complete", ranking=result["ranking"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "collect", "prepare", "mini", "all"))
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)
    freeze()
    if args.phase == "freeze":
        return
    if args.phase in ("collect", "all"):
        collect()
    if args.phase in ("prepare", "all"):
        prepare()
    if args.phase in ("mini", "all"):
        mini()


if __name__ == "__main__":
    main()
