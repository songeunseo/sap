#!/usr/bin/env python3
"""Run the frozen random-role and 50/75%-sparsity validation experiments."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from experiments.dlm_capacity_predictor.audit_existing import strict_exact_match
from experiments.dlm_dual_role_allocation.io import atomic_write_json, file_sha256, load_frozen_inputs
from experiments.dlm_loss_aggregation.core import mask_sha256, pack_mask, unpack_mask
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples, _evaluate_gsm8k, _evaluation_config_hash,
    _read_jsonl, _write_jsonl, load_config as eval_config,
)
from experiments.dlm_loss_aggregation.run import _load_model, _suffix_logits, load_config
from experiments.dlm_role_validation.core import (
    GRIDS, RANDOM_SEEDS, TARGET_INDEX, allocate_grid, deterministic_random_role,
    partition_digest, partition_sums, pool_partition, random_mini_gate,
    sparsity_mini_gate,
)
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, read_mask
from experiments.projection_capacity_followup_65.core import (
    build_selected_manifest, paired_binary_comparison,
)
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
from experiments.wanda_failure_characterization.core import masked_linear_variants
from experiments.wanda_failure_characterization.run_failure_map import model_sha, modules


ROOT = Path("experiments/dlm_role_validation")
DATA_ROOT = Path("/DATA/tmluser1/sap-dlm-role-validation")
STATS = Path("experiments/cgq_wanda_structured_diagnostic/sufficient_statistics.pt")
BASE_CONFIG = "experiments/dlm_loss_aggregation/config.yaml"
EVAL_CONFIG = "experiments/dlm_loss_aggregation/exp002/config.yaml"
OLD_ROLE_ROOT = Path("experiments/dlm_dual_role_allocation")
OLD_MINI_ROOT = Path("experiments/dlm_dual_role_mini100")
CAPACITY_ROOT = Path("experiments/projection_capacity_allocation_65")
SWEEP_ROOT = Path("experiments/uniform_wanda_sparsity_sweep")
WEIGHTS = 6_979_321_856


class _TargetReached(Exception):
    pass


def event(kind: str, **values) -> None:
    print(json.dumps({"event": kind, **values}, sort_keys=True), flush=True)


def sha(path: str | Path) -> str:
    return file_sha256(path)


def save_tensor(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _suite_labels(suite: str) -> list[str]:
    return ["actual"] + ([f"random_{seed}" for seed in RANDOM_SEEDS] if suite == "random65" else [])


def _state_partitions(suite: str, states: list[dict]) -> tuple[list[dict[str, torch.Tensor]], dict]:
    rows, audit = [], {"suite": suite, "states": []}
    for state_index, state in enumerate(states):
        actual = torch.as_tensor(state["mask"], dtype=torch.bool)[0].cpu()
        groups = {"actual": actual}
        for seed in RANDOM_SEEDS if suite == "random65" else ():
            groups[f"random_{seed}"] = deterministic_random_role(actual, seed, state_index)
        rows.append(groups)
        audit_row = {
            "state_index": state_index, "sequence_index": int(state["sequence_index"]),
            "timestep": float(state["timestep"]), "token_count": actual.numel(),
            "actual_count": int(actual.sum()), "groups": {},
        }
        actual_set = set(torch.nonzero(actual, as_tuple=False).flatten().tolist())
        for label, mask in groups.items():
            chosen = torch.nonzero(mask, as_tuple=False).flatten().tolist()
            intersection = len(actual_set.intersection(chosen))
            union = len(actual_set.union(chosen))
            audit_row["groups"][label] = {
                "count": len(chosen), "selected_indices": chosen,
                "actual_intersection": intersection,
                "actual_jaccard": intersection / union,
            }
        audit["states"].append(audit_row)
    audit["partition_sha256"] = {
        label: partition_digest([row[label] for row in rows]) for label in _suite_labels(suite)
    }
    return rows, audit


def freeze() -> dict:
    inputs = load_frozen_inputs()
    states = inputs.states["states"]
    partition_audits = {}
    for suite in GRIDS:
        _, partition_audits[suite] = _state_partitions(suite, states)
    protocol_hash, protocol = _evaluation_config_hash(eval_config(EVAL_CONFIG))
    config = {
        "status": "frozen_before_new_collection_or_evaluation",
        "model": inputs.config["model"], "dense_model_sha256": DENSE_SHA,
        "projection_count": 224, "state_count": 80, "total_prunable_weights": WEIGHTS,
        "selector": "exact historical Standard Wanda abs(W)*sqrt(overall_uniform A), stable row-wise sort",
        "suites": {
            "random65": {
                "target": .65, "grid": list(GRIDS["random65"]),
                "groups": "actual masked/unmasked plus three cardinality-matched random partitions",
                "random_seeds": list(RANDOM_SEEDS),
                "mini_gate": "actual Role > mean(random Role) and strict wins against at least 2/3 seeds",
            },
            "target50": {
                "target": .50, "grid": list(GRIDS["target50"]),
                "grid_reason": "same -15/+10 percentage-point range around target as original 65% grid",
                "mini_gate": "Role strictly exceeds both Aggregate and Uniform",
            },
            "target75": {
                "target": .75, "grid": list(GRIDS["target75"]),
                "grid_reason": "same -15/+10 percentage-point range around target as original 65% grid",
                "mini_gate": "Role strictly exceeds both Aggregate and Uniform",
            },
        },
        "role": "max(pooled group-A normalized reconstruction error, pooled group-B normalized reconstruction error)",
        "aggregate": "pooled total squared reconstruction error / pooled dense-output energy",
        "allocation": "raw next 5pp marginal cost per nominal parameter; exact target row-floor budget",
        "random_partition": "SeedSequence([seed,state_index]); exact real masked-token cardinality; labels only, inputs unchanged",
        "evaluation": {"protocol_hash": protocol_hash, "protocol": protocol, "limit": 100,
                       "full_policy": "only after each preregistered mini gate passes"},
        "source_hashes": {
            str(STATS): sha(STATS),
            str(CAPACITY_ROOT / "candidate_mask_manifest.json"): sha(CAPACITY_ROOT / "candidate_mask_manifest.json"),
            str(CAPACITY_ROOT / "state_verification.json"): sha(CAPACITY_ROOT / "state_verification.json"),
            str(OLD_ROLE_ROOT / "role_reconstruction_raw.json"): sha(OLD_ROLE_ROOT / "role_reconstruction_raw.json"),
            str(OLD_MINI_ROOT / "mini100_results.json"): sha(OLD_MINI_ROOT / "mini100_results.json"),
            str(SWEEP_ROOT / "summary.json"): sha(SWEEP_ROOT / "summary.json"),
            EVAL_CONFIG: sha(EVAL_CONFIG),
        },
        "no_retuning": True,
    }
    config_path = ROOT / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("frozen validation config differs")
    if not config_path.exists():
        atomic_write_json(config_path, config)
    verification = {
        "status": "verified", "source_state_digest": inputs.metadata["state_digest"],
        "source_receipt_sha256": inputs.receipt["receipt_sha256"],
        "input_state_disjointness": inputs.verification,
        "partitions": partition_audits,
    }
    verification_path = ROOT / "state_verification.json"
    if verification_path.exists() and json.loads(verification_path.read_text()) != verification:
        raise RuntimeError("frozen partition verification differs")
    if not verification_path.exists():
        atomic_write_json(verification_path, verification)
    event("freeze_complete", config_sha256=sha(config_path), protocol_hash=protocol_hash)
    return config


def require_frozen() -> dict:
    config = freeze()
    for path, digest in config["source_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"frozen source changed: {path}")
    return config


def load_dense():
    model, _ = _load_model(load_config(BASE_CONFIG))
    if model_sha(model) != DENSE_SHA:
        raise RuntimeError("wrong dense model")
    mapping = modules(model)
    if len(mapping) != 224:
        raise RuntimeError("expected 224 projections")
    return model, mapping


def _activation_statistics(mapping) -> dict[str, torch.Tensor]:
    payload = torch.load(STATS, map_location="cpu", weights_only=False)
    result = {name: row["overall_uniform"] for name, row in payload["statistics"].items()}
    if set(result) != set(mapping):
        raise RuntimeError("activation-statistic module mismatch")
    return result


def _transient_masks(module, activation: torch.Tensor, grid) -> tuple[list[torch.Tensor], list[str]]:
    weight = module.weight
    a = activation.float().to(weight.device)
    score = weight.float().abs() * a.sqrt()[None, :]
    order = torch.argsort(score, dim=1, stable=True)
    masks, hashes = [], []
    for sparsity in grid:
        count = int(weight.shape[1] * sparsity)
        mask = torch.zeros_like(weight, dtype=torch.bool).scatter_(1, order[:, :count], True)
        masks.append(mask)
        hashes.append(mask_sha256(pack_mask(mask.cpu())))
    return masks, hashes


def _checkpoint_expected(suite: str, row: dict, mask_hashes: list[str], audit: dict,
                         state_digest: str, selector: str) -> dict:
    return {
        "config_sha256": sha(ROOT / "config.json"), "suite": suite,
        "dense_model_sha256": DENSE_SHA, "state_digest": state_digest,
        "partition_sha256": audit["partition_sha256"], "module_index": int(row["module_index"]),
        "name": row["name"], "shape": row["shape"], "grid": list(GRIDS[suite]),
        "candidate_mask_sha256": mask_hashes, "state_count": 80,
        "selector": selector,
    }


def _validate_checkpoint(payload: dict, expected: dict) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"checkpoint {key} mismatch")
    if len(payload.get("states", [])) != 80:
        raise RuntimeError("checkpoint state count mismatch")


@torch.inference_mode()
def collect(suite: str, stop_after: int | None = None) -> dict:
    config = require_frozen()
    inputs = load_frozen_inputs()
    model, mapping = load_dense()
    names = list(mapping)
    if names != inputs.metadata["module_names"]:
        raise RuntimeError("module ordering mismatch")
    activation = _activation_statistics(mapping)
    states = inputs.states["states"]
    partitions, audit = _state_partitions(suite, states)
    device = next(model.parameters()).device
    prefixes = {block: [] for block in range(32)}
    handles = []
    for block_index, block in enumerate(model.model.transformer.blocks):
        def save_prefix(_module, inp, index=block_index):
            prefixes[index].append(inp[0].detach().clone())
        handles.append(block.register_forward_pre_hook(save_prefix))
    try:
        for state in states:
            model(torch.tensor(state["noisy_ids"], dtype=torch.long, device=device))
    finally:
        for handle in handles:
            handle.remove()
    if any(len(rows) != 80 for rows in prefixes.values()):
        raise RuntimeError("dense prefix collection incomplete")
    event("dense_prefixes_complete", suite=suite, states=80)

    started = time.monotonic()
    checkpoint_root = ROOT / "runtime" / suite
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    for row in inputs.candidate["entries"]:
        if stop_after is not None and completed >= stop_after:
            break
        module = mapping[row["name"]]
        if suite == "random65":
            masks = [read_mask(row, level, module.weight.device) for level in range(6)]
            mask_hashes = [entry["mask_sha256"] for entry in row["masks"]]
        else:
            masks, mask_hashes = _transient_masks(module, activation[row["name"]], GRIDS[suite])
        expected = _checkpoint_expected(suite, row, mask_hashes, audit,
                                        inputs.metadata["state_digest"], config["selector"])
        path = checkpoint_root / f"{row['name']}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            _validate_checkpoint(payload, expected)
            event("projection_skipped", suite=suite, module=int(row["module_index"]) + 1, total=224)
            del masks
            completed += 1
            continue
        block_index = int(row["name"].split(".")[0].removeprefix("block_"))
        records = []
        for state_index, (prefix, state, group_masks) in enumerate(zip(prefixes[block_index], states, partitions)):
            captured = {}
            def hook(mod, inp, _out):
                variants_in = inp[0]
                if variants_in.shape[0] != 7 or not torch.equal(variants_in, variants_in[:1].expand_as(variants_in)):
                    raise RuntimeError("same-path variant input mismatch")
                outputs = masked_linear_variants(variants_in, mod.weight, mod.bias, [None] + masks)
                for label, position_mask in group_masks.items():
                    captured[label] = partition_sums(outputs, position_mask)
                raise _TargetReached
            handle = module.register_forward_hook(hook)
            reached = False
            try:
                _suffix_logits(model, prefix.expand(7, -1, -1).contiguous(), block_index)
            except _TargetReached:
                reached = True
            finally:
                handle.remove()
            if not reached:
                raise RuntimeError("target Linear not reached")
            groups = {}
            for label in _suite_labels(suite):
                values = captured[label]
                groups[label] = [{
                    "sparsity": GRIDS[suite][level],
                    "num_a": values["num_a"][level], "den_a": values["den_a"],
                    "num_b": values["num_b"][level], "den_b": values["den_b"],
                } for level in range(6)]
            records.append({"state_index": state_index,
                            "sequence_index": int(state["sequence_index"]),
                            "timestep": float(state["timestep"]), "groups": groups})
        payload = {**expected, "states": records}
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["payload_sha256"] = hashlib.sha256(body).hexdigest()
        atomic_write_json(path, payload)
        _validate_checkpoint(payload, expected)
        event("projection_complete", suite=suite, module=int(row["module_index"]) + 1,
              total=224, name=row["name"], elapsed_seconds=time.monotonic() - started)
        del masks, records
        completed += 1
    if model_sha(model) != DENSE_SHA:
        raise RuntimeError("dense model changed during collection")
    result = {"status": "complete" if completed == 224 else "partial", "suite": suite,
              "projections": completed,
              "elapsed_seconds": time.monotonic() - started, "dense_model_sha256": DENSE_SHA}
    atomic_write_json(ROOT / f"collection_{suite}.json", result)
    event("collection_complete", **result)
    return result


def _load_records(suite: str) -> tuple[list[dict], list[dict]]:
    inputs = load_frozen_inputs()
    config = require_frozen()
    _, audit = _state_partitions(suite, inputs.states["states"])
    payloads = []
    for row in inputs.candidate["entries"]:
        path = ROOT / "runtime" / suite / f"{row['name']}.json"
        if not path.exists():
            raise RuntimeError(f"missing checkpoint {path}")
        payload = json.loads(path.read_text())
        expected = {
            "config_sha256": sha(ROOT / "config.json"), "suite": suite,
            "dense_model_sha256": DENSE_SHA, "state_digest": inputs.metadata["state_digest"],
            "partition_sha256": audit["partition_sha256"],
            "module_index": int(row["module_index"]), "name": row["name"],
            "shape": row["shape"], "grid": list(GRIDS[suite]),
            "candidate_mask_sha256": payload.get("candidate_mask_sha256"),
            "state_count": 80, "selector": config["selector"],
        }
        _validate_checkpoint(payload, expected)
        if payload["partition_sha256"] != audit["partition_sha256"]:
            raise RuntimeError("partition digest changed")
        payloads.append(payload)
    return inputs.candidate["entries"], payloads


def _pooled_curves(payloads: list[dict], label: str, field: str) -> np.ndarray:
    return np.asarray([pool_partition(payload["states"], label)[field] for payload in payloads], dtype=float)


def _allocation_summary(allocation: dict) -> dict:
    levels = allocation["sparsities"]
    return {"pruned": allocation["pruned"], "weights": allocation["weights"],
            "global_sparsity": allocation["pruned"] / allocation["weights"],
            "counts": {str(value): levels.count(value) for value in allocation["grid"]},
            "budget_error": allocation["budget_error"]}


def analyze(suite: str) -> dict:
    require_frozen()
    entries, payloads = _load_records(suite)
    shapes = [row["shape"] for row in entries]
    labels = _suite_labels(suite)
    curves, allocations = {}, {}
    for label in labels:
        curves[label] = {
            "aggregate": _pooled_curves(payloads, label, "aggregate"),
            "role": _pooled_curves(payloads, label, "role"),
            "a": _pooled_curves(payloads, label, "a"),
            "b": _pooled_curves(payloads, label, "b"),
        }
    if suite == "random65":
        methods = {label: curves[label]["role"] for label in labels}
        # The rerun must reproduce the already-confirmed actual Role and Aggregate allocations.
        methods["aggregate"] = curves["actual"]["aggregate"]
    else:
        methods = {"aggregate": curves["actual"]["aggregate"], "role": curves["actual"]["role"]}
    for method, values in methods.items():
        allocations[method] = allocate_grid(values, shapes, GRIDS[suite])
        if allocations[method]["budget_error"] != 0:
            raise RuntimeError("allocation missed exact row-floor target")

    if suite == "random65":
        old = json.loads((OLD_MINI_ROOT / "preparation.json").read_text())["allocations"]
        if allocations["actual"]["levels"] != [GRIDS[suite].index(x) for x in old["role"]["sparsities"]]:
            raise RuntimeError("actual Role allocation did not reproduce historical result")
        if allocations["aggregate"]["levels"] != [GRIDS[suite].index(x) for x in old["aggregate"]["sparsities"]]:
            raise RuntimeError("Aggregate allocation did not reproduce historical result")

    names = [row["name"] for row in entries]
    diagnostics = {"curve_correlations": {}, "allocation_differences": {}}
    reference = allocations["actual" if suite == "random65" else "role"]
    for method, allocation in allocations.items():
        x = methods[method].reshape(-1)
        y = methods["actual" if suite == "random65" else "role"].reshape(-1)
        diagnostics["curve_correlations"][method] = {
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(x, y).statistic),
        }
        diagnostics["allocation_differences"][method] = {
            "changed_projections": sum(a != b for a, b in zip(reference["levels"], allocation["levels"])),
            "sum_absolute_sparsity_difference": float(sum(abs(a - b) for a, b in zip(reference["sparsities"], allocation["sparsities"]))),
        }
    document = {
        "status": "frozen_before_downstream", "suite": suite, "grid": list(GRIDS[suite]),
        "target": GRIDS[suite][TARGET_INDEX], "module_names": names,
        "methods": {method: {**_allocation_summary(allocation),
                             "levels": allocation["levels"], "sparsities": allocation["sparsities"]}
                    for method, allocation in allocations.items()},
        "diagnostics": diagnostics,
    }
    path = ROOT / f"allocation_{suite}.json"
    if path.exists() and json.loads(path.read_text()) != document:
        raise RuntimeError("frozen allocation changed")
    if not path.exists():
        atomic_write_json(path, document)
    # Persist compact full curves separately; raw state statistics remain checkpointed.
    atomic_write_json(ROOT / f"curves_{suite}.json", {
        "grid": list(GRIDS[suite]), "names": names,
        "groups": {label: {field: values.tolist() for field, values in row.items()}
                   for label, row in curves.items()},
    })
    event("analysis_complete", suite=suite,
          summaries={key: _allocation_summary(value) for key, value in allocations.items()})
    return document


def _build_manifest(method: str, entries: list[dict], allocation: dict, grid) -> dict:
    manifest = build_selected_manifest(method, entries, allocation["sparsities"], grid)
    manifest.update({"config_sha256": sha(ROOT / "config.json"), "allocation_levels": allocation["levels"]})
    if manifest["pruned"] != allocation["pruned"] or manifest["weights"] != WEIGHTS:
        raise RuntimeError("manifest budget mismatch")
    return manifest


@torch.inference_mode()
def materialize(suite: str) -> dict:
    allocation_path = ROOT / f"allocation_{suite}.json"
    allocation_doc = (json.loads(allocation_path.read_text()) if allocation_path.exists()
                      else analyze(suite))
    inputs = load_frozen_inputs()
    methods = allocation_doc["methods"]
    if suite == "random65":
        for method in [f"random_{seed}" for seed in RANDOM_SEEDS]:
            manifest = _build_manifest(method, inputs.candidate["entries"], methods[method], GRIDS[suite])
            atomic_write_json(ROOT / f"manifest_{suite}_{method}.json", manifest)
        event("materialize_complete", suite=suite, reused_candidate_masks=True)
        return {"suite": suite, "methods": list(methods)}

    model, mapping = load_dense()
    activation = _activation_statistics(mapping)
    method_names = ("uniform", "aggregate", "role")
    allocations = dict(methods)
    allocations["uniform"] = {
        "levels": [TARGET_INDEX] * 224,
        "sparsities": [GRIDS[suite][TARGET_INDEX]] * 224,
        "pruned": sum(row["shape"][0] * int(row["shape"][1] * GRIDS[suite][TARGET_INDEX])
                      for row in inputs.candidate["entries"]),
    }
    built_entries = {method: [] for method in method_names}
    for index, row in enumerate(inputs.candidate["entries"]):
        module = mapping[row["name"]]
        needed = sorted({allocations[method]["levels"][index] for method in method_names})
        masks, hashes = _transient_masks(module, activation[row["name"]], GRIDS[suite])
        meta_by_level = {}
        for level in needed:
            sparsity = GRIDS[suite][level]
            path = DATA_ROOT / suite / "masks" / f"{row['name']}.{round(sparsity * 100)}.pt"
            packed = pack_mask(masks[level].cpu())
            if path.exists():
                existing = torch.load(path, map_location="cpu", weights_only=False)
                if mask_sha256(existing) != hashes[level]:
                    raise RuntimeError("existing materialized mask differs")
            else:
                save_tensor(path, packed)
            meta_by_level[level] = {
                "nominal_sparsity": sparsity, "sparsity": sparsity,
                "prune_per_row": int(row["shape"][1] * sparsity),
                "pruned": row["shape"][0] * int(row["shape"][1] * sparsity),
                "mask_sha256": hashes[level], "path": str(path), "file_sha256": sha(path),
            }
        for method in method_names:
            level = allocations[method]["levels"][index]
            built_entries[method].append({
                "module_index": index, "name": row["name"], "shape": row["shape"],
                "weights": row["weights"], "assigned_sparsity": GRIDS[suite][level],
                "level": level, "selected_mask": meta_by_level[level],
            })
        del masks
        event("mask_materialized", suite=suite, module=index + 1, total=224)
    for method in method_names:
        manifest = {"method": method, "config_sha256": sha(ROOT / "config.json"),
                    "pruned": sum(row["selected_mask"]["pruned"] for row in built_entries[method]),
                    "weights": WEIGHTS, "entries": built_entries[method]}
        if manifest["pruned"] != allocations[method]["pruned"]:
            raise RuntimeError("materialized budget differs")
        atomic_write_json(ROOT / f"manifest_{suite}_{method}.json", manifest)
    if model_sha(model) != DENSE_SHA:
        raise RuntimeError("dense model changed during materialization")
    event("materialize_complete", suite=suite, methods=list(method_names))
    return {"suite": suite, "methods": list(method_names)}


def _manifest_path(suite: str, method: str) -> Path:
    return ROOT / f"manifest_{suite}_{method}.json"


def _validate_prediction_rows(rows, reference, protocol_hash, limit):
    if len(rows) != limit:
        raise RuntimeError("prediction row count mismatch")
    _assert_same_examples(reference[:limit], rows)
    for index, row in enumerate(rows):
        if row.get("example_id") != index or row.get("evaluation_config_hash") != protocol_hash:
            raise RuntimeError("prediction identity mismatch")
        if strict_exact_match(row.get("extracted_answer"), row.get("reference_answer")) != row.get("correct"):
            raise RuntimeError("stored exact match mismatch")


def _historical_uniform(target: float) -> tuple[dict, list[dict]]:
    summary = json.loads((SWEEP_ROOT / "summary.json").read_text())
    row = next(value for value in summary["results"] if float(value["nominal_sparsity"]) == target)
    prediction = SWEEP_ROOT / "gsm8k_mini" / f"uniform_{round(target * 100)}_predictions.jsonl"
    return row, _read_jsonl(prediction)


@torch.inference_mode()
def _evaluate_methods(suite: str, method_names: list[str], limit: int) -> dict:
    config = eval_config(EVAL_CONFIG)
    protocol_hash, protocol = _evaluation_config_hash(config)
    reference = _read_jsonl(Path("experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True)
    outcomes = {}
    for method in method_names:
        manifest_path = _manifest_path(suite, method)
        manifest = json.loads(manifest_path.read_text())
        output_dir = ROOT / ("gsm8k_mini" if limit == 100 else "gsm8k_full")
        prediction_path = output_dir / f"{suite}_{method}_{limit}_predictions.jsonl"
        receipt_path = prediction_path.with_suffix(".receipt.json")
        fingerprint = {"config_sha256": sha(ROOT / "config.json"),
                       "manifest_sha256": sha(manifest_path), "protocol_sha256": protocol_hash,
                       "limit": limit, "suite": suite, "method": method}
        rows = None
        if prediction_path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("fingerprint") != fingerprint or receipt.get("predictions_sha256") != sha(prediction_path):
                raise RuntimeError("prediction receipt mismatch")
            rows = _read_jsonl(prediction_path)
            _validate_prediction_rows(rows, reference, protocol_hash, limit)
            metrics = receipt.get("metrics", {"reused": True})
        if rows is None:
            model, mapping = load_dense()
            applied = apply_manifest(model, mapping, manifest)
            if applied != int(manifest["pruned"]):
                raise RuntimeError("applied budget mismatch")
            sparse_sha = model_sha(model)
            metrics, rows = _evaluate_gsm8k(model, tokenizer, config,
                                            f"role_validation_{suite}_{method}", limit, protocol_hash)
            if model_sha(model) != sparse_sha:
                raise RuntimeError("sparse model changed during evaluation")
            _validate_prediction_rows(rows, reference, protocol_hash, limit)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            _write_jsonl(prediction_path, rows)
            atomic_write_json(receipt_path, {"status": "complete", "fingerprint": fingerprint,
                                             "predictions_sha256": sha(prediction_path),
                                             "sparse_model_sha256": sparse_sha, "metrics": metrics})
            del model
            gc.collect()
            torch.cuda.empty_cache()
        outcomes[method] = {"correct": sum(bool(row["correct"]) for row in rows),
                            "predictions": str(prediction_path), "sha256": sha(prediction_path),
                            "metrics": metrics, "rows": rows}
        event("gsm8k_method_complete", suite=suite, method=method, limit=limit,
              correct=outcomes[method]["correct"])
    return {"protocol_hash": protocol_hash, "protocol": protocol, "outcomes": outcomes}


def mini(suite: str) -> dict:
    required = ([f"random_{seed}" for seed in RANDOM_SEEDS] if suite == "random65"
                else ["uniform", "aggregate", "role"])
    if any(not _manifest_path(suite, method).exists() for method in required):
        materialize(suite)
    if suite == "random65":
        method_names = [f"random_{seed}" for seed in RANDOM_SEEDS]
        evaluated = _evaluate_methods(suite, method_names, 100)
        historical = json.loads((OLD_MINI_ROOT / "mini100_results.json").read_text())
        actual_correct = int(historical["methods"]["role"]["correct"])
        actual_rows = _read_jsonl(Path(historical["methods"]["role"]["predictions"]))
        random_correct = [evaluated["outcomes"][method]["correct"] for method in method_names]
        gate = random_mini_gate(actual_correct, random_correct)
        paired = {method: paired_binary_comparison(
            [row["correct"] for row in evaluated["outcomes"][method]["rows"]],
            [row["correct"] for row in actual_rows]) for method in method_names}
        methods = {method: {k: v for k, v in row.items() if k != "rows"}
                   for method, row in evaluated["outcomes"].items()}
        result = {"status": "complete", "suite": suite, "actual_role_correct": actual_correct,
                  "methods": methods, "paired_actual_vs_random": paired, "gate": gate,
                  "protocol_hash": evaluated["protocol_hash"]}
    else:
        evaluated = _evaluate_methods(suite, ["aggregate", "role"], 100)
        target = GRIDS[suite][TARGET_INDEX]
        uniform_summary, uniform_rows = _historical_uniform(target)
        agg = evaluated["outcomes"]["aggregate"]
        role = evaluated["outcomes"]["role"]
        gate = sparsity_mini_gate(int(uniform_summary["correct"]), agg["correct"], role["correct"])
        result = {"status": "complete", "suite": suite,
                  "uniform": {"correct": int(uniform_summary["correct"]),
                              "predictions": str(SWEEP_ROOT / "gsm8k_mini" / f"uniform_{round(target * 100)}_predictions.jsonl")},
                  "methods": {key: {k: v for k, v in value.items() if k != "rows"}
                              for key, value in evaluated["outcomes"].items()},
                  "paired": {
                      "role_vs_aggregate": paired_binary_comparison([x["correct"] for x in agg["rows"]], [x["correct"] for x in role["rows"]]),
                      "role_vs_uniform": paired_binary_comparison([x["correct"] for x in uniform_rows], [x["correct"] for x in role["rows"]]),
                  }, "gate": gate, "protocol_hash": evaluated["protocol_hash"]}
    atomic_write_json(ROOT / f"mini_{suite}.json", result)
    event("mini_complete", suite=suite, gate=result["gate"])
    return result


def full(suite: str, force: bool = False, role_only: bool = False) -> dict:
    mini_path = ROOT / f"mini_{suite}.json"
    mini_passed = (mini_path.exists()
                   and bool(json.loads(mini_path.read_text())["gate"]["passed"]))
    if not mini_passed and not force:
        event("full_skipped", suite=suite, reason="mini gate did not pass")
        return {"status": "skipped", "suite": suite}
    method_names = ([f"random_{seed}" for seed in RANDOM_SEEDS] if suite == "random65"
                    else (["role"] if role_only else ["uniform", "aggregate", "role"]))
    evaluated = _evaluate_methods(suite, method_names, 1319)
    rows = evaluated["outcomes"]
    if suite == "random65":
        historical = json.loads(Path("experiments/dlm_dual_role_full1319/full1319_results.json").read_text())
        actual_path = Path("experiments/dlm_dual_role_full1319/gsm8k/role_1319_predictions.jsonl")
        actual_rows = _read_jsonl(actual_path)
        comparisons = {method: paired_binary_comparison([x["correct"] for x in row["rows"]],
                                                         [x["correct"] for x in actual_rows])
                       for method, row in rows.items()}
        reference = {"actual_role_correct": int(historical["correct"]["role"]),
                     "actual_role_predictions": str(actual_path),
                     "actual_role_predictions_sha256": sha(actual_path)}
    else:
        comparisons = {}
        if "aggregate" in rows:
            comparisons["role_vs_aggregate"] = paired_binary_comparison(
                [x["correct"] for x in rows["aggregate"]["rows"]],
                [x["correct"] for x in rows["role"]["rows"]])
        if "uniform" in rows:
            comparisons["role_vs_uniform"] = paired_binary_comparison(
                [x["correct"] for x in rows["uniform"]["rows"]],
                [x["correct"] for x in rows["role"]["rows"]])
        reference = {}
    result = {"status": "complete", "suite": suite, **reference,
              "mini_gate_passed": mini_passed,
              "mini_gate_override": bool(force and not mini_passed),
              "requested_methods": method_names,
              "override_reason": ("user explicitly requested immediate full-1319 evaluation; "
                                  "mini screening was stopped" if force and not mini_passed else None),
              "methods": {key: {k: v for k, v in value.items() if k != "rows"} for key, value in rows.items()},
              "paired": comparisons, "protocol_hash": evaluated["protocol_hash"]}
    atomic_write_json(ROOT / f"full_{suite}.json", result)
    event("full_complete", suite=suite,
          correct={key: value["correct"] for key, value in rows.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "collect", "analyze", "materialize", "mini", "full", "screen"))
    parser.add_argument("--suite", choices=tuple(GRIDS), required=False)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--role-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)
    freeze()
    if args.phase == "freeze":
        return
    if args.suite is None:
        parser.error("--suite is required for this phase")
    if args.phase in ("collect", "screen"):
        collect(args.suite, stop_after=args.stop_after)
        if args.stop_after is not None:
            return
    if args.phase in ("analyze", "screen"):
        analyze(args.suite)
    if args.phase in ("materialize", "screen"):
        materialize(args.suite)
    if args.phase in ("mini", "screen"):
        mini(args.suite)
    if args.phase == "full":
        full(args.suite, force=args.force_full, role_only=args.role_only)


if __name__ == "__main__":
    main()
