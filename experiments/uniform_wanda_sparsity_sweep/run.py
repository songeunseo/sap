"""Characterize the fixed-100 GSM8K collapse curve for uniform DLM-Wanda.

This is diagnosis only. It reuses the frozen 50% and 75% uniform masks, creates
only the missing 60/65/70% masks with the unchanged DLM-Wanda score, and uses
the exact EXP-002 GSM8K mini protocol for every newly evaluated point.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.dlm_loss_aggregation.core import mask_sha256, pack_mask, unpack_mask
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _overall_mask_hash,
    _read_jsonl,
    _write_jsonl,
    load_config as load_exp002_config,
)
from experiments.dlm_loss_aggregation.run import _load_model, load_config as load_base_config
from experiments.oracle_allocation75.core import sha256_file
from experiments.wanda_failure_characterization.run_failure_map import model_sha, modules


ROOT = Path("experiments/uniform_wanda_sparsity_sweep")
STORE = Path("/DATA/tmluser1/sap-uniform-wanda-sparsity-sweep")
MASK_STORE = STORE / "masks"
PREREG = ROOT / "preregistered.json"
MASK_MANIFEST = ROOT / "mask_manifest.json"
RESULTS_DIR = ROOT / "gsm8k_mini"
SUMMARY = ROOT / "summary.json"
DLM_STATS = Path("experiments/cgq_wanda_structured_diagnostic/sufficient_statistics.pt")
EXP002 = Path("experiments/dlm_loss_aggregation/exp002")
CONFIG = EXP002 / "config.yaml"
SOURCE_50_MANIFEST = Path("experiments/oracle_allocation50/mask_manifest.json")
SOURCE_75_MANIFEST = Path("experiments/oracle_allocation75/mask_manifest.json")
SOURCE_75_RESULT = Path("experiments/oracle_allocation75/stage2/V0_uniform.json")
SOURCE_75_PREDICTIONS = Path("experiments/oracle_allocation75/stage2/V0_uniform_predictions.jsonl")
MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
EXPECTED_DENSE_SHA = "2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc"
TARGETS = (0.50, 0.60, 0.65, 0.70, 0.75)
NEW_TARGETS = (0.60, 0.65, 0.70)
IDENTIFIER = {
    0.50: "V0_50_uniform",
    0.60: "V0_60_uniform",
    0.65: "V0_65_uniform",
    0.70: "V0_70_uniform",
    0.75: "V0_uniform",
}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _raw_mask_sha(mask: torch.Tensor) -> str:
    return hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def freeze() -> None:
    if PREREG.exists():
        fixed = json.loads(PREREG.read_text())
        if fixed.get("status") != "frozen_before_new_masks_or_outcomes":
            raise RuntimeError("invalid existing preregistration")
        print(json.dumps({"event": "preregister_reuse", "path": str(PREREG)}), flush=True)
        return
    for path in (
        DLM_STATS,
        CONFIG,
        SOURCE_50_MANIFEST,
        SOURCE_75_MANIFEST,
        SOURCE_75_RESULT,
        SOURCE_75_PREDICTIONS,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    config = load_exp002_config(CONFIG)
    config_hash, protocol = _evaluation_config_hash(config)
    dense_records = _read_jsonl(EXP002 / "results/predictions/dense.jsonl")[:100]
    source_75 = json.loads(SOURCE_75_RESULT.read_text())
    if source_75["protocol_hash"] != config_hash or source_75["limit"] != 100:
        raise RuntimeError("75% result is incompatible with the frozen mini protocol")
    source_75_predictions = _read_jsonl(SOURCE_75_PREDICTIONS)
    _assert_same_examples(dense_records, source_75_predictions)
    write_json(
        PREREG,
        {
            "status": "frozen_before_new_masks_or_outcomes",
            "research_question": "Where between 50% and 75% does uniform DLM-Wanda GSM8K performance collapse?",
            "scope": "diagnosis_only_no_new_score_no_allocation_no_rollout_yet",
            "model": {
                "id": config["model"]["id"],
                "revision": MODEL_REVISION,
                "expected_dense_sha256": EXPECTED_DENSE_SHA,
            },
            "nominal_sparsities": list(TARGETS),
            "new_mask_sparsities": list(NEW_TARGETS),
            "score": "abs(W_ij) * sqrt(overall_uniform DLM activation statistic_j)",
            "mask_semantics": "repository-equivalent row-wise int(in_features * sparsity), stable ascending score order",
            "calibration": {
                "path": str(DLM_STATS),
                "sha256": sha256_file(DLM_STATS),
                "label": "historical 80-state matched DLM-Wanda activation cache",
            },
            "evaluation": {
                "config_path": str(CONFIG),
                "config_sha256": sha256_file(CONFIG),
                "protocol_hash": config_hash,
                "protocol": protocol,
                "limit": 100,
                "dense_example_ids_sha256": hashlib.sha256(
                    json.dumps([row.get("doc_id", row.get("id")) for row in dense_records], sort_keys=True).encode()
                ).hexdigest(),
            },
            "reuse": {
                "mask_50_manifest": {"path": str(SOURCE_50_MANIFEST), "sha256": sha256_file(SOURCE_50_MANIFEST)},
                "mask_75_manifest": {"path": str(SOURCE_75_MANIFEST), "sha256": sha256_file(SOURCE_75_MANIFEST)},
                "result_75": {"path": str(SOURCE_75_RESULT), "sha256": sha256_file(SOURCE_75_RESULT)},
                "predictions_75": {"path": str(SOURCE_75_PREDICTIONS), "sha256": sha256_file(SOURCE_75_PREDICTIONS)},
            },
        },
    )
    print(json.dumps({"event": "preregister_complete", "path": str(PREREG)}), flush=True)


def require_frozen() -> dict:
    fixed = json.loads(PREREG.read_text())
    if fixed.get("status") != "frozen_before_new_masks_or_outcomes":
        raise RuntimeError("sweep was not preregistered")
    checks = {
        DLM_STATS: fixed["calibration"]["sha256"],
        CONFIG: fixed["evaluation"]["config_sha256"],
        SOURCE_50_MANIFEST: fixed["reuse"]["mask_50_manifest"]["sha256"],
        SOURCE_75_MANIFEST: fixed["reuse"]["mask_75_manifest"]["sha256"],
        SOURCE_75_RESULT: fixed["reuse"]["result_75"]["sha256"],
        SOURCE_75_PREDICTIONS: fixed["reuse"]["predictions_75"]["sha256"],
    }
    for path, expected in checks.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen artifact changed: {path}: {actual} != {expected}")
    return fixed


def module_parts(name: str) -> tuple[int, str]:
    block, module_type = name.split(".", 1)
    return int(block.removeprefix("block_")), module_type


def recover_existing_masks(fixed: dict) -> bool:
    """Recover metadata after a completed mask write but failed manifest write."""
    reference = json.loads(SOURCE_50_MANIFEST.read_text())
    reference_rows = {
        row["canonical_name"]: row
        for row in reference["entries"]
        if row["method"] == "V0_50_uniform"
    }
    if len(reference_rows) != 224:
        raise RuntimeError("50% reference manifest does not contain 224 uniform modules")
    expected_paths = {
        MASK_STORE / IDENTIFIER[target] / f"{name}.bin"
        for target in NEW_TARGETS
        for name in reference_rows
    }
    existing_paths = set(MASK_STORE.rglob("*.bin")) if MASK_STORE.exists() else set()
    if not expected_paths.issubset(existing_paths):
        return False
    unexpected = existing_paths - expected_paths
    if unexpected:
        raise RuntimeError(f"unexpected recovered mask files: {len(unexpected)}")
    entries = []
    aggregates = {target: {"pruned": 0, "weights": 0} for target in NEW_TARGETS}
    for target in NEW_TARGETS:
        identifier = IDENTIFIER[target]
        for name in sorted(reference_rows):
            shape = reference_rows[name]["shape"]
            path = MASK_STORE / identifier / f"{name}.bin"
            bits = path.read_bytes()
            packed = {"shape": shape, "bits": bits}
            mask = unpack_mask(packed)
            prune_per_row = int(shape[1] * target)
            row_counts = mask.to("cuda").sum(dim=1)
            if not bool(torch.all(row_counts == prune_per_row)):
                raise RuntimeError(f"recovered mask row-count mismatch: {identifier} {name}")
            block_index, module_type = module_parts(name)
            entry = {
                "method": identifier,
                "nominal_sparsity": target,
                "block_index": block_index,
                "module": module_type,
                "canonical_name": name,
                "shape": shape,
                "byte_length": len(bits),
                "sha256": mask_sha256(packed),
                "raw_sha256": None,
                "runtime_path": str(path),
                "prune_per_row": prune_per_row,
                "actual_sparsity": prune_per_row / shape[1],
                "pruned": shape[0] * prune_per_row,
                "weights": shape[0] * shape[1],
            }
            entries.append(entry)
            aggregates[target]["pruned"] += entry["pruned"]
            aggregates[target]["weights"] += entry["weights"]
    summaries, hashes = {}, {}
    for target in NEW_TARGETS:
        identifier = IDENTIFIER[target]
        rows = [row for row in entries if row["method"] == identifier]
        aggregate = aggregates[target]
        achieved = aggregate["pruned"] / aggregate["weights"]
        summaries[identifier] = {
            "nominal_sparsity": target,
            "achieved_global_sparsity": achieved,
            "target_error": achieved - target,
            "total_pruned": aggregate["pruned"],
            "total_weights": aggregate["weights"],
        }
        hashes[identifier] = _overall_mask_hash(rows, identifier)
    write_json(
        MASK_MANIFEST,
        {
            "status": "complete",
            "recovery_note": "Recovered and fully revalidated the 672 completed mask files after a metadata-only KeyError before the original manifest write.",
            "model_revision": MODEL_REVISION,
            "dense_model_sha256_before": EXPECTED_DENSE_SHA,
            "dense_model_sha256_after": EXPECTED_DENSE_SHA,
            "activation_statistics_sha256": fixed["calibration"]["sha256"],
            "score": fixed["score"],
            "shared_score_sort_seconds": None,
            "peak_gpu_memory_bytes": None,
            "overall_sha256": hashes,
            "summaries": summaries,
            "entries": entries,
        },
    )
    print(json.dumps({"event": "mask_manifest_recovered", "summaries": summaries}, indent=2), flush=True)
    return True


@torch.inference_mode()
def prepare_masks() -> None:
    fixed = require_frozen()
    if MASK_MANIFEST.exists():
        print(json.dumps({"event": "mask_prepare_reuse", "path": str(MASK_MANIFEST)}), flush=True)
        return
    if recover_existing_masks(fixed):
        return
    stats = torch.load(DLM_STATS, map_location="cpu", weights_only=False)
    activation = {name: values["overall_uniform"].float() for name, values in stats["statistics"].items()}
    config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, _ = _load_model(config)
    before = model_sha(model)
    if before != EXPECTED_DENSE_SHA:
        raise RuntimeError(f"dense model SHA mismatch: {before}")
    mapping = modules(model)
    if set(mapping) != set(activation) or len(mapping) != 224:
        raise RuntimeError("activation/module mapping mismatch")
    entries = []
    aggregates = {target: {"pruned": 0, "weights": 0} for target in NEW_TARGETS}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for module_index, name in enumerate(sorted(mapping)):
        layer = mapping[name]
        a = activation[name]
        if tuple(a.shape) != (layer.weight.shape[1],) or not torch.isfinite(a).all():
            raise RuntimeError(f"bad activation statistic: {name}")
        score = layer.weight.detach().float().abs() * a.to(layer.weight.device).sqrt()[None, :]
        order = torch.argsort(score, dim=1, stable=True)
        for target in NEW_TARGETS:
            prune_per_row = int(layer.weight.shape[1] * target)
            mask = torch.zeros_like(layer.weight, dtype=torch.bool)
            mask.scatter_(1, order[:, :prune_per_row], True)
            mask_cpu = mask.cpu()
            packed = pack_mask(mask_cpu)
            output = MASK_STORE / IDENTIFIER[target] / f"{name}.bin"
            _atomic_bytes(output, packed["bits"])
            entry = {
                "method": IDENTIFIER[target],
                "nominal_sparsity": target,
                "block_index": module_parts(name)[0],
                "module": module_parts(name)[1],
                "canonical_name": name,
                "shape": list(mask_cpu.shape),
                "byte_length": len(packed["bits"]),
                "sha256": mask_sha256(packed),
                "raw_sha256": _raw_mask_sha(mask_cpu),
                "runtime_path": str(output),
                "prune_per_row": prune_per_row,
                "actual_sparsity": float(mask_cpu.float().mean()),
                "pruned": int(mask_cpu.sum()),
                "weights": mask_cpu.numel(),
            }
            entries.append(entry)
            aggregates[target]["pruned"] += entry["pruned"]
            aggregates[target]["weights"] += entry["weights"]
            del mask, mask_cpu, packed
        del score, order
        print(f"sweep masks {module_index + 1}/224 {name}", flush=True)
    after = model_sha(model)
    if before != after:
        raise RuntimeError("mask construction modified model weights")
    summaries, hashes = {}, {}
    for target in NEW_TARGETS:
        identifier = IDENTIFIER[target]
        rows = [row for row in entries if row["method"] == identifier]
        aggregate = aggregates[target]
        achieved = aggregate["pruned"] / aggregate["weights"]
        summaries[identifier] = {
            "nominal_sparsity": target,
            "achieved_global_sparsity": achieved,
            "target_error": achieved - target,
            "total_pruned": aggregate["pruned"],
            "total_weights": aggregate["weights"],
        }
        hashes[identifier] = _overall_mask_hash(rows, identifier)
    write_json(
        MASK_MANIFEST,
        {
            "status": "complete",
            "model_revision": MODEL_REVISION,
            "dense_model_sha256_before": before,
            "dense_model_sha256_after": after,
            "activation_statistics_sha256": fixed["calibration"]["sha256"],
            "score": fixed["score"],
            "shared_score_sort_seconds": time.monotonic() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "overall_sha256": hashes,
            "summaries": summaries,
            "entries": entries,
        },
    )
    print(json.dumps({"event": "mask_prepare_complete", "summaries": summaries}, indent=2), flush=True)


def source_manifest_and_identifier(target: float) -> tuple[Path, str]:
    if target == 0.50:
        return SOURCE_50_MANIFEST, "V0_50_uniform"
    if target == 0.75:
        return SOURCE_75_MANIFEST, "V0_uniform"
    return MASK_MANIFEST, IDENTIFIER[target]


def manifest_rows(target: float) -> tuple[list[dict], dict, str]:
    path, identifier = source_manifest_and_identifier(target)
    manifest = json.loads(path.read_text())
    rows = [row for row in manifest["entries"] if row["method"] == identifier]
    if len(rows) != 224:
        raise RuntimeError(f"incomplete mask set for {identifier}: {len(rows)}")
    expected_hash = manifest["overall_sha256"][identifier]
    if _overall_mask_hash(rows, identifier) != expected_hash:
        raise RuntimeError(f"overall mask hash mismatch: {identifier}")
    return rows, manifest, identifier


@torch.inference_mode()
def apply_target(model, target: float) -> dict:
    rows, manifest, identifier = manifest_rows(target)
    mapping = modules(model)
    pruned = weights = 0
    started = time.monotonic()
    for row in rows:
        bits = Path(row["runtime_path"]).read_bytes()
        packed = {"shape": row["shape"], "bits": bits}
        if len(bits) != row["byte_length"] or mask_sha256(packed) != row["sha256"]:
            raise RuntimeError(f"packed mask mismatch: {row['canonical_name']}")
        mask = unpack_mask(packed).to(mapping[row["canonical_name"]].weight.device)
        mapping[row["canonical_name"]].weight.masked_fill_(mask, 0)
        pruned += row["pruned"]
        weights += row["weights"]
        del bits, packed, mask
    return {
        "identifier": identifier,
        "nominal_sparsity": target,
        "achieved_global_sparsity": pruned / weights,
        "pruned": pruned,
        "weights": weights,
        "mask_sha256": manifest["overall_sha256"][identifier],
        "apply_seconds": time.monotonic() - started,
    }


def invalid_answers(records: list[dict]) -> int:
    return sum(row.get("extracted_answer") in (None, "") for row in records)


@torch.inference_mode()
def evaluate_target(target: float) -> dict:
    fixed = require_frozen()
    output = RESULTS_DIR / f"uniform_{int(target * 100):02d}.json"
    predictions = RESULTS_DIR / f"uniform_{int(target * 100):02d}_predictions.jsonl"
    if output.exists() and predictions.exists():
        result = json.loads(output.read_text())
        if result.get("status") == "complete":
            print(json.dumps({"event": "evaluation_reuse", "target": target}), flush=True)
            return result
    config = load_exp002_config(CONFIG)
    config_hash, protocol = _evaluation_config_hash(config)
    if config_hash != fixed["evaluation"]["protocol_hash"]:
        raise RuntimeError("evaluation protocol hash changed")
    dense_records = _read_jsonl(EXP002 / "results/predictions/dense.jsonl")[:100]
    if target == 0.75:
        source_result = json.loads(SOURCE_75_RESULT.read_text())
        source_records = _read_jsonl(SOURCE_75_PREDICTIONS)
        _assert_same_examples(dense_records, source_records)
        result = {
            **source_result,
            "nominal_sparsity": target,
            "achieved_global_sparsity": source_result["mask"]["sparsity"],
            "reuse": {"result": str(SOURCE_75_RESULT), "predictions": str(SOURCE_75_PREDICTIONS)},
        }
        print(json.dumps({"event": "evaluation_75_reused"}), flush=True)
        return result
    base_config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, _ = _load_model(base_config)
    dense_sha = model_sha(model)
    if dense_sha != EXPECTED_DENSE_SHA:
        raise RuntimeError("dense checkpoint mismatch")
    mask = apply_target(model, target)
    sparse_before = model_sha(model)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True
    )
    torch.cuda.reset_peak_memory_stats()
    measured, records = _evaluate_gsm8k(model, tokenizer, config, IDENTIFIER[target], 100, config_hash)
    sparse_after = model_sha(model)
    if sparse_before != sparse_after:
        raise RuntimeError("evaluation modified sparse weights")
    _assert_same_examples(dense_records, records)
    _write_jsonl(predictions, records)
    result = {
        "status": "complete",
        "identifier": IDENTIFIER[target],
        "nominal_sparsity": target,
        "achieved_global_sparsity": mask["achieved_global_sparsity"],
        "correct": sum(row["correct"] for row in records),
        "accuracy": sum(row["correct"] for row in records) / len(records),
        "invalid_answer_count": invalid_answers(records),
        "dense_extracted_answer_agreement": sum(
            row.get("extracted_answer") == dense.get("extracted_answer")
            for row, dense in zip(records, dense_records)
        ) / len(records),
        "limit": len(records),
        "evaluation_seconds": measured["eval_seconds"],
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "protocol_hash": config_hash,
        "protocol": protocol,
        "mask": mask,
        "dense_model_sha256": dense_sha,
        "sparse_model_sha256_before_eval": sparse_before,
        "sparse_model_sha256_after_eval": sparse_after,
        "predictions_path": str(predictions),
        "predictions_sha256": sha256_file(predictions),
    }
    write_json(output, result)
    print(json.dumps({"event": "evaluation_complete", "target": target, "correct": result["correct"]}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_all() -> None:
    prepare_masks()
    results = [evaluate_target(target) for target in TARGETS]
    write_json(ROOT / "evaluation_results.json", {"status": "complete", "results": results})


def summarize() -> None:
    fixed = require_frozen()
    results = []
    for target in TARGETS:
        if target == 0.75:
            result = evaluate_target(target)
        else:
            path = RESULTS_DIR / f"uniform_{int(target * 100):02d}.json"
            if not path.exists():
                raise RuntimeError(f"missing evaluation result: {path}")
            result = json.loads(path.read_text())
        results.append(
            {
                "nominal_sparsity": target,
                "achieved_global_sparsity": result["achieved_global_sparsity"],
                "correct": result["correct"],
                "accuracy": result["accuracy"],
                "invalid_answer_count": result["invalid_answer_count"],
                "dense_extracted_answer_agreement": result.get("dense_extracted_answer_agreement"),
                "evaluation_seconds": result["evaluation_seconds"],
                "mask_sha256": result["mask"]["mask_sha256"],
                "reused": target == 0.75,
            }
        )
    write_json(
        SUMMARY,
        {
            "status": "complete",
            "research_question": fixed["research_question"],
            "protocol_hash": fixed["evaluation"]["protocol_hash"],
            "metric": "fixed-100 GSM8K strict exact match",
            "results": results,
            "next_action_not_run": "rollout characterization near the observed transition",
        },
    )
    print(json.dumps({"event": "summary_complete", "results": results}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=("freeze", "prepare", "evaluate", "summarize", "all"))
    args = parser.parse_args()
    if args.job in ("freeze", "all"):
        freeze()
    if args.job in ("prepare", "all"):
        prepare_masks()
    if args.job in ("evaluate", "all"):
        evaluate_all()
    if args.job in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()
