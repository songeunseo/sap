"""Run the EXP-004 Reveal-50x mask-sensitivity stress test.

This diagnostic keeps the frozen EXP-004 states, reveal/remain partition, DLM
loss, ABS aggregation, and exact row-wise 50% pruning semantics.  It evaluates
Reveal-50x on the fixed GSM8K mini-100 only when the preregistered mask gate is
crossed.  Reveal-50x is intentionally extreme and is not a tuned method.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

import torch

from experiments.dlm_loss_aggregation.core import (
    mask_sha256,
    pack_mask,
    rowwise_mask,
)
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _module_map,
    _overall_mask_hash,
    _read_jsonl,
    _validated_dense_fingerprint,
    _write_jsonl,
    apply_dlm_masks,
    dense_fingerprint,
    load_config as load_exp002_config,
)
from experiments.dlm_loss_aggregation.exp004.run import (
    _ranked_comparisons,
    _tensor_sha256,
    allow_repeated_compiled_backwards,
    freeze_token_partition,
    summarize_mask_diagnostics,
    validate_partition_artifact,
    weighted_dlm_losses,
)
from experiments.dlm_loss_aggregation.run import (
    _atomic_write_json,
    _cuda_peaks,
    _initial_cached_states,
    _load_model,
    _module_specs,
    _reset_cuda_peaks,
    _suffix_logits,
    historical_state_digest,
    load_config as load_exp001_config,
    require_historical_digest,
    validate_config as validate_exp001_config,
)


ROOT = Path("experiments/dlm_loss_aggregation/exp004_strong_reveal")
DEFAULT_CONFIG = ROOT / "preregistered.json"
CONDITIONS = ("uniform_abs", "reveal2_abs", "reveal50_abs")
RATIOS = {"uniform_abs": 1.0, "reveal2_abs": 2.0, "reveal50_abs": 50.0}
PRIMARY_PAIR = "reveal50_abs/uniform_abs"


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "experiment": "EXP-004-STRONG-REVEAL",
        "status": "frozen_before_strong_reveal_scoring",
        "model.id": "GSAI-ML/LLaDA-8B-Base",
        "model.revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
        "model.dtype": "bfloat16",
        "calibration.state_count": 80,
        "calibration.state_sha256": "1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df",
        "calibration.partition.masked_observations": 10267,
        "calibration.partition.reveal_observations": 116,
        "calibration.partition.remain_observations": 10151,
        "mask.sparsity": 0.5,
        "evaluation.limit": 100,
        "evaluation.protocol_hash": "1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add",
        "evaluation.uniform_predictions.expected_correct": 58,
        "evaluation.reveal2_predictions.expected_correct": 59,
        "storage.persisted_methods": ["reveal50_abs"],
    }
    for dotted, wanted in expected.items():
        actual = config
        for key in dotted.split("."):
            actual = actual[key]
        if actual != wanted:
            raise ValueError(f"strong-reveal preregistration mismatch: {dotted}")
    observed_conditions = [row["name"] for row in config["conditions"]]
    observed_ratios = {row["name"]: row["reveal_raw"] for row in config["conditions"]}
    if observed_conditions != list(CONDITIONS) or observed_ratios != RATIOS:
        raise ValueError("strong-reveal conditions or ratios changed")
    if any(row["remain_raw"] != 1.0 for row in config["conditions"]):
        raise ValueError("remain raw weight must stay one")
    if config["diagnostics"]["primary_pair"] != ["reveal50_abs", "uniform_abs"]:
        raise ValueError("primary diagnostic pair changed")
    return config


def _verify_file(source: dict) -> Path:
    path = Path(source["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != source["file_sha256"]:
        raise RuntimeError(f"frozen source changed: {path}: {actual}")
    return path


def _load_sources(config: dict) -> dict:
    calibration = config["calibration"]
    manifest_path = _verify_file(calibration["manifest"])
    partition_path = _verify_file(calibration["partition"])
    score_metadata_path = _verify_file(calibration["score_metadata"])
    evaluation = config["evaluation"]
    evaluation_config_path = _verify_file(evaluation["config"])
    uniform_path = _verify_file(evaluation["uniform_predictions"])
    reveal2_path = _verify_file(evaluation["reveal2_predictions"])
    historical_path = _verify_file(config["historical_reference"]["source"])

    exp001_config = validate_exp001_config(
        load_exp001_config("experiments/dlm_loss_aggregation/config.yaml")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = require_historical_digest(manifest, calibration["state_sha256"])
    if (
        digest != historical_state_digest(manifest)
        or len(manifest["states"]) != calibration["state_count"]
    ):
        raise RuntimeError("frozen calibration manifest identity changed")

    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    validate_partition_artifact(partition, manifest, calibration["state_count"])
    expected_partition_counts = {
        "masked_tokens": calibration["partition"]["masked_observations"],
        "reveal_tokens": calibration["partition"]["reveal_observations"],
        "remain_tokens": calibration["partition"]["remain_observations"],
    }
    if any(partition["summary"][key] != value for key, value in expected_partition_counts.items()):
        raise RuntimeError("frozen partition counts changed")

    score_metadata = json.loads(score_metadata_path.read_text(encoding="utf-8"))
    if (
        score_metadata.get("state_count") != calibration["state_count"]
        or score_metadata.get("matrix_count") != 224
        or len(score_metadata.get("modules", [])) != 224
        or any(row.get("update_count") != 80 for row in score_metadata["modules"])
    ):
        raise RuntimeError("EXP-001 score metadata is incomplete")

    exp002_config = load_exp002_config(evaluation_config_path)
    evaluation_hash, evaluation_document = _evaluation_config_hash(exp002_config)
    if evaluation_hash != evaluation["protocol_hash"]:
        raise RuntimeError("GSM8K evaluation protocol changed")

    limit = evaluation["limit"]
    uniform_records = _read_jsonl(uniform_path)[:limit]
    reveal2_all = _read_jsonl(reveal2_path)
    reveal2_records = [
        row for row in reveal2_all
        if row.get("method") == evaluation["reveal2_predictions"]["method"]
    ][:limit]
    if len(uniform_records) != limit or len(reveal2_records) != limit:
        raise RuntimeError("fixed mini-100 baseline records are incomplete")
    _assert_same_examples(uniform_records, reveal2_records)
    for records in (uniform_records, reveal2_records):
        if any(row.get("evaluation_config_hash") != evaluation_hash for row in records):
            raise RuntimeError("mini baseline protocol hash changed")
    if sum(row["correct"] for row in uniform_records) != evaluation["uniform_predictions"]["expected_correct"]:
        raise RuntimeError("frozen Uniform mini accuracy changed")
    if sum(row["correct"] for row in reveal2_records) != evaluation["reveal2_predictions"]["expected_correct"]:
        raise RuntimeError("frozen Reveal-2x mini accuracy changed")

    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_global = {
        row["pair"]: row for row in historical["rows"] if row["scope"] == "global"
    }
    old = historical_global.get("reveal_abs/uniform_abs")
    reference = config["historical_reference"]
    if (
        old is None
        or old["spearman"] != reference["reveal2_vs_uniform_spearman"]
        or old["mask_xor"] != reference["reveal2_vs_uniform_mask_xor"]
    ):
        raise RuntimeError("historical Reveal-2x reference changed")

    return {
        "exp001_config": exp001_config,
        "manifest": manifest,
        "partition": partition,
        "score_metadata": score_metadata,
        "exp002_config": exp002_config,
        "evaluation_hash": evaluation_hash,
        "evaluation_document": evaluation_document,
        "uniform_records": uniform_records,
        "reveal2_records": reveal2_records,
        "historical": old,
    }


def token_weights_ratio(
    mask: torch.Tensor, reveal_mask: torch.Tensor, reveal_ratio: float
) -> torch.Tensor:
    if mask.dtype != torch.bool or reveal_mask.dtype != torch.bool or mask.shape != reveal_mask.shape:
        raise ValueError("mask and reveal_mask must be matching boolean tensors")
    if (reveal_mask & ~mask).any().item() or not mask.any().item():
        raise ValueError("reveal positions must be a nonempty subset of masked positions")
    if not math.isfinite(reveal_ratio) or reveal_ratio < 1.0:
        raise ValueError("reveal ratio must be finite and at least one")
    raw = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    raw[mask] = 1.0
    raw[reveal_mask] = reveal_ratio
    raw[mask] /= raw[mask].mean()
    one = torch.tensor(1.0, dtype=torch.float32, device=mask.device)
    if not torch.isclose(raw[mask].mean(), one, atol=1e-6, rtol=0):
        raise RuntimeError("normalized masked-token weight mean differs from one")
    return raw


def summarize_token_weight_ratio(partition: dict, reveal_ratio: float) -> dict:
    masked = partition["masked_count"]
    reveal = partition["reveal_count"]
    remain = partition["remain_count"]
    if masked <= 0 or reveal <= 0 or remain <= 0 or reveal + remain != masked:
        raise ValueError("partition counts are invalid")
    raw_mean = (reveal * reveal_ratio + remain) / masked
    reveal_alpha = reveal_ratio / raw_mean
    remain_alpha = 1.0 / raw_mean
    return {
        "masked_count": masked,
        "reveal_count": reveal,
        "remain_count": remain,
        "reveal_raw": reveal_ratio,
        "remain_raw": 1.0,
        "raw_mean": raw_mean,
        "reveal_alpha": reveal_alpha,
        "remain_alpha": remain_alpha,
        "normalized_mean": (reveal * reveal_alpha + remain * remain_alpha) / masked,
        "reveal_normalized_weight_mass_share": reveal * reveal_alpha / masked,
    }


def independent_weight_effects(losses, parameters, frozen_cpu_weights):
    if tuple(losses) != CONDITIONS or not parameters:
        raise ValueError("losses must contain the three preregistered conditions in order")
    values = tuple(parameters.values())
    if any(parameter.grad is not None for parameter in values):
        raise RuntimeError("parameter gradient leaked before condition scoring")
    effects = {}
    for index, condition in enumerate(CONDITIONS):
        gradients = torch.autograd.grad(
            losses[condition], values, retain_graph=index + 1 < len(CONDITIONS)
        )
        effects[condition] = {
            name: -frozen_cpu_weights[name]
            * gradient.detach().to(device="cpu", dtype=torch.float32)
            for (name, _), gradient in zip(parameters.items(), gradients)
        }
        if any(parameter.grad is not None for parameter in values):
            raise RuntimeError("condition gradient accumulated on a parameter")
    return effects


class DeviceAbsAccumulator:
    """Accumulate the unchanged FP32 ABS score without per-state PCIe copies."""

    def __init__(self, shape, device):
        self.absolute = torch.zeros(shape, dtype=torch.float32, device=device)
        self.count = 0

    def add_gradient(self, frozen_weight_fp32, gradient):
        if gradient.shape != self.absolute.shape:
            raise ValueError("gradient shape differs from the ABS accumulator")
        self.absolute.add_((frozen_weight_fp32 * gradient.detach().float()).abs())
        self.count += 1

    def finalize(self):
        if not self.count:
            raise RuntimeError("cannot finalize an empty ABS accumulator")
        score = (self.absolute / self.count).cpu()
        if not torch.isfinite(score).all().item():
            raise RuntimeError("non-finite finalized ABS score")
        return score


def accumulate_weight_effects(losses, parameters, frozen_weights, accumulators):
    """Run isolated VJPs and update GPU-resident FP32 ABS accumulators."""
    if tuple(losses) != CONDITIONS or not parameters:
        raise ValueError("losses must contain the three preregistered conditions in order")
    values = tuple(parameters.values())
    if any(parameter.grad is not None for parameter in values):
        raise RuntimeError("parameter gradient leaked before condition scoring")
    for index, condition in enumerate(CONDITIONS):
        gradients = torch.autograd.grad(
            losses[condition], values, retain_graph=index + 1 < len(CONDITIONS)
        )
        for (name, _), gradient in zip(parameters.items(), gradients):
            accumulators[condition][name].add_gradient(
                frozen_weights[name], gradient
            )
        if any(parameter.grad is not None for parameter in values):
            raise RuntimeError("condition gradient accumulated on a parameter")


def _mask_entry(method: str, block_index: int, module: str, mask: torch.Tensor) -> dict:
    payload = pack_mask(mask)
    return {
        "method": method,
        "block_index": block_index,
        "module": module,
        "shape": list(payload["shape"]),
        "byte_length": len(payload["bits"]),
        "sha256": mask_sha256(payload),
    }


def write_packed_mask(
    mask_root: Path, method: str, block_index: int, module: str, mask: torch.Tensor
) -> dict:
    payload = pack_mask(mask)
    directory = mask_root / method
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"block_{block_index:02d}__{module.replace('.', '_')}.bin"
    temporary = path.with_suffix(".bin.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload["bits"])
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    saved = {"shape": payload["shape"], "bits": path.read_bytes()}
    digest = mask_sha256(payload)
    if len(saved["bits"]) != len(payload["bits"]) or mask_sha256(saved) != digest:
        raise RuntimeError("packed Reveal-50x mask checksum readback failed")
    return {
        "method": method,
        "block_index": block_index,
        "module": module,
        "shape": list(payload["shape"]),
        "byte_length": len(payload["bits"]),
        "sha256": digest,
        "runtime_path": str(path.resolve()),
    }


def write_score_tensor(
    score_root: Path, method: str, block_index: int, module: str, score: torch.Tensor
) -> dict:
    """Persist a CPU FP32 score tensor atomically and verify exact readback."""
    directory = score_root / method
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"block_{block_index:02d}__{module.replace('.', '_')}.pt"
    temporary = path.with_suffix(".pt.tmp")
    cpu_score = score.detach().to(device="cpu", dtype=torch.float32).contiguous()
    torch.save(cpu_score, temporary)
    temporary.replace(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.equal(saved, cpu_score):
        raise RuntimeError("score tensor readback differs from persisted tensor")
    return {"runtime_path": str(path.resolve()), "file_sha256": sha256_file(path)}


def _weight_summaries(partition: dict) -> dict:
    states = []
    for state in partition["states"]:
        states.append(
            {
                "state_index": state["state_index"],
                **{
                    condition: summarize_token_weight_ratio(state, ratio)
                    for condition, ratio in RATIOS.items()
                },
            }
        )
    return {"state_count": len(states), "states": states}


def run_preflight(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    config = load_config(config_path)
    sources = _load_sources(config)
    mask_root = Path(config["storage"]["mask_root"])
    mask_root.mkdir(parents=True, exist_ok=True)
    one_method_bytes = sum(
        math.prod(row["shape"]) for row in sources["score_metadata"]["modules"]
    ) // 8
    required = one_method_bytes + config["storage"]["safety_bytes"]
    available = shutil.disk_usage(mask_root).free
    if available < required:
        raise RuntimeError(f"insufficient mask storage: need {required}, have {available}")

    weights = _weight_summaries(sources["partition"])
    strong = [row["reveal50_abs"] for row in weights["states"]]
    result = {
        "status": "passed",
        "checked_at_unix": time.time(),
        "state_sha256": config["calibration"]["state_sha256"],
        "partition_sha256": sources["partition"]["sha256"],
        "state_count": len(sources["partition"]["states"]),
        "masked_observations": sum(row["masked_count"] for row in strong),
        "reveal_observations": sum(row["reveal_count"] for row in strong),
        "reveal_fraction": sum(row["reveal_count"] for row in strong)
        / sum(row["masked_count"] for row in strong),
        "reveal_count_per_state": {
            "min": min(row["reveal_count"] for row in strong),
            "max": max(row["reveal_count"] for row in strong),
        },
        "strong_weight_ranges": {
            key: {
                "min": min(row[key] for row in strong),
                "max": max(row[key] for row in strong),
                "mean": statistics.fmean(row[key] for row in strong),
            }
            for key in (
                "reveal_alpha",
                "remain_alpha",
                "reveal_normalized_weight_mass_share",
            )
        },
        "mini_baselines": {
            "uniform_abs": {
                "correct": sum(row["correct"] for row in sources["uniform_records"]),
                "count": len(sources["uniform_records"]),
            },
            "reveal2_abs": {
                "correct": sum(row["correct"] for row in sources["reveal2_records"]),
                "count": len(sources["reveal2_records"]),
            },
        },
        "historical_reveal2_vs_uniform": sources["historical"],
        "evaluation_protocol_hash": sources["evaluation_hash"],
        "mask_storage": {
            "path": str(mask_root),
            "one_method_bytes": one_method_bytes,
            "required_bytes": required,
            "available_bytes": available,
        },
    }
    _atomic_write_json(ROOT / "token_weight_summary.json", weights)
    _atomic_write_json(ROOT / "logs" / "preflight.json", result)
    return result


def decision_from_diagnostics(primary: dict) -> dict:
    xor = primary["mask_xor"]
    spearman = primary["spearman"]
    if spearman is None:
        raise ValueError("primary Spearman is undefined")
    if xor < 0.01 and spearman > 0.99:
        interpretation = "ranking_redundant"
    elif xor >= 0.02:
        interpretation = "weighting_strength_supported"
    else:
        interpretation = "intermediate_mask_sensitivity"
    run_downstream = xor >= 0.01 or spearman <= 0.99
    return {
        "interpretation": interpretation,
        "run_downstream": run_downstream,
        "gate": "mask_xor >= 0.01 or spearman <= 0.99",
    }


def run_scoring_shard(
    config_path: Path | str = DEFAULT_CONFIG,
    block_start: int = 0,
    block_end: int = 32,
) -> dict:
    from lib.prune_llada import find_layers

    allow_repeated_compiled_backwards()
    config = load_config(config_path)
    if not (0 <= block_start < block_end <= 32):
        raise ValueError("score shard must be a nonempty block interval within [0, 32]")
    preflight = json.loads((ROOT / "logs" / "preflight.json").read_text())
    if preflight.get("status") != "passed":
        raise RuntimeError("preflight must pass before scoring")
    sources = _load_sources(config)
    model, devices = _load_model(sources["exp001_config"])
    dense_model_sha = _validated_dense_fingerprint(
        model, sources["exp002_config"], sources["score_metadata"]
    )
    if dense_model_sha != sources["partition"]["dense_fingerprint"]:
        raise RuntimeError("dense model fingerprint differs from frozen partition")
    specs = _module_specs(model)
    if len(specs) != 224:
        raise RuntimeError("prunable matrix universe changed")

    shard_name = f"blocks_{block_start:02d}_{block_end:02d}"
    shard_path = ROOT / "logs" / "shards" / f"{shard_name}.json"
    cached_states = _initial_cached_states(model, sources["manifest"])
    parameters = list(model.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    _reset_cuda_peaks(devices)
    started = time.perf_counter()
    _atomic_write_json(
        ROOT / "logs" / "shards" / f"{shard_name}_start.json",
        {
            "status": "running",
            "started_at_unix": time.time(),
            "resume_supported": True,
            "conditions": list(CONDITIONS),
            "block_start": block_start,
            "block_end": block_end,
        },
    )

    score_documents = {
        condition: {"version": 1, "method": condition, "modules": []}
        for condition in CONDITIONS
    }
    mask_entries = {condition: [] for condition in CONDITIONS}
    matrix_diagnostics = []
    loss_sums = {condition: 0.0 for condition in CONDITIONS}
    loss_count = 0
    completed_blocks = 0
    previous_runtime = 0.0
    if shard_path.exists():
        checkpoint = json.loads(shard_path.read_text())
        if (
            checkpoint.get("block_start") != block_start
            or checkpoint.get("block_end") != block_end
            or checkpoint.get("dense_model_sha256") != dense_model_sha
        ):
            raise RuntimeError("stale score shard checkpoint")
        if checkpoint.get("status") == "passed":
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return checkpoint
        if checkpoint.get("status") != "running":
            raise RuntimeError("invalid score shard checkpoint status")
        completed_blocks = checkpoint["completed_blocks"]
        if not (0 < completed_blocks < block_end - block_start):
            raise RuntimeError("invalid score shard checkpoint coverage")
        score_documents = checkpoint["score_documents"]
        mask_entries = checkpoint["mask_entries"]
        matrix_diagnostics = checkpoint["matrix_diagnostics"]
        loss_sums = checkpoint["loss_sums"]
        loss_count = checkpoint["loss_count"]
        previous_runtime = checkpoint["runtime_seconds"]
    comparison_pairs = tuple(tuple(pair) for pair in config["diagnostics"]["pairs"])
    mask_root = Path(config["storage"]["mask_root"])
    resume_from = block_start + completed_blocks

    try:
        for block_index, block in enumerate(model.model.transformer.blocks):
            if block_index < resume_from:
                next_hiddens = []
                with torch.no_grad():
                    for state in cached_states:
                        parameter = next(block.parameters())
                        output, _ = block(
                            state["hidden"].to(device=parameter.device, dtype=parameter.dtype),
                            attention_bias=None,
                            layer_past=None,
                            use_cache=False,
                            replace_position=None,
                            attn_collector=None,
                        )
                        next_hiddens.append(output.detach().cpu())
                for state, hidden in zip(cached_states, next_hiddens):
                    state["hidden"] = hidden
                continue
            if block_index >= block_end:
                break
            layers = find_layers(block)
            for layer in layers.values():
                layer.weight.requires_grad_(True)
            accumulators = {
                condition: {
                    name: DeviceAbsAccumulator(layer.weight.shape, layer.weight.device)
                    for name, layer in layers.items()
                }
                for condition in CONDITIONS
            }
            frozen_weights = {
                name: layer.weight.detach().to(dtype=torch.float32).clone()
                for name, layer in layers.items()
            }
            next_hiddens = []
            for state, partition in zip(cached_states, sources["partition"]["states"]):
                parameter = next(block.parameters())
                target_output, _ = block(
                    state["hidden"].to(device=parameter.device, dtype=parameter.dtype),
                    attention_bias=None,
                    layer_past=None,
                    use_cache=False,
                    replace_position=None,
                    attn_collector=None,
                )
                if block_index + 1 < 32:
                    next_hiddens.append(target_output.detach().cpu())
                logits = _suffix_logits(model, target_output, block_index + 1)
                mask = state["mask"].to(logits.device)
                if block_index == 0:
                    marker = torch.zeros(mask.shape, dtype=torch.long, device=mask.device)
                    marker[mask] = sources["manifest"]["mask_id"]
                    observed = freeze_token_partition(
                        logits,
                        marker,
                        sources["manifest"]["mask_id"],
                        state["p_mask"],
                        256,
                    )
                    for key in ("masked_indices", "reveal_indices", "remain_indices", "masked_predictions"):
                        if observed[key] != partition[key]:
                            raise RuntimeError("frozen reveal/remain partition changed during scoring")
                reveal_mask = torch.zeros_like(mask)
                reveal_mask[0, partition["reveal_indices"]] = True
                if (
                    int(mask.sum().item()) != partition["masked_count"]
                    or int(reveal_mask.sum().item()) != partition["reveal_count"]
                    or (reveal_mask & ~mask).any().item()
                ):
                    raise RuntimeError("frozen token partition no longer matches the state")
                alphas = {
                    condition: token_weights_ratio(mask, reveal_mask, ratio)
                    for condition, ratio in RATIOS.items()
                }
                losses = weighted_dlm_losses(
                    logits,
                    state["clean_ids"].to(logits.device),
                    mask,
                    state["p_mask"],
                    alphas,
                )
                accumulate_weight_effects(
                    losses,
                    {name: layer.weight for name, layer in layers.items()},
                    frozen_weights,
                    accumulators,
                )
                for condition in CONDITIONS:
                    if abs(alphas[condition][mask].mean().item() - 1.0) > 1e-6:
                        raise RuntimeError("masked-token alpha mean changed")
                    loss_sums[condition] += losses[condition].detach().cpu().item()
                loss_count += 1
                del target_output, logits, mask, reveal_mask, alphas, losses

            for layer in layers.values():
                layer.weight.requires_grad_(False)
            if next_hiddens:
                for state, hidden in zip(cached_states, next_hiddens):
                    state["hidden"] = hidden

            for name, layer in layers.items():
                module_type = name.rsplit(".", 1)[-1]
                scores = {}
                for condition in CONDITIONS:
                    scores[condition] = accumulators[condition].pop(name).finalize()
                masks = {
                    condition: rowwise_mask(score, config["mask"]["sparsity"])
                    for condition, score in scores.items()
                }
                for condition in CONDITIONS:
                    score = scores[condition]
                    mask = masks[condition]
                    expected_per_row = score.shape[1] // 2
                    if not mask.sum(dim=1).eq(expected_per_row).all().item():
                        raise RuntimeError("mask violates exact row-wise 50% sparsity")
                    if condition in config["storage"]["persisted_methods"]:
                        entry = write_packed_mask(
                            mask_root, condition, block_index, name, mask
                        )
                    else:
                        entry = _mask_entry(condition, block_index, name, mask)
                    mask_entries[condition].append(entry)
                    score_row = {
                            "layer": block_index,
                            "module": name,
                            "module_type": module_type,
                            "shape": list(score.shape),
                            "num_weights": score.numel(),
                            "score_sha256": _tensor_sha256(score),
                            "mean": score.mean().item(),
                            "std": score.std().item(),
                            "min": score.min().item(),
                            "max": score.max().item(),
                            "per_row_prune_count": expected_per_row,
                            "actual_sparsity": mask.float().mean().item(),
                            "mask_sha256": entry["sha256"],
                    }
                    if condition in config["storage"].get("persisted_scores", []):
                        score_row.update(
                            write_score_tensor(
                                Path(config["storage"]["score_root"]),
                                condition,
                                block_index,
                                name,
                                score,
                            )
                        )
                    score_documents[condition]["modules"].append(score_row)
                for row in _ranked_comparisons(scores, masks, comparison_pairs):
                    matrix_diagnostics.append(
                        {
                            "scope": "matrix",
                            "layer": block_index,
                            "module": name,
                            "module_type": module_type,
                            "num_weights": layer.weight.numel(),
                            **row,
                        }
                    )
                del scores, masks
            del accumulators, frozen_weights
            progress = {
                "status": "running",
                "block_start": block_start,
                "block_end": block_end,
                "completed_blocks": block_index - block_start + 1,
                "elapsed_seconds": previous_runtime + time.perf_counter() - started,
                "cuda_peak": _cuda_peaks(devices),
            }
            checkpoint = {
                "status": "running",
                "block_start": block_start,
                "block_end": block_end,
                "completed_blocks": progress["completed_blocks"],
                "runtime_seconds": progress["elapsed_seconds"],
                "score_documents": score_documents,
                "mask_entries": mask_entries,
                "matrix_diagnostics": matrix_diagnostics,
                "loss_sums": loss_sums,
                "loss_count": loss_count,
                "dense_model_sha256": dense_model_sha,
                "cuda_peak": progress["cuda_peak"],
            }
            _atomic_write_json(shard_path, checkpoint)
            _atomic_write_json(
                ROOT / "logs" / "shards" / f"{shard_name}_progress.json", progress
            )
            print(json.dumps(progress, sort_keys=True), flush=True)
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, original in zip(parameters, original_flags):
            parameter.requires_grad_(original)

    result = {
        "status": "passed",
        "block_start": block_start,
        "block_end": block_end,
        "completed_blocks": block_end - block_start,
        "runtime_seconds": previous_runtime + time.perf_counter() - started,
        "score_documents": score_documents,
        "mask_entries": mask_entries,
        "matrix_diagnostics": matrix_diagnostics,
        "loss_sums": loss_sums,
        "loss_count": loss_count,
        "dense_model_sha256": dense_model_sha,
        "cuda_peak": _cuda_peaks(devices),
    }
    _atomic_write_json(shard_path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _module_specs_from_metadata(config: dict):
    path = Path(config["calibration"]["score_metadata"]["path"])
    metadata = json.loads(path.read_text())
    return [
        (row["layer"], row["module"], row["shape"])
        for row in metadata["modules"]
    ]


def merge_scoring_shards(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    config = load_config(config_path)
    configured_shards = [tuple(row) for row in config["execution"]["score_shards"]]
    if configured_shards != [(0, 16), (16, 32)]:
        raise RuntimeError("score shard plan changed")
    shards = []
    for start, end in configured_shards:
        path = ROOT / "logs" / "shards" / f"blocks_{start:02d}_{end:02d}.json"
        shard = json.loads(path.read_text())
        if (
            shard.get("status") != "passed"
            or shard.get("block_start") != start
            or shard.get("block_end") != end
        ):
            raise RuntimeError(f"score shard is incomplete: {path}")
        shards.append(shard)
    dense_hashes = {shard["dense_model_sha256"] for shard in shards}
    if len(dense_hashes) != 1:
        raise RuntimeError("score shards used different dense checkpoints")

    score_documents = {
        condition: {"version": 1, "method": condition, "modules": []}
        for condition in CONDITIONS
    }
    mask_entries = {condition: [] for condition in CONDITIONS}
    matrix_diagnostics = []
    loss_sums = {condition: 0.0 for condition in CONDITIONS}
    loss_count = 0
    for shard in shards:
        for condition in CONDITIONS:
            score_documents[condition]["modules"].extend(
                shard["score_documents"][condition]["modules"]
            )
            mask_entries[condition].extend(shard["mask_entries"][condition])
            loss_sums[condition] += shard["loss_sums"][condition]
        matrix_diagnostics.extend(shard["matrix_diagnostics"])
        loss_count += shard["loss_count"]

    expected_keys = {
        (block, module) for block, module, _ in _module_specs_from_metadata(config)
    }
    for condition in CONDITIONS:
        score_documents[condition]["modules"].sort(
            key=lambda row: (row["layer"], row["module"])
        )
        mask_entries[condition].sort(
            key=lambda row: (row["block_index"], row["module"])
        )
        score_keys = {
            (row["layer"], row["module"])
            for row in score_documents[condition]["modules"]
        }
        mask_keys = {
            (row["block_index"], row["module"])
            for row in mask_entries[condition]
        }
        if score_keys != expected_keys or mask_keys != expected_keys or len(score_keys) != 224:
            raise RuntimeError(f"score shard coverage is not exact for {condition}")

    comparison_pairs = tuple(tuple(pair) for pair in config["diagnostics"]["pairs"])
    diagnostics = summarize_mask_diagnostics(matrix_diagnostics)
    global_rows = {row["pair"]: row for row in diagnostics if row["scope"] == "global"}
    if set(global_rows) != {f"{left}/{right}" for left, right in comparison_pairs}:
        raise RuntimeError("global diagnostic pairs are incomplete")
    decision = decision_from_diagnostics(global_rows[PRIMARY_PAIR])

    for condition in CONDITIONS:
        entries = mask_entries[condition]
        overall_hash = _overall_mask_hash(entries, condition)
        mask_document = {
            "version": 1,
            "method": condition,
            "persisted": condition in config["storage"]["persisted_methods"],
            "overall_sha256": overall_hash,
            "total_bytes": sum(entry["byte_length"] for entry in entries),
            "entries": entries,
        }
        _atomic_write_json(ROOT / f"{condition}_mask.json", mask_document)
        score_documents[condition].update(
            {
                "state_count": 80,
                "matrix_count": len(score_documents[condition]["modules"]),
                "reveal_raw_ratio": RATIOS[condition],
                "normalization": "masked-token mean one per state",
                "aggregation": "ABS",
                "mask_overall_sha256": overall_hash,
            }
        )
        _atomic_write_json(ROOT / f"{condition}_scores.json", score_documents[condition])
    _atomic_write_json(
        ROOT / "mask_diagnostics.json",
        {
            "version": 1,
            "spearman_aggregation": "element-weighted mean of exact per-matrix Spearman correlations",
            "rows": diagnostics,
            "decision": decision,
        },
    )
    result = {
        "status": "passed",
        "runtime_seconds_sum_across_workers": sum(
            shard["runtime_seconds"] for shard in shards
        ),
        "state_count": 80,
        "matrix_count": 224,
        "conditions": list(CONDITIONS),
        "gradient_isolation": "torch.autograd.grad with parameter .grad always None",
        "loss_means": {key: value / loss_count for key, value in loss_sums.items()},
        "global_diagnostics": global_rows,
        "decision": decision,
        "mask_hashes": {
            condition: _overall_mask_hash(mask_entries[condition], condition)
            for condition in CONDITIONS
        },
        "dense_model_sha256": next(iter(dense_hashes)),
        "worker_cuda_peaks": [shard["cuda_peak"] for shard in shards],
        "score_shards": [list(row) for row in configured_shards],
        "evaluation_started": False,
    }
    _atomic_write_json(ROOT / "logs" / "scoring.json", result)
    return result


def run_scoring(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    config = load_config(config_path)
    for start, end in config["execution"]["score_shards"]:
        run_scoring_shard(config_path, start, end)
    return merge_scoring_shards(config_path)


def _model_weight_sha(model) -> str:
    return dense_fingerprint(_module_map(model))


def run_evaluation(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    from transformers import AutoTokenizer

    config = load_config(config_path)
    sources = _load_sources(config)
    scoring = json.loads((ROOT / "logs" / "scoring.json").read_text())
    if scoring.get("status") != "passed":
        raise RuntimeError("scoring must pass before evaluation")
    if not scoring["decision"]["run_downstream"]:
        result = {
            "status": "skipped_by_preregistered_mask_gate",
            "decision": scoring["decision"],
            "baselines": {
                "uniform_abs": sum(row["correct"] for row in sources["uniform_records"]),
                "reveal2_abs": sum(row["correct"] for row in sources["reveal2_records"]),
            },
            "limit": config["evaluation"]["limit"],
        }
        _atomic_write_json(ROOT / "results" / "evaluation.json", result)
        return result

    output = ROOT / "results" / "evaluation.json"
    predictions = ROOT / "results" / "reveal50_abs_predictions.jsonl"
    mask_document = json.loads((ROOT / "reveal50_abs_mask.json").read_text())
    if output.exists() and predictions.exists():
        existing = json.loads(output.read_text())
        records = _read_jsonl(predictions)
        if (
            existing.get("status") == "passed"
            and existing.get("mask_sha256") == mask_document["overall_sha256"]
            and existing.get("evaluation_protocol_hash") == sources["evaluation_hash"]
            and len(records) == config["evaluation"]["limit"]
            and sum(row["correct"] for row in records) == existing.get("correct")
        ):
            _assert_same_examples(sources["uniform_records"], records)
            return existing

    model, _ = _load_model(sources["exp001_config"])
    dense_sha = _validated_dense_fingerprint(
        model, sources["exp002_config"], sources["score_metadata"]
    )
    mask_result = apply_dlm_masks(
        model,
        mask_document["entries"],
        mask_document["overall_sha256"],
        get_modules=_module_map,
    )
    sparse_before = _model_weight_sha(model)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    measured, records = _evaluate_gsm8k(
        model,
        tokenizer,
        sources["exp002_config"],
        "REVEAL-50X-ABS",
        config["evaluation"]["limit"],
        sources["evaluation_hash"],
    )
    sparse_after = _model_weight_sha(model)
    if sparse_after != sparse_before:
        raise RuntimeError("mini evaluation modified sparse weights")
    _assert_same_examples(sources["uniform_records"], records)
    _write_jsonl(predictions, records)
    if _read_jsonl(predictions) != records:
        raise RuntimeError("Reveal-50x mini prediction readback failed")

    baselines = {
        "uniform_abs": sources["uniform_records"],
        "reveal2_abs": sources["reveal2_records"],
        "reveal50_abs": records,
    }
    result = {
        "status": "passed",
        "limit": len(records),
        "metric": "GSM8K strict exact match",
        "correct": sum(row["correct"] for row in records),
        "accuracy": measured["accuracy"],
        "evaluation_seconds": measured["eval_seconds"],
        "comparison": {
            condition: {
                "correct": sum(row["correct"] for row in rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
            }
            for condition, rows in baselines.items()
        },
        "mask_sha256": mask_result["mask_hash"],
        "sparsity": mask_result["sparsity"],
        "rowwise_exact": mask_result["rowwise_exact"],
        "dense_model_sha256": dense_sha,
        "sparse_model_sha256_before_eval": sparse_before,
        "sparse_model_sha256_after_eval": sparse_after,
        "evaluation_protocol_hash": sources["evaluation_hash"],
        "predictions_path": str(predictions),
        "predictions_sha256": sha256_file(predictions),
        "interpretation_guardrail": "N=100 is a noisy mask-sensitivity diagnostic, not a method conclusion.",
    }
    _atomic_write_json(output, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _percent(value: float, digits: int = 4) -> str:
    return f"{100 * value:.{digits}f}%"


def run_report(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    config = load_config(config_path)
    preflight = json.loads((ROOT / "logs" / "preflight.json").read_text())
    scoring = json.loads((ROOT / "logs" / "scoring.json").read_text())
    evaluation = json.loads((ROOT / "results" / "evaluation.json").read_text())
    diagnostics = scoring["global_diagnostics"]
    primary = diagnostics[PRIMARY_PAIR]
    vs_two = diagnostics["reveal50_abs/reveal2_abs"]
    two_vs_uniform = diagnostics["reveal2_abs/uniform_abs"]
    decision = scoring["decision"]
    ranges = preflight["strong_weight_ranges"]

    if decision["interpretation"] == "ranking_redundant":
        headline = (
            "50×에서도 mask가 거의 움직이지 않아, 단순 weighting strength 부족보다 "
            "Reveal gradient ranking의 redundancy가 더 강하게 지지된다."
        )
    elif decision["interpretation"] == "weighting_strength_supported":
        headline = (
            "50×에서 mask가 크게 움직여, 기존 2× weighting이 decision을 바꾸기에는 "
            "너무 약했다는 설명이 지지된다."
        )
    else:
        headline = "50×는 중간 수준의 mask sensitivity를 만들었으며 두 가설을 깔끔히 분리하지 못했다."

    if evaluation["status"] == "passed":
        downstream = evaluation["comparison"]
        downstream_text = f"""
| 조건 | 정답 / 100 | 정확도 |
|---|---:|---:|
| Uniform-ABS (재사용) | {downstream['uniform_abs']['correct']} | {_percent(downstream['uniform_abs']['accuracy'], 1)} |
| Reveal-2×-ABS (재사용) | {downstream['reveal2_abs']['correct']} | {_percent(downstream['reveal2_abs']['accuracy'], 1)} |
| Reveal-50×-ABS (신규) | {downstream['reveal50_abs']['correct']} | {_percent(downstream['reveal50_abs']['accuracy'], 1)} |

이 100문제 결과는 noisy diagnostic이다. 작은 상승이나 하락만으로 method conclusion을 내리지 않는다.
"""
    else:
        downstream_text = (
            "Preregistered gate를 통과하지 않아 Reveal-50× downstream은 실행하지 않았다. "
            f"재사용 baseline은 Uniform {evaluation['baselines']['uniform_abs']}/100, "
            f"Reveal-2× {evaluation['baselines']['reveal2_abs']}/100이다."
        )

    report = f"""# EXP-004 Strong Reveal 50× Stress Test

## 기술 요약

**{headline}**

- Frozen 80 states와 10,267 masked observations를 그대로 썼고 reveal은 116개 ({_percent(preflight['reveal_fraction'], 2)})였다.
- Primary Reveal-50× vs Uniform: Spearman `{primary['spearman']:.9f}`, mask XOR `{_percent(primary['mask_xor'], 4)}`.
- Reveal-50× vs Reveal-2×: Spearman `{vs_two['spearman']:.9f}`, mask XOR `{_percent(vs_two['mask_xor'], 4)}`.
- 이 결과의 preregistered 분류는 `{decision['interpretation']}`이고, downstream gate는 `{'PASS' if decision['run_downstream'] else 'FAIL'}`였다.

## 50×가 score와 mask를 얼마나 바꿨는가

| 비교 | Spearman | Mask XOR | Mask IoU | Kept top-k overlap |
|---|---:|---:|---:|---:|
| Reveal-2× / Uniform | {two_vs_uniform['spearman']:.9f} | {_percent(two_vs_uniform['mask_xor'], 4)} | {_percent(two_vs_uniform['mask_iou'], 4)} | {_percent(two_vs_uniform['topk_overlap'], 4)} |
| Reveal-50× / Uniform | {primary['spearman']:.9f} | {_percent(primary['mask_xor'], 4)} | {_percent(primary['mask_iou'], 4)} | {_percent(primary['topk_overlap'], 4)} |
| Reveal-50× / Reveal-2× | {vs_two['spearman']:.9f} | {_percent(vs_two['mask_xor'], 4)} | {_percent(vs_two['mask_iou'], 4)} | {_percent(vs_two['topk_overlap'], 4)} |

Spearman은 각 matrix의 exact rank correlation을 weight count로 가중 평균한 값이다. Mask XOR은 6,979,321,856개 prunable weights 전체에서 서로 다른 binary decision의 비율이다.

## 동일한 calibration과 metric 정의

- Model: `GSAI-ML/LLaDA-8B-Base` revision `{config['model']['revision']}` in BF16.
- Population: EXP-001/004의 동일 frozen 80 corrupted states; 새 timestep, seed, state, partition 없음.
- Reveal/remain: low-confidence remasking schedule이 정한 EXP-004 partition을 checksum과 state identity로 재검증했다.
- Score: state별 `d_{{i,s}} = -w_i * dL_s/dw_i`; `S_i = mean_s |d_{{i,s}}|`.
- Mask: matrix별 각 row의 score 하위 50%를 정확히 prune.
- 비교: Uniform `1:1`, Reveal-2× `2:1`, Reveal-50× `50:1`; 모두 state별 masked-token mean alpha가 1.

## 50× weighting은 의도적으로 과격하다

Reveal-50× normalized alpha 범위는 reveal `{ranges['reveal_alpha']['min']:.6f}`–`{ranges['reveal_alpha']['max']:.6f}`, remain `{ranges['remain_alpha']['min']:.6f}`–`{ranges['remain_alpha']['max']:.6f}`였다. Reveal group이 normalized token-weight mass에서 차지한 비율은 state별 `{_percent(ranges['reveal_normalized_weight_mass_share']['min'], 2)}`–`{_percent(ranges['reveal_normalized_weight_mass_share']['max'], 2)}`였다.

이는 현실적인 hyperparameter 탐색이 아니라, 2×가 약해서 mask가 안 바뀌었는지를 분리하기 위한 stress test다.

## Fixed GSM8K mini-100

{downstream_text}

## 검증과 한계

- 동일 forward graph에서 세 weighted loss를 만들고 condition별 `torch.autograd.grad`를 독립 호출했다. Parameter `.grad` accumulation과 weight update는 없었다.
- 모든 신규 mask는 exact row-wise 50%를 통과했다.
- 2× same-pass reproduction은 역사적 EXP-004 수치(Spearman `{config['historical_reference']['reveal2_vs_uniform_spearman']:.9f}`, XOR `{_percent(config['historical_reference']['reveal2_vs_uniform_mask_xor'], 4)}`)와 별도로 비교해야 한다. CUDA/compiled-backward 비결정성 때문에 bit-exact 재현은 요구하지 않았다.
- Score correlation과 mask XOR은 sensitivity를 설명하지만 downstream 성능의 대체 지표가 아니다.
- Mini-100은 confidence interval이나 method superiority를 확립하기 위한 표본이 아니다.

## 다음 단계

"""
    if decision["interpretation"] == "ranking_redundant":
        report += (
            "현재 binary Reveal weighting의 ratio tuning은 중단한다. 다음 진단은 aggregation 구조가 "
            "token contrast를 hard pruning decision으로 전달하지 못하는 위치를 직접 측정해야 한다.\n"
        )
    elif evaluation["status"] == "passed" and evaluation["comparison"]["reveal50_abs"]["correct"] > evaluation["comparison"]["uniform_abs"]["correct"]:
        report += (
            "Mask sensitivity와 mini 방향이 모두 양수이므로 token-importance branch를 다시 볼 근거는 있다. "
            "다만 50×를 후보로 채택하지 말고, ratio/contrast 설계는 별도 사전등록 후 검증한다.\n"
        )
    elif evaluation["status"] == "passed":
        report += (
            "Reveal signal은 ranking을 바꿀 수 있지만 50× 지배는 mini에서 유익하지 않았다. "
            "강한 ratio 튜닝으로 이어가지 말고 이 결과를 failure-mode diagnostic으로 종료한다.\n"
        )
    else:
        report += "Mask gate 결과가 결론을 결정하며 추가 downstream은 실행하지 않는다.\n"

    report += """
## 추가 질문

Same-pass 2× reproduction과 historical EXP-004의 작은 차이가 CUDA nondeterminism 범위인지, 그리고 layer/module별 50× sensitivity가 특정 위치에 집중되는지는 `mask_diagnostics.json`의 상세 rows로 확인할 수 있다.
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")
    final = {
        "status": "complete",
        "primary": primary,
        "reveal50_vs_reveal2": vs_two,
        "reveal2_vs_uniform": two_vs_uniform,
        "decision": decision,
        "evaluation": evaluation,
        "report": str(ROOT / "report.md"),
    }
    _atomic_write_json(ROOT / "logs" / "final.json", final)
    return final


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="EXP-004 Reveal-50x stress test")
    parser.add_argument(
        "command",
        choices=("preflight", "score", "score-shard", "merge", "evaluate", "report", "all"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--block-start", type=int)
    parser.add_argument("--block-end", type=int)
    args = parser.parse_args(argv)
    if args.command in ("preflight", "all"):
        result = run_preflight(args.config)
    if args.command == "score-shard":
        if args.block_start is None or args.block_end is None:
            parser.error("score-shard requires --block-start and --block-end")
        result = run_scoring_shard(args.config, args.block_start, args.block_end)
    if args.command in ("score", "all"):
        result = run_scoring(args.config)
    if args.command == "merge":
        result = merge_scoring_shards(args.config)
    if args.command in ("evaluate", "all"):
        result = run_evaluation(args.config)
    if args.command in ("report", "all"):
        result = run_report(args.config)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
