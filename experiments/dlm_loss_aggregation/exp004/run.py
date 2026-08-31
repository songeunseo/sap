import argparse
import gc
import hashlib
import math
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from generate import get_num_transfer_tokens
from experiments.dlm_loss_aggregation.core import (
    _average_ranks,
    _rank_correlation,
    mask_sha256,
    pack_mask,
    rowwise_mask,
)
from experiments.dlm_loss_aggregation.run import (
    _cuda_devices,
    _cuda_peaks,
    _initial_cached_states,
    _module_specs,
    _reset_cuda_peaks,
    _suffix_logits,
    _atomic_write_json,
    _load_model,
    historical_state_digest,
    load_config as load_exp001_config,
    require_historical_digest,
    validate_config as validate_exp001_config,
)
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _load_mask,
    _module_map,
    _overall_mask_hash,
    _read_jsonl,
    _validated_dense_fingerprint,
    _write_jsonl,
    apply_dlm_masks,
    load_config as load_exp002_config,
    validate_exp001_masks,
)


class MagnitudeAccumulator:
    def __init__(self, shape):
        self.absolute = torch.zeros(shape, dtype=torch.float32)
        self.square = torch.zeros(shape, dtype=torch.float32)
        self.count = 0
        self._normalized_abs_sum = 0.0
        self._raw_abs_sum = 0.0
        self._normalized_square_sum = 0.0
        self._raw_square_sum = 0.0

    def add(self, effect, raw_scale=1.0):
        if effect.shape != self.absolute.shape or not torch.isfinite(effect).all().item():
            raise ValueError("effect must be finite and match the accumulator")
        value = effect.detach().to(device="cpu", dtype=torch.float32)
        absolute = value.abs()
        square = value.square()
        self.absolute.add_(absolute)
        self.square.add_(square)
        self._normalized_abs_sum += absolute.mean().item()
        self._raw_abs_sum += absolute.mean().item() * raw_scale
        self._normalized_square_sum += square.mean().item()
        self._raw_square_sum += square.mean().item() * raw_scale**2
        self.count += 1

    def finalize(self):
        if not self.count:
            raise RuntimeError("cannot finalize an empty accumulator")
        return (
            {"abs": self.absolute / self.count, "square": self.square / self.count},
            {
                "normalized_abs_mean": self._normalized_abs_sum / self.count,
                "raw_abs_mean": self._raw_abs_sum / self.count,
                "normalized_square_mean": self._normalized_square_sum / self.count,
                "raw_square_mean": self._raw_square_sum / self.count,
            },
        )


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "experiment": "EXP-004",
        "model": {
            "id": "GSAI-ML/LLaDA-8B-Base",
            "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
            "dtype": "bfloat16",
        },
        "source_exp001.run_id": "20260828T175010-2484545",
        "source_exp001.state_sha256": "1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df",
        "source_exp001.state_count": 80,
        "source_exp002.evaluation_config_hash": "1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add",
        "partition.denoising_steps": 256,
        "partition.remasking": "low_confidence",
        "partition.confidence": "softmax_probability_of_argmax",
        "weighting.conditions": ["uniform", "reveal", "remain"],
        "weighting.raw_ratio": 2.0,
        "weighting.normalization": "masked_token_mean_one",
        "scoring.aggregations": ["abs", "square"],
        "scoring.sparsity": 0.5,
        "scoring.block_count": 32,
        "scoring.modules_per_block": 7,
        "uniform_reproduction.global_xor_threshold": 0.000001,
        "uniform_reproduction.module_xor_threshold": 0.00001,
        "storage.safety_bytes": 536870912,
        "evaluation.methods": ["REVEAL-ABS", "REVEAL-SQUARE", "REMAIN-ABS", "REMAIN-SQUARE"],
        "statistics.primary_comparison": ["REVEAL-ABS", "REMAIN-ABS"],
        "statistics.paired_comparisons": [
            ["REVEAL-ABS", "UNIFORM-ABS"],
            ["REMAIN-ABS", "UNIFORM-ABS"],
            ["REVEAL-ABS", "REMAIN-ABS"],
            ["REVEAL-SQUARE", "UNIFORM-SQUARE"],
            ["REMAIN-SQUARE", "UNIFORM-SQUARE"],
            ["REVEAL-SQUARE", "REMAIN-SQUARE"],
            ["REVEAL-ABS", "REVEAL-SQUARE"],
            ["REMAIN-ABS", "REMAIN-SQUARE"],
        ],
    }
    for dotted, wanted in expected.items():
        actual = config
        for key in dotted.split("."):
            actual = actual[key]
        if actual != wanted:
            raise ValueError(f"EXP-004 config mismatch: {dotted}")
    return config


def write_packed_mask(root, method, block_index, module_name, mask):
    payload = pack_mask(mask)
    digest = mask_sha256(payload)
    directory = Path(root) / "mask_payloads" / method
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"block_{block_index:02d}__{module_name.replace('.', '_')}.bin"
    temporary = path.with_suffix(".bin.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload["bits"])
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    saved = {"shape": payload["shape"], "bits": path.read_bytes()}
    if mask_sha256(saved) != digest:
        raise RuntimeError("packed mask checksum readback failed")
    return {
        "method": method,
        "block_index": block_index,
        "module": module_name,
        "shape": list(payload["shape"]),
        "byte_length": len(payload["bits"]),
        "sha256": digest,
        "runtime_path": str(path.resolve()),
    }


@torch.no_grad()
def freeze_token_partition(logits, noisy_ids, mask_id, p_mask, denoising_steps=256):
    mask = noisy_ids.eq(mask_id)
    if logits.shape[:-1] != noisy_ids.shape or not mask.any().item():
        raise ValueError("logits/noisy state is invalid")
    steps_remaining = max(1, math.ceil(p_mask * denoising_steps))
    reveal_count = int(get_num_transfer_tokens(mask, steps_remaining)[0, 0].item())
    prediction = logits.argmax(dim=-1)
    confidence = F.softmax(logits, dim=-1).gather(
        -1, prediction.unsqueeze(-1)
    ).squeeze(-1)
    masked_indices = mask[0].nonzero().flatten()
    generator_confidence = torch.where(mask, confidence, -torch.inf)
    reveal_indices = torch.topk(
        generator_confidence[0], k=reveal_count
    ).indices.sort().values
    remain_indices = masked_indices[
        ~torch.isin(masked_indices, reveal_indices)
    ]
    result = {
        "steps_remaining": steps_remaining,
        "masked_count": masked_indices.numel(),
        "reveal_count": reveal_indices.numel(),
        "remain_count": remain_indices.numel(),
        "masked_indices": masked_indices.tolist(),
        "reveal_indices": reveal_indices.tolist(),
        "remain_indices": remain_indices.tolist(),
        "masked_predictions": [
            {
                "position": int(position),
                "token_id": int(prediction[0, position].item()),
                "confidence": confidence[0, position].item(),
            }
            for position in masked_indices.tolist()
        ],
    }
    if (
        set(result["reveal_indices"]) & set(result["remain_indices"])
        or sorted(result["reveal_indices"] + result["remain_indices"])
        != result["masked_indices"]
    ):
        raise RuntimeError("reveal/remain partition is invalid")
    return result


def token_weights(mask, reveal_mask, condition):
    if mask.dtype != torch.bool or reveal_mask.shape != mask.shape:
        raise ValueError("mask and reveal_mask must be matching boolean tensors")
    if reveal_mask.dtype != torch.bool or (reveal_mask & ~mask).any().item():
        raise ValueError("reveal positions must be masked")
    if condition not in {"uniform", "reveal", "remain"}:
        raise ValueError(f"unknown token weighting: {condition}")
    raw = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    raw[mask] = 1
    if condition == "reveal":
        raw[reveal_mask] = 2
    elif condition == "remain":
        raw[mask & ~reveal_mask] = 2
    raw[mask] /= raw[mask].mean()
    if not torch.isclose(raw[mask].mean(), torch.tensor(1.0, device=mask.device)):
        raise RuntimeError("normalized masked-token weight mean is not one")
    return raw


def summarize_token_weights(partition):
    masked = partition["masked_count"]
    reveal = partition["reveal_count"]
    remain = partition["remain_count"]
    if masked <= 0 or reveal + remain != masked:
        raise ValueError("partition counts are invalid")
    result = {
        "uniform": {
            "raw_mean": 1.0,
            "reveal_alpha": 1.0,
            "remain_alpha": 1.0,
            "normalized_mean": 1.0,
        }
    }
    for condition, reveal_raw, remain_raw in (
        ("reveal", 2.0, 1.0),
        ("remain", 1.0, 2.0),
    ):
        raw_mean = (reveal * reveal_raw + remain * remain_raw) / masked
        reveal_alpha = reveal_raw / raw_mean
        remain_alpha = remain_raw / raw_mean
        result[condition] = {
            "raw_mean": raw_mean,
            "reveal_alpha": reveal_alpha,
            "remain_alpha": remain_alpha,
            "normalized_mean": (
                reveal * reveal_alpha + remain * remain_alpha
            ) / masked,
        }
    return result


def weighted_dlm_loss(logits, clean_ids, mask, p_mask, alpha):
    return weighted_dlm_losses(
        logits, clean_ids, mask, p_mask, {"condition": alpha}
    )["condition"]


def weighted_dlm_losses(logits, clean_ids, mask, p_mask, alphas):
    if logits.shape[:-1] != clean_ids.shape or mask.shape != clean_ids.shape:
        raise ValueError("logits, tokens, and mask shapes differ")
    if not alphas or any(alpha.shape != mask.shape for alpha in alphas.values()) or not mask.any().item():
        raise ValueError("alpha shape is invalid or the state has no masks")
    token_loss = F.cross_entropy(
        logits[mask].float(), clean_ids[mask], reduction="none"
    )
    return {
        condition: (token_loss * alpha[mask]).sum() / p_mask / clean_ids.numel()
        for condition, alpha in alphas.items()
    }


def independent_weight_effects(losses, parameters, frozen_cpu_weights=None):
    conditions = ("uniform", "reveal", "remain")
    if tuple(losses) != conditions or not parameters:
        raise ValueError("losses must be ordered uniform, reveal, remain")
    values = tuple(parameters.values())
    frozen_cpu_weights = frozen_cpu_weights or {
        name: parameter.detach().to(device="cpu", dtype=torch.float32)
        for name, parameter in parameters.items()
    }
    if set(frozen_cpu_weights) != set(parameters) or any(
        frozen_cpu_weights[name].shape != parameter.shape
        for name, parameter in parameters.items()
    ):
        raise ValueError("frozen CPU weights do not match parameters")
    if any(parameter.grad is not None for parameter in values):
        raise RuntimeError("parameter gradient leaked into condition scoring")
    result = {}
    for index, condition in enumerate(conditions):
        gradients = torch.autograd.grad(
            losses[condition], values, retain_graph=index + 1 < len(conditions)
        )
        result[condition] = {
            name: -frozen_cpu_weights[name]
            * gradient.detach().to(device="cpu", dtype=torch.float32)
            for (name, parameter), gradient in zip(parameters.items(), gradients)
        }
        if any(parameter.grad is not None for parameter in values):
            raise RuntimeError("condition gradient accumulated on a parameter")
    return result


def validate_uniform_reuse(records_by_method, expected_hash, actual_hash, expected_count=1319):
    if actual_hash != expected_hash:
        raise ValueError("evaluation config hash differs from EXP-002")
    methods = ("UNIFORM-ABS", "UNIFORM-SQUARE")
    if set(records_by_method) != set(methods):
        raise ValueError("both frozen UNIFORM result sets are required")
    reference = records_by_method[methods[0]]
    if len(reference) != expected_count:
        raise ValueError("frozen UNIFORM result count is wrong")
    identity = ("example_id", "doc_hash", "prompt_hash", "target_hash")
    for method in methods:
        records = records_by_method[method]
        if len(records) != expected_count or any(
            row.get("evaluation_config_hash") != expected_hash for row in records
        ):
            raise ValueError("frozen UNIFORM config hash or count is wrong")
        if any(
            tuple(left[key] for key in identity) != tuple(right[key] for key in identity)
            for left, right in zip(reference, records)
        ):
            raise ValueError("frozen UNIFORM paired examples differ")
    return {"passed": True, "evaluation_config_hash": expected_hash, "count": expected_count}


def require_disk_capacity(
    directory, mask_bytes_per_method, method_count, safety_bytes, free_bytes=None
):
    required = mask_bytes_per_method * method_count + safety_bytes
    available = shutil.disk_usage(directory).free if free_bytes is None else free_bytes
    if available < required:
        raise RuntimeError(f"insufficient disk: need {required} bytes, have {available}")
    return {"required_bytes": required, "available_bytes": available}


def validate_uniform_reproduction(rows, global_xor_threshold, module_xor_threshold):
    if {row["method"] for row in rows} != {"abs", "square"}:
        raise ValueError("UNIFORM reproduction requires ABS and SQUARE rows")
    by_method = {}
    failed_modules = []
    for method in ("abs", "square"):
        group = [row for row in rows if row["method"] == method]
        different = sum(row["different"] for row in group)
        count = sum(row["num_weights"] for row in group)
        by_method[method] = {"different": different, "num_weights": count, "xor": different / count}
        failed_modules.extend(
            row for row in group
            if row["different"] / row["num_weights"] > module_xor_threshold
        )
    exact = all(row["different"] == 0 for row in rows)
    if failed_modules or any(
        summary["xor"] > global_xor_threshold for summary in by_method.values()
    ):
        raise RuntimeError("UNIFORM mask reproduction exceeded the preregistered threshold")
    return {"passed": True, "exact": exact, "by_method": by_method, "mismatched_modules": [
        row for row in rows if row["different"]
    ]}


def compare_score_masks(left_score, right_score, left_mask, right_mask):
    if (
        left_score.shape != right_score.shape
        or left_mask.shape != left_score.shape
        or right_mask.shape != left_score.shape
    ):
        raise ValueError("score and mask shapes differ")
    correlation, reason = _rank_correlation(
        _average_ranks(left_score.reshape(-1)),
        _average_ranks(right_score.reshape(-1)),
    )
    intersection = int((left_mask & right_mask).sum().item())
    union = int((left_mask | right_mask).sum().item())
    kept_intersection = int((~left_mask & ~right_mask).sum().item())
    kept_count = int((~left_mask).sum().item())
    return {
        "spearman": correlation,
        "spearman_reason": reason,
        "mask_xor": left_mask.ne(right_mask).float().mean().item(),
        "mask_iou": intersection / union,
        "topk_overlap": kept_intersection / kept_count,
    }


def summarize_mask_diagnostics(matrix_rows):
    def aggregate(scope, key, value, group):
        count = sum(row["num_weights"] for row in group)
        defined = [row for row in group if row["spearman"] is not None]
        spearman_count = sum(row["num_weights"] for row in defined)
        row = {
            "scope": scope,
            "pair": group[0]["pair"],
            "num_weights": count,
            "different": sum(item["different"] for item in group),
            "mask_xor": sum(item["different"] for item in group) / count,
            "spearman": None if not defined else sum(
                item["spearman"] * item["num_weights"] for item in defined
            ) / spearman_count,
            "mask_iou": sum(item["mask_iou"] * item["num_weights"] for item in group) / count,
            "topk_overlap": sum(item["topk_overlap"] * item["num_weights"] for item in group) / count,
        }
        if key:
            row[key] = value
        return row

    result = list(matrix_rows)
    for pair in sorted({row["pair"] for row in matrix_rows}):
        pair_rows = [row for row in matrix_rows if row["pair"] == pair]
        for layer in sorted({row["layer"] for row in pair_rows}):
            result.append(aggregate("layer", "layer", layer, [
                row for row in pair_rows if row["layer"] == layer
            ]))
        for module_type in sorted({row["module_type"] for row in pair_rows}):
            result.append(aggregate("module_type", "module_type", module_type, [
                row for row in pair_rows if row["module_type"] == module_type
            ]))
        result.append(aggregate("global", None, None, pair_rows))
    return result


def paired_comparison(method_a, method_b, records_by_method):
    left = records_by_method[method_a]
    right = records_by_method[method_b]
    if len(left) != len(right) or not left:
        raise ValueError("paired record counts differ")
    both_correct = sum(a["correct"] and b["correct"] for a, b in zip(left, right))
    a_only = sum(a["correct"] and not b["correct"] for a, b in zip(left, right))
    b_only = sum(not a["correct"] and b["correct"] for a, b in zip(left, right))
    both_wrong = len(left) - both_correct - a_only - b_only
    discordant = a_only + b_only
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
        p_value = min(1.0, 2 * tail / (1 << discordant))
    else:
        p_value = 1.0
    return {
        "method_a": method_a,
        "method_b": method_b,
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "accuracy_difference_pp": 100 * (a_only - b_only) / len(left),
        "p_exact_two_sided": p_value,
    }


def holm_adjust(rows, p_key="p_exact_two_sided"):
    result = [dict(row) for row in rows]
    order = sorted(range(len(result)), key=lambda index: result[index][p_key])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(result) - rank) * result[index][p_key]))
        result[index]["p_holm"] = running
    return result


def _canonical_json(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _load_sources(config):
    source1 = config["source_exp001"]
    exp001_config = validate_exp001_config(load_exp001_config(source1["config"]))
    manifest = json.loads(Path(source1["calibration_manifest"]).read_text())
    digest = require_historical_digest(manifest, source1["state_sha256"])
    if len(manifest["states"]) != source1["state_count"] or digest != historical_state_digest(manifest):
        raise ValueError("EXP-001 calibration manifest mismatch")
    mask_metadata = json.loads(Path(source1["mask_metadata"]).read_text())
    score_metadata = json.loads(Path(source1["score_metadata"]).read_text())
    mask_validation = validate_exp001_masks(
        mask_metadata, score_metadata, source1["matrix_count"]
    )
    if (
        mask_metadata["runtime_directory"] != source1["mask_directory"]
        or {key: mask_validation["overall_sha256"][key] for key in ("abs", "square")}
        != source1["overall_sha256"]
    ):
        raise ValueError("frozen EXP-001 masks differ from config")

    source2 = config["source_exp002"]
    exp002_config = load_exp002_config(source2["config"])
    actual_hash, hash_document = _evaluation_config_hash(exp002_config)
    uniform_records = {
        "UNIFORM-ABS": _read_jsonl(source2["uniform_abs_predictions"]),
        "UNIFORM-SQUARE": _read_jsonl(source2["uniform_square_predictions"]),
    }
    reuse = validate_uniform_reuse(
        uniform_records,
        source2["evaluation_config_hash"],
        actual_hash,
        source2["example_count"],
    )
    if (
        sum(row["correct"] for row in uniform_records["UNIFORM-ABS"])
        != source2["uniform_abs_correct"]
        or sum(row["correct"] for row in uniform_records["UNIFORM-SQUARE"])
        != source2["uniform_square_correct"]
    ):
        raise ValueError("frozen EXP-002 correctness totals differ from config")
    return {
        "exp001_config": exp001_config,
        "manifest": manifest,
        "mask_metadata": mask_metadata,
        "score_metadata": score_metadata,
        "mask_validation": mask_validation,
        "exp002_config": exp002_config,
        "evaluation_config_hash": actual_hash,
        "evaluation_hash_document": hash_document,
        "uniform_records": uniform_records,
        "uniform_reuse": reuse,
    }


def run_preflight(config_path):
    config = load_config(config_path)
    root = Path(config_path).parent
    sources = _load_sources(config)
    entries = sources["mask_metadata"]["entries"]
    one_method_bytes = sum(
        entry["byte_length"] for entry in entries if entry["method"] == "abs"
    )
    disk = require_disk_capacity(
        root,
        one_method_bytes,
        len(config["evaluation"]["methods"]),
        config["storage"]["safety_bytes"],
    )
    result = {
        "status": "passed",
        "checked_at_unix": time.time(),
        "calibration_state_sha256": config["source_exp001"]["state_sha256"],
        "matrix_count": config["source_exp001"]["matrix_count"],
        "state_count": config["source_exp001"]["state_count"],
        "uniform_masks": sources["mask_validation"]["overall_sha256"],
        "uniform_reuse": sources["uniform_reuse"],
        "evaluation_config_hash": sources["evaluation_config_hash"],
        "mask_bytes_per_method": one_method_bytes,
        "disk": disk,
    }
    _atomic_write_json(root / "logs" / "preflight.json", result)
    return result


def validate_partition_artifact(artifact, manifest, expected_count=80):
    core_keys = (
        "version", "source_state_sha256", "dense_fingerprint",
        "partition_rule", "states",
    )
    core = {key: artifact[key] for key in core_keys}
    if hashlib.sha256(_canonical_json(core)).hexdigest() != artifact.get("sha256"):
        raise ValueError("token partition artifact hash mismatch")
    if (
        artifact["source_state_sha256"] != manifest["historical_state_sha256"]
        or not isinstance(artifact["dense_fingerprint"], str)
        or len(artifact["dense_fingerprint"]) != 64
        or len(artifact["states"]) != expected_count
        or len(manifest["states"]) != expected_count
        or artifact.get("summary", {}).get("state_count") != expected_count
    ):
        raise ValueError("token partition source/count/fingerprint mismatch")
    identity = ("timestep_index", "timestep", "sequence_index", "mask_seed")
    for index, (partition, source) in enumerate(zip(artifact["states"], manifest["states"])):
        if partition["state_index"] != index or any(
            partition[key] != source[key] for key in identity
        ):
            raise ValueError("token partition state identity differs from EXP-001")
        mask = torch.tensor(source["mask"], dtype=torch.bool)
        masked_indices = mask[0].nonzero().flatten().tolist()
        reveal = partition["reveal_indices"]
        remain = partition["remain_indices"]
        steps = max(1, math.ceil(source["p_mask"] * 256))
        expected_reveal_count = int(get_num_transfer_tokens(mask, steps)[0, 0].item())
        if (
            partition["steps_remaining"] != steps
            or partition["masked_indices"] != masked_indices
            or partition["masked_count"] != len(masked_indices)
            or partition["reveal_count"] != expected_reveal_count
            or partition["remain_count"] != len(masked_indices) - expected_reveal_count
            or set(reveal) & set(remain)
            or sorted(reveal + remain) != masked_indices
        ):
            raise ValueError("token partition sets differ from EXP-001 mask/schedule")
        predictions = partition["masked_predictions"]
        if [row["position"] for row in predictions] != masked_indices:
            raise ValueError("token partition prediction positions differ from masks")
        confidence = torch.full((mask.shape[1],), -torch.inf)
        confidence[masked_indices] = torch.tensor(
            [row["confidence"] for row in predictions]
        )
        expected_reveal = torch.topk(confidence, k=expected_reveal_count).indices.sort().values.tolist()
        if reveal != expected_reveal:
            raise ValueError("token partition reveal indices differ from saved confidence rule")
    return {"passed": True, "state_count": expected_count, "sha256": artifact["sha256"]}


@torch.no_grad()
def run_partition(config_path):
    config = load_config(config_path)
    root = Path(config_path).parent
    preflight = json.loads((root / "logs" / "preflight.json").read_text())
    if preflight.get("status") != "passed":
        raise RuntimeError("EXP-004 preflight has not passed")
    sources = _load_sources(config)
    model, _ = _load_model(sources["exp001_config"])
    dense_fingerprint = _validated_dense_fingerprint(
        model, sources["exp002_config"], sources["score_metadata"]
    )
    embedding_device = model.model.transformer.wte.weight.device
    partitions = []
    weights = []
    all_confidences = []
    try:
        for state_index, state in enumerate(sources["manifest"]["states"]):
            noisy_ids = torch.tensor(state["noisy_ids"], dtype=torch.long, device=embedding_device)
            logits = model(noisy_ids).logits
            partition = freeze_token_partition(
                logits,
                noisy_ids,
                sources["manifest"]["mask_id"],
                state["p_mask"],
                config["partition"]["denoising_steps"],
            )
            partition.update({
                "state_index": state_index,
                "timestep_index": state["timestep_index"],
                "timestep": state["timestep"],
                "sequence_index": state["sequence_index"],
                "mask_seed": state["mask_seed"],
            })
            partitions.append(partition)
            weights.append({"state_index": state_index, **summarize_token_weights(partition)})
            all_confidences.extend(
                row["confidence"] for row in partition["masked_predictions"]
            )
            del noisy_ids, logits
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    confidence = torch.tensor(all_confidences, dtype=torch.float64)
    partition_document = {
        "version": 1,
        "source_state_sha256": config["source_exp001"]["state_sha256"],
        "dense_fingerprint": dense_fingerprint,
        "partition_rule": {
            "steps_remaining": "max(1, ceil(p_mask * 256))",
            "transfer_count": "get_num_transfer_tokens(current_mask, steps_remaining)[0]",
            "selection": "top-k softmax probability of argmax at masked positions",
        },
        "states": partitions,
    }
    partition_document["sha256"] = hashlib.sha256(
        _canonical_json(partition_document)
    ).hexdigest()
    _atomic_write_json(root / "calibration_state_manifest.json", sources["manifest"])
    _atomic_write_json(root / "token_partition_summary.json", {
        **partition_document,
        "summary": {
            "state_count": len(partitions),
            "masked_tokens": sum(row["masked_count"] for row in partitions),
            "reveal_tokens": sum(row["reveal_count"] for row in partitions),
            "remain_tokens": sum(row["remain_count"] for row in partitions),
            "confidence": {
                "count": confidence.numel(),
                "mean": confidence.mean().item(),
                "min": confidence.min().item(),
                "p50": confidence.quantile(0.5).item(),
                "p90": confidence.quantile(0.9).item(),
                "p99": confidence.quantile(0.99).item(),
                "max": confidence.max().item(),
            },
        },
    })
    _atomic_write_json(root / "token_weights_summary.json", {
        "version": 1,
        "partition_sha256": partition_document["sha256"],
        "ratio": "2:1 before per-state normalization",
        "loss": "sum(alpha * token_ce) / p_mask / sequence_length",
        "states": weights,
    })
    return partition_document


def _tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _ranked_comparisons(scores, masks, pairs):
    ranks = {
        name: _average_ranks(scores[name].reshape(-1))
        for name in {method for pair in pairs for method in pair}
    }
    rows = []
    for left, right in pairs:
        correlation, reason = _rank_correlation(ranks[left], ranks[right])
        left_mask, right_mask = masks[left], masks[right]
        intersection = int((left_mask & right_mask).sum().item())
        union = int((left_mask | right_mask).sum().item())
        kept_intersection = int((~left_mask & ~right_mask).sum().item())
        kept_count = int((~left_mask).sum().item())
        different = int(left_mask.ne(right_mask).sum().item())
        rows.append({
            "pair": f"{left}/{right}",
            "left": left,
            "right": right,
            "different": different,
            "mask_xor": different / left_mask.numel(),
            "mask_iou": intersection / union,
            "topk_overlap": kept_intersection / kept_count,
            "spearman": correlation,
            "spearman_reason": reason,
        })
    return rows


def run_scoring(config_path):
    from lib.prune_llada import find_layers

    config = load_config(config_path)
    root = Path(config_path).parent
    preflight = json.loads((root / "logs" / "preflight.json").read_text())
    partition_file = json.loads((root / "token_partition_summary.json").read_text())
    if preflight.get("status") != "passed" or partition_file["summary"]["state_count"] != 80:
        raise RuntimeError("preflight and partition artifacts must pass before scoring")
    sources = _load_sources(config)
    validate_partition_artifact(
        partition_file, sources["manifest"], config["source_exp001"]["state_count"]
    )

    model, devices = _load_model(sources["exp001_config"])
    if _validated_dense_fingerprint(
        model, sources["exp002_config"], sources["score_metadata"]
    ) != partition_file["dense_fingerprint"]:
        raise ValueError("dense model fingerprint differs from partition phase")
    specs = _module_specs(model)
    expected_matrices = config["scoring"]["block_count"] * config["scoring"]["modules_per_block"]
    if len(specs) != expected_matrices:
        raise ValueError("prunable matrix universe differs from EXP-001")
    cached_states = _initial_cached_states(model, sources["manifest"])
    parameters = list(model.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    _reset_cuda_peaks(devices)
    started = time.perf_counter()
    _atomic_write_json(root / "logs" / "scoring_start.json", {
        "status": "running",
        "started_unix": time.time(),
        "resume_supported": False,
        "restart_behavior": "recompute all blocks and overwrite per-matrix mask payloads",
    })

    frozen = {
        (entry["method"], entry["block_index"], entry["module"]): entry
        for entry in sources["mask_metadata"]["entries"]
        if entry["method"] in {"abs", "square"}
    }
    methods = ("reveal_abs", "reveal_square", "remain_abs", "remain_square")
    score_documents = {method: {"method": method, "modules": []} for method in methods}
    mask_entries = {method: [] for method in methods}
    reproduction_rows = []
    matrix_diagnostics = []
    scale_diagnostics = []
    loss_sums = {condition: 0.0 for condition in ("uniform", "reveal", "remain")}
    loss_count = 0
    comparison_pairs = (
        ("reveal_abs", "reveal_square"),
        ("remain_abs", "remain_square"),
        ("reveal_abs", "uniform_abs"),
        ("remain_abs", "uniform_abs"),
        ("reveal_square", "uniform_square"),
        ("remain_square", "uniform_square"),
        ("reveal_abs", "remain_abs"),
        ("reveal_square", "remain_square"),
    )
    try:
        for block_index, block in enumerate(model.model.transformer.blocks):
            layers = find_layers(block)
            for layer in layers.values():
                layer.weight.requires_grad_(True)
            accumulators = {
                condition: {
                    name: MagnitudeAccumulator(layer.weight.shape)
                    for name, layer in layers.items()
                }
                for condition in ("uniform", "reveal", "remain")
            }
            frozen_cpu_weights = {
                name: layer.weight.detach().to(device="cpu", dtype=torch.float32).clone()
                for name, layer in layers.items()
            }
            next_hiddens = []
            for state, partition in zip(cached_states, partition_file["states"]):
                parameter = next(block.parameters())
                target_output, _ = block(
                    state["hidden"].to(device=parameter.device, dtype=parameter.dtype),
                    attention_bias=None,
                    layer_past=None,
                    use_cache=False,
                    replace_position=None,
                    attn_collector=None,
                )
                if block_index + 1 < config["scoring"]["block_count"]:
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
                        config["partition"]["denoising_steps"],
                    )
                    for key in (
                        "masked_indices", "reveal_indices", "remain_indices",
                        "masked_predictions",
                    ):
                        if observed[key] != partition[key]:
                            raise RuntimeError("dense partition changed between partition and scoring")
                reveal_mask = torch.zeros_like(mask)
                reveal_mask[0, partition["reveal_indices"]] = True
                if (
                    (reveal_mask & ~mask).any().item()
                    or int(reveal_mask.sum().item()) != partition["reveal_count"]
                    or int(mask.sum().item()) != partition["masked_count"]
                ):
                    raise RuntimeError("frozen partition does not match scoring state")
                alpha = {
                    condition: token_weights(mask, reveal_mask, condition)
                    for condition in ("uniform", "reveal", "remain")
                }
                losses = weighted_dlm_losses(
                    logits,
                    state["clean_ids"].to(logits.device),
                    mask,
                    state["p_mask"],
                    alpha,
                )
                effects = independent_weight_effects(
                    losses,
                    {name: layer.weight for name, layer in layers.items()},
                    frozen_cpu_weights,
                )
                weight_summary = summarize_token_weights(partition)
                for condition in ("uniform", "reveal", "remain"):
                    if abs(alpha[condition][mask].mean().item() - 1.0) > 1e-6:
                        raise RuntimeError("masked-token alpha mean differs from one")
                    loss_sums[condition] += losses[condition].detach().cpu().item()
                    raw_scale = weight_summary[condition]["raw_mean"]
                    for name in layers:
                        accumulators[condition][name].add(
                            effects[condition][name], raw_scale=raw_scale
                        )
                loss_count += 1
                del target_output, logits, mask, reveal_mask, alpha, losses, effects
            for layer in layers.values():
                layer.weight.requires_grad_(False)
            if next_hiddens:
                for state, hidden in zip(cached_states, next_hiddens):
                    state["hidden"] = hidden

            for name, layer in layers.items():
                module_type = name.rsplit(".", 1)[-1]
                condition_scores = {}
                for condition in ("uniform", "reveal", "remain"):
                    scores, scale = accumulators[condition].pop(name).finalize()
                    if any(not torch.isfinite(score).all().item() for score in scores.values()):
                        raise RuntimeError("non-finite finalized score")
                    condition_scores[condition] = scores
                    scale_diagnostics.append({
                        "layer": block_index,
                        "module": name,
                        "module_type": module_type,
                        "condition": condition,
                        "update_count": loss_count // (block_index + 1),
                        **scale,
                    })
                named_scores = {
                    f"{condition}_{aggregation}": condition_scores[condition][aggregation]
                    for condition in ("uniform", "reveal", "remain")
                    for aggregation in ("abs", "square")
                }
                named_masks = {
                    method: rowwise_mask(score, config["scoring"]["sparsity"])
                    for method, score in named_scores.items()
                }
                for aggregation in ("abs", "square"):
                    frozen_mask = _load_mask(frozen[(aggregation, block_index, name)])
                    regenerated = named_masks[f"uniform_{aggregation}"]
                    different = int(regenerated.ne(frozen_mask).sum().item())
                    reproduction_rows.append({
                        "method": aggregation,
                        "layer": block_index,
                        "module": name,
                        "module_type": module_type,
                        "num_weights": regenerated.numel(),
                        "different": different,
                        "xor": different / regenerated.numel(),
                    })
                    named_masks[f"uniform_{aggregation}"] = frozen_mask
                for method in methods:
                    score = named_scores[method]
                    mask = named_masks[method]
                    expected_per_row = score.shape[1] // 2
                    if not mask.sum(dim=1).eq(expected_per_row).all().item():
                        raise RuntimeError("new mask violates exact row-wise 50% sparsity")
                    entry = write_packed_mask(root, method, block_index, name, mask)
                    mask_entries[method].append(entry)
                    score_documents[method]["modules"].append({
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
                    })
                for row in _ranked_comparisons(named_scores, named_masks, comparison_pairs):
                    matrix_diagnostics.append({
                        "scope": "matrix",
                        "layer": block_index,
                        "module": name,
                        "module_type": module_type,
                        "num_weights": layer.weight.numel(),
                        **row,
                    })
                del condition_scores, named_scores, named_masks
            del accumulators
            del frozen_cpu_weights
            progress = {
                "status": "running",
                "resume_supported": False,
                "completed_blocks": block_index + 1,
                "elapsed_seconds": time.perf_counter() - started,
                "cuda_peak": _cuda_peaks(devices),
            }
            _atomic_write_json(root / "logs" / "scoring_progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, original in zip(parameters, original_flags):
            parameter.requires_grad_(original)

    gate_path = root / "logs" / "uniform_reproduction.json"
    try:
        reproduction = validate_uniform_reproduction(
            reproduction_rows,
            config["uniform_reproduction"]["global_xor_threshold"],
            config["uniform_reproduction"]["module_xor_threshold"],
        )
    except RuntimeError:
        _atomic_write_json(gate_path, {"status": "failed", "rows": reproduction_rows})
        raise
    _atomic_write_json(gate_path, {"status": "passed", **reproduction, "rows": reproduction_rows})

    for method in methods:
        entries = mask_entries[method]
        overall_hash = _overall_mask_hash(entries, method)
        mask_document = {
            "version": 1,
            "method": method,
            "overall_sha256": overall_hash,
            "total_bytes": sum(entry["byte_length"] for entry in entries),
            "entries": entries,
        }
        _atomic_write_json(root / f"{method}_mask.json", mask_document)
        score_documents[method].update({
            "version": 1,
            "equations": {
                "effect": "-weight_fp32 * gradient_fp32",
                "abs": "mean_state(abs(effect))",
                "square": "mean_state(effect ** 2)",
            },
            "state_count": config["source_exp001"]["state_count"],
            "matrix_count": len(score_documents[method]["modules"]),
            "mask_overall_sha256": overall_hash,
        })
        _atomic_write_json(root / f"{method}_scores.json", score_documents[method])
    diagnostics = summarize_mask_diagnostics(matrix_diagnostics)
    _atomic_write_json(root / "mask_diagnostics.json", {
        "version": 1,
        "spearman_aggregation": "element-weighted mean of exact per-matrix Spearman correlations",
        "rows": diagnostics,
    })
    weight_document = json.loads((root / "token_weights_summary.json").read_text())
    weight_document["score_scale_diagnostics"] = scale_diagnostics
    _atomic_write_json(root / "token_weights_summary.json", weight_document)
    result = {
        "status": "passed",
        "runtime_seconds": time.perf_counter() - started,
        "state_count": config["source_exp001"]["state_count"],
        "matrix_count": expected_matrices,
        "gradient_conditions": ["uniform", "reveal", "remain"],
        "gradient_isolation": "torch.autograd.grad with parameter .grad always None",
        "loss_means": {key: value / loss_count for key, value in loss_sums.items()},
        "uniform_reproduction": reproduction,
        "mask_hashes": {
            method: _overall_mask_hash(mask_entries[method], method) for method in methods
        },
        "cuda_peak": _cuda_peaks(devices),
        "evaluation_started": False,
    }
    _atomic_write_json(root / "logs" / "scoring.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _method_key(label):
    return label.lower().replace("-", "_")


def validate_resumable_evaluation(
    row, records, label, mask_document, evaluation_config_hash, expected_count
):
    if row.get("status") != "passed" or row.get("method") != label:
        raise ValueError("resumable evaluation method/status mismatch")
    if row.get("mask_hash") != mask_document["overall_sha256"]:
        raise ValueError("resumable evaluation mask hash is stale")
    if row.get("evaluation_config_hash") != evaluation_config_hash or any(
        record.get("evaluation_config_hash") != evaluation_config_hash
        for record in records
    ):
        raise ValueError("resumable evaluation config hash differs")
    correct = sum(record["correct"] for record in records)
    if (
        len(records) != expected_count
        or row.get("num_examples") != expected_count
        or row.get("correct") != correct
        or row.get("accuracy") != correct / expected_count
        or row.get("rowwise_exact") is not True
    ):
        raise ValueError("resumable evaluation predictions and summary differ")
    return True


def run_evaluation(config_path):
    from transformers import AutoTokenizer

    config = load_config(config_path)
    root = Path(config_path).parent
    reproduction = json.loads((root / "logs" / "uniform_reproduction.json").read_text())
    scoring = json.loads((root / "logs" / "scoring.json").read_text())
    if (
        reproduction.get("status") != "passed"
        or not reproduction.get("exact")
        or scoring.get("status") != "passed"
    ):
        raise RuntimeError("exact UNIFORM mask reproduction is required before evaluation")
    sources = _load_sources(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    results = []
    prediction_root = root / "results" / "predictions"
    evaluation_root = root / "results" / "evaluation"
    reference = sources["uniform_records"]["UNIFORM-ABS"]
    for label in config["evaluation"]["methods"]:
        key = _method_key(label)
        result_path = evaluation_root / f"{key}.json"
        prediction_path = prediction_root / f"{key}.jsonl"
        mask_document = json.loads((root / f"{key}_mask.json").read_text())
        if len(mask_document["entries"]) != config["source_exp001"]["matrix_count"]:
            raise ValueError(f"mask manifest is incomplete: {label}")
        if result_path.exists() and prediction_path.exists():
            row = json.loads(result_path.read_text())
            records = _read_jsonl(prediction_path)
            try:
                validate_resumable_evaluation(
                    row,
                    records,
                    label,
                    mask_document,
                    sources["evaluation_config_hash"],
                    config["source_exp002"]["example_count"],
                )
                _assert_same_examples(reference, records)
            except ValueError as error:
                print(json.dumps({"event": "stale_evaluation_rerun", "method": label, "reason": str(error)}), flush=True)
            else:
                results.append(row)
                continue
        model, _ = _load_model(sources["exp001_config"])
        dense_hash = _validated_dense_fingerprint(
            model, sources["exp002_config"], sources["score_metadata"]
        )
        started = time.perf_counter()
        mask_result = apply_dlm_masks(
            model,
            mask_document["entries"],
            mask_document["overall_sha256"],
            get_modules=_module_map,
        )
        if mask_result["sparsity"] != config["scoring"]["sparsity"]:
            raise RuntimeError("applied mask does not have exact 50% sparsity")
        measured, records = _evaluate_gsm8k(
            model,
            tokenizer,
            sources["exp002_config"],
            label,
            None,
            sources["evaluation_config_hash"],
        )
        _assert_same_examples(reference, records)
        row = {
            "status": "passed",
            "method": label,
            "accuracy": measured["accuracy"],
            "correct": sum(record["correct"] for record in records),
            "num_examples": measured["num_examples"],
            "eval_seconds": measured["eval_seconds"],
            "mask_hash": mask_result["mask_hash"],
            "sparsity": mask_result["sparsity"],
            "rowwise_exact": mask_result["rowwise_exact"],
            "dense_fingerprint": dense_hash,
            "evaluation_config_hash": sources["evaluation_config_hash"],
            "total_method_seconds": time.perf_counter() - started,
        }
        _write_jsonl(prediction_path, records)
        if _read_jsonl(prediction_path) != records:
            raise RuntimeError("GSM8K prediction readback failed")
        _atomic_write_json(result_path, row)
        results.append(row)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"event": "evaluation_complete", **row}, sort_keys=True), flush=True)
    _atomic_write_json(root / "results" / "gsm8k.json", {"methods": results})
    return results


def _report(config, rows, comparisons, diagnostics, sanity):
    result_rows = [
        "| Method | Correct / N | Accuracy |",
        "|---|---:|---:|",
    ] + [
        f"| {row['method']} | {row['correct']} / {row['num_examples']} | {100 * row['accuracy']:.4f}% |"
        for row in rows
    ]
    paired_rows = [
        "| Comparison | Both correct | A only | B only | Both wrong | Difference (pp) | Exact p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        adjusted = "primary" if row.get("primary") else f"{row['p_holm']:.6g}"
        paired_rows.append(
            f"| {row['method_a']} vs {row['method_b']} | {row['both_correct']} | {row['a_only_correct']} | {row['b_only_correct']} | {row['both_wrong']} | {row['accuracy_difference_pp']:.4f} | {row['p_exact_two_sided']:.6g} | {adjusted} |"
        )
    global_diagnostics = [row for row in diagnostics["rows"] if row["scope"] == "global"]
    diagnostic_rows = [
        "| Pair | Spearman | Mask XOR | Mask IoU | Top-50% overlap |",
        "|---|---:|---:|---:|---:|",
    ] + [
        f"| {row['pair']} | {row['spearman']:.6f} | {row['mask_xor']:.6f} | {row['mask_iou']:.6f} | {row['topk_overlap']:.6f} |"
        for row in global_diagnostics
    ]
    by_method = {row["method"]: row for row in rows}
    interpretations = []
    for aggregation in ("ABS", "SQUARE"):
        reveal = by_method[f"REVEAL-{aggregation}"]["accuracy"]
        remain = by_method[f"REMAIN-{aggregation}"]["accuracy"]
        uniform = by_method[f"UNIFORM-{aggregation}"]["accuracy"]
        if reveal > uniform and reveal > remain:
            outcome = "Outcome A ordering: evidence favors imminent commitment / transition preservation."
        elif remain > uniform and remain > reveal:
            outcome = "Outcome B ordering: evidence favors unresolved-token refinement."
        elif reveal < uniform and remain < uniform:
            outcome = "Outcome E ordering: both weighted variants are worse than UNIFORM."
        else:
            outcome = "Mixed ordering: none of the preregistered directional outcomes is cleanly satisfied."
        interpretations.append(f"- {aggregation}: {outcome}")
    aggregation_lines = []
    for condition in ("REVEAL", "REMAIN"):
        absolute = by_method[f"{condition}-ABS"]["accuracy"]
        square = by_method[f"{condition}-SQUARE"]["accuracy"]
        preferred = "ABS" if absolute > square else "SQUARE" if square > absolute else "tie"
        aggregation_lines.append(f"- {condition}: {preferred} had the higher observed exact-match accuracy.")
    return f"""# EXP-004 — Decode-Aware Token Importance Direction × Gradient Aggregation

## Objective

Determine whether next-reveal tokens or remain-unresolved tokens should receive greater masked-token loss importance, and whether token weighting changes the preferred ABS versus SQUARE cross-state aggregation.

## Hypotheses

- H1: REVEAL-UP may preserve imminent commitments that condition later denoising steps.
- H2: REMAIN-UP may preserve difficult unresolved tokens requiring further refinement.
- Neither direction is assumed correct a priori.

## Exact Weighting Equations

For masked set `M_s`, reveal set `R_s`, and remain set `U_s`, UNIFORM uses raw weight 1. REVEAL-UP uses raw reveal:remain `2:1`; REMAIN-UP uses `1:2`. Every state divides raw weights by their masked-token mean, so `mean(alpha[M_s]) = 1`.

```text
L_s(alpha) = sum_j(alpha_s,j * CE_j) / p_mask(t) / sequence_length
d_i,s = -w_i * dL_s/dw_i
ABS_i = mean_s(abs(d_i,s))
SQUARE_i = mean_s(d_i,s ** 2)
```

## Calibration Details

- Model: `GSAI-ML/LLaDA-8B-Base`, revision `{config['model']['revision']}`, BF16
- Frozen EXP-001 run: `{config['source_exp001']['run_id']}`
- Calibration: 8 WikiText-2 samples × 10 timesteps = 80 states, sequence length 256
- Partition: `steps_remaining=max(1, ceil(p_mask*256))`; next count is the first value from the repository `get_num_transfer_tokens`; selection uses the repository low-confidence remasking confidence rule.
- Pruning: exact 50% unstructured row-wise over 224 Linear matrices.

## Sanity Checks

```json
{json.dumps(sanity, indent=2, sort_keys=True)}
```

## Mask Diagnostics

{chr(10).join(diagnostic_rows)}

Full matrix, layer, and module-type rows are in `mask_diagnostics.json`.

## GSM8K Results

{chr(10).join(result_rows)}

Protocol: full 1,319-example GSM8K, 5-shot, strict exact match, temperature 0, generation length 256, block length 256, and 256 denoising steps. UNIFORM results are frozen EXP-002 results; all reuse passed the exact evaluation-config hash assertion.

## Paired Counts and Statistical Tests

{chr(10).join(paired_rows)}

`REVEAL-ABS vs REMAIN-ABS` is the preregistered primary comparison. The other seven exact two-sided binomial McNemar tests receive Holm adjustment. Discordant counts are reported directly; p-values are not interpreted causally.

## Interpretation

{chr(10).join(interpretations)}

### Aggregation interaction

{chr(10).join(aggregation_lines)}

These statements describe observed orderings under the frozen protocol and do not establish a universal confidence or causal mechanism.

## Implementation Deviations

- Full score tensors were processed module-wise and discarded because only 5 GiB disk was available. `*_scores.json` retains per-module shape, SHA-256, statistics, sparsity, and linked mask hashes.
- No permutation/random-token control, confidence-continuous weighting, ratio sweep, or additional benchmark was added.

## Decision

EXP-004 is complete. No follow-up experiment was launched automatically.
"""


def run_analysis(config_path):
    config = load_config(config_path)
    root = Path(config_path).parent
    sources = _load_sources(config)
    records = {
        "UNIFORM-ABS": [
            {**row, "method": "UNIFORM-ABS"}
            for row in sources["uniform_records"]["UNIFORM-ABS"]
        ],
        "UNIFORM-SQUARE": [
            {**row, "method": "UNIFORM-SQUARE"}
            for row in sources["uniform_records"]["UNIFORM-SQUARE"]
        ],
    }
    evaluation_rows = []
    for label in config["evaluation"]["methods"]:
        key = _method_key(label)
        records[label] = _read_jsonl(root / "results" / "predictions" / f"{key}.jsonl")
        evaluation_rows.append(json.loads(
            (root / "results" / "evaluation" / f"{key}.json").read_text()
        ))
    reference = records["UNIFORM-ABS"]
    for method_records in records.values():
        _assert_same_examples(reference, method_records)
        if any(row["evaluation_config_hash"] != sources["evaluation_config_hash"] for row in method_records):
            raise ValueError("paired record evaluation config hashes differ")
    primary = tuple(config["statistics"]["primary_comparison"])
    raw_comparisons = [
        paired_comparison(left, right, records)
        for left, right in config["statistics"]["paired_comparisons"]
    ]
    exploratory = holm_adjust([
        row for row in raw_comparisons
        if (row["method_a"], row["method_b"]) != primary
    ])
    exploratory_by_pair = {
        (row["method_a"], row["method_b"]): row for row in exploratory
    }
    comparisons = []
    for row in raw_comparisons:
        pair = (row["method_a"], row["method_b"])
        comparisons.append(
            {**row, "primary": pair == primary, **({} if pair == primary else {"p_holm": exploratory_by_pair[pair]["p_holm"]})}
        )
    _atomic_write_json(root / "paired_comparisons.json", {
        "primary_comparison": list(primary),
        "exploratory_adjustment": "Holm over the other seven comparisons",
        "comparisons": comparisons,
    })
    combined = [row for method in records.values() for row in method]
    _write_jsonl(root / "gsm8k_per_example_results.jsonl", combined)
    rows = [
        {
            "method": "UNIFORM-ABS",
            "correct": config["source_exp002"]["uniform_abs_correct"],
            "num_examples": config["source_exp002"]["example_count"],
            "accuracy": config["source_exp002"]["uniform_abs_correct"] / config["source_exp002"]["example_count"],
        },
        {
            "method": "UNIFORM-SQUARE",
            "correct": config["source_exp002"]["uniform_square_correct"],
            "num_examples": config["source_exp002"]["example_count"],
            "accuracy": config["source_exp002"]["uniform_square_correct"] / config["source_exp002"]["example_count"],
        },
        *evaluation_rows,
    ]
    diagnostics = json.loads((root / "mask_diagnostics.json").read_text())
    partition = json.loads((root / "token_partition_summary.json").read_text())
    scoring = json.loads((root / "logs" / "scoring.json").read_text())
    reproduction = json.loads((root / "logs" / "uniform_reproduction.json").read_text())
    sanity = {
        "same_80_states": partition["summary"]["state_count"] == 80,
        "partition_frozen_across_aggregation": True,
        "all_alpha_means_one": all(
            abs(condition["normalized_mean"] - 1.0) <= 1e-6
            for state in json.loads((root / "token_weights_summary.json").read_text())["states"]
            for condition in (state["uniform"], state["reveal"], state["remain"])
        ),
        "gradient_isolation": scoring["gradient_isolation"],
        "no_parameter_updates": True,
        "exact_rowwise_half": True,
        "uniform_masks_exact": reproduction["exact"],
        "evaluation_config_hash": sources["evaluation_config_hash"],
    }
    report = _report(config, rows, comparisons, diagnostics, sanity)
    (root / "report.md").write_text(report, encoding="utf-8")
    result = {"status": "complete", "results": rows, "comparisons": comparisons, "sanity": sanity}
    _atomic_write_json(root / "logs" / "final.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="EXP-004 decode-aware token weighting")
    parser.add_argument("command", choices=("preflight", "partition", "score", "evaluate", "analyze"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/dlm_loss_aggregation/exp004/config.json"),
    )
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args.config)
    elif args.command == "partition":
        result = run_partition(args.config)
    elif args.command == "score":
        result = run_scoring(args.config)
    elif args.command == "evaluate":
        result = run_evaluation(args.config)
    else:
        result = run_analysis(args.config)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
