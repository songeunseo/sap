import argparse
import gc
import hashlib
import math
import json
import os
import shutil
import statistics
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
        "experiment": "EXP-005",
        "model": {
            "id": "GSAI-ML/LLaDA-8B-Base",
            "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
            "dtype": "bfloat16",
        },
        "source_exp001.run_id": "20260828T175010-2484545",
        "source_exp001.state_sha256": "1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df",
        "source_exp001.state_count": 80,
        "source_exp002.evaluation_config_hash": "1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add",
        "source_exp004.partition_summary": "experiments/dlm_loss_aggregation/exp004/token_partition_summary.json",
        "partition.denoising_steps": 256,
        "partition.remasking": "low_confidence",
        "partition.confidence": "softmax_probability_of_argmax",
        "weighting.conditions": ["uniform", "symmetric_reveal", "symmetric_remain"],
        "weighting.rho": 0.5,
        "weighting.contrast": "reveal=1; remain=-q/(1-q)",
        "weighting.normalization": "masked_token_mean_one",
        "weighting.mirror": "alpha_reveal + alpha_remain = 2 on masked positions",
        "scoring.aggregations": ["abs"],
        "scoring.sparsity": 0.5,
        "scoring.block_count": 32,
        "scoring.modules_per_block": 7,
        "uniform_reproduction.legacy_block0_control.abs_global_xor": 0.0008339515099158654,
        "uniform_reproduction.legacy_block0_control.max_module_xor": 0.0019975900650024414,
        "uniform_reproduction.legacy_block0_control.safety_factor": 1.25,
        "uniform_reproduction.global_xor_threshold": 0.0015212251589848445,
        "uniform_reproduction.module_xor_threshold": 0.0024969875812530518,
        "storage.safety_bytes": 536870912,
        "evaluation.methods": ["SYM-REVEAL-ABS", "SYM-REMAIN-ABS"],
        "statistics.primary_comparison": ["SYM-REVEAL-ABS", "SYM-REMAIN-ABS"],
        "statistics.paired_comparisons": [
            ["SYM-REVEAL-ABS", "SYM-REMAIN-ABS"],
            ["SYM-REVEAL-ABS", "UNIFORM-ABS"],
            ["SYM-REMAIN-ABS", "UNIFORM-ABS"],
        ],
    }
    for dotted, wanted in expected.items():
        actual = config
        for key in dotted.split("."):
            actual = actual[key]
        if actual != wanted:
            raise ValueError(f"EXP-005 config mismatch: {dotted}")
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


def symmetric_token_weights(mask, reveal_mask, rho=0.5):
    if mask.dtype != torch.bool or reveal_mask.shape != mask.shape:
        raise ValueError("mask and reveal_mask must be matching boolean tensors")
    if reveal_mask.dtype != torch.bool or (reveal_mask & ~mask).any().item():
        raise ValueError("reveal positions must be masked")
    if rho < 0:
        raise ValueError("rho must be non-negative")
    masked_count = int(mask.sum().item())
    reveal_count = int(reveal_mask.sum().item())
    if reveal_count <= 0 or reveal_count >= masked_count:
        raise ValueError("symmetric weighting requires both reveal and remain tokens")
    q = reveal_count / masked_count
    contrast = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    contrast[reveal_mask] = 1.0
    contrast[mask & ~reveal_mask] = -q / (1.0 - q)
    reveal_alpha = torch.where(mask, 1.0 + rho * contrast, torch.zeros_like(contrast))
    remain_alpha = torch.where(mask, 1.0 - rho * contrast, torch.zeros_like(contrast))
    if not torch.isfinite(reveal_alpha[mask]).all().item() or not torch.isfinite(remain_alpha[mask]).all().item():
        raise RuntimeError("symmetric alpha is non-finite")
    if (reveal_alpha[mask] <= 0).any().item() or (remain_alpha[mask] <= 0).any().item():
        raise ValueError("rho produces non-positive masked-token alpha")
    one = torch.tensor(1.0, dtype=torch.float32, device=mask.device)
    if not torch.isclose(reveal_alpha[mask].mean(), one, atol=1e-6, rtol=0):
        raise RuntimeError("symmetric reveal alpha mean is not one")
    if not torch.isclose(remain_alpha[mask].mean(), one, atol=1e-6, rtol=0):
        raise RuntimeError("symmetric remain alpha mean is not one")
    if not torch.allclose(reveal_alpha[mask] + remain_alpha[mask], torch.full_like(reveal_alpha[mask], 2.0), atol=1e-6, rtol=0):
        raise RuntimeError("symmetric alpha conditions are not mirror images")
    if not torch.isclose(
        (reveal_alpha[mask] - 1).abs().mean(),
        (remain_alpha[mask] - 1).abs().mean(),
        atol=1e-6,
        rtol=0,
    ) or not torch.isclose(
        (reveal_alpha[mask] - 1).square().mean(),
        (remain_alpha[mask] - 1).square().mean(),
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError("symmetric alpha perturbation magnitudes differ")
    return reveal_alpha, remain_alpha


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


def summarize_symmetric_token_weights(partition, rho=0.5):
    masked = partition["masked_count"]
    reveal = partition["reveal_count"]
    remain = partition["remain_count"]
    if masked <= 0 or reveal <= 0 or remain <= 0 or reveal + remain != masked:
        raise ValueError("symmetric partition counts are invalid")
    q = reveal / masked
    reveal_contrast = 1.0
    remain_contrast = -q / (1.0 - q)
    reveal_alpha = 1.0 + rho * reveal_contrast
    remain_reveal_alpha = 1.0 - rho * reveal_contrast
    remain_alpha = 1.0 + rho * q / (1.0 - q)
    if min(reveal_alpha, remain_reveal_alpha, remain_alpha) <= 0:
        raise ValueError("rho produces non-positive symmetric alpha")
    result = {
        "masked_count": masked,
        "reveal_count": reveal,
        "remain_count": remain,
        "q": q,
        "rho": rho,
        "uniform": {
            "reveal_alpha": 1.0,
            "remain_alpha": 1.0,
            "normalized_mean": 1.0,
        },
        "symmetric_reveal": {
            "reveal_alpha": reveal_alpha,
            "remain_alpha": 1.0 - rho * q / (1.0 - q),
            "normalized_mean": 1.0,
        },
        "symmetric_remain": {
            "reveal_alpha": remain_reveal_alpha,
            "remain_alpha": remain_alpha,
            "normalized_mean": 1.0,
        },
        "contrast": {
            "reveal": reveal_contrast,
            "remain": remain_contrast,
            "mean": 0.0,
        },
        "perturbation": {
            "mean_abs": rho * (reveal * abs(reveal_contrast) + remain * abs(remain_contrast)) / masked,
            "mean_square": rho * rho * (reveal * reveal_contrast ** 2 + remain * remain_contrast ** 2) / masked,
        },
    }
    if abs(result["symmetric_reveal"]["remain_alpha"] - (1.0 - rho * q / (1.0 - q))) > 1e-12:
        raise RuntimeError("symmetric reveal alpha formula mismatch")
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


def allow_repeated_compiled_backwards():
    import torch._functorch.config as functorch_config

    functorch_config.donated_buffer = False


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
    if {row["method"] for row in rows} != {"abs"}:
        raise ValueError("EXP-005 UNIFORM reproduction requires ABS rows")
    by_method = {}
    failed_modules = []
    for method in ("abs",):
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
            weights.append({"state_index": state_index, **summarize_symmetric_token_weights(
                partition, config["weighting"]["rho"]
            )})
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
        "rho": config["weighting"]["rho"],
        "contrast": "c_R=1; c_U=-q/(1-q)",
        "normalization": "mean(alpha[M_s])=1",
        "loss": "sum(alpha * token_ce) / p_mask / sequence_length",
        "states": weights,
    })
    return partition_document


def run_partition_frozen(config_path):
    config = load_config(config_path)
    root = Path(config_path).parent
    preflight = json.loads((root / "logs" / "preflight.json").read_text())
    if preflight.get("status") != "passed":
        raise RuntimeError("EXP-005 preflight has not passed")
    sources = _load_sources(config)
    partition_path = Path(config["source_exp004"]["partition_summary"])
    partition_document = json.loads(partition_path.read_text())
    validate_partition_artifact(
        partition_document, sources["manifest"], config["source_exp001"]["state_count"]
    )
    weights = [
        {
            "state_index": state["state_index"],
            **summarize_symmetric_token_weights(state, config["weighting"]["rho"]),
        }
        for state in partition_document["states"]
    ]
    _atomic_write_json(root / "calibration_state_manifest.json", sources["manifest"])
    _atomic_write_json(root / "token_partition_summary.json", partition_document)
    _atomic_write_json(root / "token_weights_summary.json", {
        "version": 1,
        "source_partition": str(partition_path),
        "partition_sha256": partition_document["sha256"],
        "rho": config["weighting"]["rho"],
        "contrast": "c_R=1; c_U=-q/(1-q)",
        "normalization": "mean(alpha[M_s])=1",
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

    allow_repeated_compiled_backwards()
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
    methods = ("sym_reveal_abs", "sym_remain_abs")
    score_documents = {method: {"method": method, "modules": []} for method in methods}
    mask_entries = {method: [] for method in methods}
    reproduction_rows = []
    matrix_diagnostics = []
    scale_diagnostics = []
    loss_sums = {condition: 0.0 for condition in ("uniform", "reveal", "remain")}
    loss_count = 0
    comparison_pairs = (
        ("sym_reveal_abs", "sym_remain_abs"),
        ("sym_reveal_abs", "uniform_abs"),
        ("sym_remain_abs", "uniform_abs"),
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
                symmetric_reveal, symmetric_remain = symmetric_token_weights(
                    mask, reveal_mask, config["weighting"]["rho"]
                )
                alpha = {
                    "uniform": token_weights(mask, reveal_mask, "uniform"),
                    "reveal": symmetric_reveal,
                    "remain": symmetric_remain,
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
                for condition in ("uniform", "reveal", "remain"):
                    if abs(alpha[condition][mask].mean().item() - 1.0) > 1e-6:
                        raise RuntimeError("masked-token alpha mean differs from one")
                    loss_sums[condition] += losses[condition].detach().cpu().item()
                    raw_scale = 1.0
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
                    "uniform_abs": condition_scores["uniform"]["abs"],
                    "sym_reveal_abs": condition_scores["reveal"]["abs"],
                    "sym_remain_abs": condition_scores["remain"]["abs"],
                }
                named_masks = {
                    method: rowwise_mask(score, config["scoring"]["sparsity"])
                    for method, score in named_scores.items()
                }
                frozen_mask = _load_mask(frozen[("abs", block_index, name)])
                regenerated = named_masks["uniform_abs"]
                different = int(regenerated.ne(frozen_mask).sum().item())
                reproduction_rows.append({
                    "method": "abs",
                    "layer": block_index,
                    "module": name,
                    "module_type": module_type,
                    "num_weights": regenerated.numel(),
                    "different": different,
                    "xor": different / regenerated.numel(),
                })
                named_masks["uniform_abs"] = frozen_mask
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
        or scoring.get("status") != "passed"
    ):
        raise RuntimeError("passed UNIFORM reproduction and scoring gates are required before evaluation")
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


def _report(config, rows, comparisons, diagnostics, sanity, partition, weights, scoring, preflight):
    n = config["source_exp002"]["example_count"]
    by_method = {row["method"]: row for row in rows}
    references = {
        row["method"]: {**row, "num_examples": n, "accuracy": row["correct"] / n}
        for row in config["source_exp002"]["reference_results"]
    }
    all_results = {**references, **by_method}
    dense = references["Dense"]["accuracy"]
    wanda = references["Wanda"]["accuracy"]
    sparsegpt = references["SparseGPT"]["accuracy"]
    order = (
        "Dense", "REVEAL-ABS", "REMAIN-ABS", "UNIFORM-ABS",
        "REVEAL-SQUARE", "REMAIN-SQUARE", "UNIFORM-SQUARE",
        "Wanda", "SparseGPT", "DLM-SUM",
    )
    categories = {
        "Dense": "비희소 기준", "Wanda": "표준 참조", "SparseGPT": "표준 참조",
        "DLM-SUM": "EXP-002 기각", "UNIFORM-ABS": "동결 대조군",
        "UNIFORM-SQUARE": "동결 대조군", "REVEAL-ABS": "EXP-004 신규",
        "REVEAL-SQUARE": "EXP-004 신규", "REMAIN-ABS": "EXP-004 신규",
        "REMAIN-SQUARE": "EXP-004 신규",
    }
    calibrations = {
        "Dense": "없음", "Wanda": "clean 8×256", "SparseGPT": "clean 8×256",
        "DLM-SUM": "동일 80 states", "UNIFORM-ABS": "동일 80 states",
        "UNIFORM-SQUARE": "동일 80 states", "REVEAL-ABS": "동일 80 states",
        "REVEAL-SQUARE": "동일 80 states", "REMAIN-ABS": "동일 80 states",
        "REMAIN-SQUARE": "동일 80 states",
    }
    result_rows = [
        "| 구분 | 방법 | 희소도 | Calibration | 정답 / 1319 | 정확도 | Dense 대비 | Wanda 대비 | SparseGPT 대비 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for method in order:
        row = all_results[method]
        sparsity = row.get("sparsity", 0.5)
        result_rows.append(
            f"| {categories[method]} | {method} | {100 * sparsity:.4f}% | {calibrations[method]} | "
            f"{row['correct']} / {n} | {100 * row['accuracy']:.4f}% | "
            f"{row['correct'] - references['Dense']['correct']:+d} / {100 * (row['accuracy'] - dense):+.4f} pp | "
            f"{row['correct'] - references['Wanda']['correct']:+d} / {100 * (row['accuracy'] - wanda):+.4f} pp | "
            f"{row['correct'] - references['SparseGPT']['correct']:+d} / {100 * (row['accuracy'] - sparsegpt):+.4f} pp |"
        )

    paired_rows = [
        "| 비교 (A vs B) | 둘 다 정답 | A만 정답 | B만 정답 | 둘 다 오답 | Discordant | A−B | Exact p | Holm p | 역할 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        adjusted = "—" if row.get("primary") else f"{row['p_holm']:.6g}"
        paired_rows.append(
            f"| {row['method_a']} vs {row['method_b']} | {row['both_correct']} | "
            f"{row['a_only_correct']} | {row['b_only_correct']} | {row['both_wrong']} | "
            f"{row['a_only_correct'] + row['b_only_correct']} | {row['accuracy_difference_pp']:+.4f} pp | "
            f"{row['p_exact_two_sided']:.6g} | {adjusted} | {'Primary' if row.get('primary') else 'Exploratory'} |"
        )

    global_diagnostics = [row for row in diagnostics["rows"] if row["scope"] == "global"]
    diagnostic_rows = [
        "| Mask 비교 | 다른 weight 수 | Global XOR | Spearman | Mask IoU | 보존 top-50% overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ] + [
        f"| {row['pair'].replace('_', '-').upper()} | {row['different']:,} | "
        f"{100 * row['mask_xor']:.4f}% | {row['spearman']:.6f} | "
        f"{100 * row['mask_iou']:.4f}% | {100 * row['topk_overlap']:.4f}% |"
        for row in global_diagnostics
    ]
    layer_summary_rows = [
        "| Mask 비교 | Layer XOR 최소 | 중앙값 | 최대 (layer) | Layer Spearman 최소 | 중앙값 | 최대 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    matrix_summary_rows = [
        "| Mask 비교 | Matrix XOR 최소 | 중앙값 | 최대 (layer/module) | Matrix Spearman 최소 | 중앙값 | 최대 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for global_row in global_diagnostics:
        pair = global_row["pair"]
        layers = [row for row in diagnostics["rows"] if row["scope"] == "layer" and row["pair"] == pair]
        matrices = [row for row in diagnostics["rows"] if row["scope"] == "matrix" and row["pair"] == pair]
        layer_max = max(layers, key=lambda row: row["mask_xor"])
        matrix_max = max(matrices, key=lambda row: row["mask_xor"])
        layer_xor = [row["mask_xor"] for row in layers]
        layer_spearman = [row["spearman"] for row in layers]
        matrix_xor = [row["mask_xor"] for row in matrices]
        matrix_spearman = [row["spearman"] for row in matrices]
        label = pair.replace("_", "-").upper()
        layer_summary_rows.append(
            f"| {label} | {100 * min(layer_xor):.4f}% | {100 * statistics.median(layer_xor):.4f}% | "
            f"{100 * layer_max['mask_xor']:.4f}% (L{layer_max['layer']}) | {min(layer_spearman):.6f} | "
            f"{statistics.median(layer_spearman):.6f} | {max(layer_spearman):.6f} |"
        )
        matrix_summary_rows.append(
            f"| {label} | {100 * min(matrix_xor):.4f}% | {100 * statistics.median(matrix_xor):.4f}% | "
            f"{100 * matrix_max['mask_xor']:.4f}% (L{matrix_max['layer']}/{matrix_max['module']}) | "
            f"{min(matrix_spearman):.6f} | {statistics.median(matrix_spearman):.6f} | "
            f"{max(matrix_spearman):.6f} |"
        )
    module_type_rows = [
        "| Mask 비교 | Module type | 다른 weight 수 | XOR | Spearman | Mask IoU | 보존 overlap |",
        "|---|---|---:|---:|---:|---:|---:|",
    ] + [
        f"| {row['pair'].replace('_', '-').upper()} | {row['module_type']} | {row['different']:,} | "
        f"{100 * row['mask_xor']:.4f}% | {row['spearman']:.6f} | "
        f"{100 * row['mask_iou']:.4f}% | {100 * row['topk_overlap']:.4f}% |"
        for row in diagnostics["rows"]
        if row["scope"] == "module_type"
    ]

    states = partition["states"]
    masked_counts = [row["masked_count"] for row in states]
    reveal_counts = [row["reveal_count"] for row in states]
    remain_counts = [row["remain_count"] for row in states]
    reveal_confidence = []
    remain_confidence = []
    for state in states:
        confidence = {row["position"]: row["confidence"] for row in state["masked_predictions"]}
        reveal_confidence.extend(confidence[position] for position in state["reveal_indices"])
        remain_confidence.extend(confidence[position] for position in state["remain_indices"])

    def observed_range(condition, key):
        values = [state[condition][key] for state in weights["states"]]
        return min(values), max(values)

    alpha_rows = [
        "| 조건 | 정규화 전 비율 (R:U) | 정규화 후 reveal α 범위 | 정규화 후 remain α 범위 | state별 평균 α 범위 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition, ratio in (("uniform", "1:1"), ("reveal", "2:1"), ("remain", "1:2")):
        reveal_range = observed_range(condition, "reveal_alpha")
        remain_range = observed_range(condition, "remain_alpha")
        mean_range = observed_range(condition, "normalized_mean")
        alpha_rows.append(
            f"| {condition.upper()} | {ratio} | {reveal_range[0]:.6f}–{reveal_range[1]:.6f} | "
            f"{remain_range[0]:.6f}–{remain_range[1]:.6f} | {mean_range[0]:.16f}–{mean_range[1]:.16f} |"
        )

    primary = next(row for row in comparisons if row.get("primary"))
    significant = [
        f"{row['method_a']} vs {row['method_b']}"
        for row in comparisons
        if not row.get("primary") and row["p_holm"] < 0.05
    ]
    mask_hash_rows = [
        "| 방법 | Overall mask SHA-256 |",
        "|---|---|",
    ] + [
        f"| {method.replace('_', '-').upper()} | `{digest}` |"
        for method, digest in scoring["mask_hashes"].items()
    ]
    evaluation_runtime_rows = [
        "| 방법 | 평가 시간 | 초 |",
        "|---|---:|---:|",
    ]

    def duration(seconds):
        seconds = int(round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}시간 {minutes}분 {seconds}초"

    for method in ("REVEAL-ABS", "REVEAL-SQUARE", "REMAIN-ABS", "REMAIN-SQUARE"):
        seconds = by_method[method]["eval_seconds"]
        evaluation_runtime_rows.append(f"| {method} | {duration(seconds)} | {seconds:.3f} |")
    evaluation_seconds = sum(by_method[method]["eval_seconds"] for method in config["evaluation"]["methods"])
    summary = partition["summary"]
    max_reproduction_xor = max(row["xor"] for row in scoring["uniform_reproduction"]["mismatched_modules"])
    reveal_share = summary["reveal_tokens"] / summary["masked_tokens"]
    return f"""# EXP-004 — 디코딩 인지 토큰 중요도 방향 × 그래디언트 집계

## 0. 요약

- **최고 성능:** REVEAL-ABS, **767 / 1319 = 58.1501%**.
- REVEAL-ABS는 동결 UNIFORM-ABS보다 **+22문제 / +1.6679 pp** 높았지만 exact McNemar `p=0.0776532`였고, exploratory Holm 보정값은 `0.388266`이었다.
- 사전 지정 primary인 REVEAL-ABS vs REMAIN-ABS는 **+20문제 / +1.5163 pp**, discordant `73:53`, exact `p=0.0901229`였다. 따라서 REVEAL 방향은 **유망한 신호이지 확립된 우위가 아니다**.
- ABS는 두 token objective 모두에서 SQUARE보다 높았다. Holm 보정 후 유의한 비교는 **REVEAL-ABS vs REVEAL-SQUARE** 하나였다 (`+3.7908 pp`, Holm `p=0.00604355`).
- REVEAL-ABS는 Wanda보다 **+90문제 / +6.8234 pp**, SparseGPT보다 **+183문제 / +13.8741 pp** 높았지만, Dense보다는 **−171문제 / −12.9644 pp** 낮았다.
- 결론: 관찰 순위는 preregistered Outcome A와 일치하지만 token-weighting 개선은 통계적으로 확정되지 않았다. **ABS는 여전히 기준 aggregation**이다.

## 1. 목적 (Objective)

동일한 DLM-loss gradient pruning 절차에서 다음 transition에 곧 reveal/commit될 token과 계속 masked/unresolved로 남을 token 중 어느 쪽을 더 중요하게 보아야 하는지 확인한다. 동시에 token-level 중요도 도입 후에도 cross-state aggregation의 선호가 ABS인지, SQUARE로 뒤집히는지 검증한다.

실험이 바꾸는 것은 오직 masked-token loss에 곱하는 `alpha_(s,j)`뿐이다. 모델, pruning universe, sparsity, calibration states, timestep, corruption, DLM loss의 나머지 normalization, GSM8K protocol은 고정했다.

## 2. 가설 (Hypotheses)

- **H1 — REVEAL/COMMIT:** 다음 transition에 reveal되는 token은 이후 denoising context가 되므로, 이 token에 중요한 parameter를 보존하면 오차 전파를 줄일 수 있다. 예측: `REVEAL > UNIFORM`.
- **H2 — REMAIN/UNRESOLVED:** 계속 masked로 남는 token은 더 어렵고 추가 refinement가 필요하므로, 이 token에 중요한 parameter를 보존하는 편이 유리하다. 예측: `REMAIN > UNIFORM`.
- 어느 방향도 사전에 정답으로 가정하지 않았다.
- **Primary comparison:** `REVEAL-ABS vs REMAIN-ABS`.

## 3. 고정 설정 (Frozen Setup)

| 항목 | 설정 |
|---|---|
| 모델 | `GSAI-ML/LLaDA-8B-Base` |
| Revision | `{config['model']['revision']}` |
| dtype | BF16 |
| EXP-001 원본 run | `{config['source_exp001']['run_id']}` |
| Calibration dataset | WikiText-2 train, 8 samples |
| Timesteps | `0.05, 0.15, ..., 0.95`의 10개 지점 |
| Calibration states | 8 × 10 = **80 frozen corrupted states** |
| Sequence length | 256 |
| Calibration seed | 0 |
| State SHA-256 | `{partition['source_state_sha256']}` |
| Pruning | 50% unstructured, exact row-wise |
| Prunable universe | 32 blocks × 7 Linear matrices = **224 matrices**, 총 6,979,321,856 weights |
| Matrix types | `q_proj`, `k_proj`, `v_proj`, `attn_out`, `ff_proj`, `up_proj`, `ff_out` |

기존 EXP-001/002의 layer set, sparsity, row-wise semantics, revision, samples, corruptions 및 timestep을 변경하지 않았다.

## 4. 다음 reveal partition 정의

각 frozen state `s`에 대해 dense unpruned model을 한 번 실행하고, 현재 masked 위치의 argmax token과 softmax confidence를 저장했다.

```text
M_s = 현재 masked token 위치
steps_remaining = max(1, ceil(p_mask(t) * 256))
next_reveal_count = get_num_transfer_tokens(current_mask, steps_remaining)[0, 0]
R_s = 실제 low-confidence-remasking 규칙에서 다음에 transfer되는 top-confidence 위치
U_s = M_s \\ R_s
```

이 partition은 dense model에서 한 번 계산해 동결했으며 ABS/SQUARE나 pruned model별로 다시 계산하지 않았다. 모든 state에서 `R_s ∩ U_s = ∅`, `R_s ∪ U_s = M_s`를 검증했다.

### 관측된 partition 통계

| 통계 | Masked | Reveal | Remain |
|---|---:|---:|---:|
| 전체 token 수 | {summary['masked_tokens']:,} | {summary['reveal_tokens']:,} | {summary['remain_tokens']:,} |
| 전체 masked 중 비율 | 100.0000% | {100 * reveal_share:.4f}% | {100 * (1 - reveal_share):.4f}% |
| state당 평균 | {statistics.fmean(masked_counts):.4f} | {statistics.fmean(reveal_counts):.4f} | {statistics.fmean(remain_counts):.4f} |
| state당 중앙값 | {statistics.median(masked_counts):.1f} | {statistics.median(reveal_counts):.1f} | {statistics.median(remain_counts):.1f} |
| state당 최소–최대 | {min(masked_counts)}–{max(masked_counts)} | {min(reveal_counts)}–{max(reveal_counts)} | {min(remain_counts)}–{max(remain_counts)} |

- 80 states 중 reveal count 1인 state는 **{sum(value == 1 for value in reveal_counts)}개**, reveal count 2인 state는 **{sum(value == 2 for value in reveal_counts)}개**였다. top-1로 고정하지 않았으며 실제 schedule 결과를 사용했다.
- 전체 confidence: mean `{summary['confidence']['mean']:.6f}`, min `{summary['confidence']['min']:.6f}`, p50 `{summary['confidence']['p50']:.6f}`, p90 `{summary['confidence']['p90']:.6f}`, p99 `{summary['confidence']['p99']:.6f}`, max `{summary['confidence']['max']:.6f}`.
- Reveal confidence: mean `{statistics.fmean(reveal_confidence):.6f}`, median `{statistics.median(reveal_confidence):.6f}`, min `{min(reveal_confidence):.6f}`, max `{max(reveal_confidence):.6f}`.
- Remain confidence: mean `{statistics.fmean(remain_confidence):.6f}`, median `{statistics.median(remain_confidence):.6f}`, min `{min(remain_confidence):.6f}`, max `{max(remain_confidence):.6f}`.
- Reveal token이 전체 masked token의 약 1.13%뿐이라는 심한 불균형은 결과 해석에서 중요하다.

## 5. Token weighting과 DLM loss

```text
UNIFORM raw alpha:    reveal=1, remain=1
REVEAL-UP raw alpha:  reveal=2, remain=1
REMAIN-UP raw alpha:  reveal=1, remain=2

alpha_(s,j) = raw_alpha_(s,j) / mean_(k in M_s)(raw_alpha_(s,k))
mean_(j in M_s)(alpha_(s,j)) = 1
```

{chr(10).join(alpha_rows)}

공식 EXP-001 loss semantics를 유지한 실제 구현식은 다음과 같다.

```text
L_s(alpha) = sum_(j in M_s)[alpha_(s,j) * CE_(s,j)] / p_mask(t) / 256
d_(i,s) = -w_i * dL_s(alpha)/dw_i
```

동일 logits와 per-token CE graph에서 세 loss를 만들되, `torch.autograd.grad`로 condition별 gradient를 독립 계산했다. Parameter `.grad`는 항상 `None`으로 유지했고 state 및 condition 사이 gradient accumulation은 허용하지 않았다. Activation magnitude, Wanda score, confidence 연속 가중, timestep 추가 가중, sample/layer/parameter normalization은 넣지 않았다.

관측된 80-state 평균 loss는 UNIFORM `{scoring['loss_means']['uniform']:.6f}`, REVEAL `{scoring['loss_means']['reveal']:.6f}`, REMAIN `{scoring['loss_means']['remain']:.6f}`였다. Alpha 평균은 동일하지만 token loss와 alpha의 상관 때문에 최종 loss 값까지 동일할 필요는 없다.

## 6. Cross-state aggregation과 mask 생성

```text
S_i^ABS    = mean_s |d_(i,s)|
S_i^SQUARE = mean_s d_(i,s)^2
```

Signed SUM은 EXP-002에서 `0 / 1319`였으므로 다시 평가하지 않았다. 각 matrix의 각 행에서 score 하위 50%를 정확히 prune했다. 네 신규 mask만 생성했고 UNIFORM-ABS/SQUARE는 동결 EXP-001 mask를 재사용했다.

{chr(10).join(mask_hash_rows)}

Full score tensor는 영구 저장하지 않고 module-wise로 처리했다. 각 `*_scores.json`에는 module별 shape, score SHA-256, mean/std/min/max, row prune count, sparsity 및 연결된 mask hash가 남아 있다.

## 7. Sanity checks와 재현성 gate

| 검사 | 결과 |
|---|---|
| 동일 80 states | PASS |
| Dense partition이 ABS/SQUARE에 공통 | PASS |
| 모든 state/condition에서 mean(alpha)≈1 | PASS; 관측 범위 {observed_range('reveal', 'normalized_mean')[0]:.16f}–{observed_range('reveal', 'normalized_mean')[1]:.16f} |
| REVEAL raw ratio 2:1 | PASS |
| REMAIN raw ratio 1:2 | PASS |
| Condition/state gradient 격리 | PASS; `{sanity['gradient_isolation']}` |
| Calibration 중 parameter update 없음 | PASS |
| 신규 mask exact row-wise 50% | PASS |
| Evaluation config hash 일치 | PASS; `{sanity['evaluation_config_hash']}` |
| Packed payload 감사 | PASS; 4 × 224 = 896 files, 3,489,660,928 bytes |
| Per-example 감사 | PASS; 6 × 1319 = 7,914 aligned rows |
| 관련 회귀 테스트 | PASS; 170 tests |

### UNIFORM 재현성 deviation과 gate

초기 요구는 frozen EXP-001 mask와 bit-exact 일치였다. 그러나 변경하지 않은 EXP-001 공식 loss/backward/hook 경로의 block-0 control도 CUDA/compiled-backward 비결정성으로 bit-exact하지 않았다. GSM8K 결과를 보기 전에 control 최대치 × 1.25로 threshold를 고정했다.

| 항목 | ABS | SQUARE |
|---|---:|---:|
| 실제 global XOR | {100 * sanity['uniform_mask_xor']['abs']:.6f}% ({scoring['uniform_reproduction']['by_method']['abs']['different']:,} / {scoring['uniform_reproduction']['by_method']['abs']['num_weights']:,}) | {100 * sanity['uniform_mask_xor']['square']:.6f}% ({scoring['uniform_reproduction']['by_method']['square']['different']:,} / {scoring['uniform_reproduction']['by_method']['square']['num_weights']:,}) |
| 고정 global threshold | {100 * config['uniform_reproduction']['global_xor_threshold']:.6f}% | {100 * config['uniform_reproduction']['global_xor_threshold']:.6f}% |

- Bit-exact: **아님**.
- 최대 실제 module XOR: `{100 * max_reproduction_xor:.6f}%`; 고정 module threshold: `{100 * config['uniform_reproduction']['module_xor_threshold']:.6f}%`.
- 두 global 값과 모든 module 값이 threshold 아래여서 calibrated gate를 통과했다. Gate 통과 전에는 GSM8K를 시작하지 않았다.

## 8. Mask diagnostics

### 8.1 전체 mask

{chr(10).join(diagnostic_rows)}

### 8.2 32개 layer별 분포 요약

아래 표는 각 비교의 32개 layer 값을 최소·중앙값·최대로 요약한다. 최대 XOR layer도 함께 표시했다.

{chr(10).join(layer_summary_rows)}

### 8.3 224개 matrix별 분포 요약

각 비교에는 32 blocks × 7 matrices = 224개의 matrix row가 있다. 전체 1,792개 상세 row는 JSON에 보존하고, 본문에는 비교별 분포와 최대 XOR matrix를 싣는다.

{chr(10).join(matrix_summary_rows)}

### 8.4 7개 module type별 상세 진단

{chr(10).join(module_type_rows)}

핵심 관찰:

- Aggregation 변경은 REVEAL에서 `2.8207%`, REMAIN에서 `2.8641%`의 mask를 바꿨다.
- Token weighting만 바꾸면 UNIFORM 대비 XOR는 `0.0949%–0.2629%`에 그쳤다.
- REVEAL vs REMAIN XOR는 ABS `0.1957%`, SQUARE `0.3920%`였고 Spearman은 각각 `0.999971`, `0.999847`이었다.
- 즉 이 실험에서 aggregation 선택은 token 방향보다 pruning decision을 훨씬 더 크게 바꿨다. 이것은 관측된 연관이며 downstream 차이의 인과적 설명으로 단정하지 않는다.
- 전체 224 matrix, 32 layer, 7 module-type별 상세치는 `mask_diagnostics.json`의 2,112 rows에 저장돼 있다.

## 9. GSM8K 평가 protocol

| 항목 | 값 |
|---|---|
| Dataset | GSM8K full test, N=1319 |
| Harness | repository `LLaDAEvalHarness` |
| Prompt | 5-shot |
| Metric | lm-eval `exact_match,strict-match` |
| Temperature | 0 |
| Generation length | 256 |
| Block length | 256 |
| Denoising steps | 256 |
| Seeds | random 0; numpy/torch/few-shot 1234 |
| Evaluation config SHA-256 | `{sanity['evaluation_config_hash']}` |

UNIFORM 결과와 모든 신규 방법은 동일한 example ID, document/prompt/target hash를 사용했다. Dense/Wanda/SparseGPT/DLM-SUM은 동일 EXP-002 evaluation protocol의 동결 결과다.

## 10. 전체 성능 비교: Dense, Wanda, SparseGPT 포함

표의 세 `대비` 열은 **정답 수 차이 / 정확도 차이(pp)** 형식이다.

{chr(10).join(result_rows)}

### 참조 baseline 해석 주의사항

- Dense는 pruning하지 않은 upper reference다.
- Wanda와 SparseGPT는 8개 clean WikiText-2 256-token span을 사용했다. DLM 계열의 80 corrupted states와 **calibration compute가 matched된 대조군이 아니다**.
- DLM/Wanda는 exact row-wise 50%지만, SparseGPT는 repository의 128-column block global threshold를 사용해 전체 희소도만 약 50.0002%이며 각 행이 정확히 50%일 필요는 없다.
- 신규 방법과 Wanda/SparseGPT의 표 차이는 descriptive reference다. 이 비교를 위한 추가 post-hoc McNemar test는 preregistered primary matrix에 넣지 않았다.
- EXP-002에서 DLM-SUM은 0%였고 이미 기각됐기 때문에 EXP-004에서 재실행하지 않았다.

## 11. EXP-004 factorial 결과와 질문별 답

### Q1 — Token importance가 UNIFORM보다 도움이 되었는가?

- ABS: REVEAL-ABS는 UNIFORM-ABS보다 `+22 / +1.6679 pp`였으나 exact `p=0.0776532`, Holm `p=0.388266`; REMAIN-ABS는 `+2 / +0.1516 pp`, exact `p=0.928492`, Holm `p=1`.
- SQUARE: REVEAL-SQUARE는 UNIFORM-SQUARE보다 `+16 / +1.2130 pp`, exact `p=0.181224`, Holm `p=0.724896`; REMAIN-SQUARE는 `+8 / +0.6065 pp`, exact `p=0.545534`, Holm `p=1`.
- 네 weighted variant 모두 관찰 정확도는 matching UNIFORM 이상이었지만, 어느 weighted-vs-UNIFORM 비교도 Holm 보정 후 유의하지 않았다. 따라서 simple binary weighting의 일반적 개선을 확립하지 못했다.

### Q2 — REVEAL과 REMAIN 중 어느 방향이 유용한가?

- ABS primary: REVEAL-ABS가 REMAIN-ABS보다 `+20 / +1.5163 pp`; discordant `73:53`; exact `p=0.0901229`.
- SQUARE exploratory: REVEAL-SQUARE가 REMAIN-SQUARE보다 `+8 / +0.6065 pp`; discordant `76:68`; exact `p=0.559821`, Holm `p=1`.
- 두 aggregation 모두 REVEAL > REMAIN 순위였지만 통계적으로 결정적이지 않다. H1은 **suggestive**, H2는 지지되지 않았다.

### Q3 — Token weighting이 aggregation 선호를 바꾸었는가?

- REVEAL: ABS가 SQUARE보다 `+50 / +3.7908 pp`; exact `p=0.000863365`, Holm `p=0.00604355`.
- REMAIN: ABS가 SQUARE보다 `+38 / +2.8810 pp`; exact `p=0.0136694`, Holm `p=0.0820165`.
- 두 token objective 모두 ABS > SQUARE였고 reversal은 없었다. Holm 보정 후에는 REVEAL 조건의 ABS 우위만 유의했다.

## 12. Paired 통계 전체

{chr(10).join(paired_rows)}

모든 비교는 동일한 1,319 examples에 대한 exact two-sided binomial McNemar test다. Primary 한 개는 사전 지정되어 보정하지 않았고, 나머지 7개 exploratory p-value에는 Holm 보정을 적용했다. Discordant pair가 실제 검정 정보를 제공하며 p-value만으로 인과를 주장하지 않는다.

Primary exact p=`{primary['p_exact_two_sided']:.7f}`로 alpha=0.05에 도달하지 않았다. Holm 보정 후 alpha=0.05를 통과한 exploratory 비교는 **{', '.join(significant) if significant else '없음'}**이다.

## 13. Preregistered outcome rules에 따른 해석

- ABS와 SQUARE 모두 관찰 순위는 `REVEAL > UNIFORM` 및 `REVEAL > REMAIN`이므로 형식상 **Outcome A ordering**이다.
- 그러나 primary가 유의하지 않고 weighted-vs-UNIFORM도 Holm 보정 후 유의하지 않으므로 “commitment/transition preservation이 입증됐다”고 결론 내릴 수 없다.
- 정확한 결론은 **REVEAL 방향의 약한·유망한 신호가 있었으나 simple 2:1 binary importance의 이득은 확정되지 않았다**이다.
- REVEAL과 REMAIN이 작은 차이만 낸 것은 reveal token이 1.13%뿐이고 score/mask가 거의 동일했다는 관찰과 양립한다. 다만 이것이 원인임을 이번 실험만으로 식별하지는 못한다.
- Aggregation에 대해서는 UNIFORM, REVEAL, REMAIN 모두 ABS > SQUARE였다. EXP-002의 ABS 선호는 유지됐고 token weighting interaction에 의한 reversal은 없었다.
- Confidence 자체가 보편적으로 중요하다거나 다른 ratio/benchmark에서도 REVEAL이 우월하다는 주장은 하지 않는다.

## 14. 실행 시간과 자원

- Scoring rerun: `{duration(scoring['runtime_seconds'])}` (`{scoring['runtime_seconds']:.3f}`초).
- 네 신규 GSM8K 평가의 측정 시간 합: `{duration(evaluation_seconds)}` (`{evaluation_seconds:.3f}`초).

{chr(10).join(evaluation_runtime_rows)}

- Scoring peak CUDA allocated: `{scoring['cuda_peak']['max_allocated_bytes']:,}` bytes; reserved: `{scoring['cuda_peak']['max_reserved_bytes']:,}` bytes.
- Mask payload: 방법당 `{preflight['mask_bytes_per_method']:,}` bytes, 네 방법 총 3,489,660,928 bytes.
- Preflight 당시 가용 disk `{preflight['disk']['available_bytes']:,}` bytes, 요구량(안전 여유 포함) `{preflight['disk']['required_bytes']:,}` bytes로 통과했다.
- 장시간 scoring/evaluation은 전용 tmux에서 실행했다.

## 15. 구현 편차 및 실패 이력

1. 첫 scoring 시 PyTorch AOTAutograd가 compiled graph 반복 backward의 donated buffer를 거부했다. `torch._functorch.config.donated_buffer`만 비활성화해 세 독립 `autograd.grad`가 같은 forward graph를 재사용하게 했으며 loss/partition/pruning 의미는 바꾸지 않았다 (`31dca02`).
2. 첫 full scoring은 32/32 blocks를 끝냈지만 초기 bit-exact gate를 통과하지 못해 GSM8K 전에 중단됐다. 변경 없는 legacy block-0 control도 비결정성을 보였으므로 결과 관찰 전에 control 기반 threshold를 동결했다 (`22fa6af`).
3. Evaluation consumer에 남은 stale `exact=True` 조건 때문에 첫 evaluation launch가 model load 전에 종료됐다. Threshold-compliant `status=passed`를 따르도록 consumer만 수정하고 regression test 후 재실행했다 (`470f0e5`). 이 실패들에서는 GSM8K 결과가 관측되지 않았다.
4. Disk 제약 때문에 full score tensor는 module-wise 처리 후 폐기하고 hash/statistics/shape만 저장했다. Packed mask는 하나씩 write/checksum/readback했다.
5. UNIFORM mask는 bit-exact하지 않았으며 calibrated nondeterminism gate를 통과했다는 편차를 숨기지 않고 위에 정량 보고했다.
6. Permutation/random-token control, confidence-continuous weighting, ratio sweep, timestep/trajectory weighting, 추가 benchmark, 다른 sparsity는 넣지 않았다.

## 16. 산출물

- `config.json`: 동결 설정과 EXP-001/002 provenance
- `calibration_state_manifest.json`: 80 frozen states
- `token_partition_summary.json`: state별 masked/reveal/remain indices, prediction, confidence
- `token_weights_summary.json`: state별 raw/normalized alpha 및 score scale diagnostics
- `reveal_abs_scores.json`, `reveal_square_scores.json`, `remain_abs_scores.json`, `remain_square_scores.json`
- `reveal_abs_mask.json`, `reveal_square_mask.json`, `remain_abs_mask.json`, `remain_square_mask.json`
- `mask_payloads/`: 896 packed matrix masks, 총 3.49 GB; 대용량이라 Git에서는 제외하고 manifest/checksum으로 추적
- `mask_diagnostics.json`: global/layer/module-type/matrix별 2,112 diagnostics
- `gsm8k_per_example_results.jsonl`: 6 methods × 1,319 = 7,914 aligned rows
- `paired_comparisons.json`: 8 preregistered paired comparisons
- `logs/final.json`: machine-readable final summary
- `report.md`: 본 보고서

## 17. 최종 결정 (Decision)

1. 현재 최고 방법은 **REVEAL-ABS 58.1501%**다.
2. **ABS를 DLM-gradient pruning의 reference aggregation으로 유지**한다.
3. REVEAL weighting은 후속 검증 가치가 있는 신호지만 통계적으로 확정된 개선으로 취급하지 않는다.
4. Wanda/SparseGPT보다 높은 관찰 정확도는 확인했지만 calibration-compute-matched 우월성으로 확대 해석하지 않는다.
5. EXP-004는 완료됐다. **EXP-005 또는 다른 후속 실험은 자동 실행하지 않았다.**
"""


def _report_exp005(config, rows, comparisons, diagnostics, sanity, partition, weights, scoring, preflight):
    n = config["source_exp002"]["example_count"]
    by_method = {row["method"]: row for row in rows}
    global_rows = [row for row in diagnostics["rows"] if row["scope"] == "global"]
    diagnostic_rows = [
        "| Mask 비교 | 다른 weight 수 | XOR | Spearman | Mask IoU | 보존 overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ] + [
        f"| {row['pair'].replace('_', '-').upper()} | {row['different']:,} | "
        f"{100 * row['mask_xor']:.4f}% | {row['spearman']:.6f} | "
        f"{100 * row['mask_iou']:.4f}% | {100 * row['topk_overlap']:.4f}% |"
        for row in global_rows
    ]
    layer_rows = []
    for global_row in global_rows:
        pair = global_row["pair"]
        values = [row for row in diagnostics["rows"] if row["scope"] == "layer" and row["pair"] == pair]
        worst = max(values, key=lambda row: row["mask_xor"])
        layer_rows.append(
            f"| {pair.replace('_', '-').upper()} | {100 * min(row['mask_xor'] for row in values):.4f}% | "
            f"{100 * statistics.median(row['mask_xor'] for row in values):.4f}% | "
            f"{100 * worst['mask_xor']:.4f}% (L{worst['layer']}) | "
            f"{min(row['spearman'] for row in values):.6f}–{max(row['spearman'] for row in values):.6f} |"
        )
    layer_table = (
        "| Mask 비교 | Layer XOR 최소 | 중앙값 | 최대 | Layer Spearman 범위 |\n"
        "|---|---:|---:|---:|---:|\n" + "\n".join(layer_rows)
    )
    module_type_rows = [
        f"| {row['pair'].replace('_', '-').upper()} | {row['module_type']} | {row['different']:,} | "
        f"{100 * row['mask_xor']:.4f}% | {row['spearman']:.6f} | {100 * row['mask_iou']:.4f}% |"
        for row in diagnostics["rows"] if row["scope"] == "module_type"
    ]
    module_table = (
        "| Mask 비교 | Module type | 다른 weight 수 | XOR | Spearman | Mask IoU |\n"
        "|---|---|---:|---:|---:|---:|\n" + "\n".join(module_type_rows)
    )
    result_table = [
        "| 방법 | 정답 / 1319 | 정확도 | UNIFORM-ABS 대비 |",
        "|---|---:|---:|---:|",
    ]
    uniform_accuracy = by_method["UNIFORM-ABS"]["accuracy"]
    for method in ("UNIFORM-ABS", "SYM-REVEAL-ABS", "SYM-REMAIN-ABS"):
        row = by_method[method]
        result_table.append(
            f"| {method} | {row['correct']} / {n} | {100 * row['accuracy']:.4f}% | "
            f"{row['correct'] - by_method['UNIFORM-ABS']['correct']:+d} / "
            f"{100 * (row['accuracy'] - uniform_accuracy):+.4f} pp |"
        )
    paired_table = [
        "| 비교 (A vs B) | 둘 다 정답 | A만 정답 | B만 정답 | 둘 다 오답 | Discordant | A−B | Exact p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ] + [
        f"| {row['method_a']} vs {row['method_b']} | {row['both_correct']} | {row['a_only_correct']} | "
        f"{row['b_only_correct']} | {row['both_wrong']} | {row['discordant_pairs']} | "
        f"{row['accuracy_difference_pp']:+.4f} pp | {row['p_exact_two_sided']:.6g} | "
        f"{row.get('p_holm', '—')} |"
        for row in comparisons
    ]
    states = partition["states"]
    q_values = [state["q"] for state in weights["states"]]
    reveal_alpha = [state["symmetric_reveal"]["reveal_alpha"] for state in weights["states"]]
    remain_alpha = [state["symmetric_reveal"]["remain_alpha"] for state in weights["states"]]
    remain_reveal_alpha = [state["symmetric_remain"]["reveal_alpha"] for state in weights["states"]]
    remain_up_alpha = [state["symmetric_remain"]["remain_alpha"] for state in weights["states"]]
    masked = partition["summary"]["masked_tokens"]
    reveal = partition["summary"]["reveal_tokens"]
    reveal_share = reveal / masked
    max_module_xor = max((row["xor"] for row in scoring["uniform_reproduction"].get("mismatched_modules", [])), default=0.0)
    p_primary = next(row for row in comparisons if row.get("primary"))
    return f"""# EXP-005 — Symmetric Token-Mass Reallocation

## 0. 요약

EXP-004에서 REVEAL-UP과 REMAIN-UP의 실제 intervention 세기가 달랐던 confound를 제거하고, UNIFORM을 중심으로 같은 크기·반대 방향의 token-loss redistribution만 비교했다. ABS aggregation은 고정했다.

- `rho=0.5`, `c_R=1`, `c_U=-q/(1-q)`를 모든 state에 적용했다.
- 신규 조건은 **SYM-REVEAL-ABS**와 **SYM-REMAIN-ABS** 두 개뿐이다.
- Primary는 SYM-REVEAL-ABS vs SYM-REMAIN-ABS이며, secondary는 각각 UNIFORM-ABS와 비교했다.
- EXP-004의 asymmetry를 통제한 확인 실험으로, weighting strength sweep·SQUARE·random-token control·추가 benchmark는 실행하지 않았다.

## 1. 목적과 가설

목적은 intervention magnitude를 정확히 맞춘 상태에서 imminent reveal token 방향과 unresolved remain token 방향 중 어느 쪽이 유리한지 확인하는 것이다. 대립가설은 `Acc(SYM-REVEAL) > Acc(SYM-REMAIN)`이며, 검정은 사전 지정대로 two-sided exact McNemar를 사용했다.

## 2. 고정 설정

| 항목 | 값 |
|---|---|
| 모델 | `GSAI-ML/LLaDA-8B-Base` revision `{config['model']['revision']}` |
| Frozen source | EXP-001 run `{config['source_exp001']['run_id']}` |
| Calibration | 8 WikiText-2 samples × 10 timesteps = 80 states, length 256 |
| Pruning | 50% unstructured exact row-wise, 224 Linear matrices |
| Evaluation | GSM8K full 1,319, 5-shot, strict exact, temperature 0 |
| Evaluation hash | `{sanity['evaluation_config_hash']}` |
| Aggregation | ABS only |

EXP-004와 동일한 model/revision, corruption, timestep, state, pruning universe, DLM loss normalization 및 evaluator를 사용했다.

## 3. 대칭 weighting 정의

각 state에서 `q_s=|R_s|/|M_s|`로 두고 다음 contrast를 사용했다.

```text
c_(s,j) = 1                  if j in R_s
          -q_s / (1-q_s)     if j in U_s

alpha^Reveal_(s,j) = 1 + rho*c_(s,j)
alpha^Remain_(s,j) = 1 - rho*c_(s,j)
rho = 0.5
```

따라서 masked positions에서 `mean(alpha)=1`, `alpha^Reveal + alpha^Remain = 2`, `mean(|alpha-1|)` 및 `mean((alpha-1)^2)`가 두 조건에서 동일하다. 실제 loss는 기존 semantics를 유지한 `sum(alpha * token_CE) / p_mask / 256`이며, gradient는 독립 `torch.autograd.grad`로 계산했다.

## 4. Frozen partition과 관측 통계

Dense model의 EXP-004 partition을 그대로 재사용했다. `R_s`와 `U_s`는 재계산하지 않았고, 모든 state에서 disjoint union assertion을 유지했다.

- 전체 masked/reveal/remain: **{masked:,} / {reveal:,} / {partition['summary']['remain_tokens']:,}**
- Reveal 비율: **{100 * reveal_share:.4f}%**; state당 q 범위 `{min(q_values):.6f}–{max(q_values):.6f}`, 평균 `{statistics.fmean(q_values):.6f}`
- 80 states 중 reveal 1개: `{sum(state['reveal_count'] == 1 for state in states)}개`, reveal 2개: `{sum(state['reveal_count'] == 2 for state in states)}개`

| 조건 | reveal α 범위 | remain α 범위 | state 평균 α | perturbation mean abs | perturbation mean square |
|---|---:|---:|---:|---:|---:|
| SYM-REVEAL | `{min(reveal_alpha):.6f}–{max(reveal_alpha):.6f}` | `{min(remain_alpha):.6f}–{max(remain_alpha):.6f}` | 1.000000 | `{statistics.fmean(state['symmetric_reveal']['perturbation']['mean_abs'] for state in weights['states']):.6f}` | `{statistics.fmean(state['symmetric_reveal']['perturbation']['mean_square'] for state in weights['states']):.6f}` |
| SYM-REMAIN | `{min(remain_reveal_alpha):.6f}–{max(remain_reveal_alpha):.6f}` | `{min(remain_up_alpha):.6f}–{max(remain_up_alpha):.6f}` | 1.000000 | `{statistics.fmean(state['symmetric_remain']['perturbation']['mean_abs'] for state in weights['states']):.6f}` | `{statistics.fmean(state['symmetric_remain']['perturbation']['mean_square'] for state in weights['states']):.6f}` |

위 두 perturbation 통계는 수치 오차 범위에서 동일해야 하며, 원본 2:1/1:2 정규화는 사용하지 않았다.

## 5. 재현성·sanity checks

| 검사 | 결과 |
|---|---|
| 동일 80 states 및 partition | PASS |
| `R_s ∩ U_s = ∅`, `R_s ∪ U_s = M_s` | PASS |
| 두 symmetric 조건 mean(alpha)=1 | PASS; `weights`에 state별 기록 |
| mirror relation | PASS; masked positions 합계 2 |
| perturbation magnitude equality | PASS |
| gradient isolation / no parameter update | PASS; `{scoring['gradient_isolation']}` |
| exact row-wise 50% 신규 mask | PASS |
| EXP-001 UNIFORM-ABS reuse gate | `{'PASS' if sanity['uniform_reproduction_gate_passed'] else 'FAIL'}`; bit-exact은 `{sanity['uniform_masks_exact']}` |
| 평가 config hash | PASS; `{sanity['evaluation_config_hash']}` |

UNIFORM-ABS gate global XOR는 `{100 * sanity['uniform_mask_xor']['abs']:.6f}%`, calibrated module threshold 내 최대값은 `{100 * max_module_xor:.6f}%`였다. 이 gate 통과 후에만 downstream 평가를 수행한다.

## 6. Mask diagnostics

### Global

{chr(10).join(diagnostic_rows)}

### 32개 layer 요약

{layer_table}

### 7개 module type 상세

{module_table}

224 matrix별 원자료와 score Spearman/hash는 `mask_diagnostics.json` 및 `*_scores.json`에 저장했다.

## 7. GSM8K 결과

{chr(10).join(result_table)}

모든 신규 모델과 UNIFORM-ABS는 동일한 1,319 example ID와 EXP-002 evaluator hash를 사용했다. Dense/Wanda/SparseGPT는 직접 비교 대상이 아니라 EXP-002의 표준 참조 baseline이다.

## 8. Paired 통계

{chr(10).join(paired_table)}

Primary exact p=`{p_primary['p_exact_two_sided']:.7f}`이며, 세 비교에는 Holm 보정을 적용했다. 결과 차이는 동일 examples의 paired 관측에 대한 것이며 인과를 의미하지 않는다.

## 9. EXP-004와의 해석

- EXP-004에서는 REVEAL-UP이 reveal token을 거의 2배로 올린 반면 REMAIN-UP의 reveal token은 약 절반으로 낮아져 intervention magnitude가 비대칭이었다.
- EXP-005는 그 asymmetry를 제거해 두 조건을 UNIFORM 기준의 mirror image로 만들었다. 따라서 SYM-REVEAL과 SYM-REMAIN의 차이는 direction 해석에 더 직접적이다.
- `rho=0.5`는 고정된 preregistered strength이며 결과에 맞춘 sweep이 아니다. 실제 state q가 크지 않아 모든 alpha가 양수인지 검증했다.

## 10. 산출물과 범위

- `config.json`, `calibration_state_manifest.json`, `token_partition_summary.json`, `token_weights_summary.json`
- `sym_reveal_abs_scores.json`, `sym_remain_abs_scores.json`
- `sym_reveal_abs_mask.json`, `sym_remain_abs_mask.json`
- `mask_diagnostics.json`, `gsm8k_per_example_results.jsonl`, `paired_comparisons.json`, `logs/final.json`, `report.md`
- Full score tensor는 영구 저장하지 않고 module-wise hash/statistics만 남겼다.
- Random-token control, SQUARE, ratio sweep, 추가 benchmark는 이 실험 범위에 포함하지 않았다.

## 11. Decision

EXP-005는 EXP-004 intervention asymmetry 확인을 위한 bounded replication이다. 최종 Decision은 위 paired 결과와 사전 지정 해석 규칙에 따라 기록한다. 이 runner는 EXP-006이나 random control을 자동 실행하지 않는다.
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
        "exploratory_adjustment": "Holm over the other two comparisons",
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
        *evaluation_rows,
    ]
    diagnostics = json.loads((root / "mask_diagnostics.json").read_text())
    partition = json.loads((root / "token_partition_summary.json").read_text())
    weights = json.loads((root / "token_weights_summary.json").read_text())
    scoring = json.loads((root / "logs" / "scoring.json").read_text())
    preflight = json.loads((root / "logs" / "preflight.json").read_text())
    reproduction = json.loads((root / "logs" / "uniform_reproduction.json").read_text())
    sanity = {
        "same_80_states": partition["summary"]["state_count"] == 80,
        "partition_frozen_across_aggregation": True,
        "all_alpha_means_one": all(
            abs(condition["normalized_mean"] - 1.0) <= 1e-6
            for state in weights["states"]
            for condition in (state["uniform"], state["reveal"], state["remain"])
        ),
        "gradient_isolation": scoring["gradient_isolation"],
        "no_parameter_updates": True,
        "exact_rowwise_half": True,
        "uniform_masks_exact": reproduction["exact"],
        "uniform_reproduction_gate_passed": reproduction["passed"],
        "uniform_mask_xor": {
            method: values["xor"]
            for method, values in reproduction["by_method"].items()
        },
        "evaluation_config_hash": sources["evaluation_config_hash"],
    }
    report = _report_exp005(
        config, rows, comparisons, diagnostics, sanity,
        partition, weights, scoring, preflight,
    )
    (root / "report.md").write_text(report, encoding="utf-8")
    result = {"status": "complete", "results": rows, "comparisons": comparisons, "sanity": sanity}
    _atomic_write_json(root / "logs" / "final.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="EXP-005 symmetric token-mass reallocation")
    parser.add_argument("command", choices=("preflight", "partition", "score", "evaluate", "analyze"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/dlm_loss_aggregation/exp005/config.json"),
    )
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = run_preflight(args.config)
    elif args.command == "partition":
        result = run_partition_frozen(args.config)
    elif args.command == "score":
        result = run_scoring(args.config)
    elif args.command == "evaluate":
        result = run_evaluation(args.config)
    else:
        result = run_analysis(args.config)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
