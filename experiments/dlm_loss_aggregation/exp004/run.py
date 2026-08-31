import math
import shutil

import torch
import torch.nn.functional as F

from generate import get_num_transfer_tokens
from experiments.dlm_loss_aggregation.core import _average_ranks, _rank_correlation


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


@torch.no_grad()
def freeze_token_partition(logits, noisy_ids, mask_id, p_mask, denoising_steps=256):
    mask = noisy_ids.eq(mask_id)
    if logits.shape[:-1] != noisy_ids.shape or not mask.any().item():
        raise ValueError("logits/noisy state is invalid")
    steps_remaining = max(1, math.ceil(p_mask * denoising_steps))
    reveal_count = int(get_num_transfer_tokens(mask, steps_remaining)[0, 0].item())
    prediction = logits.argmax(dim=-1)
    confidence = F.softmax(logits.float(), dim=-1).gather(
        -1, prediction.unsqueeze(-1)
    ).squeeze(-1)
    masked_indices = mask[0].nonzero().flatten()
    selected = torch.topk(confidence[0, masked_indices], k=reveal_count).indices
    reveal_indices = masked_indices[selected].sort().values
    remain_indices = masked_indices[
        ~torch.isin(masked_indices, reveal_indices)
    ]
    result = {
        "steps_remaining": steps_remaining,
        "masked_indices": masked_indices.tolist(),
        "reveal_indices": reveal_indices.tolist(),
        "remain_indices": remain_indices.tolist(),
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


def weighted_dlm_loss(logits, clean_ids, mask, p_mask, alpha):
    if logits.shape[:-1] != clean_ids.shape or mask.shape != clean_ids.shape:
        raise ValueError("logits, tokens, and mask shapes differ")
    if alpha.shape != mask.shape or not mask.any().item():
        raise ValueError("alpha shape is invalid or the state has no masks")
    token_loss = F.cross_entropy(
        logits[mask].float(), clean_ids[mask], reduction="none"
    )
    return (token_loss * alpha[mask]).sum() / p_mask / clean_ids.numel()


def independent_weight_effects(losses, parameters):
    conditions = ("uniform", "reveal", "remain")
    if tuple(losses) != conditions or not parameters:
        raise ValueError("losses must be ordered uniform, reveal, remain")
    values = tuple(parameters.values())
    if any(parameter.grad is not None for parameter in values):
        raise RuntimeError("parameter gradient leaked into condition scoring")
    result = {}
    for index, condition in enumerate(conditions):
        gradients = torch.autograd.grad(
            losses[condition], values, retain_graph=index + 1 < len(conditions)
        )
        result[condition] = {
            name: (-parameter.detach() * gradient.detach())
            .to(device="cpu", dtype=torch.float32)
            .clone()
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
