import base64
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


_MASK_ARTIFACT_VERSION = 1
_MASK_BITORDER = "big"


class NegativeSuffixCostError(ValueError):
    pass


class _DiagnosticFloat(float):
    def __new__(cls, value: float, sample_indices: torch.Tensor, reason: str | None = None):
        result = float.__new__(cls, value)
        result.sample_indices = sample_indices
        result.reason = reason
        return result


def midpoint_timesteps(count: int = 10) -> tuple[float, ...]:
    if not isinstance(count, int) or count <= 0:
        raise ValueError("count must be positive")
    return tuple((index + 0.5) / count for index in range(count))


def mask_probability(timestep: float, eps: float = 1e-3) -> float:
    if not math.isfinite(timestep) or not 0 <= timestep <= 1:
        raise ValueError("timestep must be between 0 and 1")
    if not math.isfinite(eps) or not 0 <= eps < 1:
        raise ValueError("eps must be between 0 and 1")
    return (1 - eps) * timestep + eps


def make_masked_state(
    clean_ids: torch.Tensor, timestep: float, mask_id: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if not isinstance(clean_ids, torch.Tensor) or clean_ids.numel() == 0:
        raise ValueError("clean_ids must have a positive sequence length")
    p_mask = mask_probability(timestep)
    generator = torch.Generator(device=clean_ids.device)
    attempt = 0
    while True:
        generator.manual_seed(seed + attempt)
        mask = torch.rand(
            clean_ids.shape, device=clean_ids.device, generator=generator
        ).lt(p_mask)
        if mask.any():
            masked_ids = clean_ids.clone()
            masked_ids[mask] = mask_id
            return masked_ids, mask, p_mask
        attempt += 1


def official_dlm_loss(
    logits: torch.Tensor,
    clean_ids: torch.Tensor,
    mask: torch.Tensor,
    p_mask: float,
) -> torch.Tensor:
    if clean_ids.numel() == 0:
        raise ValueError("clean_ids must have a positive sequence length")
    if logits.ndim == 0 or tuple(logits.shape[:-1]) != tuple(clean_ids.shape):
        raise ValueError("logits and clean_ids shapes do not match")
    if mask.shape != clean_ids.shape or mask.dtype != torch.bool:
        raise ValueError("mask shape and dtype do not match clean_ids")
    if not math.isfinite(p_mask) or not 0 < p_mask <= 1:
        raise ValueError("p_mask must be in (0, 1]")
    if not mask.any():
        raise ValueError("official DLM loss requires at least one masked token")
    token_loss = F.cross_entropy(logits[mask].float(), clean_ids[mask], reduction="sum")
    return token_loss / p_mask / clean_ids.numel()


def _llada_core(model):
    core = getattr(model, "model", model)
    if not hasattr(core, "transformer"):
        raise TypeError("model must contain a LLaDA transformer")
    if core.config.block_group_size != 1:
        raise ValueError("block suffix scoring requires ungrouped transformer blocks")
    return core


def _module_device_dtype(module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters())
    return parameter.device, parameter.dtype


def embed_state(model, noisy_ids: torch.Tensor) -> torch.Tensor:
    core = _llada_core(model)
    transformer = core.transformer
    device = transformer.wte.weight.device
    noisy_ids = noisy_ids.to(device=device, dtype=torch.long)
    hidden = transformer.wte(noisy_ids)
    if core.config.input_emb_norm:
        hidden = hidden * math.sqrt(core.config.d_model)
    if not (core.config.alibi or core.config.rope):
        positions = torch.arange(
            hidden.shape[1], dtype=torch.long, device=hidden.device
        ).unsqueeze(0)
        hidden = hidden + transformer.wpe(positions)
    return transformer.emb_drop(hidden)


def suffix_logits(model, hidden: torch.Tensor, start_block: int) -> torch.Tensor:
    core = _llada_core(model)
    blocks = core.transformer.blocks
    if isinstance(start_block, bool) or not isinstance(start_block, int) or not 0 <= start_block <= len(blocks):
        raise ValueError("start_block is outside the transformer")
    for block in blocks[start_block:]:
        device, dtype = _module_device_dtype(block)
        hidden = hidden.to(device=device, dtype=dtype)
        hidden, _ = block(
            hidden,
            attention_bias=None,
            layer_past=None,
            use_cache=False,
            replace_position=None,
            attn_collector=None,
        )
    norm_device, norm_dtype = _module_device_dtype(core.transformer.ln_f)
    hidden = core.transformer.ln_f(hidden.to(device=norm_device, dtype=norm_dtype))
    if core.config.weight_tying:
        output_weight = core.transformer.wte.weight
        hidden = hidden.to(device=output_weight.device, dtype=output_weight.dtype)
        logits = F.linear(hidden, output_weight, None)
    else:
        output_device, output_dtype = _module_device_dtype(core.transformer.ff_out)
        logits = core.transformer.ff_out(hidden.to(device=output_device, dtype=output_dtype))
    if core.config.scale_logits:
        logits.mul_(1 / math.sqrt(core.config.d_model))
    return logits


def block_state_gradients(
    model,
    block_index: int,
    hidden: torch.Tensor,
    clean_ids: torch.Tensor,
    mask: torch.Tensor,
    p_mask: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    from .prune_llada import find_layers

    core = _llada_core(model)
    blocks = core.transformer.blocks
    if isinstance(block_index, bool) or not isinstance(block_index, int) or not 0 <= block_index < len(blocks):
        raise ValueError("block_index is outside the transformer")
    target_block = blocks[block_index]
    layers = find_layers(target_block)
    if len(layers) != 7:
        raise ValueError(f"expected seven block linear weights, found {len(layers)}")
    parameters = list(model.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    model.zero_grad(set_to_none=True)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        for layer in layers.values():
            layer.weight.requires_grad_(True)

        device, dtype = _module_device_dtype(target_block)
        target_output, _ = target_block(
            hidden.detach().to(device=device, dtype=dtype),
            attention_bias=None,
            layer_past=None,
            use_cache=False,
            replace_position=None,
            attn_collector=None,
        )
        logits = suffix_logits(model, target_output, block_index + 1)
        loss = official_dlm_loss(
            logits,
            clean_ids.to(device=logits.device, dtype=torch.long),
            mask.to(device=logits.device, dtype=torch.bool),
            p_mask,
        )
        loss.backward()
        gradients = {}
        for name, layer in layers.items():
            if layer.weight.grad is None:
                raise RuntimeError(f"missing gradient for block weight: {name}")
            gradients[name] = layer.weight.grad.detach().to(device="cpu", dtype=torch.float32)
        return gradients, target_output.detach()
    finally:
        model.zero_grad(set_to_none=True)
        for parameter, requires_grad in zip(parameters, original_flags):
            parameter.requires_grad_(requires_grad)


def project_scoring_seconds(
    block_31_seconds: float,
    block_0_seconds: float,
    blocks: int = 32,
    states: int = 80,
) -> dict:
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 2:
        raise ValueError("blocks must be at least two")
    if isinstance(states, bool) or not isinstance(states, int) or states <= 0:
        raise ValueError("states must be positive")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        for value in (block_31_seconds, block_0_seconds)
    ):
        raise ValueError("block seconds must be finite and nonnegative")
    fixed = float(block_31_seconds)
    suffix = (float(block_0_seconds) - fixed) / (blocks - 1)
    if suffix < 0:
        raise NegativeSuffixCostError("fitted suffix cost must be nonnegative")
    per_block = [fixed + suffix * (blocks - 1 - block) for block in range(blocks)]
    return {
        "fixed_seconds": fixed,
        "suffix_seconds": suffix,
        "per_block_seconds": per_block,
        "projected_scoring_seconds": states * sum(per_block),
    }


def feasibility_gate(
    measurements: list[dict],
    projected_seconds: float,
    max_gpu_gib: float = 30,
    max_gpu_hours: float = 24,
) -> dict:
    if not measurements:
        raise ValueError("measurements must not be empty")
    memory_limit = max_gpu_gib * 1024**3
    measurement_checks = []
    for measurement in measurements:
        measurement_checks.append(
            {
                "block": measurement.get("block"),
                "checks": {
                    "allocated_memory": measurement["max_memory_allocated_bytes"] <= memory_limit,
                    "reserved_memory": measurement["max_memory_reserved_bytes"] <= memory_limit,
                    "swap": measurement["vm_swap_delta_kib"] <= 0,
                    "finite_gradients": measurement["gradient_element_count"] > 0
                    and measurement["finite_gradient_count"] == measurement["gradient_element_count"],
                    "nonzero_gradients": measurement["nonzero_gradient_count"] > 0,
                },
            }
        )
    checks = {
        "gpu_memory": all(
            item["checks"]["allocated_memory"] and item["checks"]["reserved_memory"]
            for item in measurement_checks
        ),
        "swap": all(item["checks"]["swap"] for item in measurement_checks),
        "gradients": all(
            item["checks"]["finite_gradients"] and item["checks"]["nonzero_gradients"]
            for item in measurement_checks
        ),
        "runtime": math.isfinite(projected_seconds)
        and projected_seconds <= max_gpu_hours * 60 * 60,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "measurement_checks": measurement_checks,
        "limits": {
            "max_gpu_gib": max_gpu_gib,
            "max_gpu_hours": max_gpu_hours,
            "max_swap_delta_kib": 0,
        },
    }


def rowwise_prune_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    if not isinstance(score, torch.Tensor) or score.ndim != 2:
        raise ValueError("score must be a two-dimensional tensor")
    if not torch.isfinite(score).all().item():
        raise ValueError("score must be finite")
    if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)):
        raise ValueError("sparsity must be between 0 and 1")
    if not math.isfinite(sparsity) or not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between 0 and 1")
    prune_count = math.floor(score.shape[1] * sparsity)
    mask = torch.zeros_like(score, dtype=torch.bool)
    if prune_count:
        mask.scatter_(1, torch.argsort(score, dim=1, stable=True)[:, :prune_count], True)
    return mask


def _mask_header(artifact: dict) -> dict:
    required = {"version", "shape", "dtype", "bitorder", "prune_count", "per_row_prune_count", "byte_length"}
    if not isinstance(artifact, dict) or set(artifact) != required | {"bits", "sha256"}:
        raise ValueError("invalid mask artifact")
    shape = artifact["shape"]
    if (
        isinstance(artifact["version"], bool)
        or not isinstance(artifact["version"], int)
        or artifact["version"] != _MASK_ARTIFACT_VERSION
        or artifact["dtype"] != "bool"
        or artifact["bitorder"] != _MASK_BITORDER
        or not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise ValueError("invalid mask artifact header")
    if (
        not isinstance(artifact["prune_count"], int)
        or artifact["prune_count"] < 0
        or not isinstance(artifact["per_row_prune_count"], list)
        or not isinstance(artifact["byte_length"], int)
    ):
        raise ValueError("invalid mask artifact header")
    return {key: artifact[key] for key in required}


def _mask_checksum(header: dict, bits: bytes) -> str:
    encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded_header + bits).hexdigest()


def pack_mask(mask: torch.Tensor) -> dict:
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a two-dimensional bool tensor")
    packed = np.packbits(mask.detach().cpu().numpy().reshape(-1), bitorder=_MASK_BITORDER).tobytes()
    header = {
        "version": _MASK_ARTIFACT_VERSION,
        "shape": list(mask.shape),
        "dtype": "bool",
        "bitorder": _MASK_BITORDER,
        "prune_count": int(mask.sum().item()),
        "per_row_prune_count": mask.sum(dim=1).cpu().tolist(),
        "byte_length": len(packed),
    }
    return {**header, "bits": packed, "sha256": _mask_checksum(header, packed)}


def unpack_mask(artifact: dict) -> torch.Tensor:
    header = _mask_header(artifact)
    bits = artifact["bits"]
    if not isinstance(bits, bytes) or len(bits) != header["byte_length"]:
        raise ValueError("invalid mask bit length")
    if len(bits) != math.ceil(math.prod(header["shape"]) / 8):
        raise ValueError("invalid mask bit length")
    if not isinstance(artifact["sha256"], str) or _mask_checksum(header, bits) != artifact["sha256"]:
        raise ValueError("mask checksum mismatch")
    if (
        len(header["per_row_prune_count"]) != header["shape"][0]
        or any(
            not isinstance(count, int) or count < 0 or count > header["shape"][1]
            for count in header["per_row_prune_count"]
        )
    ):
        raise ValueError("invalid mask prune counts")
    values = np.unpackbits(np.frombuffer(bits, dtype=np.uint8), bitorder=_MASK_BITORDER)
    bit_count = math.prod(header["shape"])
    if values.size != math.ceil(bit_count / 8) * 8 or values[bit_count:].any():
        raise ValueError("invalid mask trailing bits")
    mask = torch.from_numpy(values[:bit_count].reshape(header["shape"]).astype(np.bool_, copy=True))
    if int(mask.sum().item()) != header["prune_count"] or mask.sum(dim=1).tolist() != header["per_row_prune_count"]:
        raise ValueError("invalid mask prune counts")
    return mask


def _json_mask_artifact(artifact: dict) -> dict:
    return {**artifact, "bits": base64.b64encode(artifact["bits"]).decode("ascii")}


def _from_json_mask_artifact(artifact: dict) -> dict:
    if not isinstance(artifact, dict) or not isinstance(artifact.get("bits"), str):
        raise ValueError("invalid mask artifact")
    try:
        return {**artifact, "bits": base64.b64decode(artifact["bits"], validate=True)}
    except ValueError as error:
        raise ValueError("invalid mask artifact") from error


def save_mask_block(path: Path, masks: dict, metadata: dict) -> dict:
    if not isinstance(masks, dict) or not isinstance(metadata, dict):
        raise ValueError("masks and metadata must be dictionaries")
    document = {
        "version": _MASK_ARTIFACT_VERSION,
        "metadata": metadata,
        "masks": {name: _json_mask_artifact(pack_mask(mask)) for name, mask in masks.items()},
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    expected = {"sha256": hashlib.sha256(encoded).hexdigest(), "byte_length": len(encoded)}
    destination = Path(path)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(destination)
        temporary_name = None
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return expected


def load_mask_block(path: Path) -> tuple[dict, dict]:
    try:
        with Path(path).open("rb") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid mask block") from error
    if (
        not isinstance(document, dict)
        or isinstance(document.get("version"), bool)
        or not isinstance(document.get("version"), int)
        or document["version"] != _MASK_ARTIFACT_VERSION
        or not isinstance(document.get("metadata"), dict)
        or not isinstance(document.get("masks"), dict)
    ):
        raise ValueError("invalid mask block")
    return (
        {name: unpack_mask(_from_json_mask_artifact(artifact)) for name, artifact in document["masks"].items()},
        document["metadata"],
    )


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    sorted_values, order = torch.sort(values)
    starts = torch.cat(
        (torch.zeros(1, dtype=torch.long), torch.nonzero(sorted_values[1:] != sorted_values[:-1]).flatten() + 1)
    )
    ends = torch.cat((starts[1:], torch.tensor([values.numel()], dtype=torch.long)))
    average_ranks = (starts + ends + 1).to(sorted_values.dtype) / 2
    sorted_ranks = torch.repeat_interleave(average_ranks, ends - starts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def sampled_spearman(left: torch.Tensor, right: torch.Tensor, sample_size: int, seed: int) -> float:
    if (
        not isinstance(left, torch.Tensor)
        or not isinstance(right, torch.Tensor)
        or left.shape != right.shape
        or not left.numel()
        or not torch.isfinite(left).all().item()
        or not torch.isfinite(right).all().item()
    ):
        raise ValueError("inputs must be finite tensors with matching nonempty shapes")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    count = min(sample_size, left.numel())
    indices = (
        torch.arange(left.numel())
        if count == left.numel()
        else torch.randperm(left.numel(), generator=torch.Generator().manual_seed(seed))[:count]
    )
    left_ranks = _average_ranks(left.detach().cpu().reshape(-1).double()[indices])
    right_ranks = _average_ranks(right.detach().cpu().reshape(-1).double()[indices])
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = torch.sqrt(left_centered.square().sum() * right_centered.square().sum())
    if denominator.item() == 0:
        return _DiagnosticFloat(float("nan"), indices, "constant vector")
    return _DiagnosticFloat(float((left_centered * right_centered).sum() / denominator), indices)


def jaccard(left: torch.Tensor, right: torch.Tensor) -> float:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.shape != right.shape:
        raise ValueError("masks must have matching shapes")
    if left.dtype != torch.bool or right.dtype != torch.bool:
        raise ValueError("masks must be bool tensors")
    union = torch.logical_or(left, right).sum().item()
    return 1.0 if union == 0 else torch.logical_and(left, right).sum().item() / union


def row_change_fraction(left: torch.Tensor, right: torch.Tensor) -> float:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.shape != right.shape:
        raise ValueError("masks must have matching shapes")
    if left.ndim != 2:
        raise ValueError("masks must be two-dimensional")
    if not left.numel():
        raise ValueError("masks must be nonempty")
    if left.dtype != torch.bool or right.dtype != torch.bool:
        raise ValueError("masks must be bool tensors")
    return left.ne(right).any(dim=1).float().mean().item()


def top_fraction_overlap(left: torch.Tensor, right: torch.Tensor, fraction: float = 0.01) -> float:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or left.shape != right.shape:
        raise ValueError("scores must have matching shapes")
    if not left.numel():
        raise ValueError("scores must be nonempty")
    if not torch.isfinite(left).all().item() or not torch.isfinite(right).all().item():
        raise ValueError("scores must be finite")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, math.ceil(left.numel() * fraction))

    def top_mask(score: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
        mask[torch.argsort(score.reshape(-1), descending=True, stable=True)[:count]] = True
        return mask

    return jaccard(top_mask(left), top_mask(right))


class TimestepSensitivityAccumulator:
    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        split_size: int = 4,
        expected_timesteps: int = 10,
    ) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        if not isinstance(split_size, int) or split_size <= 0:
            raise ValueError("split_size must be positive")
        if not isinstance(expected_timesteps, int) or expected_timesteps <= 0:
            raise ValueError("expected_timesteps must be positive")
        self.split_size = split_size
        self.expected_timesteps = expected_timesteps
        self._weights = {
            name: weight.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, weight in weights.items()
        }
        if any(not torch.isfinite(weight).all().item() for weight in self._weights.values()):
            raise ValueError("weights must be finite")
        self._weight_sq = {name: weight.square() for name, weight in self._weights.items()}
        self._sum_sq = {
            split: {name: torch.zeros_like(weight) for name, weight in self._weights.items()}
            for split in ("a", "b")
        }
        self._count = {"a": 0, "b": 0}
        self._mean = {
            group: {name: torch.zeros_like(weight) for name, weight in self._weights.items()}
            for group in ("full", "a", "b")
        }
        self._m2 = {
            group: {name: torch.zeros_like(weight) for name, weight in self._weights.items()}
            for group in ("full", "a", "b")
        }
        self._update_count = 0

    def add_state(self, grads: dict[str, torch.Tensor], split: str) -> None:
        if split not in self._count:
            raise ValueError(f"unknown split: {split}")
        if self._count[split] >= self.split_size:
            raise ValueError(f"split {split} already has split_size states")
        if set(grads) != set(self._weights):
            raise ValueError("gradient keys do not match weights")
        cpu_grads = {}
        for name, grad in grads.items():
            if not isinstance(grad, torch.Tensor) or grad.shape != self._weights[name].shape:
                raise ValueError(f"gradient shape does not match weight: {name}")
            if not torch.isfinite(grad).all().item():
                raise ValueError("gradients must be finite")
            cpu_grads[name] = grad.detach().to(device="cpu", dtype=torch.float32).square()
        for name, grad_sq in cpu_grads.items():
            self._sum_sq[split][name].add_(grad_sq)
        self._count[split] += 1

    def finish_timestep(self) -> None:
        if self._update_count >= self.expected_timesteps:
            raise ValueError("expected timestep updates already complete")
        if any(count != self.split_size for count in self._count.values()):
            raise ValueError("cannot finish an incomplete timestep")
        f_a = {
            name: self._weight_sq[name] * self._sum_sq["a"][name] / self.split_size
            for name in self._weights
        }
        f_b = {
            name: self._weight_sq[name] * self._sum_sq["b"][name] / self.split_size
            for name in self._weights
        }
        values = {
            "a": f_a,
            "b": f_b,
            "full": {name: (f_a[name] + f_b[name]) / 2 for name in self._weights},
        }
        next_count = self._update_count + 1
        for group, group_values in values.items():
            for name, value in group_values.items():
                delta = value - self._mean[group][name]
                self._mean[group][name].add_(delta / next_count)
                self._m2[group][name].add_(delta * (value - self._mean[group][name]))
        for split in self._count:
            self._count[split] = 0
            for value in self._sum_sq[split].values():
                value.zero_()
        self._update_count = next_count

    def finalize(self) -> dict:
        if any(self._count.values()):
            raise ValueError("cannot finalize an incomplete timestep")
        if self._update_count != self.expected_timesteps:
            raise ValueError(
                f"expected {self.expected_timesteps} timestep updates, got {self._update_count}"
            )
        result = {"update_count": self._update_count}
        for group in ("full", "a", "b"):
            result[group] = {
                "mu": {name: value.clone() for name, value in self._mean[group].items()},
                "sigma": {
                    name: torch.sqrt(value / self._update_count)
                    for name, value in self._m2[group].items()
                },
            }
        return result


def _rho_diagnostic(value: _DiagnosticFloat) -> dict:
    return {
        "value": None if math.isnan(value) else float(value),
        "reason": value.reason,
        "sample_size": value.sample_indices.numel(),
        "sample_indices_sha256": hashlib.sha256(
            value.sample_indices.numpy().tobytes()
        ).hexdigest(),
    }


def score_block(model, block_index: int, cached_states: dict, config: dict) -> tuple[dict, dict]:
    from .prune_llada import find_layers

    calibration = config["calibration"]
    split_a = calibration["split_a"]
    split_b = calibration["split_b"]
    sequence_indices = calibration["sequence_indices"]
    timesteps = calibration["timesteps"]
    if (
        not isinstance(cached_states, dict)
        or not timesteps
        or len(split_a) != len(split_b)
        or set(split_a).intersection(split_b)
        or set(split_a + split_b) != set(sequence_indices)
    ):
        raise ValueError("cached states or calibration split is invalid")
    expected_keys = {
        (timestep_index, sequence_index)
        for timestep_index in range(len(timesteps))
        for sequence_index in sequence_indices
    }
    if set(cached_states) != expected_keys:
        raise ValueError("cached states do not match calibration states")

    block = _llada_core(model).transformer.blocks[block_index]
    layers = find_layers(block)
    if len(layers) != 7:
        raise ValueError(f"expected seven block linear weights, found {len(layers)}")
    accumulator = TimestepSensitivityAccumulator(
        {name: layer.weight for name, layer in layers.items()},
        split_size=len(split_a),
        expected_timesteps=len(timesteps),
    )
    next_states = {}
    _, block_dtype = _module_device_dtype(block)
    for timestep_index in range(len(timesteps)):
        for split, members in (("a", split_a), ("b", split_b)):
            for sequence_index in members:
                key = (timestep_index, sequence_index)
                state = cached_states[key]
                if (
                    state.get("timestep_index") != timestep_index
                    or state.get("sequence_index") != sequence_index
                ):
                    raise ValueError("cached state metadata does not match its key")
                gradients, target_output = block_state_gradients(
                    model,
                    block_index,
                    state["hidden"],
                    state["clean_ids"],
                    state["mask"],
                    state["p_mask"],
                )
                accumulator.add_state(gradients, split)
                next_states[key] = {
                    **state,
                    "hidden": target_output.to(device="cpu", dtype=block_dtype),
                }
        accumulator.finish_timestep()

    finalized = accumulator.finalize()
    del accumulator
    lambdas = (0.0, *map(float, config["scoring"]["lambdas"]))
    sparsities = tuple(map(float, config["scoring"]["sparsities"]))
    sample_size = config["reliability"]["spearman_sample_size"]
    seed = calibration["seed"]
    top_fraction = config["reliability"]["top_sigma_fraction"]
    ratio_eps = torch.finfo(torch.float32).eps
    statistics = {}
    masks = {}
    diagnostics = {}
    for name in layers:
        mu = finalized["full"]["mu"][name]
        sigma = finalized["full"]["sigma"][name]
        mu_a = finalized["a"]["mu"][name]
        sigma_a = finalized["a"]["sigma"][name]
        mu_b = finalized["b"]["mu"][name]
        sigma_b = finalized["b"]["sigma"][name]
        statistics[name] = {
            "mu": mu,
            "sigma": sigma,
            "sigma_A": sigma_a,
            "sigma_B": sigma_b,
        }
        rho_mu_sigma = sampled_spearman(mu, sigma, sample_size, seed)
        rho_split = sampled_spearman(sigma_a, sigma_b, sample_size, seed)
        if not torch.equal(rho_mu_sigma.sample_indices, rho_split.sample_indices):
            raise RuntimeError("Spearman diagnostics did not use the same sample")
        ratio = sigma / (mu + ratio_eps)
        quantiles = torch.quantile(
            ratio.reshape(-1), torch.tensor([0.5, 0.9, 0.99], dtype=ratio.dtype)
        )
        module_masks = {}
        for risk_lambda in lambdas:
            full_score = mu + risk_lambda * sigma
            split_a_score = mu_a + risk_lambda * sigma_a
            split_b_score = mu_b + risk_lambda * sigma_b
            for sparsity in sparsities:
                full_mask = rowwise_prune_mask(full_score, sparsity)
                split_a_mask = rowwise_prune_mask(split_a_score, sparsity)
                split_b_mask = rowwise_prune_mask(split_b_score, sparsity)
                masks[(name, risk_lambda, sparsity)] = full_mask
                mean_mask = (
                    full_mask
                    if risk_lambda == 0
                    else masks[(name, 0.0, sparsity)]
                )
                changed = full_mask.ne(mean_mask)
                key = f"lambda={risk_lambda:g}|sparsity={sparsity:g}"
                module_masks[key] = {
                    "mean_disagreement_count": int(changed.sum().item()),
                    "mean_disagreement_fraction": changed.float().mean().item(),
                    "mean_jaccard": jaccard(mean_mask, full_mask),
                    "changed_row_fraction": row_change_fraction(mean_mask, full_mask),
                    "identical_to_mean": torch.equal(mean_mask, full_mask),
                    "split_jaccard": jaccard(split_a_mask, split_b_mask),
                    "split_A_sha256": pack_mask(split_a_mask)["sha256"],
                    "split_B_sha256": pack_mask(split_b_mask)["sha256"],
                }
        diagnostics[name] = {
            "sigma_over_mu_plus_eps": {
                "epsilon": ratio_eps,
                "median": quantiles[0].item(),
                "p90": quantiles[1].item(),
                "p99": quantiles[2].item(),
            },
            "rho_mu_sigma": _rho_diagnostic(rho_mu_sigma),
            "rho_sigma_A_sigma_B": _rho_diagnostic(rho_split),
            "top_sigma_overlap": top_fraction_overlap(sigma_a, sigma_b, top_fraction),
            "top_sigma_fraction": top_fraction,
            "masks": module_masks,
        }
    return {
        "block_index": block_index,
        "update_count": finalized["update_count"],
        "statistics": statistics,
        "masks": masks,
        "diagnostics": diagnostics,
    }, next_states
