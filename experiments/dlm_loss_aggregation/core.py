import hashlib
import json
import math
import resource
import time

import numpy as np
import torch


class EffectAccumulator:
    def __init__(self, shape):
        self.shape = torch.Size(shape)
        self.signed = torch.zeros(self.shape, dtype=torch.float32)
        self.absolute = torch.zeros_like(self.signed)
        self.square = torch.zeros_like(self.signed)
        self.count = 0
        self.total_weight = 0.0

    def add(self, effect, state_weight=1.0):
        if effect.shape != self.shape or not torch.isfinite(effect).all().item():
            raise ValueError("effect must be finite and match the accumulator shape")
        if not math.isfinite(state_weight) or state_weight <= 0:
            raise ValueError("state_weight must be finite and positive")
        value = effect.detach().to(device="cpu", dtype=torch.float32).clone()
        self.signed.add_(value, alpha=state_weight)
        value.abs_()
        self.absolute.add_(value, alpha=state_weight)
        self.square.addcmul_(value, value, value=state_weight)
        self.count += 1
        self.total_weight += state_weight

    def finalize(self):
        if not self.count:
            raise RuntimeError("cannot finalize an empty accumulator")
        return {
            "sum": self.signed / self.total_weight,
            "abs": self.absolute / self.total_weight,
            "square": self.square / self.total_weight,
        }


def rowwise_mask(score, sparsity):
    if score.ndim != 2 or not torch.isfinite(score).all().item():
        raise ValueError("score must be a finite matrix")
    if not math.isfinite(sparsity) or not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between zero and one")
    count = math.floor(score.shape[1] * sparsity)
    mask = torch.zeros_like(score, dtype=torch.bool)
    if count:
        indices = torch.argsort(score, dim=1, stable=True)[:, :count]
        mask.scatter_(1, indices, True)
    return mask


def pack_mask(mask):
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean matrix")
    values = mask.detach().cpu().contiguous().numpy().reshape(-1)
    return {
        "shape": tuple(mask.shape),
        "bits": np.packbits(values, bitorder="big").tobytes(),
    }


def unpack_mask(payload):
    shape = payload.get("shape")
    bits = payload.get("bits")
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in shape)
        or not isinstance(bits, bytes)
        or len(bits) != math.ceil(math.prod(shape) / 8)
    ):
        raise ValueError("invalid packed mask")
    values = np.unpackbits(np.frombuffer(bits, dtype=np.uint8), bitorder="big")
    count = math.prod(shape)
    if values[count:].any():
        raise ValueError("packed mask has nonzero trailing bits")
    return torch.from_numpy(values[:count].reshape(shape).astype(np.bool_, copy=True))


def mask_sha256(payload):
    mask = unpack_mask(payload)
    header = json.dumps(
        {"shape": list(mask.shape), "bitorder": "big"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + payload["bits"]).hexdigest()


def _average_ranks(values):
    values = values.double()
    sorted_values, order = torch.sort(values)
    starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.nonzero(sorted_values[1:] != sorted_values[:-1]).flatten() + 1,
        )
    )
    ends = torch.cat((starts[1:], torch.tensor([values.numel()])))
    group_ranks = (starts + ends + 1).double() / 2
    sorted_ranks = torch.repeat_interleave(group_ranks, ends - starts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _rank_correlation(left, right):
    left = left.clone()
    right = right.clone()
    left.sub_(left.mean())
    right.sub_(right.mean())
    denominator = torch.sqrt(left.square().sum() * right.square().sum())
    if denominator.item() == 0:
        return None, "constant score"
    return (left * right).sum().div_(denominator).item(), None


def _pairwise_spearman(values):
    ranks = {name: _average_ranks(value) for name, value in values.items()}
    rows = []
    for left, right in (("sum", "abs"), ("sum", "square"), ("abs", "square")):
        correlation, reason = _rank_correlation(ranks[left], ranks[right])
        rows.append(
            {
                "left": left,
                "right": right,
                "spearman": correlation,
                "reason": reason,
            }
        )
    return rows


def _sample_indices(count, sample_size, seed):
    if sample_size is None or sample_size >= count:
        return None, count, hashlib.sha256(f"all:{count}".encode()).hexdigest()
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("spearman_sample_size must be positive")
    indices = np.random.default_rng(seed).choice(count, sample_size, replace=False)
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    return torch.from_numpy(indices.astype(np.int64, copy=False)), sample_size, digest


def _distribution_summary(values, seed):
    flat = values.detach().cpu().float().reshape(-1)
    indices, sample_size, digest = _sample_indices(flat.numel(), min(1_000_000, flat.numel()), seed)
    sample = flat if indices is None else flat[indices]
    quantiles = torch.quantile(sample, torch.tensor([0.5, 0.9, 0.99]))
    return {
        "mean": flat.mean().item(),
        "min": flat.min().item(),
        "p50": quantiles[0].item(),
        "p90": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "max": flat.max().item(),
        "quantile_sample_size": sample_size,
        "sample_indices_sha256": digest,
    }


def pairwise_diagnostics(scores, spearman_sample_size=None, seed=0):
    if set(scores) != {"sum", "abs", "square"}:
        raise ValueError("scores must contain sum, abs, and square")
    shapes = {tuple(score.shape) for score in scores.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("scores must be matching matrices")
    if any(not torch.isfinite(score).all().item() for score in scores.values()):
        raise ValueError("scores must be finite")

    flattened = {name: score.detach().cpu().reshape(-1) for name, score in scores.items()}
    indices, sample_size, sample_digest = _sample_indices(
        next(iter(flattened.values())).numel(), spearman_sample_size, seed
    )
    ranked_values = {
        name: values if indices is None else values[indices]
        for name, values in flattened.items()
    }
    masks = {name: rowwise_mask(score, 0.5) for name, score in scores.items()}
    pairs = _pairwise_spearman(ranked_values)
    for pair in pairs:
        pair["sample_size"] = sample_size
        pair["sample_indices_sha256"] = sample_digest
        pair["mask_xor"] = (
            masks[pair["left"]].ne(masks[pair["right"]]).float().mean().item()
        )

    epsilon = torch.finfo(torch.float32).eps
    sign_consistency = scores["sum"].abs() / (scores["abs"] + epsilon)
    spike_ratio = scores["square"].sqrt() / (scores["abs"] + epsilon)
    return {
        "pairs": pairs,
        "negative_sum_fraction": scores["sum"].lt(0).float().mean().item(),
        "sign_consistency": _distribution_summary(sign_consistency, seed),
        "spike_ratio": _distribution_summary(spike_ratio, seed),
    }


def _current_rss_kib():
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS is unavailable")


def profile_pairwise_spearman(scores):
    flattened = {
        name: value.detach().cpu().reshape(-1) for name, value in scores.items()
    }
    before_rss = _current_rss_kib()
    started = time.perf_counter()
    pairs = _pairwise_spearman(flattened)
    elapsed = time.perf_counter() - started
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "pairs": pairs,
        "elapsed_seconds": elapsed,
        "rss_before_kib": before_rss,
        "peak_rss_kib": peak_rss,
        "rss_delta_kib": max(0, peak_rss - before_rss),
        "element_count": next(iter(flattened.values())).numel(),
    }
