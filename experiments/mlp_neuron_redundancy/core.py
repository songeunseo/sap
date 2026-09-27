import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, rankdata, spearmanr


def fixed_candidates(layers, hidden_size, per_layer, seed):
    generator = np.random.default_rng(seed)
    return {
        int(layer): sorted(generator.choice(hidden_size, per_layer, replace=False).tolist())
        for layer in layers
    }


def ablate_variant_rows(hidden, neuron_ids):
    if hidden.ndim != 3 or hidden.shape[0] != len(neuron_ids):
        raise ValueError("hidden batch and variant IDs do not align")
    result = hidden.clone()
    for row, neuron in enumerate(neuron_ids):
        if neuron is not None:
            if neuron < 0 or neuron >= hidden.shape[-1]:
                raise ValueError("neuron ID is out of range")
            result[row, :, neuron] = 0
    return result


def masked_losses_and_kl(variant_logits, dense_logits, targets, mask, p_mask):
    if variant_logits.ndim != 3 or dense_logits.shape[0] != 1:
        raise ValueError("logits must be [variant, token, vocabulary]")
    positions = mask[0]
    target = targets[0, positions]
    selected = variant_logits[:, positions].float()
    repeated_target = target.unsqueeze(0).expand(selected.shape[0], -1)
    losses = F.cross_entropy(
        selected.reshape(-1, selected.shape[-1]), repeated_target.reshape(-1), reduction="none"
    ).reshape(selected.shape[:2]).sum(dim=1) / p_mask / targets.shape[1]
    dense_log_prob = F.log_softmax(dense_logits[0, positions].float(), dim=-1)
    dense_prob = dense_log_prob.exp()
    variant_log_prob = F.log_softmax(selected, dim=-1)
    kl = (dense_prob.unsqueeze(0) * (dense_log_prob.unsqueeze(0) - variant_log_prob)).sum(dim=-1).mean(dim=1)
    return losses, kl


def sham_corrected_losses(losses):
    if losses.ndim != 1 or losses.numel() < 2:
        raise ValueError("losses must contain one sham and at least one intervention")
    return losses[1:] - losses[0]


def _summary(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    q = torch.quantile(values, torch.tensor([.01, .05, .1, .25, .5, .75, .9, .95, .99], dtype=torch.float64))
    mean, std = values.mean().item(), values.std(unbiased=False).item()
    return {"count": values.numel(), "min": values.min().item(), "p01": q[0].item(),
            "p05": q[1].item(), "p10": q[2].item(), "p25": q[3].item(),
            "median": q[4].item(), "p75": q[5].item(), "p90": q[6].item(),
            "p95": q[7].item(), "p99": q[8].item(), "max": values.max().item(),
            "mean": mean, "std": std, "cv": std / mean if mean else None}


def distribution_summary(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if values.numel() == 0 or (values < 0).any().item():
        raise ValueError("distribution must be nonempty and nonnegative")
    result = _summary(values)
    sorted_values = values.sort().values
    n = values.numel()
    total = sorted_values.sum().item()
    result["gini"] = 0.0 if not total else (
        (2 * torch.arange(1, n + 1, dtype=torch.float64) - n - 1) * sorted_values
    ).sum().item() / (n * total)
    result["p90_over_p10"] = result["p90"] / result["p10"] if result["p10"] else None
    result["p90_over_median"] = result["p90"] / result["median"] if result["median"] else None
    return result


def _set(left, right, fraction, largest=False):
    count = max(1, math.ceil(left.numel() * fraction))
    a = set(torch.argsort(left, descending=largest, stable=True)[:count].tolist())
    b = set(torch.argsort(right, descending=largest, stable=True)[:count].tolist())
    intersection = len(a & b)
    return {"overlap": intersection / count, "jaccard": intersection / len(a | b),
            "crossed_boundary": 2 * (count - intersection), "count": count}


def pairwise_stability(scores, fractions=(.1, .2)):
    if scores.ndim != 2 or scores.shape[0] < 2:
        raise ValueError("at least two score vectors are required")
    pairs = list(itertools.combinations(range(scores.shape[0]), 2))
    result = {"spearman": _summary([
        spearmanr(scores[a].numpy(), scores[b].numpy()).statistic for a, b in pairs
    ])}
    for fraction in fractions:
        key = f"bottom_{round(100 * fraction)}"
        rows = [_set(scores[a], scores[b], fraction) for a, b in pairs]
        result[key] = {
            name: _summary([row[name] for row in rows]) for name in ("overlap", "jaccard", "crossed_boundary")
        }
    return result


def rank_predictability(baseline, exact, layers, fractions=(.1, .2)):
    if baseline.shape != exact.shape or baseline.shape != layers.shape:
        raise ValueError("baseline, exact, and layer vectors must align")
    baseline = baseline.detach().cpu()
    exact = exact.detach().cpu()
    layers = layers.detach().cpu()
    baseline_ranks = torch.empty_like(baseline, dtype=torch.float64)
    exact_ranks = torch.empty_like(exact, dtype=torch.float64)
    result = {}
    for layer in torch.unique(layers, sorted=True):
        selected = layers == layer
        baseline_ranks[selected] = torch.from_numpy(rankdata(baseline[selected].numpy())).double()
        exact_ranks[selected] = torch.from_numpy(rankdata(exact[selected].numpy())).double()
    result["spearman"] = float(spearmanr(baseline_ranks.numpy(), exact_ranks.numpy()).statistic)
    result["kendall_tau"] = float(kendalltau(baseline_ranks.numpy(), exact_ranks.numpy()).statistic)
    for fraction in fractions:
        key = f"{round(100 * fraction)}"
        bottoms, tops = [], []
        for layer in torch.unique(layers, sorted=True):
            selected = torch.where(layers == layer)[0]
            bottoms.append(_set(baseline[selected], exact[selected], fraction, largest=False))
            tops.append(_set(baseline[selected], exact[selected], fraction, largest=True))
        for prefix, rows in (("bottom", bottoms), ("top", tops)):
            result[f"{prefix}_{key}"] = {
                name: sum(row[name] for row in rows) / len(rows)
                for name in ("overlap", "jaccard", "crossed_boundary", "count")
            }
    return result
