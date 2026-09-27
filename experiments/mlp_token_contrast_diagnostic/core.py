import math

import torch
from scipy.stats import kendalltau, rankdata, spearmanr


def gate_directional_derivative(hidden, gradient):
    if hidden.shape != gradient.shape or hidden.ndim != 3:
        raise ValueError("hidden and gradient must be matching [batch, token, neuron] tensors")
    return -(hidden.detach().float() * gradient.detach().float()).sum(dim=(0, 1))


def symmetry_error(uniform, reveal, remain, eps=1e-30):
    if uniform.shape != reveal.shape or uniform.shape != remain.shape:
        raise ValueError("derivative shapes differ")
    residual = reveal.float() + remain.float() - 2 * uniform.float()
    scale = torch.maximum(
        torch.maximum(reveal.float().abs(), remain.float().abs()),
        2 * uniform.float().abs(),
    )
    return {
        "max_absolute": residual.abs().max().item(),
        "max_relative": (residual.abs() / scale.clamp_min(eps)).max().item(),
    }


def mirror_from_uniform_contrast(uniform, contrast):
    if uniform.shape != contrast.shape:
        raise ValueError("uniform and contrast derivative shapes differ")
    return uniform + contrast, uniform - contrast


def abs_scores(uniform, reveal, remain):
    if uniform.shape != reveal.shape or uniform.shape != remain.shape or uniform.ndim != 2:
        raise ValueError("state derivatives must be matching [state, neuron] tensors")
    return {
        "uniform": uniform.abs().mean(dim=0),
        "reveal": reveal.abs().mean(dim=0),
        "remain": remain.abs().mean(dim=0),
    }


def contrast_scores(uniform, reveal, remain):
    if uniform.shape != reveal.shape or uniform.shape != remain.shape:
        raise ValueError("state derivative shapes differ")
    signed = (reveal - remain) / 2
    return {"signed": signed, "score": signed.abs().mean(dim=0)}


def _distribution(values):
    values = values.detach().float()
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99]))
    mean = values.mean().item()
    std = values.std(unbiased=False).item()
    return {
        "mean": mean,
        "median": quantiles[2].item(),
        "std": std,
        "cv": std / mean if mean else None,
        "p01": quantiles[0].item(),
        "p10": quantiles[1].item(),
        "p50": quantiles[2].item(),
        "p90": quantiles[3].item(),
        "p99": quantiles[4].item(),
    }


def ratio_summary(numerator, denominator, relative_floor=1e-6):
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("ratio inputs must be matching vectors")
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    threshold = denominator.max().item() * relative_floor
    valid = denominator >= threshold
    if not valid.any().item():
        raise ValueError("no denominator exceeds the numerical floor")
    result = _distribution(numerator[valid] / denominator[valid])
    excluded_energy = numerator[~valid].sum().item()
    total_energy = numerator.sum().item()
    result.update({
        "floor": threshold,
        "relative_floor": relative_floor,
        "valid_count": int(valid.sum().item()),
        "excluded_count": int((~valid).sum().item()),
        "tail_fraction_from_excluded": excluded_energy / total_energy if total_energy else 0.0,
    })
    return result


def _set_stability(left, right, count, largest):
    left_idx = set(torch.argsort(left, descending=largest, stable=True)[:count].tolist())
    right_idx = set(torch.argsort(right, descending=largest, stable=True)[:count].tolist())
    intersection = len(left_idx & right_idx)
    return {
        "count": count,
        "intersection": intersection,
        "overlap": intersection / count,
        "jaccard": intersection / len(left_idx | right_idx),
        "crossed_boundary": 2 * (count - intersection),
    }


def rank_comparison(left, right, fractions=(0.05, 0.1, 0.2)):
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("rank inputs must be matching vectors")
    left_np = left.detach().float().cpu().numpy()
    right_np = right.detach().float().cpu().numpy()
    left_rank = torch.from_numpy(rankdata(left_np, method="average"))
    right_rank = torch.from_numpy(rankdata(right_np, method="average"))
    displacement = (right_rank - left_rank).abs().float()
    quantiles = torch.quantile(displacement, torch.tensor([0.5, 0.9]))
    sets = {}
    for fraction in fractions:
        count = max(1, math.ceil(left.numel() * fraction))
        suffix = str(round(fraction * 100)).replace(".", "_")
        sets[f"top_{suffix}"] = _set_stability(left, right, count, True)
        sets[f"bottom_{suffix}"] = _set_stability(left, right, count, False)
    return {
        "spearman": float(spearmanr(left_np, right_np).statistic),
        "kendall_tau": float(kendalltau(left_np, right_np).statistic),
        "rank_displacement": {
            "mean_absolute": displacement.mean().item(),
            "median_absolute": quantiles[0].item(),
            "p90_absolute": quantiles[1].item(),
            "max_absolute": displacement.max().item(),
            "mean_absolute_normalized": displacement.mean().item() / left.numel(),
        },
        "sets": sets,
    }


def timestep_scores(state_values, timestep_index):
    if state_values.ndim != 2 or timestep_index.ndim != 1 or state_values.shape[0] != timestep_index.numel():
        raise ValueError("state values and timestep indices do not align")
    return {
        int(index): state_values[timestep_index == index].abs().mean(dim=0)
        for index in torch.unique(timestep_index, sorted=True).tolist()
    }


def stability_summary(grouped_scores, global_score, top_fraction=0.1):
    if grouped_scores.ndim != 2 or global_score.ndim != 1 or grouped_scores.shape[1] != global_score.numel():
        raise ValueError("grouped and global scores do not align")
    comparisons = [rank_comparison(row, global_score, fractions=(top_fraction,)) for row in grouped_scores]
    key = f"top_{str(round(top_fraction * 100)).replace('.', '_')}"
    return {
        "spearman": _distribution(torch.tensor([row["spearman"] for row in comparisons])),
        "top_overlap": _distribution(torch.tensor([row["sets"][key]["overlap"] for row in comparisons])),
    }
