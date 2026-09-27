import math

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


def select_components(values, names):
    missing = set(names) - set(values)
    if missing:
        raise ValueError(f"missing components: {sorted(missing)}")
    return {name: values[name] for name in names}


def ratio_deviation_attribution(groups):
    components = [name for name in groups["uniform"] if name != "total"]
    uniform_total = groups["uniform"]["total"]
    terms = {
        name: groups["cgq"][name] / uniform_total
        for name in components
    }
    return {name: value - value.mean() for name, value in terms.items()}


def _linear_row_scores(weight, activation):
    return (weight.detach().abs().float().cpu() * activation.detach().float().cpu().sqrt()).sum(dim=1)


def _linear_column_scores(weight, activation):
    return weight.detach().abs().float().cpu().sum(dim=0) * activation.detach().float().cpu().sqrt()


def mlp_neuron_scores(weights, uniform, cgq):
    required = {"ff_proj", "up_proj", "ff_out"}
    if set(weights) != required or set(uniform) != required or set(cgq) != required:
        raise ValueError("MLP inputs must contain ff_proj, up_proj, and ff_out")
    result = {}
    for condition, statistics in (("uniform", uniform), ("cgq", cgq)):
        components = {
            "ff_proj": _linear_row_scores(weights["ff_proj"], statistics["ff_proj"]),
            "up_proj": _linear_row_scores(weights["up_proj"], statistics["up_proj"]),
            "ff_out": _linear_column_scores(weights["ff_out"], statistics["ff_out"]),
        }
        if len({value.numel() for value in components.values()}) != 1:
            raise ValueError("MLP component neuron counts differ")
        components["total"] = sum(components.values())
        result[condition] = components
    return result


def attention_head_scores(weights, uniform, cgq, num_heads, head_dim):
    required = {"q_proj", "k_proj", "v_proj", "attn_out"}
    if set(weights) != required or set(uniform) != required or set(cgq) != required:
        raise ValueError("attention inputs must contain q_proj, k_proj, v_proj, and attn_out")
    if any(weights[name].shape[0] != num_heads * head_dim for name in ("q_proj", "k_proj", "v_proj")):
        raise ValueError("q/k/v output dimensions do not match the head mapping")
    if weights["attn_out"].shape[1] != num_heads * head_dim:
        raise ValueError("attn_out input dimension does not match the head mapping")
    result = {}
    for condition, statistics in (("uniform", uniform), ("cgq", cgq)):
        components = {}
        for name in ("q_proj", "k_proj", "v_proj"):
            components[name] = _linear_row_scores(weights[name], statistics[name]).reshape(num_heads, head_dim).sum(dim=1)
        output_channels = weights["attn_out"].detach().abs().float().cpu().sum(dim=0)
        output_scores = output_channels * statistics["attn_out"].detach().float().cpu().sqrt()
        components["attn_out"] = output_scores.reshape(num_heads, head_dim).sum(dim=1)
        components["total"] = sum(components.values())
        result[condition] = components
    return result


def _ordinal_ranks(values):
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.long)
    ranks[order] = torch.arange(values.numel())
    return ranks


def _rank_set(left, right, count, largest):
    left_set = set(torch.topk(left, count, largest=largest).indices.tolist())
    right_set = set(torch.topk(right, count, largest=largest).indices.tolist())
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return {
        "count": count,
        "intersection": intersection,
        "overlap": intersection / count,
        "jaccard": intersection / union,
        "crossed_boundary": len(left_set ^ right_set),
    }


def structured_statistics(uniform, cgq, fractions=(), counts=()):
    uniform = uniform.detach().double().cpu().flatten()
    cgq = cgq.detach().double().cpu().flatten()
    if not uniform.numel() or uniform.shape != cgq.shape or uniform.le(0).any():
        raise ValueError("structured scores must be matching positive vectors")
    ratio = cgq / uniform
    q = torch.quantile(ratio, torch.tensor([.01, .1, .5, .9, .99], dtype=torch.float64))
    displacement = (_ordinal_ranks(cgq) - _ordinal_ranks(uniform)).abs().double()
    rank_sets = {}
    requested = [(f"{round(f * 100)}", max(1, math.floor(uniform.numel() * f))) for f in fractions]
    requested.extend((str(count), count) for count in counts if count <= uniform.numel())
    for label, count in requested:
        rank_sets[f"top_{label}"] = _rank_set(uniform, cgq, count, True)
        rank_sets[f"bottom_{label}"] = _rank_set(uniform, cgq, count, False)
    return {
        "unit_count": uniform.numel(),
        "spearman": float(spearmanr(uniform.numpy(), cgq.numpy()).statistic),
        "kendall_tau": float(kendalltau(uniform.numpy(), cgq.numpy()).statistic),
        "ratio": {"mean": float(ratio.mean()), "std": float(ratio.std(unbiased=False)),
                  "cv": float(ratio.std(unbiased=False) / ratio.mean()),
                  "p01": float(q[0]), "p10": float(q[1]), "p50": float(q[2]),
                  "p90": float(q[3]), "p99": float(q[4]),
                  "min": float(ratio.min()), "max": float(ratio.max())},
        "rank_displacement": {
            "mean_absolute": float(displacement.mean()), "median_absolute": float(displacement.median()),
            "p90_absolute": float(torch.quantile(displacement, .9)), "max_absolute": float(displacement.max()),
            "mean_absolute_normalized": float(displacement.mean() / uniform.numel()),
            "median_absolute_normalized": float(displacement.median() / uniform.numel()),
            "p90_absolute_normalized": float(torch.quantile(displacement, .9) / uniform.numel()),
            "max_absolute_normalized": float(displacement.max() / uniform.numel()),
        },
        "rank_sets": rank_sets,
    }


def aggregate_by_timestep(state_values, timestep_indices, timestep_count=10):
    if state_values.ndim != 2 or timestep_indices.shape != (state_values.shape[0],):
        raise ValueError("state statistics and timestep indices do not align")
    result = torch.zeros((timestep_count, state_values.shape[1]), dtype=state_values.dtype)
    result.index_add_(0, timestep_indices.long(), state_values)
    return result
