import math

import torch
import torch.nn.functional as F


def rowwise_wanda_mask(scores, sparsity):
    count = int(scores.shape[1] * sparsity)
    order = torch.argsort(scores, dim=1, stable=True)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, order[:, :count], True)
    return mask


def masked_linear_variants(inp, weight, bias, masks):
    if inp.shape[0] != len(masks):
        raise ValueError("variant batch and masks differ")
    outputs = []
    for row, mask in enumerate(masks):
        selected = weight if mask is None else weight.masked_fill(mask, 0)
        outputs.append(F.linear(inp[row], selected, bias))
    return torch.stack(outputs)


def reconstruction_metrics(dense, sparse, eps=1e-30):
    dense, sparse = dense.float(), sparse.float()
    difference = sparse - dense
    dense_norm = dense.square().sum()
    return {
        "relative_squared_error": (difference.square().sum() / dense_norm.clamp_min(eps)).item(),
        "relative_l2_error": (difference.norm() / dense.norm().clamp_min(eps)).item(),
        "cosine": F.cosine_similarity(dense.reshape(1, -1), sparse.reshape(1, -1)).item(),
    }


def threshold_geometry(scores, sparsity):
    count = int(scores.shape[1] * sparsity)
    ordered = scores.sort(dim=1).values
    threshold = ordered[:, count - 1]
    first_kept = ordered[:, count]
    gap = first_kept - threshold
    scale = threshold.abs().clamp_min(1e-30)
    relative = (scores - threshold[:, None]).abs() / scale[:, None]
    def summary(value):
        return {"mean": value.float().mean().item(), "std": value.float().std(unbiased=False).item()}
    return {"threshold_mean": threshold.mean().item(), "gap_mean": gap.mean().item(),
            "normalized_gap_mean": (gap / scale).mean().item(),
            "within_1pct_mean": (relative <= .01).float().mean().item(),
            "within_5pct_mean": (relative <= .05).float().mean().item(),
            "row_statistics": {"threshold": summary(threshold), "gap": summary(gap),
                               "normalized_gap": summary(gap / scale),
                               "within_1pct": summary((relative <= .01).float().mean(1)),
                               "within_5pct": summary((relative <= .05).float().mean(1))}}
