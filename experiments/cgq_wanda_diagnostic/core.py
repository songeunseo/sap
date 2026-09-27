import math

import torch


def cgq_factors(confidence, masked):
    if confidence.shape != masked.shape or masked.dtype != torch.bool:
        raise ValueError("confidence and masked must be matching tensors")
    if not torch.isfinite(confidence).all() or confidence.lt(0).any() or confidence.gt(1).any():
        raise ValueError("confidence must be finite probabilities")
    base = torch.where(masked, 1.0, 0.7).to(confidence.dtype)
    return base + confidence.sqrt()


class ActivationAccumulator:
    """Wanda-equivalent batch mean plus CGQ-weighted counterpart."""

    def __init__(self, columns):
        self.uniform = torch.zeros(columns, dtype=torch.float32)
        self.cgq = torch.zeros(columns, dtype=torch.float32)
        self.nsamples = 0
        self._token_norms = []

    def add_batch(self, inp, factors):
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        if inp.ndim != 3 or factors.shape != inp.shape[:2]:
            raise ValueError("expected [batch, token, feature] inputs and [batch, token] factors")
        batch = inp.shape[0]
        flat = inp.reshape(-1, inp.shape[-1]).float()
        flat_factors = factors.reshape(-1).to(device=flat.device, dtype=torch.float32)
        scale = self.nsamples / (self.nsamples + batch)
        self.uniform.mul_(scale)
        self.cgq.mul_(scale)
        self.nsamples += batch
        square = flat.square()
        self.uniform.add_(square.sum(dim=0).cpu(), alpha=1 / self.nsamples)
        self.cgq.add_(
            (square * flat_factors.square().unsqueeze(1)).sum(dim=0).cpu(),
            alpha=1 / self.nsamples,
        )
        self._token_norms.append(flat.norm(p=2, dim=1).reshape(inp.shape[:2]).cpu())

    @property
    def token_norms(self):
        return torch.cat(self._token_norms, dim=0)


def _ranks(values):
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64)
    return ranks


def _pearson(left, right):
    left = left.double() - left.double().mean()
    right = right.double() - right.double().mean()
    denominator = left.norm() * right.norm()
    return float((left @ right / denominator).item()) if denominator else None


def feature_comparison(uniform, weighted):
    uniform = uniform.detach().double().cpu()
    weighted = weighted.detach().double().cpu()
    if uniform.ndim != 1 or uniform.shape != weighted.shape or uniform.le(0).any():
        raise ValueError("feature statistics must be matching positive vectors")
    ratio = weighted / uniform
    mean = ratio.mean()
    std = ratio.std(unbiased=False)
    quantiles = torch.quantile(ratio, torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], dtype=torch.float64))
    return {
        "spearman": _pearson(_ranks(uniform), _ranks(weighted)),
        "log_pearson": _pearson(uniform.log(), weighted.log()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(uniform, weighted, dim=0).item()),
        "relative_l2_difference": float((weighted - uniform).norm().div(uniform.norm()).item()),
        "ratio": {
            "mean": float(mean.item()), "std": float(std.item()),
            "cv": float((std / mean).item()), "p01": float(quantiles[0]),
            "p10": float(quantiles[1]), "p50": float(quantiles[2]),
            "p90": float(quantiles[3]), "p99": float(quantiles[4]),
            "min": float(ratio.min()), "max": float(ratio.max()),
        },
    }


def masked_confidence_deciles(confidence, energy, weighted_energy, activation_norm):
    confidence = confidence.detach().double().cpu().flatten()
    energy = energy.detach().double().cpu().flatten()
    weighted_energy = weighted_energy.detach().double().cpu().flatten()
    activation_norm = activation_norm.detach().double().cpu().flatten()
    if (not confidence.numel() or confidence.shape != energy.shape
            or energy.shape != weighted_energy.shape or energy.shape != activation_norm.shape):
        raise ValueError("masked token vectors must be nonempty and matching")
    order = torch.argsort(confidence, stable=True)
    bins = torch.tensor_split(order, 10)
    rows = []
    for index, selected in enumerate(bins, 1):
        c = confidence[selected]
        e = energy[selected]
        we = weighted_energy[selected]
        norm = activation_norm[selected]
        rows.append({
            "decile": index,
            "token_count": selected.numel(),
            "mean_confidence": float(c.mean()),
            "median_confidence": float(c.median()),
            "mean_activation_norm": float(norm.mean()),
            "uniform_energy_fraction": float(e.sum() / energy.sum()),
            "cgq_energy_fraction": float(we.sum() / weighted_energy.sum()),
        })
    return rows
