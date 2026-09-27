"""Numerical helpers for super-outlier concentration and association analysis."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


SUPER_CHANNEL = 3848


def tensor_statistics(x, channel=SUPER_CHANNEL):
    """Return channel-energy concentration summaries for one [*, feature] tensor."""
    x = x.detach().float().reshape(-1, x.shape[-1])
    energy = x.square().sum(0).double().cpu().numpy()
    width = len(energy)
    total = float(energy.sum())
    if total <= 0 or not np.isfinite(energy).all():
        raise ValueError("finite nonzero feature energy required")
    p = energy / total
    top = max(1, int(np.ceil(width * .01)))
    rms = np.sqrt(energy / len(x))
    result = {
        "width": width,
        "tokens": len(x),
        "energy": total,
        "dominant_channel": int(np.argmax(energy)),
        "dominant_share": float(p.max()),
        "top1pct_share": float(np.partition(p, -top)[-top:].sum()),
        "effective_fraction": float(1.0 / (width * np.square(p).sum())),
        "outlier7_ratio": float(np.mean(rms > 7 * rms.mean())),
    }
    if channel < width:
        c = x[:, channel].double().cpu().numpy()
        rest = np.delete(energy, channel)
        rest_total = float(rest.sum())
        rp = rest / rest_total if rest_total else np.zeros_like(rest)
        rest_top = max(1, int(np.ceil(len(rest) * .01)))
        rest_rms = np.sqrt(rest / len(x))
        second = float(np.mean(c * c))
        result.update({
            "super_share": float(energy[channel] / total),
            "super_rank": int(1 + np.sum(energy > energy[channel])),
            "super_token_argmax_fraction": float(
                (x.abs().argmax(-1) == channel).float().mean().cpu()
            ),
            "super_mean_sq_ratio": float(np.mean(c) ** 2 / second) if second else None,
            "excluded_dominant_share": float(rp.max()) if rest_total else None,
            "excluded_top1pct_share": (
                float(np.partition(rp, -rest_top)[-rest_top:].sum()) if rest_total else None
            ),
            "excluded_effective_fraction": (
                float(1.0 / (len(rest) * np.square(rp).sum()))
                if rest_total and np.square(rp).sum() else None
            ),
            "excluded_outlier7_ratio": (
                float(np.mean(rest_rms > 7 * rest_rms.mean()))
                if rest_total and rest_rms.mean() else None
            ),
        })
    return result


def read_contribution(x, y, weight, bias, channel=SUPER_CHANNEL):
    """Decompose a Linear output into one input-channel contribution and remainder."""
    if channel >= x.shape[-1]:
        return None
    x = x.detach().float().reshape(-1, x.shape[-1])
    y = y.detach().float().reshape(-1, y.shape[-1])
    a = x[:, channel:channel + 1] * weight.detach().float()[:, channel][None, :]
    b = y - a
    y2 = float(y.square().sum().double().cpu())
    if y2 <= 0:
        return None
    a2 = float(a.square().sum().double().cpu())
    b2 = float(b.square().sum().double().cpu())
    cross = float((2 * a * b).sum().double().cpu())
    return {
        "super_component_over_output": a2 / y2,
        "remainder_over_output": b2 / y2,
        "cross_over_output": cross / y2,
        "decomposition_error": abs((a2 + b2 + cross) / y2 - 1.0),
        "bias_present": bias is not None,
    }


def residual_addition(before, after, channel=SUPER_CHANNEL):
    """Measure how a residual branch writes the selected residual channel."""
    before = before.detach().float().reshape(-1, before.shape[-1])[:, channel]
    after = after.detach().float().reshape(-1, after.shape[-1])[:, channel]
    branch = after - before
    after2 = float(after.square().sum().double().cpu())
    before2 = float(before.square().sum().double().cpu())
    branch2 = float(branch.square().sum().double().cpu())
    cross = float((2 * before * branch).sum().double().cpu())
    denom = after2 if after2 > 0 else 1.0
    return {
        "super_before_energy": before2,
        "super_branch_energy": branch2,
        "super_after_energy": after2,
        "super_branch_over_after": branch2 / denom,
        "super_cross_over_after": cross / denom,
        "decomposition_error": abs((before2 + branch2 + cross) / denom - (after2 / denom)),
    }


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    if (valid.sum() < 3 or np.allclose(x[valid], x[valid][0]) or
            np.allclose(y[valid], y[valid][0])):
        return {"spearman": None, "n": int(valid.sum())}
    return {"spearman": float(spearmanr(x[valid], y[valid]).statistic),
            "n": int(valid.sum())}


def partial_rank_correlation(x, y, layers, types, covariate=None):
    """Residualize ranks against layer/type dummies and an optional ranked covariate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    layers, types = np.asarray(layers), np.asarray(types)
    valid = np.isfinite(x) & np.isfinite(y)
    if covariate is not None:
        covariate = np.asarray(covariate, float)
        valid &= np.isfinite(covariate)
    ids = np.where(valid)[0]
    if len(ids) < 3:
        return {"spearman": None, "n": int(len(ids))}
    columns = [np.ones(len(ids))]
    for value in sorted(set(layers[ids]))[1:]:
        columns.append((layers[ids] == value).astype(float))
    for value in sorted(set(types[ids]))[1:]:
        columns.append((types[ids] == value).astype(float))
    if covariate is not None:
        columns.append(rankdata(covariate[ids]))
    design = np.column_stack(columns)
    rx, ry = rankdata(x[ids]), rankdata(y[ids])
    rx -= design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry -= design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return correlation(rx, ry)
