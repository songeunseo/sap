"""Frozen downstream-attribution helpers for DLMW versus Fisher geometry."""

from collections import Counter

import torch
from scipy.stats import binomtest


TARGET = "block_31.ff_out"


def build_attribution_masks(standard, dlmw_target, fg_target, target=TARGET):
    if target not in standard:
        raise KeyError(target)
    reference = standard[target]
    for name, candidate in (("DLMW", dlmw_target), ("FG", fg_target)):
        if candidate.shape != reference.shape or candidate.dtype != torch.bool:
            raise ValueError(f"{name} target mask shape/dtype mismatch")
        if not torch.equal(candidate.sum(1), reference.sum(1)):
            raise ValueError(f"{name} target mask per-row pruned count mismatch")

    ours_dlmw = dict(standard)
    ours_dlmw[target] = dlmw_target
    ours_fg = dict(standard)
    ours_fg[target] = fg_target
    changed_vs_wanda = sorted(name for name in standard if not torch.equal(ours_dlmw[name], standard[name]))
    changed_vs_fg = sorted(name for name in standard if not torch.equal(ours_dlmw[name], ours_fg[name]))
    if changed_vs_wanda != [target] or changed_vs_fg != [target]:
        raise RuntimeError("attribution configuration changed a non-target module")
    return ours_dlmw, changed_vs_wanda, changed_vs_fg


def three_way_correctness(wanda, dlmw, fg):
    if not (len(wanda) == len(dlmw) == len(fg)):
        raise ValueError("three-way sample count mismatch")
    counts = Counter((int(bool(w)), int(bool(d)), int(bool(f))) for w, d, f in zip(wanda, dlmw, fg))
    return {
        f"W{w}_D{d}_F{f}": counts[(w, d, f)]
        for w in (0, 1)
        for d in (0, 1)
        for f in (0, 1)
    }


def paired_correctness(left, right, left_name, right_name):
    if len(left) != len(right):
        raise ValueError("paired sample count mismatch")
    left = [bool(x) for x in left]
    right = [bool(x) for x in right]
    both = sum(a and b for a, b in zip(left, right))
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(not a and b for a, b in zip(left, right))
    neither = sum(not a and not b for a, b in zip(left, right))
    discordant = left_only + right_only
    pvalue = 1.0 if not discordant else float(
        binomtest(min(left_only, right_only), discordant, 0.5, alternative="two-sided").pvalue
    )
    delta = right_only - left_only
    return {
        "sample_count": len(left),
        "both_correct": both,
        f"{left_name}_only": left_only,
        f"{right_name}_only": right_only,
        "both_wrong": neither,
        f"{left_name}_correct": sum(left),
        f"{right_name}_correct": sum(right),
        f"correct_delta_{right_name}_minus_{left_name}": delta,
        f"delta_percentage_points_{right_name}_minus_{left_name}": 100 * delta / len(left),
        "mcnemar_exact_pvalue": pvalue,
    }


def attribution_decomposition(wanda_correct, dlmw_correct, fg_correct, n):
    def effect(examples):
        return {"examples": examples, "percentage_points": 100 * examples / n}

    return {
        "total_FG_minus_Wanda": effect(fg_correct - wanda_correct),
        "calibration_DLMW_minus_Wanda": effect(dlmw_correct - wanda_correct),
        "fisher_incremental_FG_minus_DLMW": effect(fg_correct - dlmw_correct),
    }
