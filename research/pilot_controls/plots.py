"""Raw-data figures for the probe reliability pilot (CPU). Reads levels_cache.npz written by controls.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)


def depth_colors():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", SEQ[1:])


def main():
    z = np.load(HERE / "levels_cache.npz")
    order = list(z["order"])
    i = {r: k for k, r in enumerate(np.round(order, 2))}
    cost = lambda key, lo, hi: z[key][:, i[hi]] - z[key][:, i[lo]]      # [block, span]
    rng = np.random.default_rng(0)
    cmap = depth_colors()
    blocks = np.arange(32)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), facecolor=SURFACE)
    # (a) old setting: A_q, delta 2, 4 vs 4 spans (first 8 spans only)
    old = cost("A_q", 0.48, 0.52)[:, :8]
    perm = rng.permutation(8)
    h1, h2 = old[:, perm[:4]].mean(1), old[:, perm[4:]].mean(1)
    ax = axes[0, 0]
    ax.scatter(h1, h2, c=blocks, cmap=cmap, s=40, edgecolors=SURFACE, linewidths=1.5)
    style(ax, f"(a) Old probe: A_q, ±2pp, 4 vs 4 spans   ρ = {spearmanr(h1, h2)[0]:+.2f}",
          "block cost, span half 1", "block cost, span half 2")
    # (b) new setting: A_m, delta 10, 16 vs 16 spans
    new = cost("A_m", 0.40, 0.60)
    perm = rng.permutation(32)
    g1, g2 = new[:, perm[:16]].mean(1), new[:, perm[16:]].mean(1)
    ax = axes[0, 1]
    sc = ax.scatter(g1, g2, c=blocks, cmap=cmap, s=40, edgecolors=SURFACE, linewidths=1.5)
    style(ax, f"(b) New probe: A_m (all masked), ±10pp, 16 vs 16 spans   ρ = {spearmanr(g1, g2)[0]:+.2f}",
          "block cost, span half 1", "block cost, span half 2")
    cb = fig.colorbar(sc, ax=axes[0, :], fraction=0.025, pad=0.01)
    cb.set_label("block index (0 = first layer)", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=7)
    # (c) depth profile with per-span spread
    ax = axes[1, 0]
    q10, q50, q90 = np.quantile(new, [.1, .5, .9], axis=1)
    ax.fill_between(blocks, q10, q90, color=SEQ[1], alpha=0.6, linewidth=0, label="10–90% of 32 spans")
    ax.plot(blocks, new.mean(1), color=CAT[0], linewidth=2, label="mean over spans")
    fit = np.polyval(np.polyfit(blocks, new.mean(1), 1), blocks)
    ax.plot(blocks, fit, color=INK2, linewidth=1, linestyle="--", label="linear depth trend")
    style(ax, "(c) Damage of pruning each block 40%→60% (A_m)", "block index", "Δ A_m (higher = more harmful)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    # (d) C vs A per block
    ax = axes[1, 1]
    ca, cc = cost("A_m", 0.40, 0.60).mean(1), cost("C_m", 0.40, 0.60).mean(1)
    ax.scatter(ca, cc, c=blocks, cmap=cmap, s=40, edgecolors=SURFACE, linewidths=1.5)
    style(ax, f"(d) C cost vs A cost per block (±10pp)   ρ = {spearmanr(ca, cc)[0]:+.2f}",
          "A_m cost (endpoint error)", "C_m cost (response error)")
    for b in (0, 31, int(np.argmax(ca)), int(np.argmin(ca))):
        ax.annotate(f"b{b}", (ca[b], cc[b]), textcoords="offset points", xytext=(5, 3), fontsize=7.5, color=INK2)
    fig.savefig(HERE / "pilot_raw_data.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")

    # dose-response for five blocks
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
    for c, b in zip(CAT, (0, 8, 16, 24, 31)):
        m = z["A_m"][b].mean(1)
        ax.plot(np.array(order) * 100, m, color=c, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"block {b}")
    style(ax, "Dose–response: A_m when one block is pruned harder (others at 50%)",
          "sparsity of the probed block (%)", "A_m (mean over 32 spans)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.savefig(HERE / "pilot_dose_response.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
