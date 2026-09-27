"""Per-block sparsity profiles: A+C (Multi) vs depth-schedule arms (CPU)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent / "output"
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
CAT = {"EIS": "#2a78d6", "DIS": "#eb6834", "AmShape": "#1baf7a", "Multi (A+C)": "#eda100", "A": "#e87ba4"}

d = json.loads((HERE / "allocation_comparison.json").read_text())
R = {k: np.array(v) * 100 for k, v in d["rates"].items()}
cost = np.array(json.loads((HERE / "config.json").read_text())["pilot_cost_Am_d10"])
t = np.arange(32)


def style(ax, title, xl, yl):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xl, color=INK2, fontsize=8.5)
    ax.set_ylabel(yl, color=INK2, fontsize=8.5)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axes[0]
for k in ("AmPerm1", "AmPerm2", "AmPerm3"):
    ax.plot(t, R[k], color=MUTED, linewidth=1, label="AmPerm ×3 (shuffled)" if k == "AmPerm1" else None)
ax.axhline(50, color=INK2, linewidth=1, linestyle="--", label="Uniform")
for name, key in (("DIS", "DIS"), ("EIS", "EIS"), ("AmShape", "AmShape"), ("A", "A"), ("Multi (A+C)", "Multi")):
    ax.plot(t, R[key], color=CAT[name], linewidth=2.2 if key == "Multi" else 1.8, label=name)
style(ax, "(a) Per-block sparsity (block = layer; 7 projections share it)", "block index (0 = first)", "sparsity (%)")
ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=2, loc="upper center")
ax = axes[1]
ax.scatter(cost, R["Multi"], s=40, color=CAT["Multi (A+C)"], edgecolors=SURFACE, linewidths=1.5, label="Multi (A+C)")
ax.scatter(cost, R["AmShape"], s=40, color=CAT["AmShape"], edgecolors=SURFACE, linewidths=1.5, label="AmShape")
style(ax, f"(b) Sparsity vs reliable damage (pilot A_m)\nMulti ρ={spearmanr(cost, R['Multi'])[0]:+.2f}, AmShape ρ=-1.00",
      "pilot damage cost of the block (higher = more fragile)", "sparsity (%)")
ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.savefig(HERE / "allocation_profiles.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
print("saved")
