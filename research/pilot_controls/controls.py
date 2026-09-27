"""Known-answer controls for the probe reliability pilot (CPU only).

1. Monotonicity: pruning a block harder should not reduce damage. Per block, is the sequence-mean level
   non-decreasing over 40 < 45 < 48 < 50(background) < 52 < 55 < 60?
2. Width proportionality: |J(hi) - J(lo)| should grow with delta (2 -> 5 -> 10).
3. Negative control: permuting block labels independently within each span destroys block identity;
   reliability must collapse to ~0 (200 shuffles).
Also caches per-(block, rate, span) levels to levels_cache.npz for plotting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.dlm_probe_reliability_pilot.analysis_c import response_levels
from experiments.dlm_probe_reliability_pilot.run import (OUT, RATES, UNIFORM_MANIFEST, read, reliability,
                                                         sequence_levels, validate)

HERE = Path(__file__).resolve().parent
KEYS = ("A_q", "A_m", "CE_m", "C_m")
ORDER = [0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]


def main():
    import torch
    config = validate(check_code=False)
    bank = read(config["bank"]["path"])
    load = lambda label: torch.load(OUT / "readouts" / f"{label}.pt", weights_only=False).double().numpy()
    dense = load("dense")
    L = {k: np.zeros((32, len(ORDER), 32)) for k in KEYS}
    ub = load("uniform50")
    ul = sequence_levels(ub, dense, bank); ul.update(response_levels(ub, dense, bank))
    for b in range(32):
        for j, r in enumerate(ORDER):
            if r == 0.50:
                lv = ul
            else:
                v = load(f"b{b:02d}_r{round(r * 100)}")
                lv = sequence_levels(v, dense, bank); lv.update(response_levels(v, dense, bank))
            for k in KEYS:
                L[k][b, j] = lv[k]
    np.savez(HERE / "levels_cache.npz", **L, order=np.array(ORDER))
    out = {}
    rng = np.random.default_rng(7)
    for k in KEYS:
        m = L[k].mean(2)                                   # [block, rate]
        mono = np.all(np.diff(m, axis=1) >= 0, axis=1)
        steps_up = (np.diff(m, axis=1) > 0).mean()
        widths = {}
        for d, (lo, hi) in {2: (2, 4), 5: (1, 5), 10: (0, 6)}.items():
            widths[d] = float(np.abs(m[:, hi] - m[:, lo]).mean())
        cost10 = L[k][:, 6] - L[k][:, 0]
        shuffled = []
        for _ in range(200):
            c = cost10.copy()
            for s in range(c.shape[1]):
                c[:, s] = rng.permutation(c[:, s])
            shuffled.append(reliability(c, rng, splits=50)["reliability"])
        out[k] = dict(blocks_fully_monotone=int(mono.sum()), frac_adjacent_steps_increasing=float(steps_up),
                      mean_abs_width={str(d): w for d, w in widths.items()},
                      width_ratio_10_over_2=widths[10] / widths[2],
                      real_reliability_d10=reliability(cost10, rng, splits=50)["reliability"],
                      shuffled_reliability_mean=float(np.mean(shuffled)),
                      shuffled_reliability_q95=float(np.quantile(shuffled, .95)))
        print(k, json.dumps(out[k]))
    (HERE / "result.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
