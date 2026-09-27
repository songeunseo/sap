"""Depth decomposition of pilot block costs (pre-registered 2026-09-27 before GPU completion; CPU only).

Motivation: in the original 8-span multiscale probes every span showed a positive depth trend in A cost
(8/8 spans, Spearman +0.10..+0.29; linear depth R^2 of mean cost 0.35) while split-half reliability of the full
ranking was ~0. So block costs may be "weak depth trend + noise", i.e. a noisy EIS-like schedule.

For every readout x delta:
  depth:    per-span Spearman(block index, cost); mean, #positive of 32, two-sided sign-test p
  residual: remove each span's least-squares linear depth fit; reliability() of the residual costs
Pre-registered reading:
  depth consistent (sign p < .01) and residual reliability < 0.5  -> criterion reduces to a depth schedule;
      compare against EIS rather than claim block-specific information
  residual reliability >= 0.8 (and half rho >= 0.6)               -> block-specific information beyond depth
  For C (Cpart): only residual reliability >= 0.8 counts as C adding non-depth information.
"""
from __future__ import annotations

import json
from math import comb

import numpy as np

from experiments.dlm_probe_reliability_pilot.analysis_c import response_levels
from experiments.dlm_probe_reliability_pilot.run import (DELTAS, OUT, RATES, SEED, UNIFORM_MANIFEST, read,
                                                         reliability, sequence_levels, sha, validate, write)

KEYS = ("A_q", "Cpart_q", "A_m", "CE_m", "Cpart_m", "Multi_m")


def sign_p(k, n):
    m = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n)


def main():
    import torch
    from scipy.stats import spearmanr
    config = validate(check_code=False)
    if not (OUT / "gpu_complete.json").exists():
        raise RuntimeError("GPU phase incomplete")
    bank, refs = read(config["bank"]["path"]), read(UNIFORM_MANIFEST)["entries"]
    load = lambda label: torch.load(OUT / "readouts" / f"{label}.pt", weights_only=False).double().numpy()
    dense = load("dense")
    levels = {}
    for b in range(32):
        for r in RATES:
            v = load(f"b{b:02d}_r{round(r * 100)}")
            lv = sequence_levels(v, dense, bank)
            lv.update(response_levels(v, dense, bank))
            lv["Cpart_q"] = list(np.asarray(lv["Multi_q"]) - np.asarray(lv["A_q"]))
            lv["Cpart_m"] = lv["C_m"]
            levels[(b, r)] = lv
    pruned = lambda b, r: sum(int(x["shape"][1] * r) * x["shape"][0] for x in refs[b * 7:(b + 1) * 7])
    depth = np.arange(32, dtype=float)
    X = np.c_[np.ones(32), depth]
    rng = np.random.default_rng(SEED + 2)
    result = dict(config_sha256=sha(OUT / "config.json"), code_sha256=sha(__file__), cells={})
    for delta, (lo, hi) in DELTAS.items():
        for key in KEYS:
            cost = np.asarray([(np.asarray(levels[(b, hi)][key]) - np.asarray(levels[(b, lo)][key]))
                               / (pruned(b, hi) - pruned(b, lo)) for b in range(32)])
            rhos = np.asarray([spearmanr(depth, cost[:, s])[0] for s in range(cost.shape[1])])
            beta = np.linalg.lstsq(X, cost, rcond=None)[0]
            resid = cost - X @ beta
            mean = cost.mean(1)
            fit = X @ np.linalg.lstsq(X, mean, rcond=None)[0]
            cell = dict(depth_rho_mean=float(rhos.mean()), depth_positive=int((rhos > 0).sum()), n_spans=len(rhos),
                        depth_sign_p=sign_p(int((rhos > 0).sum()), len(rhos)),
                        mean_cost_depth_rho=float(spearmanr(depth, mean)[0]),
                        mean_cost_depth_r2=float(1 - ((mean - fit) ** 2).sum() / ((mean - mean.mean()) ** 2).sum()),
                        residual=reliability(resid, rng))
            result["cells"][f"{key}|d{delta}"] = cell
    write(OUT / "result_depth.json", result)
    for k, v in result["cells"].items():
        r = v["residual"]
        print(f"{k:12s} depth rho={v['depth_rho_mean']:+.3f} pos={v['depth_positive']}/{v['n_spans']} p={v['depth_sign_p']:.2g} "
              f"R2={v['mean_cost_depth_r2']:.2f} | resid rel={r['reliability']:.3f} half={r['half_rho_mean']:+.3f}")


if __name__ == "__main__":
    main()
