"""A+C extension of the pilot analysis (pre-registered 2026-09-27 before GPU completion; CPU only).

Adds all-masked-position response readouts to the frozen run.py analysis without touching the GPU code hash.
C_m(edge i<j) = mean over positions masked in BOTH nodes of ((f_S-f_D)_j - (f_S-f_D)_i)^2; with monotone reveal
these are the positions still masked at the later node (query positions included). Multi_m = A_m + mean(C1,C2,C4)_m.
Cpart_m = Multi_m - A_m. Same span units, deltas, reliability estimator and seed as run.py.

A+C gate (per readout family q = query-8, m = all masked), for each delta:
  signal:   Cpart reliability at 32 spans >= 0.8 and mean 16/16 split-half rho >= 0.6
  distinct: RMS between ideal A and Multi rank-rate allocations exceeds the 97.5% quantile of the
            A-vs-A span-bootstrap RMS (same rule as the 2026-09-27 CPU audit)
  -> both: restart block-level A+C with that setting (next: reachability on a fresh bank)
  -> signal only: C is reliable but only rescales A at block level
  -> neither: block-level C is not measurable; move C to a finer unit / reconstruction objective
"""
from __future__ import annotations

import json

import numpy as np

from experiments.dlm_multiscale_ac50.core import EDGES
from experiments.dlm_probe_reliability_pilot.run import (DELTAS, MASK_ID, OUT, RATES, SEED, UNIFORM_MANIFEST,
                                                         read, reliability, sequence_levels, sha, validate, write)

SCALES = ("C1", "C2", "C4")


def response_levels(values, dense, bank):
    """Per-sequence A_m, C_m (mean of C1/C2/C4) and Multi_m for one condition."""
    n_chains = len(bank["chains"])
    e = (values - dense).reshape(n_chains, 8, -1)
    masked = np.asarray([[[t == MASK_ID for t in n["input_ids"]] for n in c["nodes"]] for c in bank["chains"]])
    a = ((e ** 2) * masked).sum(2) / masked.sum(2)          # [chain, node]
    rows = []
    for k in range(n_chains):
        c_scale = []
        for scale in SCALES:
            per_edge = []
            for i, j in EDGES[scale]:
                both = masked[k, i] & masked[k, j]
                per_edge.append(float(((e[k, j] - e[k, i]) ** 2)[both].mean()))
            c_scale.append(np.mean(per_edge))
        rows.append((float(a[k].mean()), float(np.mean(c_scale))))
    seq = [c["sequence_index"] for c in bank["chains"]]
    out = {"A_m": [], "C_m": [], "Multi_m": []}
    for s in sorted(set(seq)):
        idx = [k for k, v in enumerate(seq) if v == s]
        am, cm = np.mean([rows[k][0] for k in idx]), np.mean([rows[k][1] for k in idx])
        out["A_m"].append(float(am)); out["C_m"].append(float(cm)); out["Multi_m"].append(float(am + cm))
    return out


def rank_rates(scores):
    from scipy.stats import rankdata
    rank = (rankdata(np.asarray(scores, dtype=np.float64), method="average") - 1) / (len(scores) - 1)
    return .5 - .10 * (rank - rank.mean())


def distinctness(cost_a, cost_multi, rng, draws=2000):
    ns = cost_a.shape[1]
    ideal = float(np.sqrt(((rank_rates(cost_multi.mean(1)) - rank_rates(cost_a.mean(1))) ** 2).mean()) * 100)
    idx = rng.integers(0, ns, size=(draws, ns))
    rates_a = np.asarray([rank_rates(cost_a[:, d].mean(1)) for d in idx])
    noise = np.sqrt(((rates_a[: draws // 2] - rates_a[draws // 2:]) ** 2).mean(1)) * 100
    q975 = float(np.quantile(noise, .975))
    return dict(ideal_multi_minus_a_rms_pp=ideal, a_vs_a_rms_pp_median=float(np.median(noise)),
                a_vs_a_rms_pp_q975=q975, distinct=ideal > q975)


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
            label = f"b{b:02d}_r{round(r * 100)}"
            if read(OUT / "readouts" / f"{label}.json")["sha256"] != sha(OUT / "readouts" / f"{label}.pt"):
                raise RuntimeError(f"readout changed: {label}")
            v = load(label)
            lv = sequence_levels(v, dense, bank)
            lv.update(response_levels(v, dense, bank))
            lv["Cpart_q"] = list(np.asarray(lv["Multi_q"]) - np.asarray(lv["A_q"]))
            lv["Cpart_m"] = lv["C_m"]
            levels[(b, r)] = lv
    pruned = lambda b, r: sum(int(x["shape"][1] * r) * x["shape"][0] for x in refs[b * 7:(b + 1) * 7])
    rng = np.random.default_rng(SEED + 1)
    result = dict(config_sha256=sha(OUT / "config.json"), code_sha256=sha(__file__), cells={}, gate={})
    for delta, (lo, hi) in DELTAS.items():
        cost = {key: np.asarray([(np.asarray(levels[(b, hi)][key]) - np.asarray(levels[(b, lo)][key]))
                                 / (pruned(b, hi) - pruned(b, lo)) for b in range(32)])
                for key in ("A_q", "Multi_q", "Cpart_q", "A_m", "Multi_m", "Cpart_m")}
        for key, c in cost.items():
            result["cells"][f"{key}|d{delta}"] = reliability(c, rng)
        for fam in ("q", "m"):
            cp = result["cells"][f"Cpart_{fam}|d{delta}"]
            signal = cp["reliability"] >= 0.8 and cp["half_rho_mean"] >= 0.6
            dist = distinctness(cost[f"A_{fam}"], cost[f"Multi_{fam}"], rng)
            dist["cpart_vs_a_spearman"] = float(spearmanr(cost[f"Cpart_{fam}"].mean(1), cost[f"A_{fam}"].mean(1))[0])
            verdict = ("restart_block_AC" if signal and dist["distinct"] else
                       "C_rescales_A" if signal else "C_not_measurable_at_block")
            result["gate"][f"{fam}|d{delta}"] = dict(signal=signal, **dist, verdict=verdict)
    write(OUT / "result_c.json", result)
    for k, v in result["cells"].items():
        print(f"{k:14s} rel32={v['reliability']:.3f} rel8={v['reliability_8']:.3f} half_rho={v['half_rho_mean']:+.3f} n80={v['spans_for_0_8']}")
    print(json.dumps(result["gate"], indent=1))


if __name__ == "__main__":
    main()
