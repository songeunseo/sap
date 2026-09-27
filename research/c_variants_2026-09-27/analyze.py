"""Magnitude-decoupled response (C) variants on the probe pilot readouts (CPU; pre-registered in
Obsidian Hypotheses/2026-09-27-C-Formula-Redesign). For every block and rate 40/45/55/60 compute per-sequence
levels of: A_m (reference), C_m (current), C_rel, R_share, C_dir, C_strong, Flip. Costs = J(hi) - J(lo) for
delta 10 (primary) and 5; reliability, Spearman with A_m cost, reliability of residual after per-span linear
regression on A_m cost; depth rho as description.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.dlm_multiscale_ac50.core import EDGES
from experiments.dlm_probe_reliability_pilot.run import MASK_ID, OUT, read, reliability

HERE = Path(__file__).resolve().parent
SCALES = ("C1", "C2", "C4")
STRONG = 1.0
KEYS = ("A_m", "C_m", "C_rel", "R_share", "C_dir", "C_strong", "Flip")
RATES = (40, 45, 55, 60)


def levels(values, dense, bank, masked, seq):
    n = len(bank["chains"])
    fs, fd = values.reshape(n, 8, -1), dense.reshape(n, 8, -1)
    e = fs - fd
    out = {k: np.zeros(n) for k in KEYS}
    a = ((e ** 2) * masked).sum(2) / masked.sum(2)
    out["A_m"] = a.mean(1)
    for k in range(n):
        acc = {x: [] for x in KEYS[1:]}
        for scale in SCALES:
            per = {x: [] for x in KEYS[1:]}
            for i, j in EDGES[scale]:
                both = masked[k, i] & masked[k, j]
                de = (e[k, j] - e[k, i])[both]
                dd = (fd[k, j] - fd[k, i])[both]
                ds = (fs[k, j] - fs[k, i])[both]
                per["C_m"].append(np.mean(de ** 2))
                per["C_rel"].append(np.sum(de ** 2) / max(np.sum(dd ** 2), 1e-12))
                per["R_share"].append(np.sum(de ** 2) / max(np.sum(e[k, i][both] ** 2 + e[k, j][both] ** 2), 1e-12))
                per["C_dir"].append(1 - np.corrcoef(ds, dd)[0, 1] if dd.std() > 0 and ds.std() > 0 else np.nan)
                strong = np.abs(dd) >= STRONG
                per["C_strong"].append(np.mean(de[strong] ** 2) if strong.any() else np.nan)
                per["Flip"].append(np.mean(np.sign(ds[strong]) != np.sign(dd[strong])) if strong.any() else np.nan)
            for x in per:
                acc[x].append(np.nanmean(per[x]))
        for x in acc:
            out[x][k] = np.nanmean(acc[x])
    return {x: np.array([v[seq == s].mean() for s in np.unique(seq)]) for x, v in out.items()}


def resid_on(cost, ref):
    r = np.empty_like(cost)
    for s in range(cost.shape[1]):
        X = np.c_[np.ones(cost.shape[0]), ref[:, s]]
        r[:, s] = cost[:, s] - X @ np.linalg.lstsq(X, cost[:, s], rcond=None)[0]
    return r


def main():
    import torch
    bank = read(OUT / "bank.json")
    masked = np.asarray([[[t == MASK_ID for t in n["input_ids"]] for n in c["nodes"]] for c in bank["chains"]])
    seq = np.asarray([c["sequence_index"] for c in bank["chains"]])
    load = lambda label: torch.load(OUT / "readouts" / f"{label}.pt", weights_only=False).double().numpy()
    dense = load("dense")
    L = {k: np.zeros((32, len(RATES), 32)) for k in KEYS}
    for b in range(32):
        for j, r in enumerate(RATES):
            lv = levels(load(f"b{b:02d}_r{r}"), dense, bank, masked, seq)
            for k in KEYS:
                L[k][b, j] = lv[k]
    np.savez(HERE / "levels.npz", **L)
    rng = np.random.default_rng(20260927)
    t = np.arange(32)
    result = {}
    for delta, (lo, hi) in {10: (0, 3), 5: (1, 2)}.items():
        A = L["A_m"][:, hi] - L["A_m"][:, lo]
        for k in KEYS:
            c = L[k][:, hi] - L[k][:, lo]
            rel = reliability(c, rng, splits=300)
            res = reliability(resid_on(c, A), rng, splits=300) if k != "A_m" else None
            rho_a = float(spearmanr(c.mean(1), A.mean(1))[0])
            cell = dict(reliability=rel["reliability"], half_rho=rel["half_rho_mean"], rho_with_A=rho_a,
                        resid_on_A_reliability=None if res is None else res["reliability"],
                        resid_on_A_half_rho=None if res is None else res["half_rho_mean"],
                        depth_rho=float(spearmanr(t, c.mean(1))[0]),
                        passes=bool(k != "A_m" and rel["reliability"] >= 0.8 and abs(rho_a) <= 0.8
                                    and res["reliability"] >= 0.6))
            result[f"{k}|d{delta}"] = cell
            print(f"{k:9s} d{delta:<2d} rel={cell['reliability']:.3f} half={cell['half_rho']:+.2f} rho_A={rho_a:+.3f} "
                  f"residA_rel={'-' if res is None else format(res['reliability'], '.3f')} depth={cell['depth_rho']:+.2f} pass={cell['passes']}")
    (HERE / "result.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
