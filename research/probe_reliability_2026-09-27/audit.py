"""CPU audit: reliability of multiscale A+C block probes (C restart gate a). No model forwards.

Per-sequence marginal cost = (52% metric - 48% metric) / (pruned difference), exactly as
experiments.dlm_multiscale_ac50.analysis.allocate. Units are the 8 calibration spans.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.dlm_multiscale_ac50.core import rank_rates

SRC = Path("/home/tmluser1/sap/experiments/dlm_multiscale_ac50/output")
OUT = Path(__file__).resolve().parent
SEED, DRAWS = 20260927, 2000
ARMS = ("A", "Multi")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    alloc = json.loads((SRC / "allocation.json").read_text())
    per_seq = {arm: np.zeros((32, 8)) for arm in ARMS}
    for block in range(32):
        path = SRC / "probes" / f"block{block:02d}.json"
        if alloc["probe_sources"][str(path)] != sha(path):
            raise RuntimeError(f"probe SHA mismatch {path}")
        row = json.loads(path.read_text())
        low, high = row["conditions"]["0.48"], row["conditions"]["0.52"]
        dn = high["pruned"] - low["pruned"]
        for s, (l, h) in enumerate(zip(low["metrics"]["sequences"], high["metrics"]["sequences"], strict=True)):
            if l["sequence_index"] != h["sequence_index"]:
                raise RuntimeError("sequence order mismatch")
            for arm in ARMS:
                per_seq[arm][block, s] = (h["metrics"][arm] - l["metrics"][arm]) / dn
        for arm in ARMS:
            if not np.isclose(per_seq[arm][block].mean(), row["costs"][arm], rtol=1e-9, atol=0):
                raise RuntimeError(f"cost reconstruction failed block {block} {arm}")
    for arm in ARMS:
        if not np.allclose(per_seq[arm].mean(1), alloc["allocations"][arm]["scores"], rtol=1e-9, atol=0):
            raise RuntimeError(f"allocation score mismatch {arm}")
    per_seq["Cpart"] = per_seq["Multi"] - per_seq["A"]
    return alloc, per_seq


def split_half(per_seq):
    out = {}
    splits = [c for c in itertools.combinations(range(8), 4) if 0 in c]  # 35 unordered 4/4 splits
    for key, values in per_seq.items():
        rhos = [spearmanr(values[:, list(a)].mean(1), values[:, [i for i in range(8) if i not in a]].mean(1))[0]
                for a in splits]
        r = float(np.mean(rhos))
        sb = 2 * r / (1 + r) if r > -1 else float("nan")
        need = (0.8 * (1 - sb)) / (sb * (1 - 0.8)) * 8 if 0 < sb < 1 else float("nan")
        out[key] = dict(n_splits=len(splits), half_rho_mean=r, half_rho_min=float(np.min(rhos)),
                        half_rho_max=float(np.max(rhos)), spearman_brown_8=sb, spans_for_0_8=need)
    return out


def bootstrap(per_seq, alloc):
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, 8, size=(DRAWS, 8))
    obs = np.asarray(alloc["allocations"]["Multi"]["rates"]) - np.asarray(alloc["allocations"]["A"]["rates"])
    ideal_obs = rank_rates(per_seq["Multi"].mean(1)) - rank_rates(per_seq["A"].mean(1))
    diffs, rates_a = [], []
    for d in draws:
        ra = rank_rates(per_seq["A"][:, d].mean(1))
        rm = rank_rates(per_seq["Multi"][:, d].mean(1))
        diffs.append(rm - ra); rates_a.append(ra)
    diffs, rates_a = np.asarray(diffs), np.asarray(rates_a)
    lo, hi = np.quantile(diffs, [.025, .975], axis=0)
    l2 = np.sqrt((diffs ** 2).mean(1)) * 100
    # Noise reference: rms distance between two independent bootstrap A allocations.
    a_noise = np.sqrt(((rates_a[: DRAWS // 2] - rates_a[DRAWS // 2:]) ** 2).mean(1)) * 100
    return dict(seed=SEED, draws=DRAWS,
                realized_mean_abs_diff_pp=float(np.abs(obs).mean() * 100),
                ideal_rms_diff_pp=float(np.sqrt((ideal_obs ** 2).mean()) * 100),
                blocks_ci_excludes_zero=int(((lo > 0) | (hi < 0)).sum()),
                rms_diff_pp_ci95=[float(np.quantile(l2, .025)), float(np.quantile(l2, .975))],
                A_vs_A_bootstrap_rms_pp_median=float(np.median(a_noise)),
                A_vs_A_bootstrap_rms_pp_ci95=[float(np.quantile(a_noise, .025)), float(np.quantile(a_noise, .975))],
                A_rate_sd_pp_mean=float(rates_a.std(0).mean() * 100))


def variance_components():
    """One-way random effects: between-block variance of 8-span mean cost vs sampling noise of that mean.
    Also reports raw 48%-level and query_CE cost as sanity references (not allocation inputs)."""
    keys = ("A", "Multi", "query_CE")
    diff = {k: np.zeros((32, 8)) for k in keys}
    level48 = np.zeros((32, 8))
    for block in range(32):
        row = json.loads((SRC / "probes" / f"block{block:02d}.json").read_text())
        low, high = row["conditions"]["0.48"], row["conditions"]["0.52"]
        for s, (l, h) in enumerate(zip(low["metrics"]["sequences"], high["metrics"]["sequences"], strict=True)):
            level48[block, s] = l["metrics"]["A"]
            for k in keys:
                diff[k][block, s] = h["metrics"][k] - l["metrics"][k]
    diff["Cpart"] = diff["Multi"] - diff["A"]
    diff["A_level48"] = level48
    # Two-way block x sequence: sequence main effects shift every block equally and do not affect ranking,
    # so noise is the block-by-sequence residual.
    out = {}
    for k, d in diff.items():
        resid = d - d.mean(1, keepdims=True) - d.mean(0, keepdims=True) + d.mean()
        within = float((resid ** 2).sum() / (31 * 7))
        obs = float(d.mean(1).var(ddof=1))
        true = obs - within / 8
        out[k] = dict(between_block_var=obs, residual_var_mean8=within / 8, est_true_var=true,
                      reliability_8=max(true, 0.0) / obs,
                      spans_for_0_8=(4 * within / true) if true > 0 else None)
    return out


def main():
    alloc, per_seq = load()
    corr = {f"{a}~{b}": float(spearmanr(per_seq[a].mean(1), per_seq[b].mean(1))[0])
            for a, b in (("A", "Multi"), ("A", "Cpart"))}
    neg = {k: int((v.mean(1) < 0).sum()) for k, v in per_seq.items()}
    result = dict(source_allocation_sha256=sha(SRC / "allocation.json"),
                  full_rank_corr=corr, negative_costs=neg,
                  split_half=split_half(per_seq), variance_components=variance_components(),
                  bootstrap=bootstrap(per_seq, alloc))
    b = result["bootstrap"]
    sh = result["split_half"]["Cpart"]["spearman_brown_8"]
    within_noise = b["ideal_rms_diff_pp"] <= b["A_vs_A_bootstrap_rms_pp_ci95"][1]
    result["gate_a"] = dict(rule="hold C if Cpart SB<0.5 and ideal Multi-A rms <= 97.5% quantile of A-vs-A bootstrap rms",
                            cpart_sb=sh, multi_minus_a_within_noise=bool(within_noise),
                            blocks_excluding_zero=b["blocks_ci_excludes_zero"],
                            hold_C=bool(sh < 0.5 and within_noise))
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
