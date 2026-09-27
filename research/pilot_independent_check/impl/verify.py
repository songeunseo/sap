"""Independent re-implementation of the probe reliability pilot analysis (from SPEC.md only)."""
import json, os, time
import numpy as np
import torch
from scipy.stats import spearmanr

torch.set_num_threads(8)
T0 = time.time()
R = "/home/tmluser1/sap/experiments/dlm_probe_reliability_pilot/output"
MAN = "/home/tmluser1/sap/experiments/dlm_context_response50/uniform/mask_manifest.json"
OUT = "/home/tmluser1/sap/research/pilot_independent_check/out"
MASK_ID = 126336
RATES = [40, 45, 48, 52, 55, 60]
BLOCKS = list(range(32))
DELTAS = {2: (48, 52), 5: (45, 55), 10: (40, 60)}
EDGES = {
    "C1": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "C2": [(0, 2), (1, 3), (4, 6), (5, 7)],
    "C4": [(0, 4), (1, 5), (2, 6), (3, 7)],
}

# ---------- bank ----------
bank = json.load(open(f"{R}/bank.json"))
chains = bank["chains"]
assert len(chains) == 64
seq_ids = sorted({c["sequence_index"] for c in chains})
assert seq_ids == list(range(100, 132))
seq_pos = {s: i for i, s in enumerate(seq_ids)}
NS = 32
mask = np.zeros((512, 256), dtype=bool)
query = np.zeros((64, 8), dtype=np.int64)
chain_seq = np.zeros(64, dtype=np.int64)
for k, ch in enumerate(chains):
    chain_seq[k] = seq_pos[ch["sequence_index"]]
    query[k] = ch["query"]
    assert len(ch["nodes"]) == 8
    for node in ch["nodes"]:
        ph = node["phase"]
        ids = np.asarray(node["input_ids"])
        assert ids.shape == (256,)
        mask[8 * k + ph] = ids == MASK_ID
state_chain = np.repeat(np.arange(64), 8)
state_seq = chain_seq[state_chain]
for k in range(64):
    assert mask[8 * k:8 * k + 8][:, query[k]].all()

def per_seq(vals, owner):
    out = np.zeros(NS)
    cnt = np.zeros(NS)
    np.add.at(out, owner, vals)
    np.add.at(cnt, owner, 1)
    return out / cnt

# ---------- pruned counts ----------
man = json.load(open(MAN))["entries"]
assert len(man) == 224
def pruned(b, r):
    tot = 0
    for e in man[7 * b:7 * b + 7]:
        assert e["name"].startswith(f"block_{b:02d}.")
        out_, in_ = e["shape"]
        tot += (in_ * r // 100) * out_   # == floor(in * r/100) exactly (integer arithmetic)
    return tot

# ---------- levels ----------
def load(label):
    t = torch.load(f"{R}/readouts/{label}.pt", map_location="cpu")
    assert t.dtype == torch.float32 and tuple(t.shape) == (512, 256)
    return t.double().numpy()

fD = load("dense")
M = mask.astype(np.float64)
nmask = M.sum(1)
qidx = np.stack([query[state_chain[s]] for s in range(512)])  # [512,8]

def levels(fX):
    e = fX - fD
    e2 = e * e
    # A_q
    eq = np.take_along_axis(e, qidx, axis=1)                  # [512,8]
    Aq_chain = (eq ** 2).reshape(64, 8, 8).mean(axis=(1, 2))
    eqc = eq.reshape(64, 8, 8)                                 # chain, node, query
    Cq_chain = np.zeros(64)
    for sc, edges in EDGES.items():
        v = np.mean([((eqc[:, j] - eqc[:, i]) ** 2).mean(axis=1) for i, j in edges], axis=0)
        Cq_chain += v / 3.0
    # masked metrics
    Am_state = (e2 * M).sum(1) / nmask
    ce = np.logaddexp(0.0, -fX)                                # log(1+exp(-f))
    CEm_state = (ce * M).sum(1) / nmask
    ec = e.reshape(64, 8, 256)
    Mc = mask.reshape(64, 8, 256)
    Cm_chain = np.zeros(64)
    for sc, edges in EDGES.items():
        vals = []
        for i, j in edges:
            both = (Mc[:, i] & Mc[:, j]).astype(np.float64)
            assert (both.sum(1) > 0).all()
            vals.append((((ec[:, j] - ec[:, i]) ** 2) * both).sum(1) / both.sum(1))
        Cm_chain += np.mean(vals, axis=0) / 3.0
    A_q = per_seq(Aq_chain, chain_seq)
    Cpart_q = per_seq(Cq_chain, chain_seq)
    A_m = per_seq(Am_state, state_seq)
    CE_m = per_seq(CEm_state, state_seq)
    C_m = per_seq(Cm_chain, chain_seq)
    return {"A_q": A_q, "Multi_q": A_q + Cpart_q, "Cpart_q": Cpart_q,
            "A_m": A_m, "CE_m": CE_m, "C_m": C_m, "Multi_m": A_m + C_m, "Cpart_m": C_m}

LV = {}
for b in BLOCKS:
    for r in RATES:
        LV[(b, r)] = levels(load(f"b{b:02d}_r{r}"))
LV_uni = levels(load("uniform50"))
LV_dense = levels(fD)

OUT_LEVEL_METRICS = ["A_q", "Multi_q", "A_m", "CE_m", "C_m"]
levels_json = {}
for m in OUT_LEVEL_METRICS:
    for b in BLOCKS:
        for r in RATES:
            levels_json[f"{m}|b{b:02d}|r{r}"] = LV[(b, r)][m].tolist()

# ---------- stats ----------
def twoway(c):
    B, S = c.shape
    rm = c.mean(1); cm = c.mean(0); g = c.mean()
    resid = c - rm[:, None] - cm[None, :] + g
    W = (resid ** 2).sum() / ((B - 1) * (S - 1))
    O = rm.var(ddof=1)
    T = O - W / S
    Tp = max(T, 0.0)
    rel = Tp / (Tp + W / S) if (Tp + W / S) > 0 else None
    rel8 = Tp / (Tp + W / 8) if (Tp + W / 8) > 0 else None
    return dict(rel=rel, rel8=rel8, T=T, W=W, spans_for_0_8=(4 * W / T if T > 0 else None))

def rho(a, b):
    return float(spearmanr(a, b).correlation)

METRICS = ["A_q", "Multi_q", "Cpart_q", "A_m", "CE_m", "Multi_m", "Cpart_m"]
rng = np.random.default_rng(12345)
PERMS = [rng.permutation(NS) for _ in range(500)]
idx = np.arange(32, dtype=np.float64)
X = np.stack([np.ones(32), idx], 1)
stats = {}
for m in METRICS:
    for d, (lo, hi) in DELTAS.items():
        c = np.stack([(LV[(b, hi)][m] - LV[(b, lo)][m]) / (pruned(b, hi) - pruned(b, lo)) for b in BLOCKS])
        tw = twoway(c)
        halves = [rho(c[:, p[:16]].mean(1), c[:, p[16:]].mean(1)) for p in PERMS]
        drho = np.array([rho(idx, c[:, s]) for s in range(NS)])
        rm = c.mean(1)
        coef, *_ = np.linalg.lstsq(X, rm, rcond=None)
        fit = X @ coef
        r2 = 1 - ((rm - fit) ** 2).sum() / ((rm - rm.mean()) ** 2).sum()
        beta, *_ = np.linalg.lstsq(X, c, rcond=None)
        res = c - X @ beta
        tr = twoway(res)
        stats[f"{m}|d{d}"] = {
            "rel": tw["rel"], "rel8": tw["rel8"], "T": tw["T"], "W": tw["W"],
            "spans_for_0.8": tw["spans_for_0_8"],
            "half_rho_mean": float(np.nanmean(halves)),
            "half_rho_nan_count": int(np.isnan(halves).sum()),
            "depth_rho_mean": float(np.nanmean(drho)),
            "depth_positive": int((drho > 0).sum()),
            "meancost_depth_rho": rho(idx, rm),
            "meancost_depth_r2": float(r2),
            "resid_rel": tr["rel"], "resid_rel8": tr["rel8"],
            "resid_T": tr["T"], "resid_W": tr["W"],
            "pruned_delta_b00": int(pruned(0, hi) - pruned(0, lo)),
        }

# ---------- sanity ----------
sanity = {"A_m_mean_over_sequences": {}}
for b in (0, 31):
    d = {f"r{r}": float(LV[(b, r)]["A_m"].mean()) for r in RATES}
    d["uniform50"] = float(LV_uni["A_m"].mean())
    sanity["A_m_mean_over_sequences"][f"b{b:02d}"] = d
mono = []
mono_with_uni = []
for b in BLOCKS:
    v = [LV[(b, r)]["A_m"].mean() for r in RATES]
    mono.append(all(v[i + 1] >= v[i] for i in range(5)))
    w = v[:3] + [LV_uni["A_m"].mean()] + v[3:]
    mono_with_uni.append(all(w[i + 1] >= w[i] for i in range(6)))
sanity["A_m_monotone_fraction"] = float(np.mean(mono))
sanity["A_m_monotone_blocks_count"] = int(np.sum(mono))
sanity["A_m_nonmonotone_blocks"] = [b for b in BLOCKS if not mono[b]]
sanity["supplementary_A_m_monotone_fraction_incl_uniform50_between_48_52"] = float(np.mean(mono_with_uni))
sanity["dense_A_m_mean_should_be_0"] = float(LV_dense["A_m"].mean())
sanity["pruned_counts_b00"] = {f"r{r}": pruned(0, r) for r in RATES}
sanity["pruned_counts_equal_across_blocks"] = all(pruned(b, r) == pruned(0, r) for b in BLOCKS for r in RATES)
sanity["runtime_sec"] = time.time() - T0

json.dump(levels_json, open(f"{OUT}/levels.json", "w"))
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
json.dump(sanity, open(f"{OUT}/sanity.json", "w"), indent=1)
print("done", time.time() - T0)
