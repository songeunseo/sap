"""Trust audit of the 50% WikiText NELBO allocation baselines (experiments/dlm_ppl50). CPU only, read-only.

1. NELBO re-aggregation from 551 per-chunk checkpoints vs reported summary.
2. Shared Monte Carlo draws: same chunk ids and same per-chunk mask_sha256 across methods.
3. Mask integrity: file sha256 vs manifest, and zero counts from the stored bits (Wanda family) or the stored
   weights (EvoPress level files) vs manifest pruned counts.
4. Early-8 / late-8 block sparsity recomputed from the counted zeros.
5. Article-level paired bootstrap of Delta NELBO vs Uniform (corpus uncertainty, not only MC noise), Holm.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/home/tmluser1/sap/experiments/dlm_ppl50")
OUT = Path(__file__).resolve().parent
METHODS = ["uniform", "dsa", "evopress", "owl", "lsa_layer", "lsa_projection", "alpha", "dlp"]
SEED, DRAWS = 20260927, 10000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def chunks(method):
    rows = {}
    for f in glob.glob(str(ROOT / method / "validation" / "blocks" / "*.json")):
        r = json.loads(Path(f).read_text())
        rows[r["block_id"]] = r
    return rows


def count_wanda(entry):
    meta = entry["selected_mask"]
    if sha(meta["path"]) != meta["file_sha256"]:
        return entry["name"], None, "file_sha_mismatch"
    import torch
    payload = torch.load(meta["path"], map_location="cpu", weights_only=False)
    zeros = int(np.unpackbits(np.frombuffer(payload["bits"], dtype=np.uint8)).sum())
    return entry["name"], zeros, "ok"


def count_evopress(entry):
    import torch
    if sha(entry["path"]) != entry["file_sha256"]:
        return entry["name"], None, "file_sha_mismatch"
    w = torch.load(entry["path"], map_location="cpu", weights_only=False)
    if isinstance(w, dict):
        w = next(v for v in w.values() if hasattr(v, "shape"))
    return entry["name"], int((w == 0).sum()), "ok"


def main():
    report = dict(nelbo={}, shared_draws={}, masks={}, depth={}, bootstrap={})
    data = {m: chunks(m) for m in METHODS}
    ref_ids = sorted(data["uniform"])
    status = json.loads((ROOT / "status.json").read_text())
    for m in METHODS:
        rows = data[m]
        tok = np.array([rows[i]["tokens"] for i in ref_ids], float)
        nel = np.array([rows[i]["token_nelbo"] for i in ref_ids], float)
        agg = float((tok * nel).sum() / tok.sum())
        report["nelbo"][m] = dict(chunks=len(rows), tokens=int(tok.sum()), recomputed=agg,
                                  reported=status[m]["summary"]["token_nelbo"],
                                  abs_diff=abs(agg - status[m]["summary"]["token_nelbo"]),
                                  mc_samples_all_128=all(rows[i]["mc_samples"] == 128 for i in ref_ids))
        report["shared_draws"][m] = dict(same_ids=sorted(rows) == ref_ids,
                                         same_mask_sha=all(rows[i]["mask_sha256"] == data["uniform"][i]["mask_sha256"]
                                                           for i in ref_ids))
    # masks
    with ProcessPoolExecutor(max_workers=16) as pool:
        for m in METHODS:
            man = json.loads((ROOT / m / "mask_manifest.json").read_text())
            entries = man["entries"]
            fn = count_evopress if m == "evopress" else count_wanda
            results = dict((n, (z, s)) for n, z, s in pool.map(fn, entries))
            bad = [n for n, (z, s) in results.items() if s != "ok"]
            per_block_z, per_block_w = np.zeros(32), np.zeros(32)
            mism = 0
            for e in entries:
                b = int(e["name"][6:8])
                z = results[e["name"]][0]
                expected = e["actual_zeros"] if m == "evopress" else e["selected_mask"]["pruned"]
                mism += int(z != expected)
                per_block_z[b] += z or 0
                per_block_w[b] += e["weights"]
            rate = per_block_z / per_block_w
            report["masks"][m] = dict(files=len(entries), file_or_load_failures=bad, count_mismatches=mism,
                                      total_zeros=int(per_block_z.sum()), manifest_pruned=man["pruned"],
                                      global_sparsity=float(per_block_z.sum() / per_block_w.sum()))
            report["depth"][m] = dict(early8=float(100 * per_block_z[:8].sum() / per_block_w[:8].sum()),
                                      late8=float(100 * per_block_z[-8:].sum() / per_block_w[-8:].sum()),
                                      per_block=rate.tolist())
            print(m, report["masks"][m]["count_mismatches"], report["depth"][m]["early8"], report["depth"][m]["late8"], flush=True)
    # article-level paired bootstrap vs uniform
    art = [re.match(r"validation:article(\d+):", i).group(1) for i in ref_ids]
    arts = sorted(set(art), key=int)
    idx = {a: [k for k, x in enumerate(art) if x == a] for a in arts}
    tok = np.array([data["uniform"][i]["tokens"] for i in ref_ids], float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(arts), size=(DRAWS, len(arts)))
    a_tok = np.array([tok[idx[a]].sum() for a in arts])
    pvals = {}
    for m in METHODS[1:]:
        d = np.array([data[m][i]["token_nelbo"] - data["uniform"][i]["token_nelbo"] for i in ref_ids])
        a_sum = np.array([(tok[idx[a]] * d[idx[a]]).sum() for a in arts])
        point = a_sum.sum() / a_tok.sum()
        boot = a_sum[draws].sum(1) / a_tok[draws].sum(1)
        p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        pvals[m] = p
        report["bootstrap"][m] = dict(delta_nelbo=float(point), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                                      boot_p_two_sided=float(p), articles=len(arts),
                                      articles_better=int((a_sum < 0).sum()), articles_worse=int((a_sum > 0).sum()))
    order = sorted(pvals, key=pvals.get)
    running = 0.0
    for k, m in enumerate(order):
        running = max(running, min(1.0, (len(order) - k) * pvals[m]))
        report["bootstrap"][m]["holm_p"] = running
    slope = {m: report["depth"][m]["late8"] - report["depth"][m]["early8"] for m in METHODS}
    from scipy.stats import spearmanr
    report["slope_vs_nelbo_spearman"] = float(spearmanr([slope[m] for m in METHODS],
                                                        [report["nelbo"][m]["recomputed"] for m in METHODS])[0])
    report["slope"] = slope
    (OUT / "result.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("nelbo", "shared_draws", "bootstrap", "slope", "slope_vs_nelbo_spearman")}, indent=1))


if __name__ == "__main__":
    main()
