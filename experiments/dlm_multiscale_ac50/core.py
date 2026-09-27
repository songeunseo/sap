"""CPU-only banks and objectives. No model, torch, or CUDA imports."""
from __future__ import annotations

import itertools
import random

import numpy as np

from .artifacts import ARMS, digest

EDGES = {
    "C1": [(0, 1), (2, 3), (4, 5), (6, 7)],
    "C2": [(0, 2), (1, 3), (4, 6), (5, 7)],
    "C4": [(0, 4), (1, 5), (2, 6), (3, 7)],
    "C_path": list(zip(range(7), range(1, 8))),
    "C_all": list(itertools.combinations(range(8), 2)),
}


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:16], 16)


def clean_sequences(source):
    rows = {}
    for state in source["states"]:
        ids = state["clean_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        index = state["sequence_index"]
        if index in rows and rows[index] != ids:
            raise ValueError("Inconsistent clean sequence in source")
        if len(ids) != 256 or source["mask_id"] in ids:
            raise ValueError("Expected 256 clean, non-mask tokens")
        rows[index] = ids
    return rows


def make_bank(source, settings, split):
    sequences = clean_sequences(source)
    if len(sequences) != settings["sequences_per_split"]:
        raise ValueError("Unexpected number of clean spans")
    seed = settings["calibration_seed" if split == "calibration" else "diagnostic_seed"]
    probabilities = settings["visibility_probabilities"]
    if len(probabilities) != 8 or any(not 0 < p < 1 for p in probabilities):
        raise ValueError("Eight interior phase probabilities required")
    if sorted(set(probabilities)) != probabilities:
        raise ValueError("Phases must be strictly increasing")
    chains = []
    for index, clean in sorted(sequences.items()):
        qseed = seed_for(seed, "query", index)
        query = sorted(random.Random(qseed).sample(range(len(clean)), settings["query_positions_per_sequence"]))
        eligible = [i for i in range(len(clean)) if i not in query]
        for chain_index in range(settings["chains_per_sequence"]):
            useed = seed_for(seed, "context", index, chain_index)
            rng = random.Random(useed)
            uniforms = [rng.random() for _ in eligible]
            nodes = []
            for phase, p in enumerate(probabilities):
                visible = [i for i, u in zip(eligible, uniforms) if u <= p]
                noisy = [source["mask_id"]] * len(clean)
                for i in visible:
                    noisy[i] = clean[i]
                nodes.append(dict(phase=phase, p=p, input_ids=noisy, visible=visible,
                                  masked_count=len(clean)-len(visible),
                                  actual_mask_fraction=1-len(visible)/len(clean)))
            chains.append(dict(sequence_index=index, chain_index=chain_index, query=query,
                               gold=[clean[i] for i in query], query_seed=qseed, context_seed=useed,
                               eligible=eligible, uniforms=uniforms, nodes=nodes))
    bank = dict(split=split, mask_id=source["mask_id"], seed=seed, chains=chains,
                states=sum(len(c["nodes"]) for c in chains), settings=settings)
    validate_bank(bank)
    return bank


def validate_bank(bank):
    previous_keys = set()
    sequence_queries = {}
    for chain in bank["chains"]:
        key = (chain["sequence_index"], chain["chain_index"])
        if key in previous_keys:
            raise ValueError("Duplicate chain")
        previous_keys.add(key)
        query = chain["query"]
        index = chain["sequence_index"]
        if sequence_queries.setdefault(index, query) != query:
            raise ValueError("Query changed between chains")
        prior = set()
        for node in chain["nodes"]:
            visible = set(node["visible"])
            expected = {i for i, u in zip(chain["eligible"], chain["uniforms"]) if u <= node["p"]}
            if visible != expected or not prior <= visible or visible.intersection(query):
                raise ValueError("Invalid product monotone coupling")
            if any(node["input_ids"][i] != bank["mask_id"] for i in query):
                raise ValueError("Query was revealed")
            if sum(x == bank["mask_id"] for x in node["input_ids"]) != node["masked_count"]:
                raise ValueError("Incorrect mask count")
            prior = visible


def nodes(bank):
    for chain in bank["chains"]:
        for node in chain["nodes"]:
            yield chain, node


def metrics(prediction, teacher, bank):
    p, d = np.asarray(prediction, dtype=np.float64), np.asarray(teacher, dtype=np.float64)
    q = len(bank["chains"][0]["query"])
    shape = (bank["states"], q)
    if p.shape != shape or d.shape != shape or not np.isfinite(p).all() or not np.isfinite(d).all():
        raise ValueError(f"Expected finite readouts of shape {shape}")
    p, d = p.reshape(-1, 8, q), d.reshape(-1, 8, q)
    rows = []
    for chain, pp, dd in zip(bank["chains"], p, d, strict=True):
        e = pp-dd
        row = dict(A=float(np.mean(e**2)))
        for name, edges in EDGES.items():
            row[name] = float(np.mean([(e[j]-e[i])**2 for i, j in edges]))
        row.update(Short=row["A"]+row["C1"], Path=row["A"]+row["C_path"],
                   All=row["A"]+row["C_all"], Multi=row["A"]+(row["C1"]+row["C2"]+row["C4"])/3,
                   query_CE=float(np.logaddexp(0, -pp).mean()))
        flips, near, count = 0, 0, 0
        for i, j in EDGES["C_all"]:
            dense_delta, sparse_delta = dd[j]-dd[i], pp[j]-pp[i]
            valid = np.abs(dense_delta) > 1e-6
            flips += int(np.sum((dense_delta*sparse_delta < 0) & valid))
            near += int(np.sum(~valid))
            count += q
        row["response_sign_flip_rate"] = flips/(count-near) if count != near else 0.0
        row["near_zero_dense_response_fraction"] = near/count
        rows.append(dict(sequence_index=chain["sequence_index"], chain_index=chain["chain_index"],
                         metrics=row, phase_mse=np.mean(e**2, axis=1).tolist()))
    by_sequence = {}
    for row in rows:
        by_sequence.setdefault(row["sequence_index"], []).append(row)
    sequence_rows = [dict(sequence_index=i,
                         metrics={k:float(np.mean([r["metrics"][k] for r in group])) for k in group[0]["metrics"]},
                         phase_mse=np.mean([r["phase_mse"] for r in group], axis=0).tolist())
                     for i, group in sorted(by_sequence.items())]
    return dict(mean={k:float(np.mean([r["metrics"][k] for r in sequence_rows])) for k in rows[0]["metrics"]},
                sequences=sequence_rows, chains=rows, sign_near_zero_threshold=1e-6)


def rank_rates(scores):
    from scipy.stats import rankdata
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (32,) or not np.isfinite(values).all():
        raise ValueError("Need 32 finite signed costs")
    rank = (rankdata(values, method="average")-1)/31
    return .5-.10*(rank-rank.mean())


def marginal_cost(low, high, low_count, high_count):
    if high_count <= low_count:
        raise ValueError("Probe counts are reversed/equal")
    return {k:(high["mean"][k]-low["mean"][k])/(high_count-low_count) for k in ARMS}


def aliases(allocations):
    seen, out = {}, {}
    for arm in ("Multi", "Short", "Path", "All", "A"):
        key = tuple(allocations[arm]["row_counts"])
        out[arm] = seen.setdefault(key, arm)
    return out


def paired(reference, candidate):
    import math
    if len(reference) != len(candidate) or not reference:
        raise ValueError("Paired samples must have equal positive size")
    pairs = list(zip(reference, candidate, strict=True))
    gained = sum(not a and b for a, b in pairs)
    lost = sum(a and not b for a, b in pairs)
    n = gained+lost
    p = min(1.0, 2*sum(math.comb(n, i) for i in range(min(gained, lost)+1))/2**n) if n else 1.0
    return dict(gained=gained, lost=lost, net=gained-lost, total=len(pairs), exact_mcnemar_p=p)


def holm(comparisons):
    maximum = 0.0
    for rank, key in enumerate(sorted(comparisons, key=lambda k: comparisons[k]["exact_mcnemar_p"])):
        maximum = max(maximum, min(1.0, (len(comparisons)-rank)*comparisons[key]["exact_mcnemar_p"]))
        comparisons[key]["holm_p"] = maximum
    return comparisons
