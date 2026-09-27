from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path("experiments/dlm_role_exchange_prediction")
RUNTIME = Path("/DATA/tmluser1/sap-dlm-role-exchange-prediction")
TIMESTEPS = (0.1, 0.3, 0.5, 0.7, 0.9)
SEED = 20260912


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def feasible_exchanges(levels, entries):
    rows = []
    for i, level in enumerate(levels):
        for new in (level - 1, level + 1):
            if not 0 <= new < 6:
                continue
            before = int(entries[i]["masks"][level]["pruned"])
            after = int(entries[i]["masks"][new]["pruned"])
            rows.append({"exchange_index": len(rows), "module_index": i,
                         "name": entries[i]["name"], "layer": i // 7,
                         "projection_type": entries[i]["name"].split(".")[-1],
                         "base_level": int(level), "new_level": int(new),
                         "parameter_delta": after - before})
    return rows


def select_bundles(exchanges, per_quartile=8, seed=SEED):
    """Select exact-budget bundles using identity/count only, never damage."""
    rng = np.random.default_rng(seed)
    selected = []
    for layer_group in range(8):
        lo, hi = layer_group * 4, (layer_group + 1) * 4
        pool = [x for x in exchanges if lo <= x["layer"] < hi]
        candidates = []
        for size in (2, 3, 4):
            for combo in itertools.combinations(pool, size):
                if len({x["module_index"] for x in combo}) != size:
                    continue
                if sum(x["parameter_delta"] for x in combo) == 0:
                    candidates.append(tuple(x["exchange_index"] for x in combo))
        chosen = []
        for _trial in range(100):
            order = rng.permutation(len(candidates)); used = set(); attempt = []
            for index in order:
                combo = candidates[int(index)]
                modules = {exchanges[i]["module_index"] for i in combo}
                if modules & used: continue
                attempt.append(combo); used |= modules
                if len(attempt) == per_quartile: break
            if len(attempt) > len(chosen): chosen = attempt
            if len(chosen) == per_quartile: break
        for combo in chosen:
            selected.append({"bundle_index": len(selected), "layer_group": layer_group,
                             "layer_quartile": layer_group // 2,
                             "exchange_indices": list(combo),
                             "parameter_delta": sum(exchanges[i]["parameter_delta"] for i in combo)})
    return selected


def ridge_fit_predict(x_train, y_train, x_test, alpha=1.0):
    x_train = np.asarray(x_train, float); x_test = np.asarray(x_test, float)
    y_train = np.asarray(y_train, float)
    mean = x_train.mean(0); scale = x_train.std(0)
    scale[scale == 0] = 1
    a = (x_train - mean) / scale; b = (x_test - mean) / scale
    design = np.column_stack([np.ones(len(a)), a])
    penalty = np.eye(design.shape[1]) * alpha; penalty[0, 0] = 0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return np.column_stack([np.ones(len(b)), b]) @ beta


def cluster_bootstrap_difference(left, right, documents, quartiles,
                                 resamples=20000, seed=SEED):
    left=np.asarray(left,float); right=np.asarray(right,float)
    documents=np.asarray(documents); quartiles=np.asarray(quartiles)
    docs=np.unique(documents); qs=np.unique(quartiles); rng=np.random.default_rng(seed)
    observed=float(np.mean(left-right)); cells=np.empty((len(docs),len(qs)))
    for di,d in enumerate(docs):
        for qi,q in enumerate(qs):
            mask=(documents==d)&(quartiles==q)
            if not mask.any(): raise RuntimeError("empty document/quartile bootstrap cell")
            cells[di,qi]=(left[mask]-right[mask]).mean()
    sampled=[]
    for _ in range(resamples):
        di=rng.integers(0,len(docs),len(docs)); qi=rng.integers(0,len(qs),len(qs))
        sampled.append(cells[np.ix_(di,qi)].mean())
    return {"mean_error_difference":observed,
            "bootstrap_ci":np.quantile(sampled,[.0083333333,.9916666667]).tolist(),
            "resamples":resamples,"seed":seed}
