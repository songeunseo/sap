"""CPU-only paired-response metrics and exact allocation inputs."""
from __future__ import annotations

import numpy as np

from experiments.dlm_multiscale_ac50.core import EDGES, metrics as old_metrics

ARMS = ("A", "Multi", "Cross", "CrossMatched")
SCALES = ("C1", "C2", "C4")


def response_metrics(prediction, teacher, bank):
    """Mean over spans, both chains, same queries, then the four edges per scale."""
    p = np.asarray(prediction, dtype=np.float64)
    d = np.asarray(teacher, dtype=np.float64)
    q = len(bank["chains"][0]["query"])
    n = bank["states"] // 16
    if p.shape != (bank["states"], q) or d.shape != p.shape or n * 16 != bank["states"]:
        raise ValueError("Readout shape differs from the paired bank")
    if not np.isfinite(p).all() or not np.isfinite(d).all():
        raise ValueError("Nonfinite readout")
    if [c["sequence_index"] for c in bank["chains"]] != [i for i in sorted({c["sequence_index"] for c in bank["chains"]}) for _ in range(2)]:
        raise ValueError("Chains are not span-major pairs")
    for a, b in zip(bank["chains"][0::2], bank["chains"][1::2], strict=True):
        if a["chain_index"] != 0 or b["chain_index"] != 1 or a["query"] != b["query"] or a["gold"] != b["gold"]:
            raise ValueError("Cross-chain pair changes span, query, or gold")
    residual = (p - d).reshape(n, 2, 8, q)
    rows = []
    for s in range(n):
        e = residual[s]
        item = {"sequence_index": bank["chains"][2*s]["sequence_index"], "A": float(np.mean(e**2))}
        for name in SCALES:
            edges = EDGES[name]
            item[f"{name}_natural"] = float(np.mean([(e[:, j] - e[:, i])**2 for i, j in edges]))
            item[f"{name}_cross"] = float(np.mean([(e[1, j] - e[0, i])**2 for i, j in edges] +
                                                  [(e[0, j] - e[1, i])**2 for i, j in edges]))
        item["C_natural"] = float(np.mean([item[f"{x}_natural"] for x in SCALES]))
        item["C_cross"] = float(np.mean([item[f"{x}_cross"] for x in SCALES]))
        item["Multi"] = item["A"] + item["C_natural"]
        item["Cross"] = item["A"] + item["C_cross"]
        rows.append(item)
    keys = ["A", "C_natural", "C_cross", "Multi", "Cross"] + [f"{s}_{kind}" for s in SCALES for kind in ("natural", "cross")]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    legacy = old_metrics(prediction, teacher, bank)
    old_by_span = {r["sequence_index"]: r["metrics"] for r in legacy["sequences"]}
    for item in rows:
        old_row = old_by_span[item["sequence_index"]]
        item["query_CE"] = old_row["query_CE"]
        item["response_sign_flip_rate"] = old_row["response_sign_flip_rate"]
    means["query_CE"] = legacy["mean"]["query_CE"]
    means["response_sign_flip_rate"] = legacy["mean"]["response_sign_flip_rate"]
    # Old per-chain aggregation is equivalent to the paired natural definition.
    for name in ("A", "Multi"):
        if abs(means[name] - legacy["mean"][name]) > 2e-12:
            raise RuntimeError(f"Natural objective no longer reproduces original {name}")
    for scale in SCALES:
        if abs(means[f"{scale}_natural"] - legacy["mean"][scale]) > 2e-12:
            raise RuntimeError("Natural scale no longer reproduces old metric")
    return {"mean": means, "sequences": rows,
            "query_CE": legacy["mean"]["query_CE"],
            "response_sign_flip_rate": legacy["mean"]["response_sign_flip_rate"]}


def edge_graph(kind):
    """Undirected 16-node multigraph, one edge per scale and chain direction."""
    edges = []
    for name in SCALES:
        for c in (0, 1):
            for i, j in EDGES[name]:
                edges.append((8*c+i, 8*(c if kind == "natural" else 1-c)+j))
    return edges


def graph_invariants():
    out = []
    for kind in ("natural", "cross"):
        adjacency = np.zeros((16, 16), dtype=np.int64)
        for i, j in edge_graph(kind):
            adjacency[i, j] += 1
            adjacency[j, i] += 1
        out.append((np.diag(adjacency.sum(1)).tolist(), np.linalg.eigvalsh(adjacency).tolist()))
    if out[0][0] != out[1][0] or not np.allclose(out[0][1], out[1][1], atol=1e-12):
        raise RuntimeError("Natural and cross graphs have different degree/spectrum")
    return {"degrees": out[0][0], "max_eigenvalue_difference": float(np.max(np.abs(np.subtract(out[0][1], out[1][1]))))}
