"""Finite-space algebra checks for the multiscale A+C design.

No model, corpus, GPU, pruning, or benchmark is loaded.
"""
import itertools
import json
import math
from pathlib import Path


def states(n):
    return list(itertools.product((0, 1), repeat=n))


def mu(v, p):
    return math.prod(p if x else 1 - p for x in v)


def phi(v, subset, p):
    return math.prod((v[j] - p) / math.sqrt(p * (1 - p)) for j in subset)


def coefficients(values, p):
    n = len(next(iter(values)))
    subsets = [tuple(j for j in range(n) if z[j]) for z in states(n)]
    return {S: sum(mu(v, p) * y * phi(v, S, p)
                   for v, y in values.items()) for S in subsets}


def residual(v):
    return (0.2 + 0.3 * sum(v) - 0.8 * v[0] * v[1]
            + 0.4 * v[1] * v[2] * v[3] + 0.7 * v[3])


def check_monotone(p, q, n=4):
    vals = {v: residual(v) for v in states(n)}
    ap, aq = coefficients(vals, p), coefficients(vals, q)
    rho = math.sqrt(p * (1 - q) / (q * (1 - p)))
    direct = 0.0
    for choices in itertools.product(range(3), repeat=n):
        pairs = ((0, 0), (0, 1), (1, 1))
        x = tuple(pairs[c][0] for c in choices)
        y = tuple(pairs[c][1] for c in choices)
        prob = math.prod((1 - q, q - p, p)[c] for c in choices)
        direct += prob * (vals[y] - vals[x]) ** 2
    spectral = sum(ap[S] ** 2 + aq[S] ** 2
                   - 2 * rho ** len(S) * ap[S] * aq[S] for S in ap)
    psd = sum((1 - rho ** len(S)) * (ap[S] + aq[S]) ** 2 / 2
              + (1 + rho ** len(S)) * (ap[S] - aq[S]) ** 2 / 2
              for S in ap)
    h = math.log(q / (1 - q)) - math.log(p / (1 - p))
    errors = [abs(direct - spectral), abs(direct - psd),
              abs(rho - math.exp(-h / 2))]
    assert max(errors) < 1e-12
    return dict(p=p, q=q, rho=rho, h=h, direct=direct,
                spectral=spectral, psd=psd, max_error=max(errors))


def c_matching(e, d):
    edges = [(i, i ^ d) for i in range(len(e)) if not (i & d)]
    assert len(edges) == len(e) // 2
    assert sorted(x for pair in edges for x in pair) == list(range(len(e)))
    return sum((e[j] - e[i]) ** 2 for i, j in edges) / len(edges)


def graph_checks():
    K, m = 8, 3
    ds = [1 << b for b in range(m)]
    # Verify the exact temporal-index Walsh multiplier and the bounds.
    rows = []
    for s in range(K):
        e = [(-1.0) ** ((i & s).bit_count()) for i in range(K)]
        c = sum(c_matching(e, d) for d in ds) / m
        expected = 4 * s.bit_count() / m
        assert abs(c - expected) < 1e-12
        rows.append(dict(walsh_index=s, C=c, expected=expected))
    constant = [1.0] * K
    phase_varying = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0]
    out = []
    for name, e in (("constant", constant), ("phase_varying", phase_varying)):
        A = sum(v * v for v in e) / K
        Cs = [c_matching(e, d) for d in ds]
        C = sum(Cs) / m
        variance = A - (sum(e) / K) ** 2
        assert 4 * variance / m <= C + 1e-12
        assert C <= 4 * variance + 1e-12
        out.append(dict(name=name, A=A, C_by_scale=Cs, C_multi=C,
                        variance=variance, L=A + C))
    # On a complete graph, relation energy collapses to scaled variance.
    e = [0.4, 0.1, -0.2, 0.7, 1.0, -0.9, 0.3, -0.4]
    all_pair = sum((e[j] - e[i]) ** 2 for i in range(K)
                   for j in range(i + 1, K)) / (K * (K - 1) / 2)
    var = sum(v * v for v in e) / K - (sum(e) / K) ** 2
    assert abs(all_pair - 2 * K / (K - 1) * var) < 1e-12
    return dict(eigenmodes=rows, counterexample=out,
                complete_graph_energy=all_pair, complete_graph_variance=var)


def fork_check(p=0.3, rho=0.7):
    parent = rho * p / (1 - p + rho * p)
    reveal = p * (1 - rho)
    joint = {}
    for x, y in itertools.product((0, 1), repeat=2):
        branch = ((reveal if x else 1 - reveal)
                  * (reveal if y else 1 - reveal))
        joint[x, y] = (1 - parent) * branch + parent * (x == 1 and y == 1)
    target = {(1, 1): p*p + rho*p*(1-p),
              (0, 0): (1-p)**2 + rho*p*(1-p),
              (1, 0): p*(1-p)*(1-rho),
              (0, 1): p*(1-p)*(1-rho)}
    err = max(abs(joint[k] - target[k]) for k in joint)
    assert err < 1e-12
    return dict(p=p, rho=rho, parent_p=parent,
                conditional_reveal_p=reveal, max_joint_error=err)


result = dict(
    scope="mathematical toy verification only; no model experiments",
    monotone=[check_monotone(p, q) for p, q in
              ((0.1, 0.2), (0.2, 0.8), (0.45, 0.55), (0.8, 0.95))],
    graph=graph_checks(),
    fork=fork_check(),
)
output = Path(__file__).with_suffix(".json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "max_monotone_identity_error": max(r["max_error"] for r in result["monotone"]),
    "graph_modes_checked": len(result["graph"]["eigenmodes"]),
    "counterexample": result["graph"]["counterexample"],
    "fork_joint_error": result["fork"]["max_joint_error"],
    "output": str(output),
}, indent=2))

