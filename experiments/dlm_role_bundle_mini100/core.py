"""Label-blind exact-budget hybrid construction and paired diagnostics."""
from copy import deepcopy

import numpy as np

from experiments.dlm_capacity_predictor.audit_existing import paired_exact_mcnemar

METHODS = tuple(f"{direction}_b{i}" for direction in ("add", "revert") for i in range(6))


def signature(manifest):
    return [(e["name"], e["selected_mask"]["mask_sha256"], e["selected_mask"]["pruned"])
            for e in manifest["entries"]]


def make_hybrids(aggregate, role, bundles):
    if len(aggregate["entries"]) != len(role["entries"]) or aggregate["pruned"] != role["pruned"]:
        raise ValueError("aligned equal-budget baselines required")
    changes = set()
    for i, (a, r) in enumerate(zip(aggregate["entries"], role["entries"], strict=True)):
        if a["name"] != r["name"] or a["shape"] != r["shape"]:
            raise ValueError("module ordering/shape mismatch")
        if a["selected_mask"]["mask_sha256"] != r["selected_mask"]["mask_sha256"]:
            changes.add(i)
    flattened = [i for b in bundles for i in b["indices"]]
    if len(flattened) != len(set(flattened)) or set(flattened) != changes:
        raise ValueError("bundles must partition exactly the changed projections")
    hybrids = {}
    for b in bundles:
        indices = set(b["indices"])
        delta = sum(role["entries"][i]["selected_mask"]["pruned"] -
                    aggregate["entries"][i]["selected_mask"]["pruned"] for i in indices)
        if delta != 0:
            raise ValueError("bundle violates exact budget")
        for direction, base, donor in (("add", aggregate, role), ("revert", role, aggregate)):
            entries = [deepcopy(donor["entries"][i] if i in indices else e) for i, e in enumerate(base["entries"])]
            count = sum(e["selected_mask"]["pruned"] for e in entries)
            if count != base["pruned"]:
                raise ValueError("hybrid budget mismatch")
            name = f"{direction}_b{b['id']}"
            hybrids[name] = dict(method=name, entries=entries, pruned=count,
                weights=base["weights"], allocation_levels=[e["level"] for e in entries],
                source="frozen Aggregate/Role masks only", bundle_id=b["id"],
                changed_indices=sorted(indices), background="aggregate" if direction == "add" else "role")
    return hybrids


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    result, bound = {}, 0.0
    for i, key in enumerate(ordered):
        bound = max(bound, min(1.0, (len(ordered) - i) * pvalues[key]))
        result[key] = bound
    return result


def paired_ci(values, seed=20260913, count=20000):
    values = np.asarray(values, dtype=int)
    if len(values) == 0:
        raise ValueError("empty paired sample")
    labels, counts = np.unique(values, return_counts=True)
    samples = np.random.default_rng(seed).multinomial(len(values), counts / len(values), size=count)
    return np.quantile(samples @ labels / len(values), [.025, .975]).tolist()


def summarize(baselines, candidates):
    a = np.array([x["correct"] for x in baselines["aggregate"]], dtype=int)
    r = np.array([x["correct"] for x in baselines["role"]], dtype=int)
    pairs, pvalues = {}, {}
    for method, rows in candidates.items():
        y = np.array([x["correct"] for x in rows], dtype=int)
        forward = method.startswith("add_")
        # Both signs below mean benefit of adopting the Role-side bundle.
        left, right = (y, a) if forward else (r, y)
        pair = paired_exact_mcnemar(left.tolist(), right.tolist())
        pair["benefit_accuracy_ci95_unadjusted"] = paired_ci(left - right)
        pair["hybrid_correct"] = int(y.sum())
        pairs[method] = pair
        pvalues[method] = pair["exact_mcnemar_p"]
    adjusted = holm(pvalues)
    for method in pairs:
        pairs[method]["holm_p_family12"] = adjusted[method]
    interactions = {}
    for i in range(6):
        add = np.array([x["correct"] for x in candidates[f"add_b{i}"]], dtype=int)
        rev = np.array([x["correct"] for x in candidates[f"revert_b{i}"]], dtype=int)
        d = (r - rev) - (add - a)
        interactions[str(i)] = {"role_background_benefit_minus_aggregate_background_benefit": int(d.sum()),
            "accuracy_ci95_unadjusted_exploratory": paired_ci(d)}
    return dict(pairs=pairs, background_interactions=interactions,
                interpretation="conditional bundle effects; not unique contributions, role-path attribution, or full GSM8K confirmation")
