"""CPU-only validation and exact-quota allocation for the new screen."""

from __future__ import annotations

import math
from pathlib import Path

from experiments.dlm_multiscale_ac50.artifacts import read, sha
from experiments.dlm_owl65.core import exact_row_counts

from .integrity import freeze_checked, read_checked
from .prepare import MULTI, ROOT, validate


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _probe(path, family, block, manifest_hash, refs, expected_fingerprint=None):
    row = read_checked(path, {
        "config_sha256": manifest_hash,
        "block": block,
    })
    if row.get("family") not in (None, family):
        raise ValueError(f"Probe family mismatch: {path}")
    if set(row.get("conditions", {})) != {"0.48", "0.52"}:
        raise ValueError(f"Probe rates incomplete: {path}")
    conditions = row["conditions"]
    for key, cond in conditions.items():
        rate = float(key)
        block_refs = refs[block * 7:block * 7 + 7]
        if len(block_refs) != 7:
            raise ValueError(f"Probe block reference count changed: {path}")
        expected_counts = [int(r["shape"][1] * rate) for r in block_refs]
        expected_pruned = sum(k * r["shape"][0] for k, r in zip(expected_counts, block_refs))
        if cond.get("pruned") != expected_pruned:
            raise ValueError(f"Probe quota mismatch: {path} {key}")
        if not cond.get("mask_identity"):
            raise ValueError(f"Probe mask identity missing: {path} {key}")
        metrics_path = cond.get("metrics_path")
        metrics_sha = cond.get("metrics_sha256")
        if not metrics_path or not metrics_sha:
            raise ValueError(f"Probe metric source missing: {path} {key}")
        if sha(metrics_path) != metrics_sha:
            raise ValueError(f"Probe metric source changed: {metrics_path}")
        metrics = read_checked(metrics_path)
        fingerprint = metrics.get("fingerprint", {})
        if fingerprint.get("config_sha256") != manifest_hash:
            raise ValueError(f"Probe metric config mismatch: {metrics_path}")
        if fingerprint.get("mask_identity") != cond["mask_identity"]:
            raise ValueError(f"Probe metric mask mismatch: {metrics_path}")
        for name, expected in (expected_fingerprint or {}).items():
            if fingerprint.get(name) != expected:
                raise ValueError(f"Probe metric {name} mismatch: {metrics_path}")
        mean = metrics.get("mean", {})
        rows_metrics = metrics.get("rows", [])
        if any(not _finite(mean.get(k)) for k in ("A", "C", "AC")) or not rows_metrics:
            raise ValueError(f"Probe metric is nonfinite: {metrics_path}")
        if any(any(not _finite(row.get(k)) for k in ("A", "C", "AC")) for row in rows_metrics):
            raise ValueError(f"Probe metric rows are nonfinite: {metrics_path}")
        metric_payload = {k: metrics[k] for k in ("mean", "rows")}
        if cond.get("metrics") != metric_payload:
            raise ValueError(f"Probe metric payload mismatch: {path} {key}")
    costs = row.get("costs", {})
    for objective in ("A", "AC"):
        lo, hi = conditions["0.48"], conditions["0.52"]
        expected = (hi["metrics"]["mean"][objective] - lo["metrics"]["mean"][objective]) / (hi["pruned"] - lo["pruned"])
        if not _finite(costs.get(objective)) or not math.isclose(float(costs[objective]), expected, rel_tol=1e-12, abs_tol=1e-30):
            raise ValueError(f"Probe marginal mismatch: {path} {objective}")
    return row


def load_allocation(family):
    """Read and validate a sealed family allocation artifact and its probes."""
    from experiments.dlm_multiscale_ac50.core import rank_rates

    manifest = validate()
    path = ROOT / "allocations" / f"{family}.json"
    manifest_hash = sha(ROOT / "manifest.json")
    row = read_checked(path, {"config_sha256": manifest_hash, "family": family})
    if set(row.get("allocations", {})) != {"A", "AC"}:
        raise ValueError(f"Allocation objectives changed: {path}")
    refs = read(read(MULTI / "config.json")["legacy_manifests"]["uniform"]["path"])["entries"]
    expected_probe_paths = {str(ROOT / "probes" / family / f"block{b:02d}.json") for b in range(32)}
    probe_hashes = row.get("probe_hashes", {})
    if set(probe_hashes) != expected_probe_paths:
        raise ValueError(f"Allocation probe set incomplete: {path}")
    expected_fp = {
        "bank_sha256": manifest["banks"][f"{family}_calibration"]["sha256"],
        "readout": f"{family}-v1",
    }
    probes = [_probe(ROOT / "probes" / family / f"block{b:02d}.json", family, b, manifest_hash, refs, expected_fp) for b in range(32)]
    for probe_path, expected_sha in probe_hashes.items():
        if sha(probe_path) != expected_sha:
            raise ValueError(f"Allocation probe changed: {probe_path}")
    for objective, allocation in row["allocations"].items():
        counts = allocation.get("row_counts")
        if len(counts) != len(refs):
            raise ValueError(f"Allocation row count length changed: {path}")
        if sum(k * r["shape"][0] for k, r in zip(counts, refs)) != manifest["target"]:
            raise ValueError(f"Allocation global quota mismatch: {path}")
        if not all(isinstance(k, int) and 0 <= k <= r["shape"][1] for k, r in zip(counts, refs)):
            raise ValueError(f"Allocation row quota invalid: {path}")
        if len(allocation.get("scores", [])) != 32 or len(allocation.get("rates", [])) != 32:
            raise ValueError(f"Allocation block vectors incomplete: {path}")
        if any(not _finite(x) for x in allocation["scores"] + allocation["rates"]):
            raise ValueError(f"Allocation vector is nonfinite: {path}")
        expected_scores = [probe["costs"][objective] for probe in probes]
        expected_rates = rank_rates(expected_scores).tolist()
        expected_counts, expected_budget = exact_row_counts(refs, expected_rates, manifest["target"])
        if allocation["scores"] != expected_scores or allocation["rates"] != expected_rates or allocation["row_counts"] != expected_counts or allocation.get("budget") != expected_budget:
            raise ValueError(f"Allocation derivation changed: {path} {objective}")
    return row

def allocate(family):
    """Validate sealed probes and write a sealed exact-budget allocation."""
    from experiments.dlm_multiscale_ac50.core import rank_rates

    manifest = validate()
    c = read(MULTI / "config.json")
    refs = read(c["legacy_manifests"]["uniform"]["path"])["entries"]
    manifest_hash = sha(ROOT / "manifest.json")
    rows = []
    for block in range(32):
        path = ROOT / "probes" / family / f"block{block:02d}.json"
        rows.append(_probe(path, family, block, manifest_hash, refs, {
            "bank_sha256": manifest["banks"][f"{family}_calibration"]["sha256"],
            "readout": f"{family}-v1",
        }))

    result = {}
    for arm in ("A", "AC"):
        scores = [r["costs"][arm] for r in rows]
        rates = rank_rates(scores)
        counts, budget = exact_row_counts(refs, rates, manifest["target"])
        result[arm] = dict(scores=scores, rates=rates.tolist(), row_counts=counts, budget=budget)

    payload = dict(
        config_sha256=manifest_hash,
        family=family,
        allocations=result,
        probe_hashes={
            str(ROOT / "probes" / family / f"block{b:02d}.json"): sha(ROOT / "probes" / family / f"block{b:02d}.json")
            for b in range(32)
        },
    )
    freeze_checked(ROOT / "allocations" / f"{family}.json", payload)
    return payload
