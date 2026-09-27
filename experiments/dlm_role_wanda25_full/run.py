#!/usr/bin/env python3
"""Prepare and evaluate Role-Wanda-25 versus calibration-matched Uniform-Wanda-25.

The implementation deliberately reuses the frozen target50/75 machinery while
writing every new artifact under a separate experiment root.  Only the target
and its preregistered six-point grid differ.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_dual_role_allocation.io import atomic_write_json
from experiments.dlm_role_validation import run as base
from experiments.projection_capacity_followup_65.core import paired_binary_comparison


ROOT = Path("experiments/dlm_role_wanda25_full")
DATA_ROOT = Path("/DATA/tmluser1/sap-dlm-role-wanda25-full")
SUITE = "target25"
GRID = (.10, .15, .20, .25, .30, .35)
TARGET_INDEX = 3


def _patch_base() -> None:
    """Point the mature validation implementation at this isolated experiment."""
    base.ROOT = ROOT
    base.DATA_ROOT = DATA_ROOT
    base.GRIDS = {SUITE: GRID}
    base.TARGET_INDEX = TARGET_INDEX
    base.freeze = freeze
    base.require_frozen = require_frozen


def freeze() -> dict:
    inputs = base.load_frozen_inputs()
    states = inputs.states["states"]
    _, partition_audit = base._state_partitions(SUITE, states)
    protocol_hash, protocol = base._evaluation_config_hash(base.eval_config(base.EVAL_CONFIG))
    config = {
        "status": "frozen_before_collection_or_downstream_evaluation",
        "model": inputs.config["model"],
        "dense_model_sha256": base.DENSE_SHA,
        "projection_count": 224,
        "state_count": 80,
        "total_prunable_weights": base.WEIGHTS,
        "selector": "exact historical Standard Wanda abs(W)*sqrt(overall_uniform A), stable row-wise sort",
        "suite": SUITE,
        "target": GRID[TARGET_INDEX],
        "grid": list(GRID),
        "grid_reason": "same target-relative -15pp/+10pp range as frozen target50/75 validation",
        "role": "max(pooled masked normalized reconstruction error, pooled unmasked normalized reconstruction error)",
        "aggregate": "pooled total squared reconstruction error / pooled dense-output energy",
        "allocation": "raw adjacent 5pp marginal proxy cost per nominal parameter; exact target row-floor budget",
        "evaluation": {
            "protocol_hash": protocol_hash,
            "protocol": protocol,
            "limit": 1319,
            "methods": ["uniform", "role"],
            "mini_gate": "none; user explicitly requested full GSM8K",
        },
        "source_hashes": {
            str(base.STATS): base.sha(base.STATS),
            str(base.CAPACITY_ROOT / "candidate_mask_manifest.json"): base.sha(base.CAPACITY_ROOT / "candidate_mask_manifest.json"),
            str(base.CAPACITY_ROOT / "state_verification.json"): base.sha(base.CAPACITY_ROOT / "state_verification.json"),
            base.EVAL_CONFIG: base.sha(base.EVAL_CONFIG),
        },
        "no_retuning": True,
    }
    path = ROOT / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("frozen target25 config differs")
    if not path.exists():
        atomic_write_json(path, config)
    verification = {
        "status": "verified",
        "source_state_digest": inputs.metadata["state_digest"],
        "source_receipt_sha256": inputs.receipt["receipt_sha256"],
        "input_state_disjointness": inputs.verification,
        "partition": partition_audit,
    }
    verification_path = ROOT / "state_verification.json"
    if verification_path.exists() and json.loads(verification_path.read_text()) != verification:
        raise RuntimeError("frozen target25 state verification differs")
    if not verification_path.exists():
        atomic_write_json(verification_path, verification)
    base.event("freeze_complete", suite=SUITE, config_sha256=base.sha(path), protocol_hash=protocol_hash)
    return config


def require_frozen() -> dict:
    config = freeze()
    for path, digest in config["source_hashes"].items():
        if base.sha(path) != digest:
            raise RuntimeError(f"frozen source changed: {path}")
    return config


def prepare() -> None:
    require_frozen()
    base.collect(SUITE)
    allocation = base.analyze(SUITE)
    if allocation["target"] != GRID[TARGET_INDEX]:
        raise RuntimeError("target25 allocation target mismatch")
    for method in ("aggregate", "role"):
        row = allocation["methods"][method]
        if row["budget_error"] != 0:
            raise RuntimeError(f"{method} allocation budget error")
    base.materialize(SUITE)
    uniform = json.loads((ROOT / "manifest_target25_uniform.json").read_text())
    role = json.loads((ROOT / "manifest_target25_role.json").read_text())
    if uniform["pruned"] != role["pruned"]:
        raise RuntimeError("Uniform-25 and Role-25 pruned counts differ")
    base.event(
        "prepare_complete",
        suite=SUITE,
        pruned=int(uniform["pruned"]),
        weights=int(uniform["weights"]),
        global_sparsity=float(uniform["pruned"] / uniform["weights"]),
    )


@torch.inference_mode()
def evaluate(method: str) -> None:
    if method not in {"uniform", "role"}:
        pass
        raise ValueError(method)
    lock_path = ROOT / "locks" / f"{method}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    method_lock = lock_path.open("w")
    fcntl.flock(method_lock.fileno(), fcntl.LOCK_EX)
    require_frozen()
    manifest = ROOT / f"manifest_{SUITE}_{method}.json"
    if not manifest.exists():
        raise RuntimeError("prepare must complete before full evaluation")
    evaluated = base._evaluate_methods(SUITE, [method], 1319)
    outcome = evaluated["outcomes"][method]
    result = {
        "status": "complete",
        "suite": SUITE,
        "method": method,
        "protocol_hash": evaluated["protocol_hash"],
        "result": {key: value for key, value in outcome.items() if key != "rows"},
    }
    atomic_write_json(ROOT / f"full_{method}.json", result)
    base.event("full_worker_complete", suite=SUITE, method=method, correct=outcome["correct"])


def finalize() -> dict:
    require_frozen()
    method_results = {}
    rows = {}
    for method in ("uniform", "role"):
        path = ROOT / f"full_{method}.json"
        if not path.exists():
            raise RuntimeError(f"missing full result: {path}")
        payload = json.loads(path.read_text())
        method_results[method] = payload["result"]
        rows[method] = base._read_jsonl(Path(payload["result"]["predictions"]))
        if len(rows[method]) != 1319:
            raise RuntimeError(f"incomplete {method} prediction rows")
    paired = paired_binary_comparison(
        [row["correct"] for row in rows["uniform"]],
        [row["correct"] for row in rows["role"]],
    )
    uniform_manifest = json.loads((ROOT / "manifest_target25_uniform.json").read_text())
    role_manifest = json.loads((ROOT / "manifest_target25_role.json").read_text())
    if uniform_manifest["pruned"] != role_manifest["pruned"]:
        raise RuntimeError("final budget mismatch")
    result = {
        "status": "complete",
        "suite": SUITE,
        "target": GRID[TARGET_INDEX],
        "global_pruned_weights": int(uniform_manifest["pruned"]),
        "total_prunable_weights": int(uniform_manifest["weights"]),
        "global_sparsity": float(uniform_manifest["pruned"] / uniform_manifest["weights"]),
        "methods": method_results,
        "paired_role_vs_uniform": paired,
    }
    atomic_write_json(ROOT / "full_results.json", result)
    base.event(
        "final_complete",
        uniform_correct=method_results["uniform"]["correct"],
        role_correct=method_results["role"]["correct"],
        paired=paired,
    )
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "prepare", "evaluate", "finalize"))
    parser.add_argument("--method", choices=("uniform", "role"))
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    np.random.seed(0)
    _patch_base()
    if args.phase == "freeze":
        freeze()
    elif args.phase == "prepare":
        prepare()
    elif args.phase == "evaluate":
        if args.method is None:
            parser.error("--method is required for evaluate")
        evaluate(args.method)
    else:
        finalize()


if __name__ == "__main__":
    main()
