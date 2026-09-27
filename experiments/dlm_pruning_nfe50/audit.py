"""Independent, CPU-only evidence audit for the exposed-200 NFE comparison.

This module reads historical artifacts. It never changes them or loads a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.dlm_multiscale_ac50.artifacts import digest, mask_identity, sha, write
from experiments.dlm_multiscale_ac50.evaluation import validate_prediction

REPO = Path(__file__).resolve().parents[2]
CROSS = REPO / "experiments/dlm_crosschain_control50/output"
ROOT = Path(__file__).resolve().parent / "output"
DENSE = REPO / "experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl"
LEGACY_EVAL = REPO / "experiments/dlm_loss_aggregation/exp002/logs/evaluation_config.json"
EXPECTED = {
    "config": "59912694489062d4d7c08dece821c36ad292f95a4b42a7603851b7d14347d06b",
    "requests": "3ed3e16ecd93ab9081c40a9bd523f09e6bffb8b6da7f78803b7a729609d4f5a8",
    "manifest": "5afb660b8a005bfb7f9ec3f630de54fe13e0d56dee8b7da66c683b857903b7bc",
    "mask_identity": "fd22875b43d65e22ba2c0091b0f9b552e5f73d524cdae388f73030075c0c1d10",
    "sparse_model": "7d5dc817890f1079a6a8f2970088c7bee80afa60c11ebea30cb435d14d40922c",
    "protocol": "1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add",
    "dense_file": "1a1523c089558ee3df2ad1b49a937a3b01290060d67a51b8715e3374dc866b8e",
}


def read(path):
    return json.loads(Path(path).read_text())


def unique_map(rows, expected_ids, label):
    result = {}
    for row in rows:
        i = row.get("example_id")
        if type(i) is not int or i in result:
            raise RuntimeError(f"{label}: duplicate or invalid example_id {i!r}")
        result[i] = row
    if set(result) != set(expected_ids):
        raise RuntimeError(f"{label}: missing={sorted(set(expected_ids)-set(result))[:10]}, extra={sorted(set(result)-set(expected_ids))[:10]}")
    return result


def source_audit():
    """Return independently checked historical inputs and reusable 256-step rows."""
    paths = {
        "config": CROSS / "config.json",
        "requests": CROSS / "requests.json",
        "manifest": CROSS / "candidates/A/mask_manifest.json",
    }
    for label, path in paths.items():
        if sha(path) != EXPECTED[label]:
            raise RuntimeError(f"Historical {label} SHA changed: {path}")
    config, requests, manifest = (read(paths[k]) for k in ("config", "requests", "manifest"))
    if config["protocol_hash"] != EXPECTED["protocol"] or requests["protocol_hash"] != EXPECTED["protocol"]:
        raise RuntimeError("Historical protocol differs")
    if mask_identity(manifest) != EXPECTED["mask_identity"] or manifest["pruned"] != 3489660928:
        raise RuntimeError("A physical mask identity or prune count differs")
    model_receipt = read(CROSS / "candidates/A/model_identity.json")
    if model_receipt["mask_identity"] != EXPECTED["mask_identity"] or model_receipt["sparse_model_sha256"] != EXPECTED["sparse_model"]:
        raise RuntimeError("A sparse-model receipt differs")
    ids = config["crosschain_control"]["exposed_ids"]
    if len(ids) != 200 or len(set(ids)) != 200:
        raise RuntimeError("Exposed set is not 200 unique IDs")
    req = unique_map(requests["development"], range(1319), "historical requests")
    a_rows = {}
    for i in ids:
        path = CROSS / "gsm8k/A" / f"shard{i//128:02d}" / "examples" / f"{i:04d}.json"
        saved = read(path)
        row = saved["row"]
        if saved["row_sha256"] != digest(row):
            raise RuntimeError(f"A checkpoint row hash differs: {path}")
        fp = saved["fingerprint"]
        if fp["config_sha256"] != EXPECTED["config"] or fp["requests_sha256"] != EXPECTED["requests"] or fp["mask_identity"] != EXPECTED["mask_identity"] or fp["sparse_model_sha256"] != EXPECTED["sparse_model"] or fp["protocol_hash"] != EXPECTED["protocol"] or fp["shard"] != i//128:
            raise RuntimeError(f"A checkpoint fingerprint differs: {path}")
        validate_prediction(row, req[i], EXPECTED["protocol"])
        if row["method"] != "A":
            raise RuntimeError(f"A method differs at {i}")
        a_rows[i] = row
    legacy = read(LEGACY_EVAL)
    dense_validation = read(REPO / "experiments/dlm_loss_aggregation/exp002/logs/dense_validation.json")
    if dense_validation["dense_fingerprint"] != "a6c065970a8355d60a773fdee3d8fdc34147d4f089caa44e1e9aeecc19b51785":
        raise RuntimeError("Legacy Dense prunable-layer fingerprint differs")
    if dense_validation["record"]["evaluation_config_hash"] != EXPECTED["protocol"]:
        raise RuntimeError("Legacy Dense validation row protocol differs")
    if legacy["sha256"] != EXPECTED["protocol"] or legacy["model"] != config["model"] or legacy["evaluation"] != config["evaluation"] or legacy["gsm8k_task_sha256"] != requests["protocol"]["gsm8k_task_sha256"] or legacy["lm_eval_version"] != requests["protocol"]["lm_eval_version"]:
        raise RuntimeError("Legacy Dense evaluation protocol differs")
    if sha(DENSE) != EXPECTED["dense_file"]:
        raise RuntimeError("Legacy Dense file SHA changed")
    dense_all = unique_map([json.loads(line) for line in DENSE.open()], range(1319), "legacy Dense")
    dense_rows = {}
    for i in ids:
        row = dense_all[i]
        validate_prediction(row, req[i], EXPECTED["protocol"])
        if row["method"] != "Dense":
            raise RuntimeError(f"Dense method differs at {i}")
        dense_rows[i] = row
    return {"status": "passed", "sample": "exposed development 200", "ids": ids,
            "protocol_hash": EXPECTED["protocol"], "mask_identity": EXPECTED["mask_identity"],
            "source_sha256": {k: sha(v) for k, v in paths.items()},
            "dense_cache_path": str(DENSE), "dense_cache_sha256": sha(DENSE),
            "legacy_dense_prunable_layer_fingerprint": dense_validation["dense_fingerprint"],
            "legacy_dense_fingerprint_method": "EXP-002 BF16 prunable-layer hash; distinct algorithm from later full model_sha",
            "A256_correct": sum(a_rows[i]["correct"] for i in ids),
            "Dense256_correct": sum(dense_rows[i]["correct"] for i in ids),
            "rows_verified": {"A256": len(a_rows), "Dense256": len(dense_rows)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "review/source_audit.json")
    args = parser.parse_args()
    report = source_audit()
    write(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "ids"}, indent=2))


if __name__ == "__main__":
    main()
