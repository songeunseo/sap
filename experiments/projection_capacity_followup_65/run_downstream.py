#!/usr/bin/env python3
"""Run frozen GSM8K mini/full evaluation for the two diagnostic allocations."""
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _read_jsonl,
    _write_jsonl,
    load_config as eval_config,
)
from experiments.projection_capacity_allocation_65.run import load_dense
from experiments.projection_capacity_followup_65.core import paired_binary_comparison
from experiments.projection_capacity_followup_65.run_heldout import (
    METHOD_PATHS,
    apply_manifest,
    sha,
    validate_inputs,
    write_json,
)
from experiments.wanda_failure_characterization.run_failure_map import model_sha, modules


ROOT = Path("experiments/projection_capacity_followup_65")
SOURCE = Path("experiments/projection_capacity_allocation_65")


def event(event_type, **values):
    print(json.dumps({"event": event_type, **values}), flush=True)


def load_existing(path, limit, reference):
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    if len(rows) != limit:
        raise RuntimeError(f"partial prediction file exists: {path}")
    _assert_same_examples(reference[:limit], rows)
    return rows


def main():
    _, manifests = validate_inputs()
    heldout = json.loads((ROOT / "heldout_dlm_results.json").read_text())
    if heldout.get("status") != "complete":
        raise RuntimeError("fresh held-out DLM evaluation must finish first")
    config = eval_config("experiments/dlm_loss_aggregation/exp002/config.yaml")
    protocol_hash, protocol = _evaluation_config_hash(config)
    historical = json.loads((SOURCE / "downstream.json").read_text())
    if protocol_hash != historical["protocol_hash"]:
        raise RuntimeError("historical GSM8K protocol hash mismatch")
    reference = _read_jsonl(
        Path("experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True
    )
    output = {
        "status": "running",
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "method_manifest_sha256": {
            method: sha(METHOD_PATHS[method]) for method in ("reconstruction", "eis_type")
        },
        "evaluations": [],
    }
    output_path = ROOT / "downstream.json"
    for limit in (100, 1319):
        outcomes = {}
        for method in ("reconstruction", "eis_type"):
            prediction_path = ROOT / "gsm8k" / f"{method}_{limit}_predictions.jsonl"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            rows = load_existing(prediction_path, limit, reference)
            if rows is None:
                model, mapping = load_dense()
                apply_manifest(model, mapping, manifests[method])
                sparse_sha = model_sha(model)
                measured, rows = _evaluate_gsm8k(
                    model, tokenizer, config, f"{method}_capacity65_followup", limit,
                    protocol_hash
                )
                if model_sha(model) != sparse_sha:
                    raise RuntimeError(f"{method} model changed during GSM8K evaluation")
                if len(rows) != limit:
                    raise RuntimeError(f"{method} returned wrong GSM8K record count")
                _assert_same_examples(reference[:limit], rows)
                _write_jsonl(prediction_path, rows)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            else:
                measured = {"reused_complete_predictions": True}
            outcomes[method] = {
                "correct": int(sum(bool(row["correct"]) for row in rows)),
                "limit": limit,
                "predictions": str(prediction_path),
                "sha256": sha(prediction_path),
                "metrics": measured,
            }
            event("gsm8k_method_complete", method=method, limit=limit,
                  correct=outcomes[method]["correct"])

        historical_outcome = next(row for row in historical["evaluations"]
                                  if row["uniform"]["limit"] == limit)
        uniform_rows = _read_jsonl(Path(historical_outcome["uniform"]["predictions"]))
        capacity_rows = _read_jsonl(Path(historical_outcome["capacity"]["predictions"]))
        _assert_same_examples(reference[:limit], uniform_rows)
        _assert_same_examples(reference[:limit], capacity_rows)
        reconstruction_rows = _read_jsonl(Path(outcomes["reconstruction"]["predictions"]))
        eis_rows = _read_jsonl(Path(outcomes["eis_type"]["predictions"]))
        comparison = {
            "capacity_minus_eis_type": paired_binary_comparison(
                [row["correct"] for row in eis_rows],
                [row["correct"] for row in capacity_rows],
            ),
            "capacity_minus_reconstruction": paired_binary_comparison(
                [row["correct"] for row in reconstruction_rows],
                [row["correct"] for row in capacity_rows],
            ),
            "eis_type_minus_uniform": paired_binary_comparison(
                [row["correct"] for row in uniform_rows],
                [row["correct"] for row in eis_rows],
            ),
            "reconstruction_minus_uniform": paired_binary_comparison(
                [row["correct"] for row in uniform_rows],
                [row["correct"] for row in reconstruction_rows],
            ),
        }
        output["evaluations"].append({
            "limit": limit,
            "historical": {
                "uniform_correct": historical_outcome["uniform"]["correct"],
                "capacity_correct": historical_outcome["capacity"]["correct"],
                "uniform_predictions_sha256": historical_outcome["uniform"]["sha256"],
                "capacity_predictions_sha256": historical_outcome["capacity"]["sha256"],
            },
            "new_methods": outcomes,
            "paired_comparisons": comparison,
        })
        write_json(output_path, output)
    output["status"] = "complete"
    write_json(output_path, output)
    event("downstream_complete", full={method: output["evaluations"][-1]["new_methods"][method]["correct"]
                                       for method in ("reconstruction", "eis_type")})


if __name__ == "__main__":
    main()
