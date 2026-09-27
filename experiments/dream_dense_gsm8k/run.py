from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .harness import DreamEvalHarness


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
OUTPUT = ROOT / "output"


def canonical_json(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def atomic_json(path: Path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "experiment": "dream_dense_gsm8k_matched_llada",
        "model.id": "Dream-org/Dream-v0-Base-7B",
        "model.revision": "6572adb5535263e4d1a337b56942ba48b6dee2a9",
        "evaluation.task": "gsm8k",
        "evaluation.num_fewshot": 5,
        "evaluation.max_new_tokens": 256,
        "evaluation.diffusion_steps": 256,
        "evaluation.temperature": 0.0,
        "evaluation.algorithm": "entropy",
        "evaluation.mini_examples": 100,
        "evaluation.full_examples": 1319,
    }
    for dotted, wanted in expected.items():
        value = config
        for part in dotted.split("."):
            value = value[part]
        if value != wanted:
            raise ValueError(f"frozen config mismatch for {dotted}: {value!r}")
    if config["provenance"]["official_base_gsm8k_fewshot"] == config["evaluation"]["num_fewshot"]:
        raise ValueError("official and matched protocols must remain explicitly distinguished")
    return config


def config_receipt(config):
    task_yaml = Path(importlib.util.find_spec("lm_eval.tasks.gsm8k").submodule_search_locations[0]) / "gsm8k.yaml"
    document = {
        "config": config,
        "config_sha256": hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest(),
        "task_yaml": str(task_yaml),
        "task_yaml_sha256": hashlib.sha256(task_yaml.read_bytes()).hexdigest(),
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    document["protocol_hash"] = hashlib.sha256(canonical_json(document)).hexdigest()
    return document


def first(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value


def extract_records(samples, protocol_hash):
    records = []
    for sample in samples:
        if sample["filter"] != "strict-match":
            continue
        records.append(
            {
                "example_id": int(sample["doc_id"]),
                "doc_hash": sample["doc_hash"],
                "prompt_hash": sample["prompt_hash"],
                "target_hash": sample["target_hash"],
                "reference_answer": sample["target"],
                "generated_text": first(sample["resps"]),
                "extracted_answer": first(sample["filtered_resps"]),
                "correct": bool(sample["exact_match"]),
                "protocol_hash": protocol_hash,
            }
        )
    records.sort(key=lambda row: row["example_id"])
    if len({row["example_id"] for row in records}) != len(records):
        raise ValueError("duplicate GSM8K example ids")
    return records


def set_seeds(evaluation):
    random.seed(evaluation["random_seed"])
    np.random.seed(evaluation["numpy_seed"])
    torch.manual_seed(evaluation["torch_seed"])
    torch.cuda.manual_seed_all(evaluation["torch_seed"])


def evaluate(phase: str, config_path=DEFAULT_CONFIG):
    from lm_eval import evaluator

    config = load_config(config_path)
    evaluation = config["evaluation"]
    receipt = config_receipt(config)
    set_seeds(evaluation)
    harness = DreamEvalHarness(config)
    limit = evaluation["mini_examples"] if phase == "mini" else None
    started = time.monotonic()
    result = evaluator.simple_evaluate(
        model=harness,
        tasks=[evaluation["task"]],
        num_fewshot=evaluation["num_fewshot"],
        batch_size=evaluation["batch_size"],
        limit=limit,
        bootstrap_iters=evaluation["bootstrap_iters"],
        log_samples=True,
        random_seed=evaluation["random_seed"],
        numpy_random_seed=evaluation["numpy_seed"],
        torch_random_seed=evaluation["torch_seed"],
        fewshot_random_seed=evaluation["fewshot_seed"],
    )
    elapsed = time.monotonic() - started
    if harness.world_size > 1:
        harness.accelerator.wait_for_everyone()
    if harness.rank != 0:
        return None

    records = extract_records(result["samples"]["gsm8k"], receipt["protocol_hash"])
    expected = evaluation["mini_examples"] if phase == "mini" else evaluation["full_examples"]
    if len(records) != expected:
        raise ValueError(f"{phase} row count mismatch: {len(records)} != {expected}")
    if [row["example_id"] for row in records] != list(range(expected)):
        raise ValueError(f"{phase} is not the frozen GSM8K prefix/order")
    if any(not isinstance(row["generated_text"], str) for row in records):
        raise ValueError("missing generated text")
    accuracy = result["results"]["gsm8k"][evaluation["primary_metric"]]
    correct = sum(row["correct"] for row in records)
    if correct / expected != accuracy:
        raise ValueError("aggregate accuracy disagrees with compact records")

    phase_dir = OUTPUT / phase
    atomic_jsonl(phase_dir / "predictions.jsonl", records)
    summary = {
        "phase": phase,
        "status": "complete",
        "model": config["model"],
        "protocol_hash": receipt["protocol_hash"],
        "num_examples": expected,
        "correct": correct,
        "accuracy": accuracy,
        "eval_seconds": elapsed,
        "examples_per_second": expected / elapsed,
        "world_size": harness.world_size,
        "max_prompt_tokens": max(row["prompt_tokens"] for row in harness.generation_receipts) if harness.generation_receipts else None,
    }
    atomic_json(phase_dir / "summary.json", summary)
    atomic_json(OUTPUT / "protocol_receipt.json", receipt)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def validate_mini():
    config = load_config()
    summary = json.loads((OUTPUT / "mini" / "summary.json").read_text())
    rows = read_jsonl(OUTPUT / "mini" / "predictions.jsonl")
    if summary["status"] != "complete" or len(rows) != config["evaluation"]["mini_examples"]:
        raise ValueError("mini completion gate failed")
    if summary["correct"] != sum(row["correct"] for row in rows):
        raise ValueError("mini correctness gate failed")
    print(json.dumps({"mini_gate": "passed", "correct": summary["correct"]}), flush=True)


def finalize():
    config = load_config()
    mini = read_jsonl(OUTPUT / "mini" / "predictions.jsonl")
    full = read_jsonl(OUTPUT / "full" / "predictions.jsonl")
    n = config["evaluation"]["mini_examples"]
    keys = ("example_id", "doc_hash", "prompt_hash", "target_hash", "generated_text", "extracted_answer", "correct")
    if [[row[key] for key in keys] for row in mini] != [[row[key] for key in keys] for row in full[:n]]:
        raise ValueError("full first-100 predictions do not exactly reproduce mini")
    mini_summary = json.loads((OUTPUT / "mini" / "summary.json").read_text())
    full_summary = json.loads((OUTPUT / "full" / "summary.json").read_text())
    if mini_summary["protocol_hash"] != full_summary["protocol_hash"]:
        raise ValueError("mini/full protocol hash mismatch")
    final = {
        "status": "complete",
        "mini": mini_summary,
        "full": full_summary,
        "mini_full_prefix_exact": True,
    }
    atomic_json(OUTPUT / "final.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "validate-mini", "finalize"))
    parser.add_argument("--phase", choices=("mini", "full"))
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "evaluate":
        if not args.phase:
            parser.error("evaluate requires --phase")
        evaluate(args.phase)
    elif args.command == "validate-mini":
        validate_mini()
    else:
        finalize()


if __name__ == "__main__":
    main()
