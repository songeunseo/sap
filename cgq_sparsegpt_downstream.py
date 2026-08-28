import argparse
import gc
import json
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from lm_eval import evaluator
from lm_eval.api.model import LM
from transformers import AutoTokenizer

from cgq_sparsegpt import (
    _cache_calibration_weights,
    _load_model,
    _write_json,
    compare_model_masks,
    resolve_mask_id,
    save_plain_model_masks,
    state_digest,
    summarize_mask_cells,
)
from eval_llada import LLaDAEvalHarness, set_seed
from lib.prune_llada import prune_sparsegpt


def make_in_memory_harness(model, tokenizer, device):
    harness = object.__new__(LLaDAEvalHarness)
    LM.__init__(harness)
    harness.model = model
    harness.tokenizer = tokenizer
    harness.device = device
    harness.mask_id = 126336
    harness.max_length = 4096
    harness.batch_size = 8
    harness.mc_num = 128
    harness.sampling_eps = 0.0
    harness.is_check_greedy = False
    harness.cfg = 0.0
    harness.steps = 1024
    harness.gen_length = 1024
    harness.block_length = 1024
    harness.remasking = "low_confidence"
    harness.accelerator = None
    return harness


def extract_winogrande_records(samples):
    records = []
    for sample in samples:
        scores = [float(response[0]) for response in sample["filtered_resps"]]
        records.append(
            {
                "doc_id": sample["doc_id"],
                "doc_hash": sample["doc_hash"],
                "prompt_hash": sample["prompt_hash"],
                "target_hash": sample["target_hash"],
                "prediction_index": max(range(len(scores)), key=scores.__getitem__),
                "target_index": int(sample["doc"]["answer"]) - 1,
                "correct": bool(sample["acc"]),
                "scores": scores,
            }
        )
    return records


def assert_same_examples(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("sample counts differ")
    for expected, actual in zip(reference, candidate):
        for field in ("doc_id", "doc_hash", "prompt_hash", "target_hash"):
            if expected[field] != actual[field]:
                raise ValueError(f"WinoGrande {field} differs")


def summarize_paired_records(plain, cgq):
    assert_same_examples(plain, cgq)
    plain_correct = sum(record["correct"] for record in plain)
    cgq_correct = sum(record["correct"] for record in cgq)
    count = len(plain)
    return {
        "sample_count": count,
        "plain_correct": plain_correct,
        "cgq_correct": cgq_correct,
        "plain_accuracy": plain_correct / count,
        "cgq_accuracy": cgq_correct / count,
        "delta_percentage_points": (cgq_correct - plain_correct) / count * 100,
        "plain_wrong_cgq_correct": sum(
            not left["correct"] and right["correct"]
            for left, right in zip(plain, cgq)
        ),
        "plain_correct_cgq_wrong": sum(
            left["correct"] and not right["correct"]
            for left, right in zip(plain, cgq)
        ),
        "both_correct": sum(
            left["correct"] and right["correct"]
            for left, right in zip(plain, cgq)
        ),
        "both_wrong": sum(
            not left["correct"] and not right["correct"]
            for left, right in zip(plain, cgq)
        ),
    }


def load_cached_calibration_states(state_path, previous_result_path):
    states = torch.load(state_path, map_location="cpu", weights_only=True)[
        "calibration"
    ]
    digest = state_digest(states)
    expected = json.loads(Path(previous_result_path).read_text())[
        "calibration_state_digest"
    ]
    if digest != expected:
        raise ValueError("cached calibration state digest differs")
    return states, digest


def validate_mask_reproduction(actual, expected, tolerance):
    if abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"mask XOR {actual:.8f} differs materially from {expected:.8f}"
        )


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_existing_dense(config):
    result_path = Path(config["dense_winogrande"]["result_path"])
    sample_path = Path(config["dense_winogrande"]["sample_path"])
    result = json.loads(result_path.read_text())
    records = extract_winogrande_records(
        [json.loads(line) for line in sample_path.read_text().splitlines()]
    )
    count = config["evaluation"]["sample_count"]
    accuracy = result["results"]["winogrande"]["acc,none"]
    if len(records) != count or result["n-samples"]["winogrande"]["effective"] != count:
        raise ValueError("existing Dense WinoGrande sample count differs")
    if sum(record["correct"] for record in records) / count != accuracy:
        raise ValueError("existing Dense WinoGrande samples do not match aggregate")
    if result["configs"]["winogrande"]["num_fewshot"] != 5:
        raise ValueError("existing Dense WinoGrande result is not 5-shot")
    return {
        "source_result": str(result_path),
        "source_samples": str(sample_path),
        "sample_count": count,
        "correct": sum(record["correct"] for record in records),
        "accuracy": accuracy,
    }, records


def _evaluate_winogrande(model, tokenizer, device):
    set_seed(1234)
    started = time.monotonic()
    result = evaluator.simple_evaluate(
        model=make_in_memory_harness(model, tokenizer, device),
        tasks=["winogrande"],
        num_fewshot=5,
        batch_size=8,
        bootstrap_iters=100000,
        log_samples=True,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    records = extract_winogrande_records(result["samples"]["winogrande"])
    correct = sum(record["correct"] for record in records)
    accuracy = result["results"]["winogrande"]["acc,none"]
    if correct / len(records) != accuracy:
        raise ValueError("WinoGrande samples do not match aggregate accuracy")
    return {
        "sample_count": len(records),
        "correct": correct,
        "accuracy": accuracy,
        "stderr": result["results"]["winogrande"]["acc_stderr,none"],
        "evaluation_seconds": time.monotonic() - started,
    }, records


def _prune(model, args, loader, weights):
    started = time.monotonic()
    prune_sparsegpt(
        args,
        model,
        tokenizer=None,
        dev=torch.device("cuda:0"),
        calibration_loader=loader,
        token_weights=weights,
    )
    return time.monotonic() - started


def _release():
    gc.collect()
    torch.cuda.empty_cache()


def run_experiment(config_path, output_dir):
    config = json.loads(Path(config_path).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states, calibration_digest = load_cached_calibration_states(
        config["source_experiment"]["states_path"],
        config["source_experiment"]["results_path"],
    )
    if len(states) != 16:
        raise ValueError(f"expected 16 cached calibration states, found {len(states)}")
    loader = [(state["input_ids"],) for state in states]
    args = SimpleNamespace(nsamples=len(states), seed=0, sparsity_ratio=0.5)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    dense, dense_records = _load_existing_dense(config)

    confidence_model = _load_model(config)
    mask_id = resolve_mask_id(tokenizer, confidence_model)
    cgq_weights, _ = _cache_calibration_weights(
        confidence_model, states, mask_id, device
    )
    del confidence_model
    _release()

    mask_dir = Path(
        tempfile.mkdtemp(prefix="cgq-downstream-plain-masks-", dir="/dev/shm")
    )
    try:
        plain_model = _load_model(config)
        plain_pruning_seconds = _prune(plain_model, args, loader, None)
        plain_sparsity = save_plain_model_masks(plain_model, mask_dir)
        plain_eval, plain_records = _evaluate_winogrande(
            plain_model, tokenizer, device
        )
        assert_same_examples(dense_records, plain_records)
        plain = {
            "pruning": plain_sparsity,
            "pruning_seconds": plain_pruning_seconds,
            "winogrande": plain_eval,
        }
        _write_json(output_dir / "plain.json", plain)
        _write_jsonl(output_dir / "plain_predictions.jsonl", plain_records)
        del plain_model
        _release()

        if state_digest(states) != calibration_digest:
            raise RuntimeError("calibration states changed before CGQ pruning")
        cgq_model = _load_model(config)
        cgq_pruning_seconds = _prune(cgq_model, args, loader, cgq_weights)
        cgq_sparsity, mask_cells = compare_model_masks(cgq_model, mask_dir)
        mask_comparison = summarize_mask_cells(mask_cells)
        expected_mask = config["verification"]["expected_mask_xor_fraction"]
        validate_mask_reproduction(
            mask_comparison["global_xor_fraction"],
            expected_mask,
            config["verification"]["mask_xor_absolute_tolerance"],
        )
        cgq_eval, cgq_records = _evaluate_winogrande(cgq_model, tokenizer, device)
        assert_same_examples(dense_records, cgq_records)
        paired = summarize_paired_records(plain_records, cgq_records)
        cgq = {
            "pruning": cgq_sparsity,
            "pruning_seconds": cgq_pruning_seconds,
            "winogrande": cgq_eval,
        }
        _write_json(output_dir / "cgq.json", cgq)
        _write_jsonl(output_dir / "cgq_predictions.jsonl", cgq_records)
        del cgq_model
        _release()
    finally:
        shutil.rmtree(mask_dir)

    sample_ids = [
        {
            field: record[field]
            for field in ("doc_id", "doc_hash", "prompt_hash", "target_hash")
        }
        for record in plain_records
    ]
    _write_json(output_dir / "sample_ids.json", sample_ids)
    report = {
        "model": config["model"],
        "calibration_state_count": len(states),
        "calibration_state_digest": calibration_digest,
        "dense_winogrande": dense,
        "plain": plain,
        "cgq": cgq,
        "mask_comparison": mask_comparison,
        "paired_winogrande": paired,
        "excluded_prior_plain_accuracy": config["context"][
            "excluded_prior_plain_accuracy"
        ],
        "gsm8k": "not run (excluded by approved WinoGrande-only scope)",
    }
    _write_json(output_dir / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="codex/cgq_sparsegpt_downstream_50/config.json"
    )
    parser.add_argument(
        "--output-dir", default="codex/cgq_sparsegpt_downstream_50/results"
    )
    args = parser.parse_args()
    run_experiment(args.config, args.output_dir)


if __name__ == "__main__":
    main()
