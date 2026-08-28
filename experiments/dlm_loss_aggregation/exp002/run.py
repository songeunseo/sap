import hashlib
import json
import argparse
import gc
import importlib.metadata
import resource
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from experiments.dlm_loss_aggregation.core import mask_sha256, pack_mask, unpack_mask
from experiments.dlm_loss_aggregation.run import _atomic_write_csv, _atomic_write_json, _load_model


METHODS = ("sum", "abs", "square")
METHOD_ORDER = ("Dense", "DLM-SUM", "DLM-ABS", "DLM-SQUARE", "Wanda", "SparseGPT")
CALIBRATION = {
    "Dense": ("none", 0),
    "DLM-SUM": ("dlm_8x10x256", 80),
    "DLM-ABS": ("dlm_8x10x256", 80),
    "DLM-SQUARE": ("dlm_8x10x256", 80),
    "Wanda": ("clean_8x256", 8),
    "SparseGPT": ("clean_8x256", 8),
}


def _canonical_json(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _overall_mask_hash(entries, method):
    hasher = hashlib.sha256()
    for entry in entries:
        if entry["method"] == method:
            hasher.update(
                _canonical_json(
                    {
                        key: entry[key]
                        for key in ("block_index", "module", "shape", "sha256")
                    }
                )
            )
    return hasher.hexdigest()


def _load_mask(entry):
    bits = Path(entry["runtime_path"]).read_bytes()
    payload = {"shape": entry["shape"], "bits": bits}
    if len(bits) != entry["byte_length"] or mask_sha256(payload) != entry["sha256"]:
        raise ValueError(f"mask hash mismatch: {entry['runtime_path']}")
    return unpack_mask(payload)


def validate_exp001_masks(metadata, score_metadata, expected_matrices=224):
    entries = metadata["entries"]
    runtime_directory = Path(metadata["runtime_directory"])
    files = {str(path) for path in runtime_directory.rglob("*") if path.is_file()}
    expected_files = {entry["runtime_path"] for entry in entries}
    if files != expected_files:
        raise ValueError("mask file set mismatch")
    counts = Counter(entry["method"] for entry in entries)
    if counts != Counter({method: expected_matrices for method in METHODS}):
        raise ValueError("mask method count mismatch")
    modules = {
        (row["layer"], row["module"]): row for row in score_metadata["modules"]
    }
    if (
        score_metadata["matrix_count"] != expected_matrices
        or score_metadata["state_count"] != 80
        or len(modules) != expected_matrices
        or any(row["update_count"] != 80 for row in modules.values())
    ):
        raise ValueError("EXP-001 score metadata mismatch")
    total_bytes = 0
    for entry in entries:
        key = (entry["block_index"], entry["module"])
        if key not in modules or entry["shape"] != modules[key]["shape"]:
            raise ValueError("mask shape mismatch")
        mask = _load_mask(entry)
        if not mask.sum(dim=1).eq(mask.shape[1] // 2).all().item():
            raise ValueError("mask row sparsity mismatch")
        total_bytes += entry["byte_length"]
        del mask
    if total_bytes != metadata["total_bytes"]:
        raise ValueError("mask byte total mismatch")
    overall = {method: _overall_mask_hash(entries, method) for method in METHODS}
    if overall != metadata["overall_sha256"]:
        raise ValueError("overall mask hash mismatch")
    return {"counts": dict(counts), "total_bytes": total_bytes, "overall_sha256": overall}


@torch.no_grad()
def apply_dlm_masks(model, entries, expected_hash, get_modules):
    modules = get_modules(model)
    applied = []
    zero_count = 0
    weight_count = 0
    for entry in entries:
        key = (entry["block_index"], entry["module"])
        if key not in modules:
            raise ValueError(f"model module missing: {key}")
        layer = modules[key]
        mask = _load_mask(entry)
        if list(layer.weight.shape) != entry["shape"]:
            raise ValueError(f"model shape mismatch: {key}")
        layer.weight.masked_fill_(mask.to(layer.weight.device), 0)
        actual = layer.weight.eq(0).detach().cpu()
        if not torch.equal(actual, mask):
            raise ValueError(f"applied mask mismatch: {key}")
        payload = pack_mask(actual)
        reproduced = {**entry, "sha256": mask_sha256(payload)}
        applied.append(reproduced)
        zero_count += int(actual.sum().item())
        weight_count += actual.numel()
        del actual, mask, payload
    method = entries[0]["method"]
    actual_hash = _overall_mask_hash(applied, method)
    if actual_hash != expected_hash:
        raise ValueError("applied overall mask hash mismatch")
    return {"mask_hash": actual_hash, "sparsity": zero_count / weight_count}


def _first(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return value


def extract_gsm8k_records(samples, method, evaluation_config_hash):
    records = []
    for sample in samples:
        if sample["filter"] != "strict-match":
            continue
        records.append(
            {
                "example_id": sample["doc_id"],
                "doc_hash": sample["doc_hash"],
                "prompt_hash": sample["prompt_hash"],
                "target_hash": sample["target_hash"],
                "reference_answer": sample["target"],
                "generated_text": _first(sample["resps"]),
                "extracted_answer": _first(sample["filtered_resps"]),
                "correct": bool(sample["exact_match"]),
                "method": method,
                "evaluation_config_hash": evaluation_config_hash,
            }
        )
    if len({record["example_id"] for record in records}) != len(records):
        raise ValueError("duplicate GSM8K examples")
    return records


def paired_correctness(left, right):
    if len(left) != len(right) or any(
        (a["example_id"], a["doc_hash"]) != (b["example_id"], b["doc_hash"])
        for a, b in zip(left, right)
    ):
        raise ValueError("paired examples differ")
    return {
        "both_correct": sum(a["correct"] and b["correct"] for a, b in zip(left, right)),
        "a_correct_b_wrong": sum(a["correct"] and not b["correct"] for a, b in zip(left, right)),
        "a_wrong_b_correct": sum(not a["correct"] and b["correct"] for a, b in zip(left, right)),
        "both_wrong": sum(not a["correct"] and not b["correct"] for a, b in zip(left, right)),
    }


def dense_fingerprint(modules, rows_per_chunk=64):
    hasher = hashlib.sha256()
    for (block_index, name), layer in sorted(modules.items()):
        weight = layer.weight.detach()
        hasher.update(
            _canonical_json(
                {
                    "block_index": block_index,
                    "module": name,
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype),
                }
            )
        )
        for start in range(0, weight.shape[0], rows_per_chunk):
            chunk = weight[start : start + rows_per_chunk].contiguous().view(torch.uint8)
            hasher.update(chunk.cpu().numpy().tobytes())
    return hasher.hexdigest()


def validate_timing_gate(rows, methods, max_cuda_bytes, slowdown_factor):
    if [row["method"] for row in rows] != list(methods):
        raise ValueError("timing methods are missing or out of order")
    if any(row["peak_cuda_reserved_bytes"] > max_cuda_bytes for row in rows):
        raise RuntimeError("timing CUDA memory exceeds the safe limit")
    speeds = [row["examples_per_second"] for row in rows]
    if any(not isinstance(speed, (int, float)) or speed <= 0 for speed in speeds):
        raise RuntimeError("timing speed is invalid")
    ratio = max(speeds) / min(speeds)
    if ratio > slowdown_factor:
        raise RuntimeError("timing speed is pathologically different across methods")
    return {"passed": True, "slowest_to_fastest_ratio": ratio}


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected = {
        "experiment": "exp002_gsm8k_aggregation_evaluation",
        "model": {
            "id": "GSAI-ML/LLaDA-8B-Base",
            "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
            "dtype": "bfloat16",
        },
        "order": list(METHOD_ORDER),
        "task": "gsm8k",
        "num_fewshot": 5,
        "temperature": 0,
        "generation_length": 256,
        "block_length": 256,
        "denoising_steps": 256,
        "timing_examples": 32,
    }
    for name, wanted in expected.items():
        if name in ("order", "task", "num_fewshot", "temperature", "generation_length", "block_length", "denoising_steps", "timing_examples"):
            actual = config["evaluation"][name]
        else:
            actual = config[name]
        if actual != wanted:
            raise ValueError(f"EXP-002 config mismatch: {name}")
    source = config["source_exp001"]
    if (
        source["run_id"] != "20260828T175010-2484545"
        or source["mask_directory"] != "/dev/shm/dlm_loss_aggregation_20260828T175010-2484545"
        or source["matrix_count"] != 224
        or source["state_count"] != 80
    ):
        raise ValueError("EXP-001 source config mismatch")
    return config


def _module_map(model):
    from lib.prune_llada import find_layers

    return {
        (block_index, name): layer
        for block_index, block in enumerate(model.model.transformer.blocks)
        for name, layer in find_layers(block).items()
    }


def _validate_dense_model(model, config, score_metadata):
    revision = getattr(model.config, "_commit_hash", None)
    if revision != config["model"]["revision"]:
        raise ValueError(f"model revision mismatch: {revision}")
    modules = _module_map(model)
    expected = {
        (row["layer"], row["module"]): row["shape"]
        for row in score_metadata["modules"]
    }
    actual = {key: list(layer.weight.shape) for key, layer in modules.items()}
    if len(modules) != config["source_exp001"]["matrix_count"] or actual != expected:
        raise ValueError("dense model module shapes differ from EXP-001")
    return modules


def _clean_calibration_digest(config, tokenizer):
    from lib.data import get_loaders

    baseline = config["baselines"]
    loader, _ = get_loaders(
        baseline["dataset"],
        nsamples=baseline["calibration_states"],
        seed=baseline["seed"],
        seqlen=baseline["sequence_length"],
        tokenizer=tokenizer,
    )
    hasher = hashlib.sha256()
    for index, (input_ids, _) in enumerate(loader):
        hasher.update(_canonical_json({"index": index, "shape": list(input_ids.shape)}))
        hasher.update(input_ids.contiguous().view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _zero_mask_summary(model, method):
    entries = []
    zero_count = 0
    weight_count = 0
    for (block_index, name), layer in sorted(_module_map(model).items()):
        mask = layer.weight.eq(0).detach().cpu()
        if not mask.sum(dim=1).eq(mask.shape[1] // 2).all().item():
            raise ValueError(f"wrong row sparsity after {method}: {(block_index, name)}")
        payload = pack_mask(mask)
        entries.append(
            {
                "method": method,
                "block_index": block_index,
                "module": name,
                "shape": list(mask.shape),
                "sha256": mask_sha256(payload),
            }
        )
        zero_count += int(mask.sum().item())
        weight_count += mask.numel()
        del mask, payload
    return {
        "mask_hash": _overall_mask_hash(entries, method),
        "sparsity": zero_count / weight_count,
        "matrix_count": len(entries),
    }


def _evaluation_config_hash(config):
    task_yaml = Path(importlib.import_module("lm_eval.tasks.gsm8k").__path__[0]) / "gsm8k.yaml"
    document = {
        "model": config["model"],
        "evaluation": config["evaluation"],
        "lm_eval_version": importlib.metadata.version("lm_eval"),
        "gsm8k_task_sha256": hashlib.sha256(task_yaml.read_bytes()).hexdigest(),
    }
    return hashlib.sha256(_canonical_json(document)).hexdigest(), document


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
    temporary.replace(path)


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _evaluate_gsm8k(model, tokenizer, config, method, limit, config_hash):
    from lm_eval import evaluator
    from eval_llada import LLaDAEvalHarness, set_seed

    evaluation = config["evaluation"]
    set_seed(evaluation["torch_seed"])
    harness = LLaDAEvalHarness(
        model=model,
        tokenizer=tokenizer,
        mask_id=model.config.mask_token_id,
        batch_size=1,
        mc_num=1,
        is_check_greedy=False,
        cfg=0.0,
        steps=evaluation["denoising_steps"],
        gen_length=evaluation["generation_length"],
        block_length=evaluation["block_length"],
        device="cuda",
    )
    started = time.monotonic()
    result = evaluator.simple_evaluate(
        model=harness,
        tasks=[evaluation["task"]],
        num_fewshot=evaluation["num_fewshot"],
        batch_size=1,
        limit=limit,
        bootstrap_iters=evaluation["bootstrap_iters"],
        log_samples=True,
        random_seed=evaluation["random_seed"],
        numpy_random_seed=evaluation["numpy_seed"],
        torch_random_seed=evaluation["torch_seed"],
        fewshot_random_seed=evaluation["fewshot_seed"],
    )
    elapsed = time.monotonic() - started
    records = extract_gsm8k_records(result["samples"]["gsm8k"], method, config_hash)
    if not records or any(not isinstance(row["generated_text"], str) for row in records):
        raise ValueError("GSM8K generations are missing or corrupt")
    accuracy = result["results"]["gsm8k"][evaluation["primary_metric"]]
    if sum(row["correct"] for row in records) / len(records) != accuracy:
        raise ValueError("GSM8K compact samples do not match aggregate exact-match")
    original = result["n-samples"]["gsm8k"]["original"]
    del result, harness
    return {
        "accuracy": accuracy,
        "num_examples": len(records),
        "eval_seconds": elapsed,
        "examples_per_second": len(records) / elapsed,
        "full_num_examples": original,
    }, records


def _prune(model, tokenizer, method, config, mask_metadata):
    if method.startswith("DLM-"):
        key = method.removeprefix("DLM-").lower()
        entries = [entry for entry in mask_metadata["entries"] if entry["method"] == key]
        return apply_dlm_masks(
            model,
            entries,
            mask_metadata["overall_sha256"][key],
            _module_map,
        )
    from lib.prune_llada import prune_sparsegpt, prune_wanda

    baseline = config["baselines"]
    args = SimpleNamespace(
        nsamples=baseline["calibration_states"],
        seed=baseline["seed"],
        sparsity_ratio=config["pruning"]["sparsity"],
        use_variant=baseline["wanda_variant"],
    )
    if method == "Wanda":
        prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"))
    elif method == "SparseGPT":
        prune_sparsegpt(args, model, tokenizer, dev=torch.device("cuda:0"))
    else:
        raise ValueError(f"unsupported pruning method: {method}")
    return _zero_mask_summary(model, method.lower())


def _release():
    gc.collect()
    torch.cuda.empty_cache()


def _method_run(
    method,
    config,
    score_metadata,
    mask_metadata,
    tokenizer,
    dense_hash,
    config_hash,
    limit,
):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    method_started = time.monotonic()
    model, _ = _load_model({
        "model": config["model"],
        "calibration": {"sequence_length": config["baselines"]["sequence_length"]},
    })
    modules = _validate_dense_model(model, config, score_metadata)
    actual_dense_hash = dense_fingerprint(modules)
    if actual_dense_hash != dense_hash:
        raise ValueError(f"dense fingerprint mismatch before {method}")
    pruning_seconds = 0.0
    mask_result = {"mask_hash": "", "sparsity": 0.0, "matrix_count": len(modules)}
    if method != "Dense":
        pruning_started = time.monotonic()
        mask_result = _prune(model, tokenizer, method, config, mask_metadata)
        pruning_seconds = time.monotonic() - pruning_started
        if mask_result["sparsity"] != config["pruning"]["sparsity"]:
            raise ValueError(f"wrong achieved sparsity for {method}")
    measured, records = _evaluate_gsm8k(
        model, tokenizer, config, method, limit, config_hash
    )
    row = {
        "method": method,
        "sparsity": mask_result["sparsity"],
        **measured,
        "projected_full_seconds": measured["full_num_examples"] / measured["examples_per_second"],
        "pruning_seconds": pruning_seconds,
        "method_seconds": time.monotonic() - method_started,
        "model_revision": config["model"]["revision"],
        "dense_fingerprint": actual_dense_hash,
        "mask_hash": mask_result["mask_hash"],
        "calibration_type": CALIBRATION[method][0],
        "calibration_states": CALIBRATION[method][1],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "evaluation_config_hash": config_hash,
    }
    del model
    _release()
    return row, records


def _assert_same_examples(reference, candidate):
    if len(reference) != len(candidate) or any(
        (left["example_id"], left["doc_hash"], left["prompt_hash"], left["target_hash"])
        != (right["example_id"], right["doc_hash"], right["prompt_hash"], right["target_hash"])
        for left, right in zip(reference, candidate)
    ):
        raise ValueError("GSM8K evaluator examples or prompts differ across methods")


def _run_phase(config, score_metadata, mask_metadata, tokenizer, dense_hash, config_hash, limit):
    rows = []
    records_by_method = {}
    reference = None
    for method in METHOD_ORDER:
        print(json.dumps({"event": "method_start", "method": method, "limit": limit}), flush=True)
        row, records = _method_run(
            method, config, score_metadata, mask_metadata, tokenizer, dense_hash, config_hash, limit
        )
        if reference is None:
            reference = records
        else:
            _assert_same_examples(reference, records)
        rows.append(row)
        records_by_method[method] = records
        print(json.dumps({"event": "method_complete", **row}, sort_keys=True), flush=True)
    return rows, records_by_method


def _paired_rows(records):
    pairs = (
        ("DLM-SUM", "DLM-ABS"),
        ("DLM-SUM", "DLM-SQUARE"),
        ("DLM-ABS", "DLM-SQUARE"),
        ("DLM-ABS", "Wanda"),
        ("DLM-ABS", "SparseGPT"),
        ("DLM-SQUARE", "Wanda"),
        ("DLM-SQUARE", "SparseGPT"),
    )
    return [
        {"method_a": left, "method_b": right, **paired_correctness(records[left], records[right])}
        for left, right in pairs
    ]


def _report(config, rows, paired):
    by_method = {row["method"]: row for row in rows}
    report_order = ("Dense", "Wanda", "SparseGPT", "DLM-SUM", "DLM-ABS", "DLM-SQUARE")
    table = [
        "| Method | Sparsity | Accuracy | Correct / N | Eval seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in report_order:
        row = by_method[method]
        correct = round(row["accuracy"] * row["num_examples"])
        table.append(
            f"| {method} | {row['sparsity']:.1%} | {row['accuracy']:.6f} | {correct} / {row['num_examples']} | {row['eval_seconds']:.1f} |"
        )
    paired_table = [
        "| Pair | Both correct | A correct / B wrong | A wrong / B correct | Both wrong |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in paired:
        paired_table.append(
            f"| {row['method_a']} vs {row['method_b']} | {row['both_correct']} | {row['a_correct_b_wrong']} | {row['a_wrong_b_correct']} | {row['both_wrong']} |"
        )
    sum_acc = by_method["DLM-SUM"]["accuracy"]
    abs_acc = by_method["DLM-ABS"]["accuracy"]
    square_acc = by_method["DLM-SQUARE"]["accuracy"]
    if sum_acc > abs_acc and sum_acc >= square_acc:
        decision = "signed directionality appears useful"
    elif square_acc > abs_acc:
        decision = "square appears useful"
    elif abs_acc >= square_acc:
        decision = "square not justified"
    else:
        decision = "inconclusive / requires additional evaluation"
    return f"""# EXP-002 — GSM8K Aggregation Evaluation

## Question

Does SUM, ABS, or SQUARE aggregation of the exact same official DLM gradients produce better 50% pruned LLaDA generation performance?

## Fixed Protocol

- Model: `GSAI-ML/LLaDA-8B-Base` revision `{config['model']['revision']}` in BF16
- Task: GSM8K, 5-shot, repository `LLaDAEvalHarness` and lm-eval `gsm8k` strict-match exact match
- Generation: temperature 0, length 256, block length 256, denoising steps 256
- Pruning: 50% unstructured row-wise over the same 224 Linear matrices
- DLM methods: the frozen EXP-001 masks from run `20260828T175010-2484545`
- Wanda/SparseGPT: standard reference baselines using eight clean WikiText-2 256-token spans, seed 0
- Baseline caveat: Wanda/SparseGPT are not calibration-compute-matched to the DLM methods' 80 corrupted states.

## EXP-001 Context

| Pair | Spearman | Mask XOR |
|---|---:|---:|
| SUM / ABS | -0.000048 | 0.500045 |
| SUM / SQUARE | -0.000045 | 0.500046 |
| ABS / SQUARE | 0.992708 | 0.028463 |

- Sign consistency mean: 0.127214
- Spike ratio mean: 1.249317

## GSM8K Results

{chr(10).join(table)}

## Paired Correctness

{chr(10).join(paired_table)}

## Interpretation

SUM versus ABS directly tests whether retaining signed directionality helps or whether cross-state cancellation harms the ranking. ABS versus SQUARE tests whether the 2.85% mask disagreement induced by emphasizing large state-wise effects changes downstream exact match. These are observed performance comparisons on one benchmark and do not establish causality. Wanda and SparseGPT are standard reference baselines, not calibration-compute-matched controls.

## Decision

**{decision}.** This label follows the observed ordering only; paired counts should be considered before treating a small accuracy difference as robust.

## Next Step

No additional experiment is launched automatically.
"""


def run_experiment(config_path, output_root):
    from transformers import AutoTokenizer

    config = load_config(config_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source = config["source_exp001"]
    mask_metadata = json.loads(Path(source["mask_metadata"]).read_text())
    score_metadata = json.loads(Path(source["score_metadata"]).read_text())
    validation = validate_exp001_masks(mask_metadata, score_metadata, source["matrix_count"])
    if (
        mask_metadata["runtime_directory"] != source["mask_directory"]
        or validation["overall_sha256"] != source["overall_sha256"]
    ):
        raise ValueError("EXP-001 run directory or overall hashes mismatch")
    _atomic_write_json(output_root / "logs" / "mask_validation.json", validation)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    calibration_digest = _clean_calibration_digest(config, tokenizer)
    config_hash, config_hash_document = _evaluation_config_hash(config)
    _atomic_write_json(
        output_root / "logs" / "evaluation_config.json",
        {"sha256": config_hash, **config_hash_document, "clean_calibration_sha256": calibration_digest},
    )

    model, _ = _load_model({
        "model": config["model"],
        "calibration": {"sequence_length": config["baselines"]["sequence_length"]},
    })
    modules = _validate_dense_model(model, config, score_metadata)
    dense_hash = dense_fingerprint(modules)
    dense_validation, dense_records = _evaluate_gsm8k(
        model, tokenizer, config, "Dense", 1, config_hash
    )
    _atomic_write_json(
        output_root / "logs" / "dense_validation.json",
        {**dense_validation, "dense_fingerprint": dense_hash, "record": dense_records[0]},
    )
    del model
    _release()

    timing_rows, timing_records = _run_phase(
        config, score_metadata, mask_metadata, tokenizer, dense_hash, config_hash,
        config["evaluation"]["timing_examples"],
    )
    _atomic_write_csv(output_root / "results" / "timing.csv", timing_rows, list(timing_rows[0]))
    if len(list(__import__("csv").DictReader((output_root / "results" / "timing.csv").open()))) != len(METHOD_ORDER):
        raise ValueError("timing CSV readback failed")
    gate = validate_timing_gate(
        timing_rows,
        METHOD_ORDER,
        int(config["evaluation"]["max_cuda_gib"] * 1024**3),
        config["evaluation"]["pathological_slowdown_factor"],
    )
    _atomic_write_json(
        output_root / "logs" / "timing_gate.json",
        {**gate, "timing": timing_rows},
    )
    print(json.dumps({"event": "timing_gate", **gate}, sort_keys=True), flush=True)

    baseline_timing = {
        method: timing_rows[METHOD_ORDER.index(method)]["mask_hash"]
        for method in ("Wanda", "SparseGPT")
    }
    full_rows, full_records = _run_phase(
        config, score_metadata, mask_metadata, tokenizer, dense_hash, config_hash, None
    )
    for method in ("Wanda", "SparseGPT"):
        if full_rows[METHOD_ORDER.index(method)]["mask_hash"] != baseline_timing[method]:
            raise ValueError(f"{method} mask hash differs between timing and full runs")
    prediction_dir = output_root / "results" / "predictions"
    for method, records in full_records.items():
        path = prediction_dir / f"{method.lower().replace('-', '_')}.jsonl"
        _write_jsonl(path, records)
        if _read_jsonl(path) != records:
            raise ValueError(f"prediction readback failed: {method}")
    gsm_fields = (
        "method", "sparsity", "accuracy", "num_examples", "eval_seconds",
        "examples_per_second", "model_revision", "mask_hash", "calibration_type",
        "calibration_states", "evaluation_config_hash", "dense_fingerprint",
    )
    _atomic_write_csv(output_root / "results" / "gsm8k.csv", full_rows, gsm_fields)
    paired = _paired_rows(full_records)
    _atomic_write_csv(output_root / "results" / "paired_correctness.csv", paired, list(paired[0]))
    _atomic_write_json(
        output_root / "masks" / "baseline_hashes.json",
        {
            "model_revision": config["model"]["revision"],
            "dense_fingerprint": dense_hash,
            "clean_calibration_sha256": calibration_digest,
            "calibration_type": "clean_8x256",
            "calibration_states": 8,
            "seed": 0,
            "hashes": baseline_timing,
        },
    )
    report = _report(config, full_rows, paired)
    report_path = output_root / "report.md"
    report_path.write_text(report, encoding="utf-8")
    if len(list(__import__("csv").DictReader((output_root / "results" / "gsm8k.csv").open()))) != len(METHOD_ORDER):
        raise ValueError("GSM8K CSV readback failed")
    return {"timing": timing_rows, "gsm8k": full_rows, "paired": paired}


def main(argv=None):
    parser = argparse.ArgumentParser(description="EXP-002 GSM8K aggregation evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/dlm_loss_aggregation/exp002/config.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/dlm_loss_aggregation/exp002"),
    )
    args = parser.parse_args(argv)
    run_experiment(args.config, args.output_root)


if __name__ == "__main__":
    main()
