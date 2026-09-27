"""Freeze native lm-eval prompts on CPU; generate only selected documents."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from .artifacts import checked, digest, freeze, read, sha, write

IDENTITY = ("example_id", "doc_hash", "prompt_hash", "target_hash", "reference_answer")


def task_and_protocol(legacy_config):
    from lm_eval.tasks import TaskManager, get_task_dict
    import lm_eval.tasks.gsm8k as gsm8k
    task_yaml = Path(gsm8k.__path__[0]) / "gsm8k.yaml"
    protocol = dict(model=legacy_config["model"], evaluation=legacy_config["evaluation"],
                    lm_eval_version=importlib.metadata.version("lm_eval"),
                    gsm8k_task_sha256=sha(task_yaml))
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if protocol_hash != legacy_config["protocol_hash"]:
        raise RuntimeError("Installed lm-eval/task/config differs from historical protocol")
    task = get_task_dict(["gsm8k"], task_manager=TaskManager())["gsm8k"]
    task.set_config(key="num_fewshot", value=legacy_config["evaluation"]["num_fewshot"])
    task.set_fewshot_seed(legacy_config["evaluation"]["fewshot_seed"])
    return task, protocol, protocol_hash


def freeze_requests(root, plan, legacy_config):
    from lm_eval.utils import hash_string, handle_non_serializable
    task, protocol, protocol_hash = task_and_protocol(legacy_config)
    if len(task.eval_docs) != plan["gsm8k"]["expected_total"]:
        raise RuntimeError("GSM8K dataset size changed")
    # Build ALL prompts in original order to preserve few-shot RNG consumption.
    # This function never constructs an LM and never generates answers.
    task.build_all_requests(limit=None, rank=0, world_size=1, cache_requests=False)
    wanted = set(plan["gsm8k"]["development_doc_ids"] + plan["gsm8k"]["confirmation_doc_ids"])
    rows = {}
    for instance in task.instances:
        if instance.doc_id not in wanted:
            continue
        if instance.request_type != "generate_until" or instance.repeats != 1 or instance.doc_id in rows:
            raise RuntimeError("Expected one generation request per document")
        target = task.doc_to_target(instance.doc)
        rows[instance.doc_id] = dict(example_id=instance.doc_id, doc=instance.doc,
            prompt=instance.args[0], generation_kwargs=instance.args[1], reference_answer=target,
            doc_hash=hash_string(json.dumps(instance.doc, indent=2, default=handle_non_serializable, ensure_ascii=False)),
            prompt_hash=hash_string(instance.args[0]), target_hash=hash_string(str(target)))
    if set(rows) != wanted:
        raise RuntimeError("Missing selected document requests")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan["model"]["id"], revision=plan["model"]["revision"],
                                              trust_remote_code=True, local_files_only=True)
    for row in rows.values():
        row["input_ids_sha256"] = digest(tokenizer(row["prompt"])["input_ids"])
    result = dict(protocol=protocol, protocol_hash=protocol_hash,
                  dataset_fingerprint=getattr(task.eval_docs, "_fingerprint", None),
                  development=[rows[i] for i in plan["gsm8k"]["development_doc_ids"]],
                  confirmation=[rows[i] for i in plan["gsm8k"]["confirmation_doc_ids"]])
    freeze(Path(root) / "requests.json", result)
    return task, result


def grade(task, request, text):
    from lm_eval.api.instance import Instance
    instance = Instance(request_type="generate_until", doc=request["doc"],
                        arguments=(request["prompt"], request["generation_kwargs"]), idx=0,
                        metadata=("gsm8k", request["example_id"], 1))
    instance.resps = [text]
    old_instances = getattr(task, "_instances", [])
    try:
        task._instances = [instance]
        task.apply_filters()
        extracted = instance.filtered_resps["strict-match"]
        result = task.process_results(request["doc"], [extracted])
    finally:
        task._instances = old_instances
    return dict(extracted_answer=extracted, correct=bool(result["exact_match"]))


def validate_prediction(row, request, protocol_hash):
    from experiments.dlm_capacity_predictor.audit_existing import strict_exact_match
    for key in IDENTITY:
        if row.get(key) != request[key]:
            raise RuntimeError(f"Prediction identity differs at document {request['example_id']}: {key}")
    if row.get("evaluation_config_hash") != protocol_hash:
        raise RuntimeError("Evaluation protocol mismatch")
    if not isinstance(row.get("generated_text"), str) or type(row.get("correct")) is not bool:
        raise RuntimeError("Invalid generation/correctness")
    if strict_exact_match(row["extracted_answer"], row["reference_answer"]) != row["correct"]:
        raise RuntimeError("Strict exact-match does not match stored correctness")


def read_predictions(folder, requests, fingerprint, protocol_hash, allow_partial=False):
    folder = Path(folder)
    found = {int(p.stem):p for p in (folder / "examples").glob("*.json")}
    wanted = {r["example_id"] for r in requests}
    if not set(found) <= wanted:
        raise RuntimeError("Unexpected document checkpoint")
    rows = []
    for request in requests:
        path = found.get(request["example_id"])
        if path is None:
            if allow_partial:
                continue
            raise RuntimeError(f"Missing prediction {request['example_id']}")
        saved = read(path)
        if saved["fingerprint"] != fingerprint or saved["row_sha256"] != digest(saved["row"]):
            raise RuntimeError(f"Prediction checkpoint mismatch: {path}")
        validate_prediction(saved["row"], request, protocol_hash)
        rows.append(saved["row"])
    return rows


def evaluate_requests(folder, requests, fingerprint, protocol_hash, method, generate_one, grade_one, progress):
    """Only missing documents are generated; each atomic checkpoint survives a crash."""
    folder = Path(folder)
    freeze(folder / "identity.json", dict(fingerprint=fingerprint, document_ids=[r["example_id"] for r in requests]))
    cached = {r["example_id"]:r for r in read_predictions(folder, requests, fingerprint, protocol_hash, allow_partial=True)}
    progress("gsm8k", completed=len(cached), total=len(requests), candidate=method)
    for request in requests:
        if request["example_id"] in cached:
            continue
        started = time.monotonic()
        text = generate_one(request)
        if not isinstance(text, str):
            raise RuntimeError("Generation must return text")
        row = {key:request[key] for key in IDENTITY}
        row.update(generated_text=text, **grade_one(request, text), method=method,
                   evaluation_config_hash=protocol_hash, eval_seconds=time.monotonic()-started)
        validate_prediction(row, request, protocol_hash)
        freeze(folder / "examples" / f"{request['example_id']:04d}.json",
               dict(fingerprint=fingerprint, row=row, row_sha256=digest(row)))
        cached[request["example_id"]] = row
        progress("gsm8k", completed=len(cached), total=len(requests), candidate=method)
    rows = read_predictions(folder, requests, fingerprint, protocol_hash)
    prediction_path = folder / "predictions.json"
    freeze(prediction_path, rows)
    result = dict(status="complete", fingerprint=fingerprint, total=len(rows),
                  correct=sum(r["correct"] for r in rows), predictions_sha256=sha(prediction_path),
                  generation_seconds=sum(r.get("eval_seconds", 0) for r in rows))
    freeze(folder / "results.json", result)
    return result


def generator(model, tokenizer, evaluation, mask_id):
    """Use the repository's native generate_until without invoking full evaluator."""
    from eval_llada import LLaDAEvalHarness, set_seed
    from lm_eval.api.instance import Instance
    set_seed(evaluation["torch_seed"])
    harness = LLaDAEvalHarness(model=model, tokenizer=tokenizer, mask_id=mask_id,
        batch_size=1, mc_num=1, is_check_greedy=False, cfg=0.0,
        steps=evaluation["denoising_steps"], gen_length=evaluation["generation_length"],
        block_length=evaluation["block_length"], device="cuda")

    def one(request):
        if digest(tokenizer(request["prompt"])["input_ids"]) != request["input_ids_sha256"]:
            raise RuntimeError("Tokenized prompt differs from CPU preparation")
        instance = Instance(request_type="generate_until", doc=request["doc"],
            arguments=(request["prompt"], request["generation_kwargs"]), idx=0,
            metadata=("gsm8k", request["example_id"], 1))
        return harness.generate_until([instance])[0]

    return one
