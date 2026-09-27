"""Checkpointed Dense/A NFE workers; one tmux worker and one GPU per arm."""
from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_multiscale_ac50.artifacts import Progress, checked, digest, freeze, lock, read, sha, write
from experiments.dlm_multiscale_ac50.evaluation import (IDENTITY, grade, generator,
    read_predictions, task_and_protocol, validate_prediction)
from experiments.dlm_pruning_nfe50 import diagnostic, prepare

ROOT = Path(__file__).parent / "output"
STEPS = (256, 64, 32)
EXPECTED = {"Dense": "2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc",
            "A": "7d5dc817890f1079a6a8f2970088c7bee80afa60c11ebea30cb435d14d40922c"}
MASK_HASH = "fd22875b43d65e22ba2c0091b0f9b552e5f73d524cdae388f73030075c0c1d10"
PRUNED = 3_489_660_928


class MeteredModel:
    def __init__(self, model, counter, capture=None):
        self.model, self.counter, self.capture = model, counter, capture
        self.call_index = 0
        self.states = []

    @property
    def device(self):
        return self.model.device

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, ids, *args, **kwargs):
        # Count attempted calls before forwarding, so failed forwards remain visible.
        self.counter["forwards"] += 1
        self.counter["forward_tokens"] += int(ids.numel())
        output = self.model(ids, *args, **kwargs)
        if self.capture and self.call_index in self.capture:
            self.states.append(capture_state(ids, output.logits, self.call_index))
        self.call_index += 1
        return output

    def reset(self):
        self.call_index = 0
        self.states = []


def capture_state(ids, logits, step):
    """Use the decoder's native BF16 softmax, argmax and masked-position Top-k."""
    if ids.shape[0] != 1:
        raise RuntimeError("Diagnostic requires batch size one")
    mask = ids == 126336
    x0 = torch.argmax(logits, dim=-1)
    confidence = F.softmax(logits, dim=-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(mask, confidence, -torch.inf)
    _, selected = torch.topk(confidence[0], k=8)
    positions = selected.tolist()
    tokens = x0[0, selected].tolist()
    scores = confidence[0, selected].tolist()
    before = ids[0].tolist()
    after = before.copy()
    after[positions[0]] = tokens[0]
    items = [dict(position=int(p), token_id=int(t), native_confidence=float(c))
             for p, t, c in zip(positions, tokens, scores)]
    return dict(step=step, input_ids=before, input_ids_sha256=digest(before),
                after_input_ids_sha256=digest(after), source=items[0], targets=items[1:],
                mask_fraction=float(mask[0].sum().item() / ids.numel()),
                response_mask_fraction=float(mask[0].sum().item() / 256),
                masked_count=int(mask[0].sum().item()), input_length=int(ids.numel()))


def ensure_environment():
    if not os.environ.get("TMUX"):
        raise RuntimeError("Worker must run inside tmux")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(gpu.split(",")) != 1 or not gpu.strip().isdigit():
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU index")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA device must be visible")
    return gpu


def validate_config(config_path):
    config_path = Path(config_path).resolve()
    cfg = prepare.validate(config_path)
    if config_path.parent != ROOT.resolve():
        raise RuntimeError("Unexpected output directory")
    identity = cfg["identity"]
    if identity["mask_hash"] != MASK_HASH or int(identity["pruned_count"]) != PRUNED:
        raise RuntimeError("A mask identity/budget mismatch")
    if identity["model_sha256_dense"] != EXPECTED["Dense"] or identity["model_sha256_A"] != EXPECTED["A"]:
        raise RuntimeError("Expected model identity differs")
    if cfg["diagnostic"] != dict(steps=32, capture_steps=[0, 8, 16, 24], top_k=8, readout_dtype="float32"):
        raise RuntimeError("Diagnostic protocol differs")
    for key in ("config_path", "requests_path", "manifest_path", "selected_requests_path"):
        checked(cfg["source"][key], cfg["source"][key.replace("_path", "_sha256")])
    return cfg


def load_model(cfg, arm):
    from experiments.projection_capacity_allocation_65.run import load_dense
    from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    from experiments.dlm_multiscale_ac50.artifacts import mask_identity
    from transformers import AutoTokenizer
    model, mapping = load_dense()
    if model_sha(model) != EXPECTED["Dense"]:
        raise RuntimeError("Dense physical model hash mismatch")
    if arm == "A":
        manifest = read(cfg["source"]["manifest_path"])
        if mask_identity(manifest) != MASK_HASH or int(manifest["pruned"]) != PRUNED:
            raise RuntimeError("A physical manifest mismatch")
        if apply_manifest(model, mapping, manifest) != PRUNED:
            raise RuntimeError("A prune count mismatch")
    actual_hash = model_sha(model)
    if actual_hash != EXPECTED[arm]:
        raise RuntimeError("Loaded model weights mismatch")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"],
        revision=cfg["model"]["revision"], trust_remote_code=True, local_files_only=True)
    return model, tokenizer, actual_hash


def attempt_path(root, arm, step, example_id):
    return root / "attempts" / f"{arm}_{step}_{example_id:04d}_{time.time_ns()}.json"


def run_cell(root, cfg, arm, step, requests, task, model, tokenizer, counter, progress):
    cell = prepare.cell_info(cfg, arm, step)
    evaluation = cell["evaluation"]
    protocol_hash = cell["protocol_hash"]
    fingerprint = cell["fingerprint"]
    folder = root / "gsm8k" / f"{arm}_{step}"
    freeze(folder / "identity.json", dict(fingerprint=fingerprint, document_ids=[r["example_id"] for r in requests]))
    cached = {r["example_id"]: r for r in read_predictions(folder, requests, fingerprint, protocol_hash, allow_partial=True)}
    wrapper = MeteredModel(model, counter, capture=(0, 8, 16, 24) if arm == "Dense" and step == 32 else None)
    generate_one = generator(wrapper, tokenizer, evaluation, 126336)
    dense_cache_receipt = root / "instrumentation" / "Dense256_cache_parity.json"
    if arm == "Dense" and step == 256 and len(cached) == len(requests) and not dense_cache_receipt.exists():
        anchor = requests[0]
        if anchor["example_id"] != 0:
            raise RuntimeError("Dense256 cache parity anchor changed")
        start = time.monotonic()
        before = counter.copy()
        status = "failed"
        try:
            text = generate_one(anchor)
            if text != cached[0]["generated_text"]:
                raise RuntimeError("Dense256 cache parity changed generated text")
            actual_grade = grade(task, anchor, text)
            if actual_grade != {k: cached[0][k] for k in ("correct", "extracted_answer")}:
                raise RuntimeError("Dense256 cache parity changed official grade")
            freeze(dense_cache_receipt, dict(example_id=0, generated_text_sha256=digest(text),
                row_sha256=digest(cached[0]), protocol_hash=protocol_hash,
                model_sha256=EXPECTED["Dense"], validation_forwards=256))
            status = "complete"
        finally:
            write(attempt_path(root, arm, "cache_validation", 0),
                  dict(arm=arm, stage="cache_validation", example_id=0, status=status,
                       seconds=time.monotonic()-start,
                       forwards=counter["forwards"]-before["forwards"],
                       forward_tokens=counter["forward_tokens"]-before["forward_tokens"]))
    parity_text = None
    parity_id = None
    parity_receipt = root / "instrumentation" / "Dense32_parity.json"
    if arm == "Dense" and step == 32 and not parity_receipt.exists():
        parity_request = next((r for r in requests if r["example_id"] not in cached), None)
        if parity_request is not None:
            reference_wrapper = MeteredModel(model, counter)
            reference_one = generator(reference_wrapper, tokenizer, evaluation, 126336)
            start = time.monotonic()
            before = counter.copy()
            parity_id = parity_request["example_id"]
            status = "failed"
            try:
                parity_text = reference_one(parity_request)
                status = "complete"
            finally:
                write(attempt_path(root, arm, "instrumentation", parity_id),
                      dict(arm=arm, stage="instrumentation", example_id=parity_id,
                           status=status, seconds=time.monotonic()-start,
                           forwards=counter["forwards"]-before["forwards"],
                           forward_tokens=counter["forward_tokens"]-before["forward_tokens"]))
    progress("generation", candidate=f"{arm}_{step}", completed=len(cached), total=len(requests))
    for request in requests:
        example_id = request["example_id"]
        trace_file = diagnostic.trace_path(root, example_id)
        if example_id in cached and not (arm == "Dense" and step == 32 and not trace_file.exists()):
            continue
        recapture = example_id in cached
        wrapper.reset()
        started = time.monotonic()
        count_before = counter.copy()
        status = "failed"
        try:
            text = generate_one(request)
            if example_id == parity_id:
                if text != parity_text:
                    raise RuntimeError("Dense32 trace hook changed native generation")
                freeze(parity_receipt, dict(example_id=example_id,
                    reference_text_sha256=digest(parity_text), hooked_text_sha256=digest(text),
                    model_sha256=EXPECTED["Dense"], protocol_hash=protocol_hash,
                    reference_forwards=32))
            if arm == "Dense" and step == 32:
                if len(wrapper.states) != 4 or wrapper.call_index != 32:
                    raise RuntimeError("Incomplete Dense32 trace")
                trace = dict(example_id=example_id, prompt_hash=request["prompt_hash"],
                             config_identity=cfg["config_identity"], states=wrapper.states,
                             generated_text_sha256=digest(text))
                diagnostic.check_trace(root, request, trace, cfg["config_identity"])
                freeze(trace_file, trace)  # Trace precedes the answer checkpoint.
            if recapture:
                if text != cached[example_id]["generated_text"]:
                    raise RuntimeError("Trace recapture changed generated text")
                status = "trace_recapture"
                continue
            row = {key: request[key] for key in IDENTITY}
            row.update(generated_text=text, **grade(task, request, text), method=arm,
                       evaluation_config_hash=protocol_hash, eval_seconds=time.monotonic()-started,
                       denoising_steps=step)
            validate_prediction(row, request, protocol_hash)
            freeze(folder / "examples" / f"{example_id:04d}.json",
                   dict(fingerprint=fingerprint, row=row, row_sha256=digest(row)))
            cached[example_id] = row
            status = "complete"
            progress("generation", candidate=f"{arm}_{step}", completed=len(cached), total=len(requests))
        finally:
            write(attempt_path(root, arm, step, example_id),
                  dict(arm=arm, step=step, example_id=example_id, status=status,
                       seconds=time.monotonic()-started,
                       forwards=counter["forwards"]-count_before["forwards"],
                       forward_tokens=counter["forward_tokens"]-count_before["forward_tokens"],
                       gpu=os.environ["CUDA_VISIBLE_DEVICES"]))
    rows = read_predictions(folder, requests, fingerprint, protocol_hash)
    freeze(folder / "predictions.json", rows)
    result_path = folder / "results.json"
    if result_path.exists():
        existing = read(result_path)
        if (existing["total"] != len(rows) or existing["correct"] != sum(r["correct"] for r in rows)
                or existing["predictions_sha256"] != sha(folder / "predictions.json")
                or existing["fingerprint"] != fingerprint):
            raise RuntimeError("Existing cell result mismatch")
    else:
        freeze(result_path, dict(status="complete", arm=arm, denoising_steps=step,
            total=len(rows), correct=sum(r["correct"] for r in rows),
            predictions_sha256=sha(folder / "predictions.json"), fingerprint=fingerprint))


def diagnostic_pass(root, cfg, arm, requests, model, counter, progress):
    remaining = requests.copy()
    finished = set()
    while remaining:
        pending = []
        for request in remaining:
            trace_file = diagnostic.trace_path(root, request["example_id"])
            if not trace_file.exists():
                if arm == "A":
                    pending.append(request)
                    continue
                raise RuntimeError("Dense32 generation is missing a trace")
            trace = read(trace_file)
            if arm == "Dense":
                if diagnostic.teacher_paths(root, request["example_id"])[1].exists():
                    diagnostic.collect_dense(root, request, model, trace, cfg["config_identity"], counter)
                else:
                    before = counter.copy()
                    started = time.monotonic()
                    status = "failed"
                    try:
                        diagnostic.collect_dense(root, request, model, trace, cfg["config_identity"], counter)
                        status = "complete"
                    finally:
                        write(attempt_path(root, arm, "diagnostic", request["example_id"]),
                              dict(arm=arm, stage="diagnostic", example_id=request["example_id"],
                                   status=status, seconds=time.monotonic()-started,
                                   forwards=counter["forwards"]-before["forwards"],
                                   forward_tokens=counter["forward_tokens"]-before["forward_tokens"]))
            else:
                before = counter.copy()
                started = time.monotonic()
                status = "failed"
                try:
                    ready = diagnostic.collect_sparse(root, request, model, trace, cfg["config_identity"], counter)
                    if not ready:
                        pending.append(request)
                        continue
                    status = "complete"
                finally:
                    if counter != before or status == "complete":
                        write(attempt_path(root, arm, "diagnostic", request["example_id"]),
                              dict(arm=arm, stage="diagnostic", example_id=request["example_id"],
                                   status=status, seconds=time.monotonic()-started,
                                   forwards=counter["forwards"]-before["forwards"],
                                   forward_tokens=counter["forward_tokens"]-before["forward_tokens"]))
            finished.add(request["example_id"])
            progress("diagnostic", candidate=arm, completed=len(finished), total=len(requests))
        remaining = pending
        if remaining:
            time.sleep(15)


def _interrupt(signum, _frame):
    raise InterruptedError(f"Worker received signal {signum}")


def worker(arm, config_path):
    gpu = ensure_environment()
    root = Path(config_path).resolve().parent
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    with lock(root / "locks" / f"{arm}.lock"):
        cfg = validate_config(config_path)
        if (root / f"complete_{arm}.json").exists():
            existing = read(root / f"complete_{arm}.json")
            if existing["config_identity"] != cfg["config_identity"] or existing["model_sha256"] != EXPECTED[arm]:
                raise RuntimeError("Completed worker identity mismatch")
            return
        session_started = time.monotonic()
        session_path = root / "runs" / f"{arm}_{time.time_ns()}.json"
        counter = dict(forwards=0, forward_tokens=0)
        status = "failed"
        phase = {"stage": "preparing", "load_seconds": None}
        try:
            _worker_body(root, cfg, arm, gpu, counter, phase)
            status = "complete"
        finally:
            write(session_path, dict(arm=arm, status=status, stage=phase["stage"],
                load_seconds=phase["load_seconds"], gpu=gpu, model_sha256=EXPECTED[arm],
                config_identity=cfg["config_identity"], seconds=time.monotonic()-session_started,
                forwards=counter["forwards"], forward_tokens=counter["forward_tokens"]))


def _worker_body(root, cfg, arm, gpu, counter, phase):
    requests = prepare.requests_for(cfg)
    source_cfg = read(cfg["source"]["config_path"])
    task, _, old_hash = task_and_protocol(source_cfg)
    if old_hash != cfg["identity"]["protocol_hash"]:
        raise RuntimeError("Historical grading protocol mismatch")
    progress = Progress(root, arm)
    progress("loading", candidate=arm, completed=0, total=1)
    started = time.monotonic()
    phase["stage"] = "loading"
    try:
        model, tokenizer, model_hash = load_model(cfg, arm)
    finally:
        phase["load_seconds"] = time.monotonic()-started
    load_seconds = phase["load_seconds"]
    freeze(root / "models" / f"{arm}.json", dict(arm=arm, model_sha256=model_hash,
           gpu=gpu, gpu_name=torch.cuda.get_device_name(0),
           mask_hash=MASK_HASH if arm == "A" else None, pruned_count=PRUNED if arm == "A" else 0,
           config_identity=cfg["config_identity"]))
    for step in STEPS:
        phase["stage"] = f"generation_{step}"
        run_cell(root, cfg, arm, step, requests, task, model, tokenizer, counter, progress)
    phase["stage"] = "diagnostic"
    diagnostic_pass(root, cfg, arm, requests, model, counter, progress)
    attempts = [read(p) for p in sorted((root / "attempts").glob(f"{arm}_*.json"))]
    summary = dict(arm=arm, status="complete", model_sha256=model_hash,
                   gpu=gpu, gpu_name=torch.cuda.get_device_name(0), config_identity=cfg["config_identity"],
                   load_seconds=load_seconds, attempts=len(attempts),
                   forwards=sum(x["forwards"] for x in attempts),
                   forward_tokens=sum(x["forward_tokens"] for x in attempts),
                   attempted_seconds=sum(x["seconds"] for x in attempts))
    freeze(root / f"complete_{arm}.json", summary)
    phase["stage"] = "complete"
    progress("complete", candidate=arm, completed=1, total=1)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("worker")
    run.add_argument("--arm", choices=("Dense", "A"), required=True)
    run.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.arm, args.config)


if __name__ == "__main__":
    main()
