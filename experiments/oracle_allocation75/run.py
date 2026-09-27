"""Execute the frozen 75% projection-allocation oracle experiment.

Jobs are resumable and intentionally separated:
  prepare -> stage1 -> stage2 -> stage3 -> summarize
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cgq_sparsegpt_downstream import (
    _evaluate_winogrande,
    assert_same_examples as assert_same_wino,
)
from experiments.dlm_loss_aggregation.core import mask_sha256, pack_mask, unpack_mask
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _overall_mask_hash,
    _read_jsonl,
    _write_jsonl,
    load_config as load_exp002_config,
)
from experiments.dlm_loss_aggregation.run import _load_model, load_config as load_base_config
from experiments.oracle_allocation75.core import sha256_file
from experiments.oracle_allocation75.freeze import DLM_STATS, MODEL_REVISION, ROOT, STORE, write_json
from experiments.wanda_failure_characterization.run_failure_map import (
    model_sha,
    modules,
    state_tensors,
)


PLANS_PATH = ROOT / "allocation_plans.json"
PREREG = ROOT / "preregistered.json"
MASK_MANIFEST = ROOT / "mask_manifest.json"
HELDOUT = Path("experiments/wanda_failure_characterization/heldout_state_manifest.json")
EXP002 = Path("experiments/dlm_loss_aggregation/exp002")
MASK_STORE = STORE / "masks"
DENSE_REFERENCE = STORE / "stage1_dense_reference.pt"
EXPECTED_DENSE_SHA = "2ef1a6e04a2963b4458efeeb6a932df04b3bd394086bb6757bc8690cb9c614bc"


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _raw_mask_sha(mask):
    return hashlib.sha256(mask.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _require_frozen():
    fixed = json.loads(PREREG.read_text())
    if fixed["status"] != "frozen_before_any_75pct_mask_or_outcome":
        raise RuntimeError("oracle allocation was not preregistered")
    if fixed["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("model revision mismatch")
    if fixed["dlm_calibration"]["source_sha256"] != sha256_file(DLM_STATS):
        raise RuntimeError("DLM calibration statistics changed")
    return fixed


def _entry_lookup(plan):
    return {row["module"]: row for row in plan["entries"]}


@torch.inference_mode()
def prepare_masks():
    fixed = _require_frozen()
    if MASK_MANIFEST.exists():
        print(json.dumps({"event": "mask_prepare_reuse", "path": str(MASK_MANIFEST)}), flush=True)
        return
    plans = json.loads(PLANS_PATH.read_text())
    stats = torch.load(DLM_STATS, map_location="cpu", weights_only=False)
    if stats["source_state_sha256"] != fixed["dlm_calibration"]["state_sha256"]:
        raise RuntimeError("DLM state digest mismatch")
    activation = {
        name: values["overall_uniform"].float()
        for name, values in stats["statistics"].items()
    }
    config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, _ = _load_model(config)
    before = model_sha(model)
    if before != EXPECTED_DENSE_SHA:
        raise RuntimeError(f"dense model SHA mismatch: {before}")
    mapping = modules(model)
    if set(mapping) != set(activation) or len(mapping) != 224:
        raise RuntimeError("activation/module mapping mismatch")
    MASK_STORE.mkdir(parents=True, exist_ok=True)
    plan_rows = {identifier: _entry_lookup(plan) for identifier, plan in plans.items()}
    identifiers = ["V0_uniform"] + sorted(name for name in plans if name != "V0_uniform")
    entries = []
    aggregate = {
        identifier: {"pruned": 0, "weights": 0, "xor": 0, "intersection": 0, "union": 0}
        for identifier in identifiers
    }
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for module_index, name in enumerate(sorted(mapping)):
        layer = mapping[name]
        a = activation[name]
        if a.shape != (layer.weight.shape[1],) or not torch.isfinite(a).all():
            raise RuntimeError(f"bad activation statistic for {name}")
        score = layer.weight.detach().float().abs() * a.to(layer.weight.device).sqrt()[None, :]
        order = torch.argsort(score, dim=1, stable=True)
        uniform_count = plan_rows["V0_uniform"][name]["prune_per_row"]
        uniform = torch.zeros_like(layer.weight, dtype=torch.bool)
        uniform.scatter_(1, order[:, :uniform_count], True)
        for identifier in identifiers:
            count = plan_rows[identifier][name]["prune_per_row"]
            mask = torch.zeros_like(layer.weight, dtype=torch.bool)
            if count:
                mask.scatter_(1, order[:, :count], True)
            mask_cpu = mask.cpu()
            payload = pack_mask(mask_cpu)
            output = MASK_STORE / identifier / f"{name}.bin"
            _atomic_bytes(output, payload["bits"])
            row = plan_rows[identifier][name]
            intersection = int((mask & uniform).sum())
            union = int((mask | uniform).sum())
            xor = int((mask != uniform).sum())
            entry = {
                "method": identifier,
                "block_index": row["layer"],
                "module": row["module_type"],
                "canonical_name": name,
                "shape": list(mask.shape),
                "byte_length": len(payload["bits"]),
                "sha256": mask_sha256(payload),
                "raw_sha256": _raw_mask_sha(mask_cpu),
                "runtime_path": str(output),
                "requested_sparsity": row["requested_sparsity"],
                "actual_sparsity": float(mask_cpu.float().mean()),
                "prune_per_row": count,
                "pruned": int(mask_cpu.sum()),
                "weights": mask_cpu.numel(),
                "status": row["status"],
                "xor_with_v0": xor / mask_cpu.numel(),
            }
            entries.append(entry)
            current = aggregate[identifier]
            current["pruned"] += entry["pruned"]
            current["weights"] += entry["weights"]
            current["xor"] += xor
            current["intersection"] += intersection
            current["union"] += union
            del mask, mask_cpu, payload
        del uniform, order, score
        print(f"mask scores {module_index + 1}/224 {name}", flush=True)
    elapsed = time.monotonic() - started
    after = model_sha(model)
    if before != after:
        raise RuntimeError("mask construction modified model weights")
    overall = {}
    summaries = {}
    for identifier in identifiers:
        method_entries = [row for row in entries if row["method"] == identifier]
        current = aggregate[identifier]
        sparsity = current["pruned"] / current["weights"]
        if sparsity != 0.75:
            raise RuntimeError(f"non-exact global sparsity for {identifier}: {sparsity}")
        overall[identifier] = _overall_mask_hash(method_entries, identifier)
        summaries[identifier] = {
            "global_sparsity": sparsity,
            "total_pruned": current["pruned"],
            "total_weights": current["weights"],
            "mask_xor_with_v0": current["xor"] / current["weights"],
            "pruned_set_jaccard_with_v0": current["intersection"] / current["union"],
        }
    manifest = {
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "dense_model_sha256_before": before,
        "dense_model_sha256_after": after,
        "score": "abs(W)*sqrt(overall_uniform DLM activation statistic)",
        "activation_statistics_sha256": sha256_file(DLM_STATS),
        "activation_state_sha256": stats["source_state_sha256"],
        "variant_count": len(identifiers),
        "matrix_count_per_variant": 224,
        "shared_score_sort_seconds": elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "overall_sha256": overall,
        "summaries": summaries,
        "entries": entries,
    }
    write_json(MASK_MANIFEST, manifest)
    print(json.dumps({"event": "mask_prepare_complete", "seconds": elapsed, "variants": len(identifiers)}), flush=True)


def _manifest_entries(identifier):
    manifest = json.loads(MASK_MANIFEST.read_text())
    rows = [row for row in manifest["entries"] if row["method"] == identifier]
    if len(rows) != 224:
        raise RuntimeError(f"incomplete mask set for {identifier}")
    if _overall_mask_hash(rows, identifier) != manifest["overall_sha256"][identifier]:
        raise RuntimeError(f"overall mask hash mismatch for {identifier}")
    return rows, manifest


@torch.inference_mode()
def _apply_variant(model, identifier):
    rows, manifest = _manifest_entries(identifier)
    mapping = modules(model)
    started = time.monotonic()
    pruned = weights = 0
    for row in rows:
        bits = Path(row["runtime_path"]).read_bytes()
        payload = {"shape": row["shape"], "bits": bits}
        if len(bits) != row["byte_length"] or mask_sha256(payload) != row["sha256"]:
            raise RuntimeError(f"packed mask mismatch: {row['canonical_name']}")
        mask = unpack_mask(payload).to(mapping[row["canonical_name"]].weight.device)
        parameter = mapping[row["canonical_name"]].weight
        parameter.masked_fill_(mask, 0)
        if int(parameter.eq(0).sum()) != row["pruned"]:
            raise RuntimeError(f"applied mask zero-count mismatch: {row['canonical_name']}")
        pruned += row["pruned"]
        weights += row["weights"]
        del mask, payload, bits
    result = {
        "seconds": time.monotonic() - started,
        "pruned": pruned,
        "weights": weights,
        "sparsity": pruned / weights,
        "mask_sha256": manifest["overall_sha256"][identifier],
    }
    if result["sparsity"] != 0.75:
        raise RuntimeError("applied global sparsity is not exactly 75%")
    return result


@torch.inference_mode()
def build_dense_reference():
    fixed = _require_frozen()
    if DENSE_REFERENCE.exists():
        print(json.dumps({"event": "dense_reference_reuse", "sha256": sha256_file(DENSE_REFERENCE)}), flush=True)
        return torch.load(DENSE_REFERENCE, map_location="cpu", weights_only=False)
    held = json.loads(HELDOUT.read_text())
    if held["historical_state_sha256"] != fixed["vulnerability"]["heldout_state_sha256"]:
        raise RuntimeError("held-out state digest mismatch")
    config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, _ = _load_model(config)
    before = model_sha(model)
    if before != EXPECTED_DENSE_SHA:
        raise RuntimeError("dense reference model hash mismatch")
    dev = model.model.transformer.wte.weight.device
    records = []
    for index, state in enumerate(held["states"]):
        noisy, clean, mask = state_tensors(state, dev)
        selected = model(noisy).logits[0, mask[0]].detach().cpu()
        target = clean[0, mask[0]].cpu()
        loss = F.cross_entropy(selected.float(), target, reduction="sum") / state["p_mask"] / 256
        records.append(
            {
                "state_index": index,
                "sequence_index": state["sequence_index"],
                "timestep_index": state["timestep_index"],
                "timestep": state["timestep"],
                "p_mask": state["p_mask"],
                "masked_count": int(mask.sum()),
                "target": target,
                "logits": selected,
                "dense_loss": float(loss),
            }
        )
        print(f"dense fidelity reference {index + 1}/40", flush=True)
    after = model_sha(model)
    if before != after:
        raise RuntimeError("dense reference collection modified weights")
    payload = {
        "heldout_state_sha256": held["historical_state_sha256"],
        "model_sha256_before": before,
        "model_sha256_after": after,
        "records": records,
    }
    DENSE_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, DENSE_REFERENCE)
    write_json(ROOT / "stage1_dense_reference.json", {
        "status": "complete", "path": str(DENSE_REFERENCE), "sha256": sha256_file(DENSE_REFERENCE),
        "states": len(records), "model_sha256_before": before, "model_sha256_after": after,
    })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


@torch.inference_mode()
def evaluate_stage1_variant(identifier, dense_reference, held):
    output = ROOT / "stage1" / f"{identifier}.json"
    if output.exists():
        result = json.loads(output.read_text())
        if result.get("status") == "complete":
            print(json.dumps({"event": "stage1_reuse", "variant": identifier}), flush=True)
            return result
    config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, _ = _load_model(config)
    dense_sha = model_sha(model)
    if dense_sha != EXPECTED_DENSE_SHA:
        raise RuntimeError("stage1 dense checkpoint mismatch")
    torch.cuda.reset_peak_memory_stats()
    applied = _apply_variant(model, identifier)
    sparse_before = model_sha(model)
    dev = model.model.transformer.wte.weight.device
    per_state = []
    token_kls = []
    started = time.monotonic()
    for index, (state, reference) in enumerate(zip(held["states"], dense_reference["records"])):
        noisy, clean, mask = state_tensors(state, dev)
        logits = model(noisy).logits[0, mask[0]].float().cpu()
        dense_logits = reference["logits"].float()
        if logits.shape != dense_logits.shape:
            raise RuntimeError("dense/sparse masked logit shape mismatch")
        logp = F.log_softmax(logits, -1)
        dense_logp = F.log_softmax(dense_logits, -1)
        kl_tokens = (dense_logp.exp() * (dense_logp - logp)).sum(-1)
        target = reference["target"]
        sparse_loss = F.cross_entropy(logits, target, reduction="sum") / state["p_mask"] / 256
        state_mean = float(kl_tokens.mean())
        token_kls.append(kl_tokens)
        per_state.append(
            {
                "state_index": index,
                "sequence_index": state["sequence_index"],
                "timestep_index": state["timestep_index"],
                "timestep": state["timestep"],
                "masked_count": int(mask.sum()),
                "mean_kl": state_mean,
                "p90_token_kl": float(torch.quantile(kl_tokens, 0.9)),
                "dense_loss": reference["dense_loss"],
                "sparse_loss": float(sparse_loss),
                "nll_gap": float(sparse_loss) - reference["dense_loss"],
                "top1_agreement": float((logits.argmax(-1) == dense_logits.argmax(-1)).float().mean()),
            }
        )
        print(f"stage1 {identifier} state {index + 1}/40", flush=True)
    evaluation_seconds = time.monotonic() - started
    sparse_after = model_sha(model)
    if sparse_before != sparse_after:
        raise RuntimeError("stage1 evaluation modified sparse weights")
    all_kl = torch.cat(token_kls)
    by_timestep = []
    for timestep in sorted({row["timestep"] for row in per_state}):
        group = [row for row in per_state if row["timestep"] == timestep]
        by_timestep.append(
            {
                "timestep": timestep,
                "mean_kl": sum(row["mean_kl"] for row in group) / len(group),
                "mean_nll_gap": sum(row["nll_gap"] for row in group) / len(group),
            }
        )
    summary = {
        "mean_kl": sum(row["mean_kl"] for row in per_state) / len(per_state),
        "p90_masked_token_kl": float(torch.quantile(all_kl, 0.9)),
        "mean_nll_gap": sum(row["nll_gap"] for row in per_state) / len(per_state),
        "mean_top1_agreement": sum(row["top1_agreement"] for row in per_state) / len(per_state),
    }
    result = {
        "status": "complete",
        "variant": identifier,
        "model_revision": MODEL_REVISION,
        "dense_model_sha256": dense_sha,
        "sparse_model_sha256_before_eval": sparse_before,
        "sparse_model_sha256_after_eval": sparse_after,
        "mask": applied,
        "summary": summary,
        "per_timestep": by_timestep,
        "per_state": per_state,
        "evaluation_seconds": evaluation_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(output, result)
    print(json.dumps({"event": "stage1_complete", "variant": identifier, **summary}), flush=True)
    del model, token_kls, all_kl
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_stage1():
    _require_frozen()
    if not MASK_MANIFEST.exists():
        raise RuntimeError("run prepare first")
    dense = build_dense_reference()
    held = json.loads(HELDOUT.read_text())
    plans = json.loads(PLANS_PATH.read_text())
    results = [evaluate_stage1_variant(identifier, dense, held) for identifier in plans]
    random_rows = [row for row in results if row["variant"].startswith("V1_")]
    reconstruction = [row for row in results if row["variant"].startswith("V2_")]
    functional = [row for row in results if row["variant"].startswith("V3_")]
    key = lambda row: (row["summary"]["mean_kl"], row["summary"]["p90_masked_token_kl"], row["variant"])
    selected = [
        "V0_uniform",
        min(random_rows, key=key)["variant"],
        min(reconstruction, key=key)["variant"],
        min(functional, key=key)["variant"],
    ]
    document = {
        "status": "complete",
        "selection_rule": "V0; lowest mean-KL random; lowest mean-KL reconstruction; lowest mean-KL functional-KL schedule",
        "selected_for_stage2": selected,
        "results": [{"variant": row["variant"], **row["summary"]} for row in results],
    }
    write_json(ROOT / "stage1_summary.json", document)
    print(json.dumps({"event": "stage1_all_complete", "selected": selected}, indent=2), flush=True)


def _load_model_with_variant(identifier):
    config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, _ = _load_model(config)
    dense_sha = model_sha(model)
    if dense_sha != EXPECTED_DENSE_SHA:
        raise RuntimeError("downstream dense model SHA mismatch")
    applied = _apply_variant(model, identifier)
    return model, dense_sha, applied


def _invalid_gsm(records):
    return sum(row.get("extracted_answer") in (None, "") for row in records)


def run_stage2():
    _require_frozen()
    stage1 = json.loads((ROOT / "stage1_summary.json").read_text())
    selected = stage1["selected_for_stage2"]
    config = load_exp002_config(EXP002 / "config.yaml")
    config_hash, protocol = _evaluation_config_hash(config)
    dense_records = _read_jsonl(EXP002 / "results/predictions/dense.jsonl")[:100]
    results = []
    reference = None
    for identifier in selected:
        summary_path = ROOT / "stage2" / f"{identifier}.json"
        predictions_path = ROOT / "stage2" / f"{identifier}_predictions.jsonl"
        if summary_path.exists() and predictions_path.exists():
            result = json.loads(summary_path.read_text())
            records = _read_jsonl(predictions_path)
            print(json.dumps({"event": "stage2_reuse", "variant": identifier}), flush=True)
        else:
            model, dense_sha, mask = _load_model_with_variant(identifier)
            sparse_before = model_sha(model)
            tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True
            )
            measured, records = _evaluate_gsm8k(model, tokenizer, config, identifier, 100, config_hash)
            sparse_after = model_sha(model)
            if sparse_before != sparse_after:
                raise RuntimeError("GSM8K mini evaluation modified weights")
            _assert_same_examples(dense_records, records)
            result = {
                "status": "complete",
                "variant": identifier,
                "protocol_hash": config_hash,
                "protocol": protocol,
                "limit": 100,
                "correct": sum(row["correct"] for row in records),
                "accuracy": sum(row["correct"] for row in records) / 100,
                "invalid_answer_count": _invalid_gsm(records),
                "dense_extracted_answer_agreement": sum(
                    row.get("extracted_answer") == dense.get("extracted_answer")
                    for row, dense in zip(records, dense_records)
                ) / 100,
                "evaluation_seconds": measured["eval_seconds"],
                "mask": mask,
                "dense_model_sha256": dense_sha,
                "sparse_model_sha256_before_eval": sparse_before,
                "sparse_model_sha256_after_eval": sparse_after,
            }
            _write_jsonl(predictions_path, records)
            result["prediction_sha256"] = sha256_file(predictions_path)
            write_json(summary_path, result)
            print(json.dumps({"event": "stage2_complete", "variant": identifier, "correct": result["correct"]}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if reference is None:
            reference = records
        else:
            _assert_same_examples(reference, records)
        results.append(result)
    by_name = {row["variant"]: row for row in results}
    v0 = by_name["V0_uniform"]
    stage1_by_name = {row["variant"]: row for row in stage1["results"]}
    positive_signals = []
    for row in results:
        if not row["variant"].startswith("V3_"):
            continue
        fidelity = stage1_by_name[row["variant"]]
        base_fidelity = stage1_by_name["V0_uniform"]
        checks = {
            "mini_accuracy_better": row["correct"] > v0["correct"],
            "mean_kl_better": fidelity["mean_kl"] < base_fidelity["mean_kl"],
            "p90_kl_better": fidelity["p90_masked_token_kl"] < base_fidelity["p90_masked_token_kl"],
        }
        if sum(checks.values()) >= 2:
            positive_signals.append(row["variant"])
        row["stage3_gate_checks"] = checks
    positive_signals.sort(key=lambda name: (-by_name[name]["correct"], stage1_by_name[name]["mean_kl"], name))
    document = {
        "status": "complete",
        "results": results,
        "go_signal_variants": positive_signals,
        "go_signal_definition": "At least two of: lower mean KL, lower P90 KL, higher fixed-100 GSM8K accuracy than V0.",
        "full_downstream_status": "deferred_by_user_regardless_of_signal",
    }
    write_json(ROOT / "stage2_summary.json", document)
    print(json.dumps({"event": "stage2_all_complete", "go_signals": positive_signals, "full_downstream": "deferred"}, indent=2), flush=True)


def _paired_correctness(left, right):
    if len(left) != len(right):
        raise ValueError("paired record lengths differ")
    lc = [bool(row["correct"]) for row in left]
    rc = [bool(row["correct"]) for row in right]
    return {
        "both_correct": sum(a and b for a, b in zip(lc, rc)),
        "left_only": sum(a and not b for a, b in zip(lc, rc)),
        "right_only": sum(not a and b for a, b in zip(lc, rc)),
        "both_wrong": sum(not a and not b for a, b in zip(lc, rc)),
        "left_correct": sum(lc),
        "right_correct": sum(rc),
        "right_minus_left": sum(rc) - sum(lc),
        "right_minus_left_pp": 100 * (sum(rc) - sum(lc)) / len(lc),
    }


def _run_full_gsm(identifier, config, config_hash):
    summary_path = ROOT / "stage3" / f"gsm8k_{identifier}.json"
    predictions_path = ROOT / "stage3" / f"gsm8k_{identifier}_predictions.jsonl"
    if summary_path.exists() and predictions_path.exists():
        return json.loads(summary_path.read_text()), _read_jsonl(predictions_path)
    model, dense_sha, mask = _load_model_with_variant(identifier)
    sparse_before = model_sha(model)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True)
    measured, records = _evaluate_gsm8k(model, tokenizer, config, identifier, None, config_hash)
    sparse_after = model_sha(model)
    if sparse_before != sparse_after:
        raise RuntimeError("full GSM8K changed weights")
    _write_jsonl(predictions_path, records)
    result = {
        "status": "complete", "variant": identifier, "correct": sum(row["correct"] for row in records),
        "accuracy": sum(row["correct"] for row in records) / len(records), "invalid_answer_count": _invalid_gsm(records),
        "evaluation": measured, "mask": mask, "prediction_sha256": sha256_file(predictions_path),
        "dense_model_sha256": dense_sha, "sparse_model_sha256_before_eval": sparse_before,
        "sparse_model_sha256_after_eval": sparse_after,
    }
    write_json(summary_path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result, records


def _run_wino(identifier):
    summary_path = ROOT / "stage3" / f"winogrande_{identifier}.json"
    predictions_path = ROOT / "stage3" / f"winogrande_{identifier}_predictions.jsonl"
    if summary_path.exists() and predictions_path.exists():
        return json.loads(summary_path.read_text()), _read_jsonl(predictions_path)
    model, dense_sha, mask = _load_model_with_variant(identifier)
    sparse_before = model_sha(model)
    config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True)
    measured, records = _evaluate_winogrande(model, tokenizer, torch.device("cuda:0"))
    sparse_after = model_sha(model)
    if sparse_before != sparse_after:
        raise RuntimeError("WinoGrande changed weights")
    _write_jsonl(predictions_path, records)
    result = {
        "status": "complete", "variant": identifier, "correct": sum(row["correct"] for row in records),
        "accuracy": sum(row["correct"] for row in records) / len(records), "evaluation": measured,
        "mask": mask, "prediction_sha256": sha256_file(predictions_path), "dense_model_sha256": dense_sha,
        "sparse_model_sha256_before_eval": sparse_before, "sparse_model_sha256_after_eval": sparse_after,
    }
    write_json(summary_path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result, records


def run_stage3():
    document = {
        "status": "deferred",
        "reason": "User froze this run at Stage 1 plus fixed-100 GSM8K screening; full GSM8K/WinoGrande is not executed automatically.",
    }
    write_json(ROOT / "stage3_summary.json", document)
    print(json.dumps({"event": "stage3_deferred"}), flush=True)


def summarize():
    prereg = _require_frozen()
    plans = json.loads(PLANS_PATH.read_text())
    masks = json.loads(MASK_MANIFEST.read_text())
    stage1 = json.loads((ROOT / "stage1_summary.json").read_text())
    stage2 = json.loads((ROOT / "stage2_summary.json").read_text())
    stage3 = json.loads((ROOT / "stage3_summary.json").read_text())
    s1 = {row["variant"]: row for row in stage1["results"]}
    s2 = {row["variant"]: row for row in stage2["results"]}
    random_by_schedule = {}
    for schedule in ("mild", "medium", "aggressive"):
        rows = [row for name, row in s1.items() if name.startswith("V1_") and name.endswith("_" + schedule)]
        random_by_schedule[schedule] = {
            "mean_kl_mean": statistics.mean(row["mean_kl"] for row in rows),
            "mean_kl_std": statistics.stdev(row["mean_kl"] for row in rows),
            "p90_kl_mean": statistics.mean(row["p90_masked_token_kl"] for row in rows),
            "p90_kl_std": statistics.stdev(row["p90_masked_token_kl"] for row in rows),
        }
    result = {
        "status": "complete",
        "preregistered": prereg,
        "mask_summaries": masks["summaries"],
        "stage1": stage1,
        "random_seed_summary": random_by_schedule,
        "stage2": stage2,
        "stage3": stage3,
        "decision": "pending report classification",
        "no_new_score_or_recovery": True,
    }
    write_json(ROOT / "results.json", result)
    print(json.dumps({"event": "summary_complete", "stage3_status": stage3["status"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["prepare", "stage1", "stage2", "stage3", "summarize", "all"])
    args = parser.parse_args()
    if args.job in ("prepare", "all"):
        prepare_masks()
    if args.job in ("stage1", "all"):
        run_stage1()
    if args.job in ("stage2", "all"):
        run_stage2()
    if args.job in ("stage3", "all"):
        run_stage3()
    if args.job in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()
