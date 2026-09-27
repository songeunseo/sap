"""Run the frozen 50% allocation re-validation (mask generation + Stage 1 only)."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_loss_aggregation.core import mask_sha256, pack_mask, unpack_mask
from experiments.dlm_loss_aggregation.exp002.run import _overall_mask_hash
from experiments.dlm_loss_aggregation.run import _load_model, load_config as load_base_config
from experiments.oracle_allocation50.core import GLOBAL_SPARSITY
from experiments.oracle_allocation50.freeze import (
    DLM_STATS,
    HELDOUT,
    MODEL_REVISION,
    ROOT,
    STORE,
    write_json,
)
from experiments.oracle_allocation75.core import sha256_file
from experiments.wanda_failure_characterization.run_failure_map import model_sha, modules, state_tensors


PLANS_PATH = ROOT / "allocation_plans.json"
PREREG = ROOT / "preregistered.json"
MASK_MANIFEST = ROOT / "mask_manifest.json"
MASK_STORE = STORE / "masks"
DENSE_REFERENCE_75 = Path("/DATA/tmluser1/sap-oracle-allocation75/stage1_dense_reference.pt")
DENSE_REFERENCE_META_75 = Path("experiments/oracle_allocation75/stage1_dense_reference.json")
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
    if fixed["status"] != "frozen_before_any_50pct_mask_or_outcome":
        raise RuntimeError("50% experiment was not frozen")
    if fixed["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("model revision mismatch")
    if fixed["dlm_calibration"]["source_sha256"] != sha256_file(DLM_STATS):
        raise RuntimeError("DLM calibration statistics changed")
    if fixed["heldout"]["sha256"] != sha256_file(HELDOUT):
        raise RuntimeError("held-out manifest changed")
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
    activation = {name: values["overall_uniform"].float() for name, values in stats["statistics"].items()}
    config = load_base_config("experiments/dlm_loss_aggregation/config.yaml")
    model, _ = _load_model(config)
    before = model_sha(model)
    if before != EXPECTED_DENSE_SHA:
        raise RuntimeError("dense model SHA mismatch")
    mapping = modules(model)
    if set(mapping) != set(activation) or len(mapping) != 224:
        raise RuntimeError("activation/module mapping mismatch")
    MASK_STORE.mkdir(parents=True, exist_ok=True)
    plan_rows = {identifier: _entry_lookup(plan) for identifier, plan in plans.items()}
    identifiers = ["V0_50_uniform"] + sorted(name for name in plans if name != "V0_50_uniform")
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
        score = layer.weight.detach().float().abs() * a.to(layer.weight.device).sqrt()[None, :]
        order = torch.argsort(score, dim=1, stable=True)
        uniform_count = plan_rows["V0_50_uniform"][name]["prune_per_row"]
        uniform = torch.zeros_like(layer.weight, dtype=torch.bool)
        uniform.scatter_(1, order[:, :uniform_count], True)
        for identifier in identifiers:
            row = plan_rows[identifier][name]
            count = row["prune_per_row"]
            mask = torch.zeros_like(layer.weight, dtype=torch.bool)
            if count:
                mask.scatter_(1, order[:, :count], True)
            mask_cpu = mask.cpu()
            payload = pack_mask(mask_cpu)
            output = MASK_STORE / identifier / f"{name}.bin"
            _atomic_bytes(output, payload["bits"])
            intersection = int((mask & uniform).sum())
            union = int((mask | uniform).sum())
            xor = int((mask != uniform).sum())
            entry = {
                "method": identifier,
                "canonical_name": name,
                "block_index": row["layer"],
                "module": row["module_type"],
                "shape": list(mask.shape),
                "byte_length": len(payload["bits"]),
                "sha256": mask_sha256(payload),
                "raw_sha256": _raw_mask_sha(mask_cpu),
                "runtime_path": str(output),
                "requested_sparsity": row["requested_sparsity"],
                "actual_sparsity": float(mask_cpu.float().mean()),
                "sparsity_deviation_from_uniform": row["sparsity_deviation_from_uniform"],
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
        print(f"mask50 scores {module_index + 1}/224 {name}", flush=True)
    elapsed = time.monotonic() - started
    after = model_sha(model)
    if before != after:
        raise RuntimeError("mask construction modified model weights")
    overall, summaries = {}, {}
    for identifier in identifiers:
        method_entries = [row for row in entries if row["method"] == identifier]
        current = aggregate[identifier]
        sparsity = current["pruned"] / current["weights"]
        if abs(sparsity - GLOBAL_SPARSITY) > 1e-5:
            raise RuntimeError(f"global sparsity mismatch for {identifier}: {sparsity}")
        deviations = [row["sparsity_deviation_from_uniform"] for row in method_entries]
        overall[identifier] = _overall_mask_hash(method_entries, identifier)
        summaries[identifier] = {
            "global_sparsity": sparsity,
            "total_pruned": current["pruned"],
            "total_weights": current["weights"],
            "mask_xor_with_v0": current["xor"] / current["weights"],
            "pruned_set_jaccard_with_v0": current["intersection"] / current["union"],
            "minimum_projection_deviation": min(deviations),
            "maximum_projection_deviation": max(deviations),
        }
    write_json(MASK_MANIFEST, {
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "dense_model_sha256_before": before,
        "dense_model_sha256_after": after,
        "global_target_sparsity": GLOBAL_SPARSITY,
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
    })
    print(json.dumps({"event": "mask50_prepare_complete", "seconds": elapsed}), flush=True)


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
    pruned = weights = 0
    started = time.monotonic()
    for row in rows:
        bits = Path(row["runtime_path"]).read_bytes()
        payload = {"shape": row["shape"], "bits": bits}
        if len(bits) != row["byte_length"] or mask_sha256(payload) != row["sha256"]:
            raise RuntimeError(f"packed mask mismatch: {row['canonical_name']}")
        mask = unpack_mask(payload).to(mapping[row["canonical_name"]].weight.device)
        mapping[row["canonical_name"]].weight.masked_fill_(mask, 0)
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
    if abs(result["sparsity"] - GLOBAL_SPARSITY) > 1e-5:
        raise RuntimeError("applied global sparsity mismatch")
    return result


def load_dense_reference():
    fixed = _require_frozen()
    meta = json.loads(DENSE_REFERENCE_META_75.read_text())
    if meta["sha256"] != sha256_file(DENSE_REFERENCE_75):
        raise RuntimeError("75% dense reference hash mismatch")
    payload = torch.load(DENSE_REFERENCE_75, map_location="cpu", weights_only=False)
    if payload["heldout_state_sha256"] != fixed["heldout"]["state_sha256"]:
        raise RuntimeError("dense-reference held-out hash mismatch")
    if payload["model_sha256_before"] != EXPECTED_DENSE_SHA or payload["model_sha256_after"] != EXPECTED_DENSE_SHA:
        raise RuntimeError("dense-reference model hash mismatch")
    print(json.dumps({"event": "dense_reference_75_reused", "sha256": meta["sha256"]}), flush=True)
    return payload


@torch.inference_mode()
def evaluate_variant(identifier, dense_reference, held):
    output = ROOT / "stage1" / f"{identifier}.json"
    if output.exists():
        result = json.loads(output.read_text())
        if result.get("status") == "complete":
            print(json.dumps({"event": "stage1_50_reuse", "variant": identifier}), flush=True)
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
    per_state, token_kls = [], []
    started = time.monotonic()
    for index, (state, reference) in enumerate(zip(held["states"], dense_reference["records"])):
        noisy, clean, mask = state_tensors(state, dev)
        logits = model(noisy).logits[0, mask[0]].float().cpu()
        dense_logits = reference["logits"].float()
        logp = F.log_softmax(logits, -1)
        dense_logp = F.log_softmax(dense_logits, -1)
        kl_tokens = (dense_logp.exp() * (dense_logp - logp)).sum(-1)
        target = reference["target"]
        sparse_loss = F.cross_entropy(logits, target, reduction="sum") / state["p_mask"] / 256
        token_kls.append(kl_tokens)
        per_state.append({
            "state_index": index,
            "sequence_index": state["sequence_index"],
            "timestep_index": state["timestep_index"],
            "timestep": state["timestep"],
            "masked_count": int(mask.sum()),
            "mean_kl": float(kl_tokens.mean()),
            "p90_token_kl": float(torch.quantile(kl_tokens, 0.90)),
            "p95_token_kl": float(torch.quantile(kl_tokens, 0.95)),
            "dense_loss": reference["dense_loss"],
            "sparse_loss": float(sparse_loss),
            "nll_gap": float(sparse_loss) - reference["dense_loss"],
            "top1_agreement": float((logits.argmax(-1) == dense_logits.argmax(-1)).float().mean()),
        })
        print(f"stage1-50 {identifier} state {index + 1}/40", flush=True)
    all_kl = torch.cat(token_kls)
    summary = {
        "mean_kl": statistics.mean(row["mean_kl"] for row in per_state),
        "p90_masked_token_kl": float(torch.quantile(all_kl, 0.90)),
        "p95_masked_token_kl": float(torch.quantile(all_kl, 0.95)),
        "mean_nll_gap": statistics.mean(row["nll_gap"] for row in per_state),
        "mean_top1_agreement": statistics.mean(row["top1_agreement"] for row in per_state),
        "achieved_global_sparsity": applied["sparsity"],
    }
    sparse_after = model_sha(model)
    if sparse_before != sparse_after:
        raise RuntimeError("evaluation modified sparse weights")
    result = {
        "status": "complete",
        "variant": identifier,
        "summary": summary,
        "per_state": per_state,
        "mask": applied,
        "model_revision": MODEL_REVISION,
        "dense_model_sha256": dense_sha,
        "sparse_model_sha256_before_eval": sparse_before,
        "sparse_model_sha256_after_eval": sparse_after,
        "evaluation_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(output, result)
    print(json.dumps({"event": "stage1_50_complete", "variant": identifier, **summary}), flush=True)
    del model, token_kls, all_kl
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _beats(left, right):
    """Return metric-level booleans for left being better than right."""
    return {
        "mean_kl": left["mean_kl"] < right["mean_kl"],
        "p90_kl": left["p90_masked_token_kl"] < right["p90_masked_token_kl"],
        "nll_gap": left["mean_nll_gap"] < right["mean_nll_gap"],
        "top1_agreement": left["mean_top1_agreement"] > right["mean_top1_agreement"],
    }


def run_stage1():
    fixed = _require_frozen()
    if not MASK_MANIFEST.exists():
        raise RuntimeError("run prepare first")
    dense = load_dense_reference()
    held = json.loads(HELDOUT.read_text())
    plans = json.loads(PLANS_PATH.read_text())
    results = [evaluate_variant(identifier, dense, held) for identifier in plans]
    by_name = {row["variant"]: row["summary"] for row in results}
    oracle = by_name["V3_50_functional_kl_mild"]
    uniform = by_name["V0_50_uniform"]
    random_names = sorted(name for name in by_name if name.startswith("V1_50_random_"))
    best_random = {
        "mean_kl": min(by_name[name]["mean_kl"] for name in random_names),
        "p90_masked_token_kl": min(by_name[name]["p90_masked_token_kl"] for name in random_names),
        "mean_nll_gap": min(by_name[name]["mean_nll_gap"] for name in random_names),
        "mean_top1_agreement": max(by_name[name]["mean_top1_agreement"] for name in random_names),
    }
    versus_uniform = _beats(oracle, uniform)
    versus_best_random = _beats(oracle, best_random)
    pass_count_uniform = sum(versus_uniform.values())
    pass_count_random = sum(versus_best_random.values())
    directional_validity = pass_count_uniform >= 3 and pass_count_random >= 3
    mask_manifest = json.loads(MASK_MANIFEST.read_text())
    rows = []
    for result in results:
        name = result["variant"]
        rows.append({"variant": name, **result["summary"], **{
            "minimum_projection_deviation": mask_manifest["summaries"][name]["minimum_projection_deviation"],
            "maximum_projection_deviation": mask_manifest["summaries"][name]["maximum_projection_deviation"],
        }})
    output = {
        "status": "complete",
        "raw_results": rows,
        "comparison": {
            "functional_kl_vs_uniform": versus_uniform,
            "functional_kl_vs_best_per_metric_random": versus_best_random,
            "metrics_won_vs_uniform": pass_count_uniform,
            "metrics_won_vs_best_random": pass_count_random,
        },
        "directional_validity_gate": directional_validity,
        "interpretation": (
            "Causal-KL ranking is directionally valid at 50%; 75% likely introduced budget-tightness/interactions."
            if directional_validity else
            "Static per-projection causal-KL-to-Wanda budget allocation is not validated at 50%, independent of 75% budget tightness."
        ),
        "recommended_next_step": (
            "Investigate why the 75% joint-pruning regime breaks the otherwise valid allocation, without running GSM8K at 50%."
            if directional_validity else
            "Abandon this allocation-conversion actuator and move to outlier-pathway or individual-weight-criterion diagnostics."
        ),
        "medium_aggressive_tested": False,
        "gsm8k_or_downstream_run": False,
        "frozen_config": fixed,
    }
    write_json(ROOT / "stage1_summary.json", output)
    csv_path = ROOT / "stage1_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "stage1_50_all_complete", "directional_validity_gate": directional_validity}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["prepare", "stage1", "all"])
    args = parser.parse_args()
    if args.job in ("prepare", "all"):
        prepare_masks()
    if args.job in ("stage1", "all"):
        run_stage1()


if __name__ == "__main__":
    main()
