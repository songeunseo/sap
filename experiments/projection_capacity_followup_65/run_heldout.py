#!/usr/bin/env python3
"""Evaluate four jointly sparse allocations on the fresh disjoint DLM split."""
import gc
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_loss_aggregation.core import mask_sha256, unpack_mask
from experiments.projection_capacity_allocation_65.run import (
    DENSE_SHA,
    diagnostics,
    load_dense,
    summarize_states,
)
from experiments.projection_capacity_followup_65.core import paired_comparison
from experiments.wanda_failure_characterization.run_failure_map import (
    model_sha,
    modules,
    state_tensors,
)


ROOT = Path("experiments/projection_capacity_followup_65")
SOURCE = Path("experiments/projection_capacity_allocation_65")
STATE_PATH = ROOT / "new_heldout_state_manifest.json"
VERIFY_PATH = ROOT / "new_state_verification.json"
METHOD_PATHS = {
    "uniform": SOURCE / "uniform65_mask_manifest.json",
    "capacity": SOURCE / "capacity65_mask_manifest.json",
    "reconstruction": ROOT / "reconstruction65_mask_manifest.json",
    "eis_type": ROOT / "eis_type65_mask_manifest.json",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def event(event_type, **values):
    print(json.dumps({"event": event_type, **values}), flush=True)


def validate_inputs():
    verification = json.loads(VERIFY_PATH.read_text())
    states = json.loads(STATE_PATH.read_text())
    if verification["status"] != "verified" or not verification["all_three_splits_disjoint"]:
        raise RuntimeError("fresh state split is not verified disjoint")
    if sha(STATE_PATH) != verification["manifest_sha256"]:
        raise RuntimeError("fresh state manifest changed after verification")
    if len(states["states"]) != 40 or states["historical_state_sha256"] != verification["state_sha256"]:
        raise RuntimeError("fresh state manifest count/digest mismatch")
    manifests = {name: json.loads(path.read_text()) for name, path in METHOD_PATHS.items()}
    target = None
    expected_names = None
    for name, manifest in manifests.items():
        names = [row["name"] for row in manifest["entries"]]
        if len(names) != 224 or len(set(names)) != 224:
            raise RuntimeError(f"{name} manifest does not contain 224 unique modules")
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError(f"{name} module order mismatch")
        if target is None:
            target = int(manifest["pruned"])
        elif int(manifest["pruned"]) != target:
            raise RuntimeError(f"{name} pruned budget mismatch")
    if target != 4_536_008_704:
        raise RuntimeError("unexpected target pruned count")
    return states, manifests


def selected_mask(row, device):
    meta = row["selected_mask"]
    path = Path(meta["path"])
    if sha(path) != meta["file_sha256"]:
        raise RuntimeError(f"mask file changed: {path}")
    packed = torch.load(path, map_location="cpu", weights_only=False)
    if mask_sha256(packed) != meta["mask_sha256"]:
        raise RuntimeError(f"mask payload digest mismatch: {path}")
    return unpack_mask(packed).to(device)


@torch.inference_mode()
def apply_manifest(model, mapping, manifest):
    entries = manifest["entries"]
    if list(mapping) != [row["name"] for row in entries]:
        raise RuntimeError("model/manifest module order mismatch")
    counted = 0
    for row in entries:
        module = mapping[row["name"]]
        mask = selected_mask(row, module.weight.device)
        if tuple(mask.shape) != tuple(module.weight.shape):
            raise RuntimeError(f"mask shape mismatch for {row['name']}")
        actual = int(mask.sum().item())
        expected = int(row["selected_mask"]["pruned"])
        if actual != expected:
            raise RuntimeError(f"mask pruned count mismatch for {row['name']}")
        module.weight.masked_fill_(mask, 0)
        counted += actual
        del mask
    if counted != int(manifest["pruned"]):
        raise RuntimeError("applied mask budget mismatch")
    return counted


def method_artifact_valid(path, state_sha, manifest_sha):
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    return (row.get("status") == "complete"
            and row.get("heldout_state_sha256") == state_sha
            and row.get("mask_manifest_sha256") == manifest_sha
            and len(row.get("per_state", [])) == 40)


@torch.inference_mode()
def main():
    states_document, manifests = validate_inputs()
    states = states_document["states"]
    state_sha = states_document["historical_state_sha256"]
    model, mapping = load_dense()
    device = next(model.parameters()).device
    dense_references, dense_sham_max = [], 0.0
    for index, state in enumerate(states):
        noisy, _, mask = state_tensors(state, device)
        dense = model(noisy).logits
        sham = model(noisy).logits
        difference = float((dense - sham).abs().max().item())
        dense_sham_max = max(dense_sham_max, difference)
        if difference != 0:
            raise RuntimeError("dense repeated same-path sham differs")
        dense_references.append(dense[0, mask[0]].cpu())
        event("dense_reference", state=index + 1, total=len(states))
        del dense, sham
    if model_sha(model) != DENSE_SHA:
        raise RuntimeError("dense model changed while collecting references")
    dense_weights = {name: module.weight.detach().cpu().clone()
                     for name, module in mapping.items()}

    results = {}
    for method in ("uniform", "capacity", "reconstruction", "eis_type"):
        manifest_path = METHOD_PATHS[method]
        manifest_digest = sha(manifest_path)
        result_path = ROOT / f"heldout_{method}.json"
        if method_artifact_valid(result_path, state_sha, manifest_digest):
            results[method] = json.loads(result_path.read_text())
            event("heldout_method_reused", method=method)
            continue
        for name, module in mapping.items():
            module.weight.copy_(dense_weights[name])
        if model_sha(model) != DENSE_SHA:
            raise RuntimeError("failed to restore dense model")
        pruned = apply_manifest(model, mapping, manifests[method])
        sparse_sha = model_sha(model)
        rows, token_kl = [], []
        for index, state in enumerate(states):
            noisy, clean, mask = state_tensors(state, device)
            sparse = model(noisy).logits[0, mask[0]]
            measured, kl = diagnostics(
                sparse, dense_references[index].to(device), clean[0, mask[0]], state["p_mask"]
            )
            measured.update(state_index=index, sequence_index=state["sequence_index"],
                            timestep=state["timestep"])
            rows.append(measured)
            token_kl.append(kl)
            event("heldout_state", method=method, state=index + 1,
                  total=len(states), mean_kl=measured["mean_kl"])
        if model_sha(model) != sparse_sha:
            raise RuntimeError(f"{method} weights changed during evaluation")
        result = {
            "status": "complete",
            "method": method,
            "heldout_state_sha256": state_sha,
            "mask_manifest_sha256": manifest_digest,
            "summary": summarize_states(rows, token_kl),
            "per_state": rows,
            "pruned": pruned,
            "weights": int(manifests[method]["weights"]),
            "global_sparsity": pruned / int(manifests[method]["weights"]),
            "model_sha256": sparse_sha,
        }
        write_json(result_path, result)
        results[method] = result
        event("heldout_method_complete", method=method, mean_kl=result["summary"]["mean_kl"])

    comparisons = {
        "capacity_minus_eis_type_primary": paired_comparison(
            results["eis_type"]["per_state"], results["capacity"]["per_state"]
        ),
        "capacity_minus_reconstruction_secondary": paired_comparison(
            results["reconstruction"]["per_state"], results["capacity"]["per_state"]
        ),
        "capacity_minus_uniform_reference": paired_comparison(
            results["uniform"]["per_state"], results["capacity"]["per_state"]
        ),
        "eis_type_minus_uniform": paired_comparison(
            results["uniform"]["per_state"], results["eis_type"]["per_state"]
        ),
        "reconstruction_minus_uniform": paired_comparison(
            results["uniform"]["per_state"], results["reconstruction"]["per_state"]
        ),
    }
    output = {
        "status": "complete",
        "heldout_state_sha256": state_sha,
        "dense_sham_max_abs": dense_sham_max,
        "method_manifest_sha256": {name: sha(path) for name, path in METHOD_PATHS.items()},
        "methods": results,
        "comparisons": comparisons,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    write_json(ROOT / "heldout_dlm_results.json", output)
    write_json(ROOT / "analysis_status.json", {
        "status": "fresh_heldout_complete",
        "single_anchor_crossfit_passed": json.loads(
            (ROOT / "analysis_status.json").read_text()
        )["single_anchor_crossfit_passed"],
        "gpu_evaluation_started": True,
        "gpu_evaluation_complete": True,
    })
    event("fresh_heldout_complete", means={name: row["summary"]["mean_kl"]
                                            for name, row in results.items()})
    del model, dense_weights, dense_references
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
