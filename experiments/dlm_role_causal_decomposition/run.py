#!/usr/bin/env python3
"""Collect role-specific causal effects of 65% single-projection Wanda errors."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from experiments.dlm_dual_role_allocation.io import atomic_write_json, file_sha256, load_frozen_inputs
from experiments.dlm_loss_aggregation.run import _suffix_logits
from experiments.dlm_role_causal_decomposition.core import (
    CONDITIONS, logit_additivity_metrics, role_intervention_outputs,
)
from experiments.dlm_role_validation.core import deterministic_random_role, partition_digest
from experiments.dlm_role_validation.run import load_dense
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, read_mask
from experiments.wanda_failure_characterization.run_failure_map import metrics, model_sha, state_tensors


ROOT = Path("experiments/dlm_role_causal_decomposition")
RESULTS = Path("/DATA/tmluser1/sap-dlm-role-causal-decomposition/results.pt")
LAYERS = (3, 11, 19, 27)
SEEDS = (101, 202, 303)
LEVEL = 3


def event(kind: str, **values) -> None:
    print(json.dumps({"event": kind, "time": time.time(), **values}, sort_keys=True), flush=True)


def sha(path) -> str:
    return file_sha256(path)


def save_tensor(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def selected_entries(inputs) -> list[dict]:
    wanted = {f"block_{layer:02d}" for layer in LAYERS}
    rows = [row for row in inputs.candidate["entries"] if row["name"].split(".")[0] in wanted]
    if len(rows) != 28:
        raise RuntimeError("expected four layers x seven repository projections")
    for layer in LAYERS:
        if sum(row["name"].startswith(f"block_{layer:02d}.") for row in rows) != 7:
            raise RuntimeError("incomplete selected layer")
    return rows


def freeze() -> dict:
    inputs = load_frozen_inputs()
    entries = selected_entries(inputs)
    partitions = {seed: [] for seed in SEEDS}
    actual = []
    for index, state in enumerate(inputs.states["states"]):
        mask = torch.as_tensor(state["mask"], dtype=torch.bool)[0]
        actual.append(mask)
        for seed in SEEDS:
            partitions[seed].append(deterministic_random_role(mask, seed, index))
    config = {
        "status": "frozen_before_causal_collection",
        "model": inputs.config["model"], "dense_model_sha256": DENSE_SHA,
        "state_digest": inputs.metadata["state_digest"], "state_count": 80,
        "projection_selection": "repository-order seven projections in each fixed layer 03/11/19/27",
        "projection_names": [row["name"] for row in entries],
        "sparsity": .65, "candidate_level": LEVEL,
        "selector": "persisted exact historical Standard Wanda mask",
        "conditions": list(CONDITIONS),
        "energy_matching": "scale the larger actual-role output delta down to the smaller role norm; never amplify",
        "primary": "same-path final masked-token Dense||intervention KL",
        "controls": "three cardinality-matched deterministic random partitions; direct-vs-mixed Both reproduction",
        "random_seeds": list(SEEDS),
        "partition_sha256": {"actual": partition_digest(actual), **{
            f"random_{seed}": partition_digest(partitions[seed]) for seed in SEEDS}},
        "source_hashes": {
            "experiments/projection_capacity_allocation_65/config.json": sha("experiments/projection_capacity_allocation_65/config.json"),
            "experiments/projection_capacity_allocation_65/state_verification.json": sha("experiments/projection_capacity_allocation_65/state_verification.json"),
            "experiments/projection_capacity_allocation_65/candidate_mask_manifest.json": sha("experiments/projection_capacity_allocation_65/candidate_mask_manifest.json"),
        },
        "analysis_policy": "sequence-cluster bootstrap; path tracing only after the role causal gate",
        "no_downstream_or_allocation_selection": True,
    }
    path = ROOT / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("frozen config differs")
    if not path.exists():
        atomic_write_json(path, config)
    event("freeze_complete", config_sha256=sha(path), projections=len(entries))
    return config


def require_frozen() -> dict:
    config = freeze()
    for path, digest in config["source_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"frozen source changed: {path}")
    return config


def allocate_payload(states: int, modules: int) -> dict:
    shape = (states, modules, len(CONDITIONS))
    fields = {name: torch.full(shape, torch.nan) for name in
              ("delta_loss", "kl", "raw_kl", "top1_agreement", "confidence_mae")}
    local_shape = (states, modules, 4, 4)  # partition: actual/3 random; num_a,den_a,num_b,den_b
    return {
        "completed_states": 0, "condition_names": list(CONDITIONS),
        "local_partition_names": ["actual", "random_101", "random_202", "random_303"],
        "local_field_names": ["num_a", "den_a", "num_b", "den_b"],
        "fields": fields, "local": torch.full(local_shape, torch.nan),
        "matched_scales": torch.full((states, modules, 2), torch.nan),
        "logit_mechanism": {name: torch.full((states, modules), torch.nan) for name in
                            ("masked_logit_delta_norm", "unmasked_logit_delta_norm",
                             "both_logit_delta_norm", "additivity_residual_norm",
                             "relative_additivity_residual")},
        "mixed_direct_max_abs": torch.full((states, modules), torch.nan),
        "external_sham_max_abs": torch.full((states, modules), torch.nan),
    }


def validate_resume(payload: dict, config: dict) -> None:
    for key, expected in (("config_sha256", sha(ROOT / "config.json")),
                          ("state_digest", config["state_digest"]),
                          ("module_names", config["projection_names"]),
                          ("condition_names", list(CONDITIONS))):
        if payload.get(key) != expected:
            raise RuntimeError(f"resume {key} mismatch")
    if not 0 <= int(payload["completed_states"]) <= 80:
        raise RuntimeError("invalid resume state count")


def local_partition_stats(dense: torch.Tensor, sparse: torch.Tensor,
                          group: torch.Tensor) -> list[float]:
    group = group.to(dense.device).reshape(-1)
    delta = (sparse[0].float() - dense[0].float()).square()
    energy = dense[0].float().square()
    return [float(delta[group].sum().cpu()), float(energy[group].sum().cpu()),
            float(delta[~group].sum().cpu()), float(energy[~group].sum().cpu())]


@torch.inference_mode()
def collect(stop_after_states: int | None = None) -> dict:
    config = require_frozen()
    inputs = load_frozen_inputs()
    entries = selected_entries(inputs)
    model, mapping = load_dense()
    device = next(model.parameters()).device
    if RESULTS.exists():
        payload = torch.load(RESULTS, map_location="cpu", weights_only=False)
        validate_resume(payload, config)
    else:
        payload = allocate_payload(80, 28)
        payload.update(config_sha256=sha(ROOT / "config.json"), state_digest=config["state_digest"],
                       module_names=config["projection_names"], sequence_index=torch.tensor([
                           state["sequence_index"] for state in inputs.states["states"]]),
                       timestep=torch.tensor([state["timestep"] for state in inputs.states["states"]]))
    start = int(payload["completed_states"])
    target = min(80, start + stop_after_states) if stop_after_states else 80
    started = time.monotonic()
    for state_index in range(start, target):
        state = inputs.states["states"][state_index]
        noisy, clean, token_mask = state_tensors(state, device)
        actual = token_mask[0]
        random = {seed: deterministic_random_role(actual.cpu(), seed, state_index).to(device)
                  for seed in SEEDS}
        prefixes = {}
        handles = []
        for layer in LAYERS:
            def capture(_module, inp, layer_index=layer):
                prefixes[layer_index] = inp[0].detach()
            handles.append(model.model.transformer.blocks[layer].register_forward_pre_hook(capture))
        external_dense = model(noisy).logits
        for handle in handles:
            handle.remove()
        if set(prefixes) != set(LAYERS):
            raise RuntimeError("dense prefix capture incomplete")
        dense_loss = metrics(external_dense, external_dense, clean, token_mask, state["p_mask"])[0][0]
        for module_index, entry in enumerate(entries):
            layer = int(entry["name"].split(".")[0].removeprefix("block_"))
            module = mapping[entry["name"]]
            mask = read_mask(entry, LEVEL, module.weight.device)
            box = {}
            def intervention(mod, inp, out):
                masked_weight = mod.weight.masked_fill(mask, 0)
                sparse = F.linear(inp[0], masked_weight, mod.bias)
                composed, audit = role_intervention_outputs(out, sparse, actual, random)
                box["audit"] = audit
                box["local"] = [local_partition_stats(out, sparse, actual)] + [
                    local_partition_stats(out, sparse, random[seed]) for seed in SEEDS]
                return composed
            handle = module.register_forward_hook(intervention)
            try:
                batch = prefixes[layer].expand(len(CONDITIONS), -1, -1).contiguous()
                logits = _suffix_logits(model, batch, layer)
            finally:
                handle.remove()
            loss, kl, agreement, confidence = metrics(logits, logits[:1], clean, token_mask, state["p_mask"])
            raw_kl = metrics(logits, external_dense, clean, token_mask, state["p_mask"])[1]
            payload["fields"]["delta_loss"][state_index, module_index] = (loss - loss[0]).cpu()
            payload["fields"]["kl"][state_index, module_index] = kl.cpu()
            payload["fields"]["raw_kl"][state_index, module_index] = raw_kl.cpu()
            payload["fields"]["top1_agreement"][state_index, module_index] = agreement.cpu()
            payload["fields"]["confidence_mae"][state_index, module_index] = confidence.cpu()
            payload["local"][state_index, module_index] = torch.tensor(box["local"])
            payload["matched_scales"][state_index, module_index] = torch.tensor([
                box["audit"]["masked_scale"], box["audit"]["unmasked_scale"]])
            mechanism = logit_additivity_metrics(logits, actual)
            for name, value in mechanism.items():
                payload["logit_mechanism"][name][state_index, module_index] = value
            payload["mixed_direct_max_abs"][state_index, module_index] = (
                logits[3].float() - logits[4].float()).abs().max().cpu()
            payload["external_sham_max_abs"][state_index, module_index] = (
                logits[0].float() - external_dense[0].float()).abs().max().cpu()
            del logits, batch, mask
        payload["completed_states"] = state_index + 1
        save_tensor(RESULTS, payload)
        event("state_complete", state=state_index + 1, total=80,
              elapsed_seconds=time.monotonic() - started)
        del prefixes, external_dense
        gc.collect()
        torch.cuda.empty_cache()
    if int(payload["completed_states"]) == 80:
        if model_sha(model) != DENSE_SHA:
            raise RuntimeError("dense model changed during intervention collection")
        if float(payload["mixed_direct_max_abs"].max()) != 0:
            raise RuntimeError("Both mixed/direct reproduction gate failed")
        manifest = {
            "status": "complete", "config_sha256": sha(ROOT / "config.json"),
            "result_path": str(RESULTS), "result_sha256": sha(RESULTS),
            "states": 80, "modules": 28, "conditions": len(CONDITIONS),
            "mixed_direct_max_abs": float(payload["mixed_direct_max_abs"].max()),
            "external_sham_max_abs": float(payload["external_sham_max_abs"].max()),
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "transformers": transformers.__version__},
        }
        atomic_write_json(ROOT / "collection_manifest.json", manifest)
        event("collection_complete", result_sha256=manifest["result_sha256"])
    return {"completed_states": int(payload["completed_states"]), "target_states": 80}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "collect"))
    parser.add_argument("--stop-after-states", type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    freeze()
    if args.phase == "collect":
        collect(args.stop_after_states)


if __name__ == "__main__":
    main()
