#!/usr/bin/env python3
"""Collect dense residual and projection statistics on the frozen 80 DLM states."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from experiments.dlm_loss_aggregation.run import historical_state_digest
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, load_dense
from experiments.wanda_failure_characterization.run_failure_map import model_sha, state_tensors
from experiments.dlm_super_outlier_statistics.core import (
    SUPER_CHANNEL, read_contribution, residual_addition, tensor_statistics,
)

ROOT = Path("experiments/dlm_super_outlier_statistics")
CAL = Path("experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json")
CANDIDATES = Path("experiments/projection_capacity_allocation_65/candidate_mask_manifest.json")
RAW = ROOT / "raw_statistics.pt"
STATE_VERIFICATION = ROOT / "state_verification.json"
READ_TYPES = {"q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_state_verification(payload):
    """Persist a compact, human-readable receipt for the frozen collection."""
    states = payload["states"]
    receipt = {
        "status": payload["status"],
        "state_count": len(states),
        "sequence_count": len({int(row["sequence_index"]) for row in states}),
        "timestep_count": len({int(row["timestep_index"]) for row in states}),
        "state_keys": [[int(row["sequence_index"]), int(row["timestep_index"])]
                       for row in states],
        "projection_count": len(payload["fingerprint"]["module_names"]),
        "fingerprint": payload["fingerprint"],
        "validation": payload["validation"],
        "timing": payload["timing"],
        "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
        "environment": payload["environment"],
    }
    atomic_json(STATE_VERIFICATION, receipt)
    return receipt


def event(kind, **values):
    print(json.dumps({"event": kind, **values}, sort_keys=True), flush=True)


def _fingerprint(config, manifest, names):
    return {
        "config_sha256": sha(ROOT / "config.json"),
        "calibration_sha256": sha(CAL),
        "historical_state_sha256": historical_state_digest(manifest),
        "candidate_manifest_sha256": sha(CANDIDATES),
        "dense_model_sha256": DENSE_SHA,
        "module_names": names,
        "super_channel": SUPER_CHANNEL,
    }


def _validate_inputs(config, manifest, mapping):
    if config["sources"][str(CAL)] != sha(CAL):
        raise RuntimeError("calibration manifest changed")
    if config["sources"][str(CANDIDATES)] != sha(CANDIDATES):
        raise RuntimeError("candidate manifest changed")
    if historical_state_digest(manifest) != manifest["historical_state_sha256"]:
        raise RuntimeError("historical state digest mismatch")
    entries = json.loads(CANDIDATES.read_text())["entries"]
    if [row["name"] for row in entries] != list(mapping):
        raise RuntimeError("projection order mismatch")
    if len(entries) != 224:
        raise RuntimeError("expected 224 projections")


@torch.inference_mode()
def collect_state(model, mapping, state):
    blocks = model.model.transformer.blocks
    captured = {"linear": {}, "residual": {}}
    handles = []

    for name, module in mapping.items():
        ptype = name.split(".", 1)[1]
        def linear_hook(mod, inp, out, module_name=name, projection_type=ptype):
            x, y = inp[0], out
            row = {"input": tensor_statistics(x), "output": tensor_statistics(y)}
            if projection_type in READ_TYPES:
                row["read"] = read_contribution(x, y, mod.weight, mod.bias)
            captured["linear"][module_name] = row
        handles.append(module.register_forward_hook(linear_hook))

    for layer, block in enumerate(blocks):
        slot = captured["residual"].setdefault(str(layer), {})
        def block_pre(_, inp, target=slot):
            target["block_input_tensor"] = inp[0].detach()
            target["block_input"] = tensor_statistics(inp[0])
        def attention_residual_pre(_, inp, target=slot):
            target["attention_residual_tensor"] = inp[0].detach()
            target["attention_residual"] = tensor_statistics(inp[0])
        def block_post(_, __, out, target=slot):
            value = out[0] if isinstance(out, tuple) else out
            target["block_output_tensor"] = value.detach()
            target["block_output"] = tensor_statistics(value)
        handles.extend([
            block.register_forward_pre_hook(block_pre),
            block.ff_norm.register_forward_pre_hook(attention_residual_pre),
            block.register_forward_hook(block_post),
        ])

    dev = next(model.parameters()).device
    try:
        noisy, _, _ = state_tensors(state, dev)
        logits = model(noisy).logits
    finally:
        for handle in handles:
            handle.remove()

    if set(captured["linear"]) != set(mapping) or len(captured["residual"]) != 32:
        raise RuntimeError("incomplete hook capture")
    for layer in range(32):
        row = captured["residual"][str(layer)]
        row["attention_addition"] = residual_addition(
            row["block_input_tensor"], row["attention_residual_tensor"])
        row["mlp_addition"] = residual_addition(
            row["attention_residual_tensor"], row["block_output_tensor"])
        for key in list(row):
            if key.endswith("_tensor"):
                del row[key]
        captured["linear"][f"block_{layer:02d}.attn_out"]["write"] = row["attention_addition"]
        captured["linear"][f"block_{layer:02d}.ff_out"]["write"] = row["mlp_addition"]
    return captured, logits.detach()


@torch.inference_mode()
def collect():
    config = json.loads((ROOT / "config.json").read_text())
    manifest = json.loads(CAL.read_text())
    states = manifest["states"]
    keys = [(int(s["sequence_index"]), int(s["timestep_index"])) for s in states]
    if sorted(keys) != [(s, t) for s in range(8) for t in range(10)]:
        raise RuntimeError("expected frozen 8 x 10 state grid")
    load_started = time.monotonic()
    model, mapping = load_dense()
    _validate_inputs(config, manifest, mapping)
    fingerprint = _fingerprint(config, manifest, list(mapping))
    load_seconds = time.monotonic() - load_started
    if RAW.exists():
        payload = torch.load(RAW, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") != fingerprint or len(payload.get("states", [])) != 80:
            raise RuntimeError("existing raw artifact has a different identity or is incomplete")
        write_state_verification(payload)
        return payload

    baseline = None
    rows = []
    started = time.monotonic()
    for sequence in range(8):
        checkpoint = ROOT / "runtime" / f"sequence_{sequence}.pt"
        selected = sorted((s for s in states if int(s["sequence_index"]) == sequence),
                          key=lambda row: int(row["timestep_index"]))
        if checkpoint.exists():
            cached = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if cached.get("fingerprint") != fingerprint or len(cached.get("states", [])) != 10:
                raise RuntimeError("invalid sequence checkpoint")
            rows.extend(cached["states"])
            event("sequence_reused", sequence=sequence)
            continue
        sequence_rows = []
        for state in selected:
            captured, logits = collect_state(model, mapping, state)
            if baseline is None:
                dev = next(model.parameters()).device
                noisy, _, _ = state_tensors(state, dev)
                reference = model(noisy).logits
                baseline = float((reference.float() - logits.float()).abs().max().cpu())
                if baseline != 0.0:
                    raise RuntimeError("read-only hooks changed logits")
            sequence_rows.append({
                "sequence_index": int(state["sequence_index"]),
                "timestep_index": int(state["timestep_index"]),
                "timestep": float(state["timestep"]),
                **captured,
            })
            event("state_complete", sequence=sequence,
                  timestep=int(state["timestep_index"]))
        atomic_save(checkpoint, {"fingerprint": fingerprint, "states": sequence_rows})
        rows.extend(sequence_rows)
        event("sequence_complete", sequence=sequence,
              elapsed_seconds=time.monotonic() - started)

    if model_sha(model) != DENSE_SHA:
        raise RuntimeError("dense model changed during collection")
    payload = {
        "status": "complete", "fingerprint": fingerprint, "states": rows,
        "validation": {"hook_logits_max_abs_difference": baseline,
                       "model_hash_after_collection": DENSE_SHA},
        "timing": {"model_load_seconds": load_seconds,
                   "collection_seconds": time.monotonic() - started},
        "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated()
                                   if torch.cuda.is_available() else 0),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
    }
    atomic_save(RAW, payload)
    write_state_verification(payload)
    return payload


if __name__ == "__main__":
    result = collect()
    event("collection_complete", states=len(result["states"]), timing=result["timing"])
