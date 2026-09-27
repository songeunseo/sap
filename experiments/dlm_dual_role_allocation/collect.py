"""GPU collection of masked/unmasked local reconstruction sufficient statistics."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Callable

import torch

from experiments.dlm_dual_role_allocation.core import partition_role_sums
from experiments.dlm_dual_role_allocation.io import (
    FrozenInputs,
    atomic_write_json,
    json_sha256,
    load_frozen_inputs,
    validate_checkpoint,
)
from experiments.dlm_loss_aggregation.run import _load_model, _suffix_logits, load_config
from experiments.projection_capacity_allocation_65.core import GRID
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, read_mask
from experiments.wanda_failure_characterization.core import masked_linear_variants
from experiments.wanda_failure_characterization.run_failure_map import model_sha, modules

ROOT = Path("experiments/dlm_dual_role_allocation")
MODEL_CONFIG = "experiments/dlm_loss_aggregation/config.yaml"


class _TargetReached(Exception):
    """Private non-error control flow used to stop immediately after a target Linear."""


def event(event_type: str, **values) -> None:
    print(json.dumps({"event": event_type, **values}, sort_keys=True), flush=True)


@torch.inference_mode()
def _collect_with_runner(run_forward: Callable[[torch.Tensor], object], module: torch.nn.Module,
                         prefixes: list[torch.Tensor], states: list[dict],
                         masks: list[torch.Tensor]) -> list[dict]:
    if len(prefixes) != len(states) or len(masks) != 6:
        raise ValueError("collector requires aligned states and six masks")
    records = []
    for state_index, (prefix, state) in enumerate(zip(prefixes, states)):
        captured = {}
        position_mask = torch.as_tensor(state["mask"], dtype=torch.bool,
                                        device=module.weight.device)
        if position_mask.ndim != 2 or position_mask.shape[0] != 1:
            raise ValueError("persisted mask must have shape [1, sequence]")

        def hook(mod, inp, _out):
            variant_input = inp[0]
            if variant_input.ndim != 3 or variant_input.shape[0] != 7:
                raise RuntimeError("target did not receive seven variants")
            reference = variant_input[:1].expand_as(variant_input)
            if not torch.equal(variant_input, reference):
                raise RuntimeError("target variant inputs differ before intervention")
            variants = masked_linear_variants(variant_input, mod.weight, mod.bias,
                                              [None] + masks)
            captured.update(partition_role_sums(variants, position_mask[0]))
            raise _TargetReached

        handle = module.register_forward_hook(hook)
        reached = False
        try:
            run_forward(prefix.expand(7, -1, -1).contiguous())
        except _TargetReached:
            reached = True
        finally:
            handle.remove()
        if not reached:
            raise RuntimeError("target Linear was not reached")

        levels = []
        for level in range(6):
            num_masked = captured["num_masked"][level]
            num_unmasked = captured["num_unmasked"][level]
            den_masked = captured["den_masked"]
            den_unmasked = captured["den_unmasked"]
            levels.append({
                "sparsity": GRID[level],
                "num_masked": num_masked,
                "den_masked": den_masked,
                "num_unmasked": num_unmasked,
                "den_unmasked": den_unmasked,
                "reconstructed_aggregate_error": (
                    (num_masked + num_unmasked) / (den_masked + den_unmasked)
                ),
            })
        records.append({
            "state_index": state_index,
            "sequence_index": int(state["sequence_index"]),
            "timestep": float(state["timestep"]),
            "p_mask": float(state["p_mask"]),
            "levels": levels,
        })
    return records


@torch.inference_mode()
def collect_projection_from_block(block: torch.nn.Module, module: torch.nn.Module,
                                  prefixes: list[torch.Tensor], states: list[dict],
                                  masks: list[torch.Tensor]) -> list[dict]:
    """Testable block-local target-stop collector."""
    return _collect_with_runner(block, module, prefixes, states, masks)


@torch.inference_mode()
def collect_projection(model, module: torch.nn.Module, block_index: int,
                       prefixes: list[torch.Tensor], states: list[dict],
                       masks: list[torch.Tensor]) -> list[dict]:
    return _collect_with_runner(
        lambda hidden: _suffix_logits(model, hidden, block_index),
        module,
        prefixes,
        states,
        masks,
    )


def checkpoint_expected(inputs: FrozenInputs, row: dict) -> dict:
    return {
        "source_receipt_sha256": inputs.receipt["receipt_sha256"],
        "dense_model_sha256": inputs.metadata["dense_model_sha256"],
        "state_digest": inputs.metadata["state_digest"],
        "module_index": int(row["module_index"]),
        "name": row["name"],
        "shape": row["shape"],
        "grid": list(GRID),
        "mask_sha256": [mask["mask_sha256"] for mask in row["masks"]],
        "state_count": inputs.metadata["state_count"],
        "level_count": len(GRID),
    }


@torch.inference_mode()
def collect_all(destination: str | Path = ROOT, stop_after: int | None = None) -> dict:
    """Collect all projections, resuming only identity-validated checkpoints."""
    destination = Path(destination)
    checkpoints = destination / "runtime" / "role_stats"
    checkpoints.mkdir(parents=True, exist_ok=True)
    inputs = load_frozen_inputs()
    model, _ = _load_model(load_config(MODEL_CONFIG))
    mapping = modules(model)
    if list(mapping) != inputs.metadata["module_names"]:
        raise RuntimeError("loaded model module ordering differs from frozen artifacts")
    dense_before = model_sha(model)
    if dense_before != DENSE_SHA or dense_before != inputs.metadata["dense_model_sha256"]:
        raise RuntimeError("loaded dense model hash mismatch")
    states = inputs.states["states"]
    device = next(model.parameters()).device

    prefixes = {block: [] for block in range(32)}
    handles = []
    for block_index, block in enumerate(model.model.transformer.blocks):
        def save_prefix(_module, inp, index=block_index):
            prefixes[index].append(inp[0].detach().clone())
        handles.append(block.register_forward_pre_hook(save_prefix))
    try:
        for state in states:
            model(torch.tensor(state["noisy_ids"], dtype=torch.long, device=device))
    finally:
        for handle in handles:
            handle.remove()
    if any(len(rows) != len(states) for rows in prefixes.values()):
        raise RuntimeError("dense prefix collection is incomplete")
    event("dense_prefixes_complete", states=len(states), blocks=len(prefixes))

    completed = 0
    started = time.monotonic()
    try:
        for row in inputs.candidate["entries"]:
            if stop_after is not None and completed >= stop_after:
                break
            expected = checkpoint_expected(inputs, row)
            path = checkpoints / f"{row['name']}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                validate_checkpoint(payload, expected)
                event("role_projection_skipped", module=int(row["module_index"]) + 1,
                      total=len(inputs.candidate["entries"]), name=row["name"])
                completed += 1
                continue
            module = mapping[row["name"]]
            candidate_masks = [read_mask(row, level, module.weight.device)
                               for level in range(6)]
            records = collect_projection(
                model,
                module,
                int(row["name"].split(".")[0].removeprefix("block_")),
                prefixes[int(row["name"].split(".")[0].removeprefix("block_"))],
                states,
                candidate_masks,
            )
            payload = {**expected, "states": records}
            payload["payload_sha256"] = json_sha256(payload)
            atomic_write_json(path, payload)
            validate_checkpoint(payload, expected)
            completed += 1
            event("role_projection_complete", module=int(row["module_index"]) + 1,
                  total=len(inputs.candidate["entries"]), name=row["name"],
                  elapsed_seconds=time.monotonic() - started)
            del candidate_masks, records, payload
    finally:
        del prefixes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    dense_after = model_sha(model)
    if dense_after != dense_before:
        raise RuntimeError("dense weights changed during role collection")
    result = {
        "completed_projections": completed,
        "requested_projections": stop_after or len(inputs.candidate["entries"]),
        "dense_sha_before": dense_before,
        "dense_sha_after": dense_after,
        "destination": str(destination),
    }
    atomic_write_json(destination / "collection_manifest.json", result)
    event("collection_complete", **result)
    return result
