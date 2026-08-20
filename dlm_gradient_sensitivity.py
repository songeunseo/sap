import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

from lib.dlm_gradient_sensitivity import (
    NegativeSuffixCostError,
    TimestepSensitivityAccumulator,
    block_state_gradients,
    embed_state,
    feasibility_gate,
    load_mask_block,
    make_masked_state,
    midpoint_timesteps,
    project_scoring_seconds,
    save_mask_block,
    score_block,
)


_SCORE_ARTIFACT_VERSION = 1
_REPEAT_GRADIENT_ATOL = 1e-6
_REPEAT_GRADIENT_RTOL = 1e-5


def _process_memory() -> dict[str, int]:
    values = {"vm_rss_kib": 0, "vm_swap_kib": 0}
    names = {"VmRSS:": "vm_rss_kib", "VmSwap:": "vm_swap_kib"}
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            fields = line.split()
            if fields and fields[0] in names:
                values[names[fields[0]]] = int(fields[1])
    return values


def _model_cuda_devices(model) -> tuple[int, ...]:
    devices = set()
    for target in getattr(model, "hf_device_map", {}).values():
        if isinstance(target, int):
            devices.add(target)
        else:
            target = torch.device(target) if target != "disk" else target
            if target == "disk" or target.type != "cuda":
                raise RuntimeError("CPU/disk offload is invalid for the Stage 0 feasibility gate")
            devices.add(target.index if target.index is not None else torch.cuda.current_device())
    for parameter in model.parameters():
        if parameter.device.type != "cuda":
            raise RuntimeError("CPU/disk offload is invalid for the Stage 0 feasibility gate")
        devices.add(parameter.device.index if parameter.device.index is not None else torch.cuda.current_device())
    if not devices:
        raise RuntimeError("Stage 0 model has no CUDA parameters")
    return tuple(sorted(devices))


def _reset_cuda_peaks(devices: tuple[int, ...]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def _read_cuda_peaks(devices: tuple[int, ...]) -> dict:
    readings = [
        {
            "device": f"cuda:{device}",
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        for device in devices
    ]
    return {
        "devices": readings,
        "max_memory_allocated_bytes": max(
            (reading["max_memory_allocated_bytes"] for reading in readings), default=0
        ),
        "max_memory_reserved_bytes": max(
            (reading["max_memory_reserved_bytes"] for reading in readings), default=0
        ),
    }


def _synchronize(devices: tuple[int, ...] = ()) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


@torch.no_grad()
def _cache_block_inputs(model, noisy_ids: torch.Tensor, block_indices: list[int]) -> dict[int, torch.Tensor]:
    hidden = embed_state(model, noisy_ids)
    cached = {}
    blocks = model.model.transformer.blocks
    for block_index in range(max(block_indices) + 1):
        if block_index in block_indices:
            cached[block_index] = hidden.detach().cpu()
        block = blocks[block_index]
        parameter = next(block.parameters())
        hidden, _ = block(
            hidden.to(device=parameter.device, dtype=parameter.dtype),
            attention_bias=None,
            layer_past=None,
            use_cache=False,
            replace_position=None,
            attn_collector=None,
        )
    return cached


def _measure_block(
    model,
    block_index,
    hidden,
    clean_ids,
    mask,
    p_mask,
    cuda_devices,
    clock=time.perf_counter,
) -> dict:
    from lib.prune_llada import find_layers

    memory_before = _process_memory()
    weights = {
        name: layer.weight
        for name, layer in find_layers(model.model.transformer.blocks[block_index]).items()
    }
    accumulator = TimestepSensitivityAccumulator(weights)
    _reset_cuda_peaks(cuda_devices)
    _synchronize(cuda_devices)
    started = clock()
    gradients, next_cache = block_state_gradients(
        model, block_index, hidden, clean_ids, mask, p_mask
    )
    accumulator.add_state(gradients, "a")
    _synchronize(cuda_devices)
    state_seconds = clock() - started

    for _ in range(3):
        accumulator.add_state(gradients, "a")
    for _ in range(4):
        accumulator.add_state(gradients, "b")
    finish_started = clock()
    accumulator.finish_timestep()
    finish_timestep_seconds = clock() - finish_started
    memory_after = _process_memory()
    cuda_peaks = _read_cuda_peaks(cuda_devices)
    element_count = sum(gradient.numel() for gradient in gradients.values())
    return {
        "block": block_index,
        "state_seconds": state_seconds,
        "finish_timestep_seconds": finish_timestep_seconds,
        "replay": {
            "purpose": "timing-only working-buffer fill",
            "observation_count": 1,
            "replayed_accumulator_slots": 7,
        },
        "cuda_peak_memory": cuda_peaks["devices"],
        "max_memory_allocated_bytes": cuda_peaks["max_memory_allocated_bytes"],
        "max_memory_reserved_bytes": cuda_peaks["max_memory_reserved_bytes"],
        "vm_rss_before_kib": memory_before["vm_rss_kib"],
        "vm_rss_after_kib": memory_after["vm_rss_kib"],
        "vm_swap_before_kib": memory_before["vm_swap_kib"],
        "vm_swap_after_kib": memory_after["vm_swap_kib"],
        "vm_swap_delta_kib": memory_after["vm_swap_kib"] - memory_before["vm_swap_kib"],
        "gradient_element_count": element_count,
        "finite_gradient_count": sum(torch.isfinite(gradient).sum().item() for gradient in gradients.values()),
        "nonzero_gradient_count": sum(torch.count_nonzero(gradient).item() for gradient in gradients.values()),
        "next_block_cache_shape": list(next_cache.shape),
    }


def _project_stage0_seconds(
    last_seconds: float, first_seconds: float, blocks: int, states: int
) -> dict:
    try:
        projection = project_scoring_seconds(
            last_seconds, first_seconds, blocks=blocks, states=states
        )
        fit_status = "unconstrained"
        fit_method = "two_point_linear"
    except NegativeSuffixCostError:
        constant = max(last_seconds, first_seconds)
        projection = {
            "fixed_seconds": constant,
            "suffix_seconds": 0.0,
            "per_block_seconds": [constant] * blocks,
            "projected_scoring_seconds": states * blocks * constant,
        }
        fit_status = "constrained"
        fit_method = "max_endpoint_constant"
    raw_difference = first_seconds - last_seconds
    projection.update(
        {
            "fit_status": fit_status,
            "fit_method": fit_method,
            "raw_endpoint_difference_seconds": raw_difference,
            "raw_unconstrained_suffix_seconds": raw_difference / (blocks - 1),
        }
    )
    return projection


def _run_loaded_feasibility(
    model,
    config: dict,
    clean_ids: torch.Tensor,
    cuda_devices: tuple[int, ...] | None = None,
    clock=time.perf_counter,
) -> dict:
    calibration = config["calibration"]
    mask_id = model.config.mask_token_id
    if mask_id is None:
        raise ValueError("model config does not define mask_token_id")
    noisy_ids, mask, p_mask = make_masked_state(
        clean_ids,
        calibration["timesteps"][0],
        mask_id,
        calibration["seed"],
    )
    measured_blocks = config["stage0"]["measured_blocks"]
    if len(model.model.transformer.blocks) != config["stage0"]["model_block_count"]:
        raise ValueError("configured model block count does not match loaded model")
    if measured_blocks != [config["stage0"]["model_block_count"] - 1, 0]:
        raise ValueError("Stage 0 must measure the last block then the first block")
    if cuda_devices is None:
        cuda_devices = _model_cuda_devices(model)
    cached = _cache_block_inputs(model, noisy_ids, measured_blocks)
    measurements = [
        _measure_block(
            model,
            block,
            cached[block],
            clean_ids,
            mask,
            p_mask,
            cuda_devices,
            clock,
        )
        for block in measured_blocks
    ]
    by_block = {measurement["block"]: measurement for measurement in measurements}
    last_block, first_block = measured_blocks
    blocks = config["stage0"]["model_block_count"]
    states = calibration["sequence_count"] * len(calibration["timesteps"])
    last_seconds = by_block[last_block]["state_seconds"]
    first_seconds = by_block[first_block]["state_seconds"]
    projection = _project_stage0_seconds(
        last_seconds, first_seconds, blocks, states
    )
    finish_per_block = max(
        measurement["finish_timestep_seconds"] for measurement in measurements
    )
    finish_total = (
        len(calibration["timesteps"])
        * blocks
        * finish_per_block
    )
    projected_total = projection["projected_scoring_seconds"] + finish_total
    gate = feasibility_gate(
        measurements,
        projected_total,
        max_gpu_gib=config["stage0"]["max_gpu_memory_gib"],
        max_gpu_hours=config["stage0"]["max_projected_gpu_hours"],
    )
    return {
        "stage": 0,
        "model": config["model"],
        "state": {
            "dataset": config["dataset"],
            "sequence_index": 0,
            "timestep": calibration["timesteps"][0],
            "mask_seed": calibration["seed"],
            "p_mask": p_mask,
        },
        "measurements": measurements,
        "projection": {
            **projection,
            "finish_timestep_seconds_per_block": finish_per_block,
            "projected_statistic_finalization_seconds": finish_total,
            "projected_total_seconds": projected_total,
            "projected_total_gpu_hours": projected_total / 3600,
            "equation": "80 * sum(block state costs) + 10 * 32 * max(measured finish_timestep)",
        },
        "gate": gate,
    }


def run_feasibility(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 0 feasibility requires CUDA")

    from transformers import AutoTokenizer
    from lib.data import get_loaders
    from model import LLaDAModelLM

    model_config = config["model"]
    model = LLaDAModelLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    cuda_devices = _model_cuda_devices(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"], trust_remote_code=True
    )
    calibration = config["calibration"]
    loader, _ = get_loaders(
        config["dataset"]["loader_name"],
        nsamples=1,
        seed=calibration["seed"],
        seqlen=calibration["sequence_length"],
        tokenizer=tokenizer,
    )
    return _run_loaded_feasibility(model, config, loader[0][0], cuda_devices)


def _canonical_json(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _atomic_write_json(path: Path, document: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(_canonical_json(document))
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(destination)
        temporary_name = None
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_digest(model, config: dict) -> str:
    model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config)
    digest = hashlib.sha256(
        _canonical_json({"model": config["model"], "loaded_config": model_config})
    )
    for name, parameter in model.named_parameters():
        digest.update(
            _canonical_json(
                {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            )
        )
        raw = parameter.detach().to(device="cpu").contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _scoring_state_document(model, config: dict, clean_ids: list[torch.Tensor]) -> dict:
    calibration = config["calibration"]
    sequence_indices = calibration["sequence_indices"]
    if len(clean_ids) != calibration["sequence_count"] or len(clean_ids) != len(sequence_indices):
        raise ValueError("clean calibration sequence count does not match configuration")
    mask_id = model.config.mask_token_id
    if mask_id is None:
        raise ValueError("model config does not define mask_token_id")
    states = []
    for timestep_index, timestep in enumerate(calibration["timesteps"]):
        for sequence_position, (sequence_index, clean) in enumerate(zip(sequence_indices, clean_ids)):
            clean = clean.detach().to(device="cpu", dtype=torch.long)
            if clean.ndim != 2 or clean.shape[0] != 1 or clean.shape[1] != calibration["sequence_length"]:
                raise ValueError("clean calibration sequence shape does not match configuration")
            mask_seed = (
                calibration["seed"]
                + timestep_index * calibration["sequence_count"]
                + sequence_position
            )
            noisy, mask, p_mask = make_masked_state(clean, timestep, mask_id, mask_seed)
            states.append(
                {
                    "timestep_index": timestep_index,
                    "timestep": timestep,
                    "sequence_index": sequence_index,
                    "mask_seed": mask_seed,
                    "p_mask": p_mask,
                    "clean_ids": clean.tolist(),
                    "noisy_ids": noisy.tolist(),
                    "mask": mask.tolist(),
                }
            )
    return {"version": _SCORE_ARTIFACT_VERSION, "states": states}


def _cache_scoring_states(model, document: dict) -> dict:
    if document.get("version") != _SCORE_ARTIFACT_VERSION or not isinstance(document.get("states"), list):
        raise ValueError("invalid scoring state artifact")
    cached = {}
    embedding_dtype = model.model.transformer.wte.weight.dtype
    for saved in document["states"]:
        key = (saved["timestep_index"], saved["sequence_index"])
        if key in cached:
            raise ValueError("duplicate scoring state")
        clean_ids = torch.tensor(saved["clean_ids"], dtype=torch.long)
        noisy_ids = torch.tensor(saved["noisy_ids"], dtype=torch.long)
        mask = torch.tensor(saved["mask"], dtype=torch.bool)
        hidden = embed_state(model, noisy_ids).detach().to(device="cpu", dtype=embedding_dtype)
        cached[key] = {
            "timestep_index": saved["timestep_index"],
            "sequence_index": saved["sequence_index"],
            "clean_ids": clean_ids,
            "noisy_ids": noisy_ids,
            "mask": mask,
            "p_mask": saved["p_mask"],
            "hidden": hidden,
        }
    return cached


@torch.no_grad()
def _advance_cached_states(model, cached_states: dict, completed_blocks: int) -> dict:
    advanced = cached_states
    for block_index in range(completed_blocks):
        block = model.model.transformer.blocks[block_index]
        parameter = next(block.parameters())
        next_states = {}
        for key, state in advanced.items():
            hidden, _ = block(
                state["hidden"].to(device=parameter.device, dtype=parameter.dtype),
                attention_bias=None,
                layer_past=None,
                use_cache=False,
                replace_position=None,
                attn_collector=None,
            )
            next_states[key] = {
                **state,
                "hidden": hidden.detach().to(device="cpu", dtype=parameter.dtype),
            }
        advanced = next_states
    return advanced


def _repeat_gradient_square_check(model, config: dict, cached_states: dict) -> dict:
    block_count = len(model.model.transformer.blocks)
    if block_count != 32:
        return {"status": "not_required", "reason": "model does not have 32 blocks"}
    first_key = (0, config["calibration"]["sequence_indices"][0])
    state = _advance_cached_states(model, {first_key: cached_states[first_key]}, block_count - 1)[first_key]
    first, _ = block_state_gradients(
        model,
        block_count - 1,
        state["hidden"],
        state["clean_ids"],
        state["mask"],
        state["p_mask"],
    )
    second, _ = block_state_gradients(
        model,
        block_count - 1,
        state["hidden"],
        state["clean_ids"],
        state["mask"],
        state["p_mask"],
    )
    max_absolute = 0.0
    max_relative = 0.0
    finite = True
    within_tolerance = True
    for name in first:
        left = first[name].square()
        right = second[name].square()
        finite = finite and torch.isfinite(left).all().item() and torch.isfinite(right).all().item()
        difference = (left - right).abs()
        max_absolute = max(max_absolute, difference.max().item())
        relative = difference / torch.maximum(
            torch.maximum(left.abs(), right.abs()),
            torch.tensor(torch.finfo(left.dtype).tiny),
        )
        max_relative = max(max_relative, relative.max().item())
        within_tolerance = within_tolerance and torch.allclose(
            left, right, atol=_REPEAT_GRADIENT_ATOL, rtol=_REPEAT_GRADIENT_RTOL
        )
    result = {
        "status": "passed" if finite and within_tolerance else "failed",
        "block_index": block_count - 1,
        "state": {"timestep_index": first_key[0], "sequence_index": first_key[1]},
        "maximum_absolute_gradient_square_difference": max_absolute,
        "maximum_relative_gradient_square_difference": max_relative,
        "absolute_tolerance": _REPEAT_GRADIENT_ATOL,
        "relative_tolerance": _REPEAT_GRADIENT_RTOL,
        "finite": finite,
        "within_tolerance": within_tolerance,
    }
    return result


def _load_manifest(path: Path) -> dict:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid score manifest") from error
    if not isinstance(manifest, dict) or manifest.get("version") != _SCORE_ARTIFACT_VERSION:
        raise ValueError("invalid score manifest")
    return manifest


def _validate_block_artifact(
    model,
    artifact_dir: Path,
    entry: dict,
    manifest: dict,
    expected_masks: dict | None = None,
    expected_metadata: dict | None = None,
    include_masks: bool = False,
) -> dict | tuple[dict, dict]:
    block_index = entry.get("block_index")
    if isinstance(block_index, bool) or not isinstance(block_index, int):
        raise ValueError("score block index is invalid")
    expected_name = f"block-{block_index:03d}.json"
    if entry.get("path") != expected_name:
        raise ValueError("score block path mismatch")
    block_path = artifact_dir / expected_name
    if not block_path.is_file() or block_path.stat().st_size != entry.get("byte_length"):
        raise ValueError("score block size mismatch")
    if _file_sha256(block_path) != entry.get("sha256"):
        raise ValueError("score block checksum mismatch")
    masks, metadata = load_mask_block(block_path)
    if expected_metadata is not None and not _same_typed_value(
        metadata, expected_metadata
    ):
        raise ValueError("written block metadata mismatch")
    if expected_masks is not None and (
        set(masks) != set(expected_masks)
        or any(not torch.equal(masks[name], expected_masks[name]) for name in masks)
    ):
        raise ValueError("written block masks mismatch")

    for key in ("model_revision", "model_digest", "config_digest", "state_digest"):
        if not _same_typed_value(metadata.get(key), manifest[key]):
            raise ValueError(f"score block {key} mismatch")
    expected_modules = manifest["modules_per_block"]
    expected_updates = manifest["updates_per_module"]
    expected_welford = manifest["welford_updates_per_module"]
    expected_variants = manifest["mask_variant_count"]
    expected_entries = expected_modules * expected_variants
    from lib.prune_llada import find_layers

    block = model.model.transformer.blocks[block_index]
    layers = find_layers(block)
    if len(layers) != expected_modules:
        raise ValueError("score block model module count mismatch")
    expected_module_shapes = {
        name: list(layer.weight.shape) for name, layer in layers.items()
    }
    module_shapes = metadata.get("module_shapes")
    module_diagnostics = metadata.get("module_diagnostics")
    if not isinstance(module_shapes, dict) or not isinstance(module_diagnostics, dict):
        raise ValueError("score block module metadata is invalid")
    module_names = set(expected_module_shapes)
    expected_mask_names = {
        f"{name}|lambda={risk_lambda:g}|sparsity={sparsity:g}"
        for name in module_names
        for risk_lambda in manifest["lambdas"]
        for sparsity in manifest["sparsities"]
    }
    if (
        not _same_typed_value(metadata.get("block_index"), block_index)
        or not _same_typed_value(metadata.get("module_count"), expected_modules)
        or not _same_typed_value(metadata.get("update_count"), expected_updates)
        or not _same_typed_value(metadata.get("welford_updates"), expected_welford)
        or not _same_typed_value(metadata.get("mask_variant_count"), expected_variants)
        or not _same_typed_value(metadata.get("lambdas"), manifest["lambdas"])
        or not _same_typed_value(metadata.get("sparsities"), manifest["sparsities"])
        or not _same_typed_value(
            metadata.get("score_definition"), manifest["score_definition"]
        )
        or not _same_typed_value(metadata.get("calibration"), manifest["calibration"])
        or not _same_typed_value(module_shapes, expected_module_shapes)
        or set(module_diagnostics) != module_names
        or not _diagnostics_have_exact_types(
            module_diagnostics, module_names, expected_mask_names
        )
        or set(masks) != expected_mask_names
        or len(masks) != expected_entries
        or not _same_typed_value(entry.get("module_count"), metadata.get("module_count"))
        or not _same_typed_value(entry.get("update_count"), metadata.get("update_count"))
        or not _same_typed_value(
            entry.get("welford_updates"), metadata.get("welford_updates")
        )
        or not _same_typed_value(
            entry.get("mask_variant_count"), metadata.get("mask_variant_count")
        )
        or not _same_typed_value(entry.get("mask_entry_count"), len(masks))
    ):
        raise ValueError("score block audit metadata mismatch")
    for module_name, shape in expected_module_shapes.items():
        if any(
            list(mask.shape) != shape
            for mask_name, mask in masks.items()
            if mask_name.startswith(f"{module_name}|lambda=")
        ):
            raise ValueError("score block module shape mismatch")
    return (metadata, masks) if include_masks else metadata


def _validated_completed_blocks(
    model, artifact_dir: Path, manifest: dict, expected: dict
) -> tuple[list[dict], list[tuple[dict, dict]]]:
    for key, value in expected.items():
        if not _same_typed_value(manifest.get(key), value):
            raise ValueError(f"score manifest {key} mismatch")
    completed = manifest.get("completed_blocks")
    if not isinstance(completed, list):
        raise ValueError("invalid completed block manifest")
    if [entry.get("block_index") for entry in completed] != list(range(len(completed))):
        raise ValueError("completed score blocks must be contiguous")
    if len(completed) > manifest["block_count"]:
        raise ValueError("score manifest has too many completed blocks")
    validated = []
    for entry in completed:
        validated.append(
            (entry, _validate_block_artifact(model, artifact_dir, entry, manifest))
        )
    return completed, validated


@torch.no_grad()
def _apply_mean_masks(model, artifact_dir: Path, sparsity: float) -> dict:
    artifact_dir = Path(artifact_dir)
    manifest = _load_manifest(artifact_dir / "manifest.json")
    if manifest.get("score_definition") != "mu + lambda * population_std_timestep_sensitivity":
        raise ValueError("score manifest does not contain Mean DLM sensitivity")
    if 0.0 not in manifest.get("lambdas", []):
        raise ValueError("score manifest does not contain lambda=0 Mean masks")
    if sparsity not in manifest.get("sparsities", []):
        raise ValueError(f"sparsity {sparsity} is not present in score artifacts")
    blocks = model.model.transformer.blocks
    completed = manifest.get("completed_blocks", [])
    if manifest.get("block_count") != len(blocks) or len(completed) != len(blocks):
        raise ValueError("score artifacts do not cover every model block")
    if _model_digest(model, {"model": manifest["model"]}) != manifest.get("model_digest"):
        raise ValueError("score manifest model_digest mismatch")

    from lib.prune_llada import find_layers

    modules = []
    pruned_weight_count = 0
    total_weight_count = 0
    for entry in completed:
        metadata, masks = _validate_block_artifact(
            model, artifact_dir, entry, manifest, include_masks=True
        )
        block_index = entry["block_index"]
        layers = find_layers(blocks[block_index])
        for name, layer in layers.items():
            key = f"{name}|lambda=0|sparsity={sparsity:g}"
            mean_diagnostic = metadata["module_diagnostics"][name]["masks"][
                f"lambda=0|sparsity={sparsity:g}"
            ]
            if (
                mean_diagnostic["identical_to_mean"] is not True
                or mean_diagnostic["mean_disagreement_count"] != 0
                or mean_diagnostic["mean_disagreement_fraction"] != 0.0
                or mean_diagnostic["mean_jaccard"] != 1.0
            ):
                raise ValueError("lambda=0 mask is not identical to Mean sensitivity")
            mask = masks[key]
            weight = layer.weight.data
            if torch.count_nonzero(weight == 0).item():
                raise ValueError("Mean masks must be applied to the unpruned dense model")
            device_mask = mask.to(weight.device)
            weight[device_mask] = 0
            zero_count = torch.count_nonzero(weight == 0).item()
            if zero_count != mask.sum().item():
                raise RuntimeError("applied Mean mask zero count mismatch")
            row_counts = mask.sum(dim=1)
            module_total = mask.numel()
            pruned_weight_count += zero_count
            total_weight_count += module_total
            modules.append(
                {
                    "block": block_index,
                    "module": name,
                    "pruned_weight_count": zero_count,
                    "total_weight_count": module_total,
                    "actual_sparsity": zero_count / module_total,
                    "pruned_per_row_min": row_counts.min().item(),
                    "pruned_per_row_max": row_counts.max().item(),
                }
            )
        del masks, metadata

    actual_sparsity = pruned_weight_count / total_weight_count
    calibration = manifest["calibration"]
    return {
        "method": "mean_dlm_sensitivity",
        "score_variant": "lambda=0",
        "model": manifest["model"],
        "model_digest": manifest["model_digest"],
        "config_digest": manifest["config_digest"],
        "state_digest": manifest["state_digest"],
        "lambda_zero_equals_mean": True,
        "aggregation": {
            "sequence_count": len(calibration["split_A"]) + len(calibration["split_B"]),
            "timestep_count": len(calibration["timesteps"]),
            "updates_per_module": manifest["updates_per_module"],
            "welford_updates_per_module": manifest["welford_updates_per_module"],
        },
        "requested_sparsity": sparsity,
        "pruned_weight_count": pruned_weight_count,
        "total_weight_count": total_weight_count,
        "actual_sparsity": actual_sparsity,
        "sparsity_delta": actual_sparsity - sparsity,
        "module_count": len(modules),
        "modules": modules,
    }


def _materialize_loaded_mean(
    model,
    tokenizer,
    artifact_dir: Path,
    sparsity: float,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        **_apply_mean_masks(model, artifact_dir, sparsity),
        "checkpoint": str(output_dir),
    }
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    _atomic_write_json(output_dir / "mean_dlm_pruning.json", report)
    return report


def _materialize_loaded_mean_direct(
    model,
    tokenizer,
    config: dict,
    clean_ids: list[torch.Tensor],
    sparsity: float,
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    if isinstance(sparsity, bool) or not isinstance(sparsity, (int, float)) or not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between 0 and 1")

    state_document = _scoring_state_document(model, config, clean_ids)
    cached_states = _cache_scoring_states(model, state_document)
    score_config = {
        **config,
        "scoring": {**config["scoring"], "lambdas": [], "sparsities": [sparsity]},
    }
    model_digest = _model_digest(model, config)
    modules = []
    pruned_weight_count = 0
    total_weight_count = 0
    from lib.prune_llada import find_layers

    for block_index, block in enumerate(model.model.transformer.blocks):
        block_result, cached_states = score_block(
            model, block_index, cached_states, score_config
        )
        layers = find_layers(block)
        with torch.no_grad():
            for name, layer in layers.items():
                mask = block_result["masks"][(name, 0.0, float(sparsity))]
                weight = layer.weight.data
                if torch.count_nonzero(weight == 0).item():
                    raise ValueError("Mean masks must be applied to the unpruned dense model")
                weight[mask.to(weight.device)] = 0
                zero_count = torch.count_nonzero(weight == 0).item()
                if zero_count != mask.sum().item():
                    raise RuntimeError("applied Mean mask zero count mismatch")
                row_counts = mask.sum(dim=1)
                module_total = mask.numel()
                pruned_weight_count += zero_count
                total_weight_count += module_total
                modules.append(
                    {
                        "block": block_index,
                        "module": name,
                        "pruned_weight_count": zero_count,
                        "total_weight_count": module_total,
                        "actual_sparsity": zero_count / module_total,
                        "pruned_per_row_min": row_counts.min().item(),
                        "pruned_per_row_max": row_counts.max().item(),
                    }
                )
        del block_result

    actual_sparsity = pruned_weight_count / total_weight_count
    calibration = config["calibration"]
    report = {
        "method": "mean_dlm_sensitivity",
        "source": "direct_dlm_gradient_scoring",
        "score_variant": "lambda=0",
        "model": config["model"],
        "model_digest": model_digest,
        "config_digest": hashlib.sha256(_canonical_json(config)).hexdigest(),
        "state_digest": hashlib.sha256(_canonical_json(state_document)).hexdigest(),
        "lambda_zero_equals_mean": True,
        "aggregation": {
            "sequence_count": calibration["sequence_count"],
            "timestep_count": len(calibration["timesteps"]),
            "updates_per_module": len(calibration["timesteps"]),
            "welford_updates_per_module": {
                group: len(calibration["timesteps"]) for group in ("A", "B", "full")
            },
        },
        "requested_sparsity": sparsity,
        "pruned_weight_count": pruned_weight_count,
        "total_weight_count": total_weight_count,
        "actual_sparsity": actual_sparsity,
        "sparsity_delta": actual_sparsity - sparsity,
        "module_count": len(modules),
        "modules": modules,
        "checkpoint": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    _atomic_write_json(output_dir / "mean_dlm_pruning.json", report)
    return report


def run_materialize(
    config: dict,
    artifact_dir: Path,
    sparsity: float,
    output_dir: Path,
) -> dict:
    _validate_production_score_config(config)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    manifest = _load_manifest(Path(artifact_dir) / "manifest.json")
    if hashlib.sha256(_canonical_json(config)).hexdigest() != manifest.get("config_digest"):
        raise ValueError("materialization config does not match score artifacts")

    from transformers import AutoTokenizer
    from main_llada import copy_llada_support_files
    from model import LLaDAModelLM

    model_config = config["model"]
    model = LLaDAModelLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    _model_cuda_devices(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"], trust_remote_code=True
    )
    report = _materialize_loaded_mean(
        model, tokenizer, artifact_dir, sparsity, output_dir
    )
    copy_llada_support_files(model_config["id"], str(output_dir))
    return report


def _stage1_gate(
    module_diagnostics: list[dict], config: dict, all_useful_masks_identical: bool
) -> dict:
    reliability = config["reliability"]
    decision_sparsities = config["scoring"]["decision_sparsities"]
    rho_values = [
        module["rho_sigma_A_sigma_B"]["value"]
        for module in module_diagnostics
        if module["rho_sigma_A_sigma_B"]["value"] is not None
    ]
    undefined_rho_count = len(module_diagnostics) - len(rho_values)
    median_rho = statistics.median(rho_values) if rho_values else None
    mean_jaccard = {
        f"{sparsity:g}": statistics.mean(
            module["masks"][f"lambda=1|sparsity={sparsity:g}"]["split_jaccard"]
            for module in module_diagnostics
        )
        for sparsity in decision_sparsities
    }
    rho_passed = (
        undefined_rho_count == 0
        and median_rho is not None
        and median_rho >= reliability["minimum_median_split_sigma_spearman"]
    )
    mask_checks = {
        sparsity: value >= reliability["minimum_mean_split_mask_jaccard"]
        for sparsity, value in mean_jaccard.items()
    }
    reliability_passed = rho_passed and all(mask_checks.values())
    passed = reliability_passed and not all_useful_masks_identical
    if not reliability_passed:
        stop_reason = "split-half reliability gate failed"
    elif all_useful_masks_identical:
        stop_reason = "all useful Time-Risk masks are identical to Mean"
    else:
        stop_reason = None
    return {
        "passed": passed,
        "reliability_passed": reliability_passed,
        "all_useful_masks_identical": all_useful_masks_identical,
        "median_module_split_sigma_spearman": median_rho,
        "undefined_split_sigma_spearman_modules": undefined_rho_count,
        "mean_split_mask_jaccard": mean_jaccard,
        "checks": {"split_sigma_spearman": rho_passed, "split_mask_jaccard": mask_checks},
        "thresholds": {
            "minimum_median_split_sigma_spearman": reliability[
                "minimum_median_split_sigma_spearman"
            ],
            "minimum_mean_split_mask_jaccard": reliability[
                "minimum_mean_split_mask_jaccard"
            ],
        },
        "stop_reason": stop_reason,
    }


def _same_typed_value(actual, expected) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_typed_value(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _diagnostics_have_exact_types(
    diagnostics: dict, module_names: set[str], mask_names: set[str]
) -> bool:
    try:
        normalized = {}
        for module_name in module_names:
            module = diagnostics[module_name]

            def rho(name):
                value = module[name]
                return {
                    "value": None if value["value"] is None else float(value["value"]),
                    "reason": None if value["reason"] is None else str(value["reason"]),
                    "sample_size": int(value["sample_size"]),
                    "sample_indices_sha256": str(value["sample_indices_sha256"]),
                }

            module_mask_names = {
                name.split("|", 1)[1]
                for name in mask_names
                if name.startswith(f"{module_name}|")
            }
            normalized_masks = {}
            for name in module_mask_names:
                value = module["masks"][name]
                normalized_masks[name] = {
                    "mean_disagreement_count": int(value["mean_disagreement_count"]),
                    "mean_disagreement_fraction": float(
                        value["mean_disagreement_fraction"]
                    ),
                    "mean_jaccard": float(value["mean_jaccard"]),
                    "changed_row_fraction": float(value["changed_row_fraction"]),
                    "identical_to_mean": bool(value["identical_to_mean"]),
                    "split_jaccard": float(value["split_jaccard"]),
                    "split_A_sha256": str(value["split_A_sha256"]),
                    "split_B_sha256": str(value["split_B_sha256"]),
                }
            ratio = module["sigma_over_mu_plus_eps"]
            normalized[module_name] = {
                "sigma_over_mu_plus_eps": {
                    **{
                        key: float(ratio[key])
                        for key in ("epsilon", "median", "p90", "p99")
                    },
                    "sample_size": int(ratio["sample_size"]),
                    "requested_sample_size": int(ratio["requested_sample_size"]),
                    "sample_indices_sha256": str(ratio["sample_indices_sha256"]),
                },
                "rho_mu_sigma": rho("rho_mu_sigma"),
                "rho_sigma_A_sigma_B": rho("rho_sigma_A_sigma_B"),
                "top_sigma_overlap": float(module["top_sigma_overlap"]),
                "top_sigma_fraction": float(module["top_sigma_fraction"]),
                "masks": normalized_masks,
            }
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return _same_typed_value(diagnostics, normalized)


def _validate_production_score_config(config: dict) -> None:
    expected = {
        "model.id": "GSAI-ML/LLaDA-8B-Base",
        "model.revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
        "dataset.id": "Salesforce/wikitext",
        "dataset.configuration": "wikitext-2-raw-v1",
        "dataset.loader_name": "wikitext2",
        "dataset.calibration_split": "train",
        "dataset.validation_split": "validation",
        "calibration.seed": 0,
        "calibration.sequence_indices": list(range(8)),
        "calibration.sequence_count": 8,
        "calibration.sequence_length": 256,
        "calibration.mask_id_source": "model.config.mask_token_id",
        "calibration.epsilon": 0.001,
        "calibration.timesteps": list(midpoint_timesteps()),
        "calibration.split_a": [0, 1, 2, 3],
        "calibration.split_b": [4, 5, 6, 7],
        "scoring.gradient_microbatch_size": 1,
        "scoring.lambdas": [0.25, 0.5, 1.0],
        "scoring.sparsities": [0.5, 0.6, 0.7, 0.75],
        "scoring.decision_sparsities": [0.5, 0.6, 0.7],
        "stage0.model_block_count": 32,
        "stage0.measured_blocks": [31, 0],
        "stage0.max_gpu_memory_gib": 30,
        "stage0.max_swap_delta_kib": 0,
        "stage0.max_projected_gpu_hours": 24,
        "reliability.spearman_sample_size": 262144,
        "reliability.minimum_median_split_sigma_spearman": 0.5,
        "reliability.minimum_mean_split_mask_jaccard": 0.95,
        "reliability.top_sigma_fraction": 0.01,
    }
    for path, value in expected.items():
        try:
            actual = config
            for key in path.split("."):
                actual = actual[key]
        except (KeyError, TypeError):
            raise ValueError(f"{path} is required for production scoring") from None
        if not _same_typed_value(actual, value):
            raise ValueError(f"{path} does not match the predeclared production score contract")
    for section in ("model", "dataset", "calibration", "scoring", "stage0", "reliability"):
        expected_keys = {
            path.split(".", 1)[1] for path in expected if path.startswith(f"{section}.")
        }
        if set(config[section]) != expected_keys:
            raise ValueError(f"{section} keys do not match the predeclared production score contract")
    state_count = config["calibration"]["sequence_count"] * len(
        config["calibration"]["timesteps"]
    )
    if state_count != 80:
        raise ValueError("calibration.state_count must be exactly 80")


def _run_loaded_score(model, config: dict, clean_ids: list[torch.Tensor], artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "manifest.json"
    states_path = artifact_dir / "states.json"
    block_count = len(model.model.transformer.blocks)
    if block_count != config["stage0"]["model_block_count"]:
        raise ValueError("configured model block count does not match loaded model")
    state_document = _scoring_state_document(model, config, clean_ids)
    state_bytes = _canonical_json(state_document)
    state_digest = hashlib.sha256(state_bytes).hexdigest()
    config_digest = hashlib.sha256(_canonical_json(config)).hexdigest()
    model_digest = _model_digest(model, config)
    header = {
        "version": _SCORE_ARTIFACT_VERSION,
        "model": config["model"],
        "model_revision": config["model"]["revision"],
        "model_digest": model_digest,
        "config_digest": config_digest,
        "state_digest": state_digest,
        "state_artifact": {
            "path": states_path.name,
            "sha256": state_digest,
            "byte_length": len(state_bytes),
            "state_count": len(state_document["states"]),
        },
        "block_count": block_count,
        "modules_per_block": 7,
        "updates_per_module": len(config["calibration"]["timesteps"]),
        "welford_updates_per_module": {
            group: len(config["calibration"]["timesteps"])
            for group in ("full", "A", "B")
        },
        "lambdas": [0.0, *config["scoring"]["lambdas"]],
        "sparsities": config["scoring"]["sparsities"],
        "score_definition": "mu + lambda * population_std_timestep_sensitivity",
        "calibration": {
            "sequence_indices": config["calibration"]["sequence_indices"],
            "seed": config["calibration"]["seed"],
            "timesteps": config["calibration"]["timesteps"],
            "split_A": config["calibration"]["split_a"],
            "split_B": config["calibration"]["split_b"],
        },
        "mask_variant_count": (len(config["scoring"]["lambdas"]) + 1)
        * len(config["scoring"]["sparsities"]),
    }
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        if not states_path.is_file() or states_path.stat().st_size != len(state_bytes):
            raise ValueError("scoring state artifact size mismatch")
        if _file_sha256(states_path) != state_digest or states_path.read_bytes() != state_bytes:
            raise ValueError("scoring state checksum mismatch")
        completed, validated_blocks = _validated_completed_blocks(
            model, artifact_dir, manifest, header
        )
        repeat_check = manifest.get("repeat_gradient_square")
        if not isinstance(repeat_check, dict) or (
            block_count == 32 and repeat_check.get("status") != "passed"
        ):
            raise RuntimeError("repeated gradient-square check is not valid")
        cached_states = _cache_scoring_states(model, state_document)
    else:
        orphaned = sorted(artifact_dir.glob("block-*.json"))
        if states_path.exists() or orphaned:
            raise ValueError("score artifacts exist without a manifest")
        _atomic_write_json(states_path, state_document)
        cached_for_repeat = _cache_scoring_states(model, state_document)
        repeat_check = _repeat_gradient_square_check(model, config, cached_for_repeat)
        manifest = {**header, "repeat_gradient_square": repeat_check, "completed_blocks": []}
        _atomic_write_json(manifest_path, manifest)
        if repeat_check["status"] == "failed":
            raise RuntimeError("repeated gradient-square check failed")
        completed = []
        validated_blocks = []
        cached_states = cached_for_repeat

    cached_states = _advance_cached_states(model, cached_states, len(completed))
    for block_index in range(len(completed), block_count):
        block_path = artifact_dir / f"block-{block_index:03d}.json"
        if block_path.exists():
            raise ValueError("untracked score block artifact")
        block_result, cached_states = score_block(
            model, block_index, cached_states, config
        )
        flat_masks = {
            f"{name}|lambda={risk_lambda:g}|sparsity={sparsity:g}": mask
            for (name, risk_lambda, sparsity), mask in block_result["masks"].items()
        }
        metadata = {
            "block_index": block_index,
            "model_revision": header["model_revision"],
            "model_digest": model_digest,
            "config_digest": config_digest,
            "state_digest": state_digest,
            "module_count": len(block_result["statistics"]),
            "update_count": block_result["update_count"],
            "welford_updates": header["welford_updates_per_module"],
            "mask_variant_count": header["mask_variant_count"],
            "lambdas": header["lambdas"],
            "sparsities": header["sparsities"],
            "score_definition": header["score_definition"],
            "calibration": header["calibration"],
            "module_shapes": {
                name: list(values["mu"].shape)
                for name, values in block_result["statistics"].items()
            },
            "module_diagnostics": block_result["diagnostics"],
        }
        serialized = save_mask_block(block_path, flat_masks, metadata)
        entry = {
            "block_index": block_index,
            "path": block_path.name,
            "sha256": serialized["sha256"],
            "byte_length": serialized["byte_length"],
            "module_count": metadata["module_count"],
            "update_count": metadata["update_count"],
            "welford_updates": metadata["welford_updates"],
            "mask_variant_count": metadata["mask_variant_count"],
            "mask_entry_count": len(flat_masks),
        }
        validated_metadata = _validate_block_artifact(
            model,
            artifact_dir,
            entry,
            manifest,
            expected_masks=flat_masks,
            expected_metadata=metadata,
        )
        manifest["completed_blocks"].append(entry)
        _atomic_write_json(manifest_path, manifest)
        validated_blocks.append((entry, validated_metadata))
        del block_result, flat_masks, metadata

    module_diagnostics = []
    all_useful_masks_identical = True
    for entry, metadata in validated_blocks:
        for module_name, diagnostic in metadata["module_diagnostics"].items():
            module_diagnostics.append(
                {"block_index": entry["block_index"], "module": module_name, **diagnostic}
            )
            for risk_lambda in config["scoring"]["lambdas"]:
                for sparsity in config["scoring"]["decision_sparsities"]:
                    all_useful_masks_identical = all_useful_masks_identical and diagnostic[
                        "masks"
                    ][f"lambda={risk_lambda:g}|sparsity={sparsity:g}"]["identical_to_mean"]
    gate = _stage1_gate(module_diagnostics, config, all_useful_masks_identical)
    expected_modules = block_count * header["modules_per_block"]
    return {
        "stage": 1,
        "model": config["model"],
        "model_digest": model_digest,
        "config_digest": config_digest,
        "state_digest": state_digest,
        "repeat_gradient_square": manifest["repeat_gradient_square"],
        "audit": {
            "expected_blocks": block_count,
            "completed_blocks": len(manifest["completed_blocks"]),
            "expected_module_entries": expected_modules,
            "module_entries": len(module_diagnostics),
            "updates_per_module": header["updates_per_module"],
            "welford_updates_per_module": header["welford_updates_per_module"],
            "mask_variant_count": header["mask_variant_count"],
            "packed_mask_entries": sum(
                entry["mask_entry_count"] for entry in manifest["completed_blocks"]
            ),
        },
        "module_diagnostics": module_diagnostics,
        "gate": gate,
    }


def run_score(config: dict, artifact_dir: Path) -> dict:
    _validate_production_score_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 scoring requires CUDA")

    from transformers import AutoTokenizer
    from lib.data import get_loaders
    from model import LLaDAModelLM

    model_config = config["model"]
    model = LLaDAModelLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    _model_cuda_devices(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"], revision=model_config["revision"], trust_remote_code=True
    )
    calibration = config["calibration"]
    loader, _ = get_loaders(
        config["dataset"]["loader_name"],
        nsamples=calibration["sequence_count"],
        seed=calibration["seed"],
        seqlen=calibration["sequence_length"],
        tokenizer=tokenizer,
    )
    return _run_loaded_score(model, config, [sample[0] for sample in loader], artifact_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    feasibility = subparsers.add_parser("feasibility")
    feasibility.add_argument("--config", type=Path, required=True)
    feasibility.add_argument("--output", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--artifact-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--config", type=Path, required=True)
    materialize.add_argument("--artifact-dir", type=Path, required=True)
    materialize.add_argument("--sparsity", type=float, required=True)
    materialize.add_argument("--output-dir", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if args.command == "feasibility":
        report = run_feasibility(config)
    elif args.command == "score":
        report = run_score(config, args.artifact_dir)
    else:
        report = run_materialize(
            config, args.artifact_dir, args.sparsity, args.output_dir
        )
    _atomic_write_json(args.output, report)
    return 0 if args.command == "materialize" or report["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
