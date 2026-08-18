import argparse
import json
import sys
import time
from pathlib import Path

import torch

from lib.dlm_gradient_sensitivity import (
    TimestepSensitivityAccumulator,
    block_state_gradients,
    embed_state,
    feasibility_gate,
    make_masked_state,
    project_scoring_seconds,
)


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
    projection = project_scoring_seconds(
        by_block[last_block]["state_seconds"],
        by_block[first_block]["state_seconds"],
        blocks=config["stage0"]["model_block_count"],
        states=calibration["sequence_count"] * len(calibration["timesteps"]),
    )
    finish_per_block = max(
        measurement["finish_timestep_seconds"] for measurement in measurements
    )
    finish_total = (
        len(calibration["timesteps"])
        * config["stage0"]["model_block_count"]
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    feasibility = subparsers.add_parser("feasibility")
    feasibility.add_argument("--config", type=Path, required=True)
    feasibility.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    report = run_feasibility(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
