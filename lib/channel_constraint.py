import json
import math
from pathlib import Path

import torch


INPUT_COLUMN_MODULES = {"q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"}
OUTPUT_ROW_MODULES = {"attn_out", "ff_out"}
LLADA_8B_MODULE_SHAPES = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 4096),
    "v_proj": (4096, 4096),
    "attn_out": (4096, 4096),
    "ff_proj": (12288, 4096),
    "up_proj": (12288, 4096),
    "ff_out": (4096, 12288),
}


def protected_axis(module_name: str) -> str:
    leaf = module_name.rsplit(".", 1)[-1]
    if leaf in INPUT_COLUMN_MODULES:
        return "column"
    if leaf in OUTPUT_ROW_MODULES:
        return "row"
    raise ValueError(f"Unsupported prunable module: {module_name}")


def validate_llada_8b_module_shape(module_name: str, shape: tuple[int, ...]) -> None:
    leaf = module_name.rsplit(".", 1)[-1]
    expected = LLADA_8B_MODULE_SHAPES.get(leaf)
    if expected is None:
        raise ValueError(f"Unsupported LLaDA-8B prunable module: {module_name}")
    if tuple(shape) != expected:
        raise ValueError(f"Unexpected {module_name} weight shape {tuple(shape)}; expected {expected}")


def validate_delta_request(
    sparsity_ratio: float,
    sparsity_type: str,
    use_variant: bool,
    prune_method: str,
    channel: int,
    hidden_size: int,
) -> None:
    if sparsity_ratio != 0.75:
        raise ValueError("Channel delta generation requires exactly 75% sparsity")
    if sparsity_type != "unstructured":
        raise ValueError("Channel delta generation supports only unstructured pruning")
    if use_variant:
        raise ValueError("Channel delta generation does not support the Wanda variant")
    if prune_method not in {"wanda", "sink"}:
        raise ValueError("Channel delta generation supports only Wanda or Sink-Aware")
    if not 0 <= channel < hidden_size:
        raise ValueError("Protected channel is outside the residual hidden size")


def _lowest_flat_indices(score: torch.Tensor, candidates: torch.Tensor, count: int) -> torch.Tensor:
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    flat_score = score.flatten().masked_fill(~candidates.flatten(), torch.inf)
    if torch.isfinite(flat_score).sum().item() < count:
        raise ValueError("Not enough module-local compensation candidates")
    threshold = torch.kthvalue(flat_score, count).values
    lower = (flat_score < threshold).nonzero().flatten()
    tied = (flat_score == threshold).nonzero().flatten()
    return torch.cat((lower, tied[: count - lower.numel()]))


def apply_delta_to_mask(baseline_mask: torch.Tensor, delta: dict) -> torch.Tensor:
    if tuple(baseline_mask.shape) != tuple(delta["shape"]):
        raise ValueError("Baseline mask shape does not match delta")
    constrained = baseline_mask.clone()
    flat = constrained.flatten()
    flat[delta["restore_indices"].to(flat.device)] = False
    flat[delta["compensation_indices"].to(flat.device)] = True
    return constrained


def build_channel_delta(
    module_name: str,
    weight: torch.Tensor,
    score: torch.Tensor,
    baseline_mask: torch.Tensor,
    channel: int,
) -> dict:
    if weight.ndim != 2 or score.shape != weight.shape or baseline_mask.shape != weight.shape:
        raise ValueError("Weight, score, and baseline mask must have the same 2D shape")
    if baseline_mask.dtype != torch.bool:
        raise ValueError("Baseline mask must be boolean")

    rows, columns = weight.shape
    axis = protected_axis(module_name)
    limit = columns if axis == "column" else rows
    if not 0 <= channel < limit:
        raise ValueError(f"Channel {channel} is outside protected {axis} range {limit}")

    if axis == "column":
        restore_rows = baseline_mask[:, channel].nonzero().flatten()
        restore_indices = restore_rows * columns + channel
        candidates = ~baseline_mask
        candidates[:, channel] = False
        candidate_score = score.masked_fill(~candidates, torch.inf)
        compensation_columns = candidate_score.argmin(dim=1)[restore_rows]
        if restore_rows.numel() and not torch.isfinite(
            candidate_score[restore_rows, compensation_columns]
        ).all():
            raise ValueError("Not enough row-local compensation candidates")
        compensation_indices = restore_rows * columns + compensation_columns
        protected_total = rows
    else:
        restore_columns = baseline_mask[channel].nonzero().flatten()
        restore_indices = channel * columns + restore_columns
        candidates = ~baseline_mask
        candidates[channel] = False
        compensation_indices = _lowest_flat_indices(score, candidates, restore_columns.numel())
        protected_total = columns

    restore_indices = restore_indices.to(torch.long)
    compensation_indices = compensation_indices.to(torch.long)
    delta = {
        "shape": tuple(weight.shape),
        "restore_indices": restore_indices.detach().cpu(),
        "restore_values": weight.flatten()[restore_indices].detach().cpu().clone(),
        "compensation_indices": compensation_indices.detach().cpu(),
    }
    constrained = apply_delta_to_mask(baseline_mask, delta)
    restored = restore_indices.numel()
    compensated = compensation_indices.numel()
    baseline_pruned = baseline_mask.sum().item()
    constrained_pruned = constrained.sum().item()
    difference = (baseline_mask != constrained).sum().item()

    if restored != compensated:
        raise AssertionError("protected_restored must equal compensation_pruned")
    if difference != 2 * restored:
        raise AssertionError("mask_difference_count must equal twice protected_restored")
    if constrained_pruned != baseline_pruned:
        raise AssertionError("constraint changed the module pruning count")
    if axis == "column" and constrained[:, channel].any():
        raise AssertionError("protected column still contains pruned weights")
    if axis == "row" and constrained[channel].any():
        raise AssertionError("protected row still contains pruned weights")

    already_survived = protected_total - restored
    total = baseline_mask.numel()
    delta["stats"] = {
        "total_weights": total,
        "baseline_pruned": baseline_pruned,
        "constrained_pruned": constrained_pruned,
        "baseline_sparsity": baseline_pruned / total,
        "constrained_sparsity": constrained_pruned / total,
        "protected_total": protected_total,
        "protected_already_survived_in_baseline": already_survived,
        "protected_already_survived_fraction": already_survived / protected_total,
        "protected_restored": restored,
        "protected_restored_fraction": restored / protected_total,
        "compensation_pruned": compensated,
        "mask_difference_count": difference,
        "mask_difference_fraction": difference / total,
    }
    return delta


def select_experiment_channels(
    hidden_size: int, protected: int, random_count: int, seed: int
) -> list[int]:
    if hidden_size <= 0 or not 0 <= protected < hidden_size:
        raise ValueError("Protected channel must be inside the residual hidden size")
    if not 0 <= random_count <= hidden_size - 1:
        raise ValueError("Random channel count exceeds available control channels")
    candidates = torch.cat((torch.arange(protected), torch.arange(protected + 1, hidden_size)))
    generator = torch.Generator().manual_seed(seed)
    random_channels = candidates[torch.randperm(candidates.numel(), generator=generator)[:random_count]]
    return [protected, *random_channels.tolist()]


def new_delta_bundles(channels: list[int]) -> dict[int, dict]:
    if len(channels) != len(set(channels)):
        raise ValueError("Experiment channels must be distinct")
    return {channel: {"channel": channel, "modules": {}} for channel in channels}


def record_module_deltas(
    bundles: dict[int, dict],
    module_name: str,
    weight: torch.Tensor,
    score: torch.Tensor,
    baseline_mask: torch.Tensor,
) -> None:
    for channel, bundle in bundles.items():
        if module_name in bundle["modules"]:
            raise ValueError(f"Duplicate module delta: {module_name}")
        bundle["modules"][module_name] = build_channel_delta(
            module_name, weight, score, baseline_mask, channel
        )


def _bundle_summary(bundle: dict, metadata: dict) -> dict:
    module_stats = {
        name: {"shape": list(delta["shape"]), **delta["stats"]}
        for name, delta in bundle["modules"].items()
    }
    total = sum(stats["total_weights"] for stats in module_stats.values())
    baseline_pruned = sum(stats["baseline_pruned"] for stats in module_stats.values())
    constrained_pruned = sum(stats["constrained_pruned"] for stats in module_stats.values())
    protected_total = sum(stats["protected_total"] for stats in module_stats.values())
    survived = sum(
        stats["protected_already_survived_in_baseline"] for stats in module_stats.values()
    )
    restored = sum(stats["protected_restored"] for stats in module_stats.values())
    compensated = sum(stats["compensation_pruned"] for stats in module_stats.values())
    difference = sum(stats["mask_difference_count"] for stats in module_stats.values())
    return {
        **metadata,
        "channel": bundle["channel"],
        "total_prunable_weights": total,
        "total_pruned_weights": constrained_pruned,
        "baseline_total_pruned_weights": baseline_pruned,
        "actual_global_sparsity": constrained_pruned / total,
        "protected_total": protected_total,
        "protected_weight_fraction": protected_total / total,
        "protected_already_survived_in_baseline": survived,
        "protected_already_survived_fraction": survived / protected_total,
        "protected_restored": restored,
        "protected_restored_fraction": restored / protected_total,
        "compensation_pruned": compensated,
        "mask_difference_count": difference,
        "mask_difference_fraction": difference / total,
        "module_wise": module_stats,
    }


def save_delta_bundles(
    bundles: dict[int, dict], output_dir: Path | str, metadata: dict
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for channel, bundle in bundles.items():
        summary = _bundle_summary(bundle, metadata)
        artifact = {
            "channel": channel,
            "metadata": dict(metadata),
            "modules": bundle["modules"],
            "summary": summary,
        }
        path = output_dir / f"channel-{channel}.pt"
        torch.save(artifact, path)
        (output_dir / f"channel-{channel}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        paths.append(path)
    return paths


def _select_delta_positions(modules: dict, block: int | None, top_fraction: float) -> dict:
    if not 0 < top_fraction <= 1:
        raise ValueError("Delta top fraction must be in (0, 1]")
    prefix = None if block is None else f"model.transformer.blocks.{block}."
    selected = {
        name: delta for name, delta in modules.items() if prefix is None or name.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"No delta modules found for block {block}")

    magnitudes = torch.cat(
        [delta["restore_values"].float().abs().flatten() for delta in selected.values()]
    )
    count = math.ceil(magnitudes.numel() * top_fraction)
    chosen = torch.zeros(magnitudes.numel(), dtype=torch.bool)
    chosen[torch.argsort(magnitudes, descending=True, stable=True)[:count]] = True

    positions = {}
    offset = 0
    for name, delta in selected.items():
        size = delta["restore_indices"].numel()
        positions[name] = chosen[offset : offset + size].nonzero().flatten()
        offset += size
    return positions


def apply_channel_delta(
    model: torch.nn.Module,
    artifact_path: Path | str,
    block: int | None = None,
    top_fraction: float = 1.0,
) -> dict:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    named_modules = dict(model.named_modules())
    prepared = []
    positions = _select_delta_positions(artifact["modules"], block, top_fraction)
    source_restorable = sum(
        artifact["modules"][name]["restore_indices"].numel() for name in positions
    )

    for name, selected_positions in positions.items():
        if not selected_positions.numel():
            continue
        delta = artifact["modules"][name]
        module = named_modules.get(name)
        if module is None or not hasattr(module, "weight"):
            raise ValueError(f"Delta module not found: {name}")
        weight = module.weight.data
        if tuple(weight.shape) != tuple(delta["shape"]):
            raise ValueError(f"Weight shape mismatch for {name}")
        flat = weight.flatten()
        restore = delta["restore_indices"][selected_positions].to(flat.device)
        compensation = delta["compensation_indices"][selected_positions].to(flat.device)
        restore_values = delta["restore_values"][selected_positions]
        if flat[restore].count_nonzero().item():
            raise ValueError(f"Restore targets are not pruned in baseline module {name}")
        if compensation.numel() and flat[compensation].eq(0).any():
            raise ValueError(f"Compensation targets are already pruned in baseline module {name}")
        prepared.append((name, flat, restore, compensation, restore_values))

    module_wise = {}
    for name, flat, restore, compensation, restore_values in prepared:
        before = flat.eq(0).sum().item()
        flat[restore] = restore_values.to(device=flat.device, dtype=flat.dtype)
        flat[compensation] = 0
        after = flat.eq(0).sum().item()
        if before != after:
            raise AssertionError(f"Actual zero count changed for {name}")
        module_wise[name] = {
            "total_weights": flat.numel(),
            "zeros_before": before,
            "zeros_after": after,
            "sparsity_before": before / flat.numel(),
            "sparsity_after": after / flat.numel(),
        }

    restored = sum(len(restore) for _, _, restore, _, _ in prepared)
    return {
        **artifact["summary"],
        "application": {
            "block": block,
            "top_fraction": top_fraction,
            "source_restorable": source_restorable,
            "restored": restored,
            "compensation_pruned": restored,
            "mask_difference_count": 2 * restored,
            "modules_changed": len(prepared),
        },
        "application_module_wise": module_wise,
    }
