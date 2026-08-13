import torch


INPUT_COLUMN_MODULES = {"q_proj", "k_proj", "v_proj", "ff_proj", "up_proj"}
OUTPUT_ROW_MODULES = {"attn_out", "ff_out"}


def protected_axis(module_name: str) -> str:
    leaf = module_name.rsplit(".", 1)[-1]
    if leaf in INPUT_COLUMN_MODULES:
        return "column"
    if leaf in OUTPUT_ROW_MODULES:
        return "row"
    raise ValueError(f"Unsupported prunable module: {module_name}")


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
