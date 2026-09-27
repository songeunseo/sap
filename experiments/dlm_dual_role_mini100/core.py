"""Pure validation helpers for the dual-role GSM8K mini experiment."""
from __future__ import annotations

from pathlib import Path


def selected_mask_identity(manifest: dict) -> list[tuple[str, str, int]]:
    entries = manifest.get("entries", [])
    return [
        (row["name"], row["selected_mask"]["mask_sha256"],
         int(row["selected_mask"]["pruned"]))
        for row in entries
    ]


def same_selected_masks(left: dict, right: dict) -> bool:
    return selected_mask_identity(left) == selected_mask_identity(right)


def validate_manifest(manifest: dict, expected_names: list[str], target: int,
                      weights: int) -> None:
    entries = manifest.get("entries", [])
    if [row.get("name") for row in entries] != expected_names:
        raise RuntimeError("mask manifest module ordering mismatch")
    if len(entries) != 224 or len(set(expected_names)) != 224:
        raise RuntimeError("mask manifest must contain 224 unique projections")
    counted = sum(int(row["selected_mask"]["pruned"]) for row in entries)
    if int(manifest.get("pruned", -1)) != target or counted != target:
        raise RuntimeError("mask manifest exact budget mismatch")
    if int(manifest.get("weights", -1)) != weights:
        raise RuntimeError("mask manifest weight count mismatch")
    for row in entries:
        meta = row["selected_mask"]
        if not Path(meta["path"]).is_file() or not meta.get("file_sha256") or not meta.get("mask_sha256"):
            raise RuntimeError("mask manifest payload metadata incomplete")


def mini_decision(aggregate_correct: int, role_correct: int,
                  uniform_correct: int) -> dict:
    primary = role_correct > aggregate_correct
    return {
        "decision": "KEEP FOR FULL GSM8K" if primary else "STOP",
        "primary_role_better_than_aggregate": primary,
        "secondary_role_better_than_uniform": role_correct > uniform_correct,
        "aggregate_correct": int(aggregate_correct),
        "role_correct": int(role_correct),
        "uniform_correct": int(uniform_correct),
        "role_minus_aggregate": int(role_correct - aggregate_correct),
        "role_minus_uniform": int(role_correct - uniform_correct),
    }
