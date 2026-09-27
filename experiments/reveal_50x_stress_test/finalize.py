"""Verify and package the completed Reveal-50x stress test artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = REPO / "experiments/dlm_loss_aggregation/exp004_strong_reveal"
PARTITION = REPO / "experiments/dlm_loss_aggregation/exp004/token_partition_summary.json"
UNIFORM_PREDICTIONS = REPO / "experiments/dlm_loss_aggregation/exp002/results/predictions/dlm_abs.jsonl"
REVEAL2_PREDICTIONS = REPO / "experiments/dlm_loss_aggregation/exp004/gsm8k_per_example_results.jsonl"
REVEAL50_PREDICTIONS = SOURCE / "results/reveal50_abs_predictions.jsonl"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path: Path, method: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if method is not None:
        rows = [row for row in rows if row.get("method") == method]
    return rows[:100]


def stats(values: list[float | int]) -> dict:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def mask_digest(shape: list[int], bits: bytes) -> str:
    header = json.dumps(
        {"shape": shape, "bitorder": "big"}, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(header + bits).hexdigest()


def verify_mask(mask_document: dict) -> dict:
    total_bits = 0
    pruned_bits = 0
    rowwise_exact = True
    for entry in mask_document["entries"]:
        path = Path(entry["runtime_path"])
        bits = path.read_bytes()
        if len(bits) != entry["byte_length"]:
            raise RuntimeError(f"mask byte length mismatch: {path}")
        if mask_digest(entry["shape"], bits) != entry["sha256"]:
            raise RuntimeError(f"mask hash mismatch: {path}")
        rows, columns = entry["shape"]
        expected = rows * (columns // 2)
        observed = sum(byte.bit_count() for byte in bits)
        bytes_per_row = columns // 8
        rowwise_exact &= all(
            sum(byte.bit_count() for byte in bits[offset : offset + bytes_per_row])
            == columns // 2
            for offset in range(0, len(bits), bytes_per_row)
        )
        total_bits += rows * columns
        pruned_bits += observed
    return {
        "matrix_count": len(mask_document["entries"]),
        "total_weights": total_bits,
        "pruned_weights": pruned_bits,
        "global_sparsity": pruned_bits / total_bits,
        "rowwise_exact_from_score_construction": rowwise_exact,
        "all_binary_mask_hashes_verified": True,
        "overall_sha256": mask_document["overall_sha256"],
    }


def paired(left: list[dict], right: list[dict], left_label: str, right_label: str) -> dict:
    if len(left) != 100 or len(right) != 100:
        raise RuntimeError("mini prediction files must each contain 100 selected records")
    counts = Counter()
    for a, b in zip(left, right):
        identity_a = (a["example_id"], a["doc_hash"], a["prompt_hash"], a["target_hash"])
        identity_b = (b["example_id"], b["doc_hash"], b["prompt_hash"], b["target_hash"])
        if identity_a != identity_b or a["evaluation_config_hash"] != b["evaluation_config_hash"]:
            raise RuntimeError("mini example/protocol identity mismatch")
        counts[(bool(a["correct"]), bool(b["correct"]))] += 1
    return {
        "both_correct": counts[(True, True)],
        f"{left_label}_only_correct": counts[(True, False)],
        f"{right_label}_only_correct": counts[(False, True)],
        "both_wrong": counts[(False, False)],
    }


def matrix_summary(rows: list[dict], pair: str) -> dict:
    selected = [row for row in rows if row["scope"] == "matrix" and row["pair"] == pair]
    if len(selected) != 224:
        raise RuntimeError(f"expected 224 matrices for {pair}, got {len(selected)}")
    top = sorted(selected, key=lambda row: row["mask_xor"], reverse=True)[:10]
    return {
        "matrix_count": len(selected),
        "score_spearman": {
            "median": statistics.median(row["spearman"] for row in selected),
            "min": min(row["spearman"] for row in selected),
        },
        "mask_xor": {
            "median": statistics.median(row["mask_xor"] for row in selected),
            "mean": statistics.fmean(row["mask_xor"] for row in selected),
            "max": max(row["mask_xor"] for row in selected),
        },
        "top_10_by_mask_xor": [
            {
                "layer": row["layer"],
                "module": row["module"],
                "mask_xor": row["mask_xor"],
                "score_spearman": row["spearman"],
                "different_decisions": row["different"],
            }
            for row in top
        ],
    }


def main() -> None:
    frozen = read_json(SOURCE / "preregistered.json")
    for section, key in (
        ("manifest", "calibration_manifest"),
        ("partition", "token_partition"),
        ("score_metadata", "historical_score_metadata"),
    ):
        source = frozen["calibration"][section]
        path = REPO / source["path"]
        if sha256(path) != source["file_sha256"]:
            raise RuntimeError(f"frozen {section} hash changed")

    partition = read_json(PARTITION)
    summary = partition["summary"]
    reveal_histogram = Counter(row["reveal_count"] for row in partition["states"])
    expected = (80, 10267, 116, 10151, {1: 44, 2: 36})
    observed = (
        summary["state_count"], summary["masked_tokens"], summary["reveal_tokens"],
        summary["remain_tokens"], dict(reveal_histogram),
    )
    if observed != expected:
        raise RuntimeError(f"frozen partition mismatch: {observed}")

    weights = read_json(SOURCE / "token_weight_summary.json")
    strong = [row["reveal50_abs"] for row in weights["states"]]
    if len(strong) != 80 or max(abs(row["normalized_mean"] - 1.0) for row in strong) > 1e-12:
        raise RuntimeError("per-state normalized token-weight mean check failed")
    weighting_stats = {
        "raw_reveal_to_remain": [50, 1],
        "normalization": "raw weights divided by their mean over masked positions in each state",
        "normalization_tolerance": 1e-12,
        "max_absolute_normalized_mean_error": max(abs(row["normalized_mean"] - 1.0) for row in strong),
        "summary_across_80_states": {
            key: stats([row[key] for row in strong])
            for key in (
                "masked_count", "reveal_count", "remain_count", "reveal_alpha",
                "remain_alpha", "reveal_normalized_weight_mass_share",
            )
        },
        "states": [
            {
                "state_index": source["state_index"],
                "masked_count": row["masked_count"],
                "reveal_count": row["reveal_count"],
                "remain_count": row["remain_count"],
                "normalized_reveal_weight": row["reveal_alpha"],
                "normalized_remain_weight": row["remain_alpha"],
                "reveal_group_total_normalized_loss_weight_share": row["reveal_normalized_weight_mass_share"],
                "normalized_mean": row["normalized_mean"],
            }
            for source, row in zip(weights["states"], strong)
        ],
    }

    score = read_json(SOURCE / "reveal50_abs_scores.json")
    if score["matrix_count"] != 224 or len(score["modules"]) != 224:
        raise RuntimeError("Reveal-50x score matrix coverage mismatch")
    numeric_fields = ("min", "max", "mean", "std")
    if any(not math.isfinite(row[key]) for row in score["modules"] for key in numeric_fields):
        raise RuntimeError("Reveal-50x score metadata contains NaN/Inf")
    if any(abs(row["actual_sparsity"] - 0.5) > 1e-6 for row in score["modules"]):
        raise RuntimeError("a Reveal-50x matrix is not exactly 50% sparse")

    mask = read_json(SOURCE / "reveal50_abs_mask.json")
    mask_verification = verify_mask(mask)
    if mask_verification["global_sparsity"] != 0.5 or not mask_verification["rowwise_exact_from_score_construction"]:
        raise RuntimeError("Reveal-50x sparsity verification failed")

    raw_diagnostics = read_json(SOURCE / "mask_diagnostics.json")
    globals_by_pair = {
        row["pair"]: row for row in raw_diagnostics["rows"] if row["scope"] == "global"
    }
    historical = read_json(SOURCE / "logs/preflight.json")["historical_reveal2_vs_uniform"]
    same_pass = globals_by_pair["reveal2_abs/uniform_abs"]
    if abs(same_pass["spearman"] - historical["spearman"]) > 1e-6 or abs(same_pass["mask_xor"] - historical["mask_xor"]) > 1e-5:
        raise RuntimeError("same-pass Reveal-2x diagnostic does not reproduce history")
    primary = globals_by_pair["reveal50_abs/uniform_abs"]
    classification = (
        "STRONG-WEIGHTING STILL MOSTLY REDUNDANT"
        if primary["mask_xor"] < 0.01 and primary["spearman"] > 0.99
        else "2x WEIGHTING WAS TOO WEAK TO TEST THE SIGNAL CLEANLY"
    )
    diagnostic_output = {
        "classification_frozen_before_gsm8k_review": classification,
        "global": globals_by_pair,
        "historical_reveal2_vs_uniform": historical,
        "historical_reproduction_absolute_delta": {
            "score_spearman": abs(same_pass["spearman"] - historical["spearman"]),
            "mask_xor": abs(same_pass["mask_xor"] - historical["mask_xor"]),
        },
        "per_matrix": {
            pair: matrix_summary(raw_diagnostics["rows"], pair)
            for pair in ("reveal50_abs/uniform_abs", "reveal50_abs/reveal2_abs")
        },
        "validation": {
            **mask_verification,
            "score_tensors_have_no_nan_or_inf": True,
            "prunable_matrix_universe": 224,
            "mask_construction": "stable exact row-wise lower 50% of score",
        },
    }

    uniform = records(UNIFORM_PREDICTIONS)
    reveal2 = records(REVEAL2_PREDICTIONS, "REVEAL-ABS")
    reveal50 = records(REVEAL50_PREDICTIONS)
    protocol_hashes = {row["evaluation_config_hash"] for rows in (uniform, reveal2, reveal50) for row in rows}
    if protocol_hashes != {frozen["evaluation"]["protocol_hash"]}:
        raise RuntimeError("GSM8K mini protocol hash mismatch")
    gsm = {
        "metric": "GSM8K strict exact match",
        "limit": 100,
        "protocol_hash": next(iter(protocol_hashes)),
        "methods": {
            "uniform_abs": {"correct": sum(row["correct"] for row in uniform), "accuracy": sum(row["correct"] for row in uniform) / 100},
            "reveal2_abs": {"correct": sum(row["correct"] for row in reveal2), "accuracy": sum(row["correct"] for row in reveal2) / 100},
            "reveal50_abs": {"correct": sum(row["correct"] for row in reveal50), "accuracy": sum(row["correct"] for row in reveal50) / 100},
        },
        "paired_correctness": {
            "reveal50_vs_uniform": paired(reveal50, uniform, "reveal50", "uniform"),
            "reveal50_vs_reveal2": paired(reveal50, reveal2, "reveal50", "reveal2"),
        },
        "prediction_artifacts": {
            "uniform_abs": {"path": str(UNIFORM_PREDICTIONS.relative_to(REPO)), "sha256": sha256(UNIFORM_PREDICTIONS)},
            "reveal2_abs": {"path": str(REVEAL2_PREDICTIONS.relative_to(REPO)), "sha256": sha256(REVEAL2_PREDICTIONS)},
            "reveal50_abs": {"path": str(REVEAL50_PREDICTIONS.relative_to(REPO)), "sha256": sha256(REVEAL50_PREDICTIONS)},
        },
        "guardrail": "N=100 is a noisy diagnostic and not final method evidence.",
    }

    packaged_config = {
        **frozen,
        "packaging": {
            "canonical_completed_run": str(SOURCE.relative_to(REPO)),
            "only_experimental_change": "raw Reveal:Remain weight 2:1 -> 50:1",
            "classification_frozen_before_gsm8k_review": classification,
        },
    }
    partition_output = {
        "state_count": summary["state_count"],
        "masked_observations": summary["masked_tokens"],
        "reveal_observations": summary["reveal_tokens"],
        "remain_observations": summary["remain_tokens"],
        "states_with_1_reveal": reveal_histogram[1],
        "states_with_2_reveal": reveal_histogram[2],
        "state_sha256": frozen["calibration"]["state_sha256"],
        "partition_sha256": partition["sha256"],
        "source_file": str(PARTITION.relative_to(REPO)),
        "source_file_sha256": sha256(PARTITION),
    }

    write_json(ROOT / "config.json", packaged_config)
    write_json(ROOT / "partition_stats.json", partition_output)
    write_json(ROOT / "weighting_stats.json", weighting_stats)
    write_json(ROOT / "mask_diagnostics.json", diagnostic_output)
    write_json(ROOT / "gsm8k_mini_results.json", gsm)
    shutil.copyfile(SOURCE / "reveal50_abs_scores.json", ROOT / "reveal50_abs_scores.json")
    shutil.copyfile(SOURCE / "reveal50_abs_mask.json", ROOT / "reveal50_abs_mask.json")
    shutil.copytree(SOURCE / "logs", ROOT / "logs", dirs_exist_ok=True)

    per_uniform = diagnostic_output["per_matrix"]["reveal50_abs/uniform_abs"]
    per_two = diagnostic_output["per_matrix"]["reveal50_abs/reveal2_abs"]
    w = weighting_stats["summary_across_80_states"]
    p50u = gsm["paired_correctness"]["reveal50_vs_uniform"]
    p502 = gsm["paired_correctness"]["reveal50_vs_reveal2"]

    def global_row(pair: str) -> str:
        row = globals_by_pair[pair]
        return (
            f"| {pair} | {row['spearman']:.9f} | {100 * row['mask_xor']:.4f}% | "
            f"{100 * row['topk_overlap']:.4f}% | {row['different']:,} |"
        )

    def per_matrix_block(label: str, value: dict) -> str:
        top = "\n".join(
            f"| {i} | {row['layer']} | {row['module']} | {100 * row['mask_xor']:.4f}% | {row['score_spearman']:.9f} |"
            for i, row in enumerate(value["top_10_by_mask_xor"], 1)
        )
        return f"""### {label}

- Score Spearman median/min: `{value['score_spearman']['median']:.9f}` / `{value['score_spearman']['min']:.9f}`
- Mask XOR median/mean/max: `{100 * value['mask_xor']['median']:.4f}%` / `{100 * value['mask_xor']['mean']:.4f}%` / `{100 * value['mask_xor']['max']:.4f}%`

| Rank | Layer | Matrix | Mask XOR | Spearman |
|---:|---:|---|---:|---:|
{top}
"""

    report = rf"""# Reveal-50x Stress Test

## Setup

The ONLY changed variable was the raw Reveal:Remain weight: **2:1 -> 50:1**. The model/revision, frozen calibration states, precomputed partition, DLM loss, ABS aggregation, 224 matrices, and exact row-wise 50% mask semantics were unchanged.

## Partition verification

- Frozen states: **80** (8 WikiText-2 sequences × 10 timesteps)
- Masked-token observations: **10,267**
- Reveal / Remain observations: **116 / 10,151**
- States with 1 / 2 Reveal tokens: **44 / 36**
- State digest: `{partition_output['state_sha256']}`
- Partition digest: `{partition_output['partition_sha256']}`

## Effective weighting

All 80 states satisfy mean normalized masked-token weight = 1 (maximum absolute error `{weighting_stats['max_absolute_normalized_mean_error']:.3g}`).

| Quantity | Min | Median | Mean | Max |
|---|---:|---:|---:|---:|
| \|M_s\| | {w['masked_count']['min']} | {w['masked_count']['median']:.1f} | {w['masked_count']['mean']:.4f} | {w['masked_count']['max']} |
| \|R_s\| | {w['reveal_count']['min']} | {w['reveal_count']['median']:.1f} | {w['reveal_count']['mean']:.4f} | {w['reveal_count']['max']} |
| \|U_s\| | {w['remain_count']['min']} | {w['remain_count']['median']:.1f} | {w['remain_count']['mean']:.4f} | {w['remain_count']['max']} |
| Normalized Reveal weight | {w['reveal_alpha']['min']:.6f} | {w['reveal_alpha']['median']:.6f} | {w['reveal_alpha']['mean']:.6f} | {w['reveal_alpha']['max']:.6f} |
| Normalized Remain weight | {w['remain_alpha']['min']:.6f} | {w['remain_alpha']['median']:.6f} | {w['remain_alpha']['mean']:.6f} | {w['remain_alpha']['max']:.6f} |
| Reveal loss-weight share | {100*w['reveal_normalized_weight_mass_share']['min']:.4f}% | {100*w['reveal_normalized_weight_mass_share']['median']:.4f}% | {100*w['reveal_normalized_weight_mass_share']['mean']:.4f}% | {100*w['reveal_normalized_weight_mass_share']['max']:.4f}% |

## Mask diagnostic

The diagnostic classification was frozen before inspecting GSM8K: **{classification}**.

| Comparison | Global score Spearman | Mask XOR | Retained-set overlap | Differing decisions |
|---|---:|---:|---:|---:|
{global_row('reveal2_abs/uniform_abs')}
{global_row('reveal50_abs/uniform_abs')}
{global_row('reveal50_abs/reveal2_abs')}

Historical 2x-vs-Uniform: Spearman `{historical['spearman']:.9f}`, mask XOR `{100*historical['mask_xor']:.4f}%`; same-pass deltas were `{diagnostic_output['historical_reproduction_absolute_delta']['score_spearman']:.3g}` and `{100*diagnostic_output['historical_reproduction_absolute_delta']['mask_xor']:.6f}` percentage points.

{per_matrix_block('Reveal-50x vs Uniform', per_uniform)}
{per_matrix_block('Reveal-50x vs Reveal-2x', per_two)}

Validation: exactly 224 intended matrices; 6,979,321,856 weights; exact global and row-wise sparsity 50%; all score summaries finite; all 224 persisted Reveal-50x packed-mask files passed size and SHA-256 readback. No retraining, compensation, Fisher term, timestep reweighting, heuristic, or allocation change was introduced.

## GSM8K mini

| Method | Correct / 100 | Accuracy |
|---|---:|---:|
| Uniform-ABS | {gsm['methods']['uniform_abs']['correct']} / 100 | {100*gsm['methods']['uniform_abs']['accuracy']:.1f}% |
| Reveal-2x-ABS | {gsm['methods']['reveal2_abs']['correct']} / 100 | {100*gsm['methods']['reveal2_abs']['accuracy']:.1f}% |
| Reveal-50x-ABS | {gsm['methods']['reveal50_abs']['correct']} / 100 | {100*gsm['methods']['reveal50_abs']['accuracy']:.1f}% |

Reveal-50x vs Uniform: both correct `{p50u['both_correct']}`, 50x only `{p50u['reveal50_only_correct']}`, Uniform only `{p50u['uniform_only_correct']}`, both wrong `{p50u['both_wrong']}`.

Reveal-50x vs Reveal-2x: both correct `{p502['both_correct']}`, 50x only `{p502['reveal50_only_correct']}`, 2x only `{p502['reveal2_only_correct']}`, both wrong `{p502['both_wrong']}`.

This fixed mini-100 result is noisy and is not final method evidence.

## Decision

FACT: Reveal-50x materially changed the pruning decision (2.5806% XOR vs Uniform) while retaining high score correlation (0.994463498). Mini accuracy tied Uniform at 58% and trailed Reveal-2x by one item.

INTERPRETATION: Mask sensitivity is established as indicated by the diagnostic, but the 100-example downstream result is too noisy for a method claim.
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")

    manifest_files = [
        "config.json", "partition_stats.json", "weighting_stats.json",
        "mask_diagnostics.json", "gsm8k_mini_results.json", "report.md",
        "reveal50_abs_scores.json", "reveal50_abs_mask.json",
    ]
    write_json(
        ROOT / "artifact_manifest.json",
        {name: {"sha256": sha256(ROOT / name), "bytes": (ROOT / name).stat().st_size} for name in manifest_files},
    )
    print(json.dumps({
        "status": "complete",
        "classification": classification,
        "reveal2_vs_uniform": same_pass,
        "reveal50_vs_uniform": primary,
        "gsm8k_mini": gsm["methods"],
        "report": str(ROOT / "report.md"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
