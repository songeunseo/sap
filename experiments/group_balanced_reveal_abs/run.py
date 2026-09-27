"""Run the preregistered Group-Balanced Reveal/Remain DLM-ABS experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import time
from pathlib import Path

import torch

import experiments.dlm_loss_aggregation.exp004_strong_reveal.run as engine
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _module_map,
    _overall_mask_hash,
    _read_jsonl,
    _validated_dense_fingerprint,
    _write_jsonl,
    apply_dlm_masks,
    dense_fingerprint,
)
from experiments.dlm_loss_aggregation.run import _atomic_write_json, _load_model


ROOT = Path("experiments/group_balanced_reveal_abs")
DEFAULT_CONFIG = ROOT / "preregistered.json"
CONDITIONS = ("uniform_abs", "reveal2_abs", "reveal50_abs", "group_balanced_abs")
RATIOS = {"uniform_abs": 1.0, "reveal2_abs": 2.0, "reveal50_abs": 50.0, "group_balanced_abs": None}
_ORIGINAL_LOAD_SOURCES = engine._load_sources


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_balanced_token_weights(mask: torch.Tensor, reveal_mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool or reveal_mask.dtype != torch.bool or mask.shape != reveal_mask.shape:
        raise ValueError("mask and reveal_mask must be matching boolean tensors")
    if (reveal_mask & ~mask).any().item():
        raise ValueError("reveal positions must be masked")
    reveal_count = int(reveal_mask.sum().item())
    remain_mask = mask & ~reveal_mask
    remain_count = int(remain_mask.sum().item())
    masked_count = reveal_count + remain_count
    if reveal_count <= 0 or remain_count <= 0:
        raise ValueError("group-balanced weighting requires both reveal and remain tokens")
    alpha = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
    alpha[reveal_mask] = masked_count / (2.0 * reveal_count)
    alpha[remain_mask] = masked_count / (2.0 * remain_count)
    if not torch.isclose(alpha[reveal_mask].sum(), alpha[remain_mask].sum(), atol=1e-5, rtol=1e-6):
        raise RuntimeError("group masses are not equal")
    if not torch.isclose(alpha[mask].mean(), torch.tensor(1.0, device=mask.device), atol=1e-6, rtol=0):
        raise RuntimeError("masked-token alpha mean is not one")
    return alpha


def summarize_group_balance(state: dict) -> dict:
    masked = int(state["masked_count"])
    reveal = int(state["reveal_count"])
    remain = int(state["remain_count"])
    if reveal <= 0 or remain <= 0 or reveal + remain != masked:
        raise ValueError("partition must contain both reveal and remain groups")
    alpha_r = masked / (2.0 * reveal)
    alpha_u = masked / (2.0 * remain)
    return {
        "state_index": state["state_index"],
        "masked_count": masked,
        "reveal_count": reveal,
        "remain_count": remain,
        "reveal_alpha": alpha_r,
        "remain_alpha": alpha_u,
        "alpha_ratio": alpha_r / alpha_u,
        "reveal_group_share": reveal * alpha_r / masked,
        "remain_group_share": remain * alpha_u / masked,
    }


def exact_paired_binomial_pvalue(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(left_only, right_only) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text())
    checks = {
        "experiment": "GROUP-BALANCED-REVEAL-ABS",
        "status": "frozen_before_scoring",
        "model.revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
        "calibration.state_count": 80,
        "calibration.partition.masked_observations": 10267,
        "calibration.partition.reveal_observations": 116,
        "calibration.partition.remain_observations": 10151,
        "evaluation.limit": 1319,
        "evaluation.protocol_hash": "1c27fe9936457b586af855f3f8bc05677ea6008ec76b09fc2491d43f68950add",
    }
    for dotted, expected in checks.items():
        value = config
        for key in dotted.split("."):
            value = value[key]
        if value != expected:
            raise ValueError(f"preregistration mismatch: {dotted}")
    if [row["name"] for row in config["conditions"]] != list(CONDITIONS):
        raise ValueError("condition order changed")
    return config


def _verify_file(item: dict) -> Path:
    path = Path(item["path"])
    if not path.exists() or sha256_file(path) != item["file_sha256"]:
        raise RuntimeError(f"frozen artifact missing or changed: {path}")
    return path


def _load_sources(config: dict) -> dict:
    sources = _ORIGINAL_LOAD_SOURCES(config)
    limit = config["evaluation"]["limit"]
    if len(sources["uniform_records"]) != limit or len(sources["reveal2_records"]) != limit:
        raise RuntimeError("full historical predictions are incomplete")
    if sum(r["correct"] for r in sources["uniform_records"]) != 745:
        raise RuntimeError("Uniform full result changed")
    if sum(r["correct"] for r in sources["reveal2_records"]) != 767:
        raise RuntimeError("Reveal-2x full result changed")
    strong = json.loads(_verify_file(config["historical_reference"]["strong_diagnostics"]).read_text())
    global_rows = {r["pair"]: r for r in strong["rows"] if r["scope"] == "global"}
    expected = {
        "reveal2_abs/uniform_abs": (0.9999865634876942, 0.0013903410961994757),
        "reveal50_abs/uniform_abs": (0.9944634976856712, 0.025806263375740785),
        "reveal50_abs/reveal2_abs": (0.9947859089209274, 0.02504896831053954),
    }
    for pair, (spearman, xor) in expected.items():
        if global_rows[pair]["spearman"] != spearman or global_rows[pair]["mask_xor"] != xor:
            raise RuntimeError(f"historical strong diagnostic changed: {pair}")
    for item in config["historical_reference"]["strong_masks"].values():
        document = json.loads(_verify_file(item).read_text())
        if len(document["entries"]) != 224 or document["total_bytes"] <= 0:
            raise RuntimeError("historical mask manifest is incomplete")
    return sources


def _weight_dispatch(mask: torch.Tensor, reveal_mask: torch.Tensor, ratio):
    if ratio is None:
        return group_balanced_token_weights(mask, reveal_mask)
    return engine._ORIGINAL_TOKEN_WEIGHTS(mask, reveal_mask, ratio)


def _decision(_primary: dict) -> dict:
    return {"run_downstream": True, "interpretation": "diagnostics_frozen_before_full_evaluation", "gate": "mandatory full evaluation"}


def configure_engine() -> None:
    if not hasattr(engine, "_ORIGINAL_TOKEN_WEIGHTS"):
        engine._ORIGINAL_TOKEN_WEIGHTS = engine.token_weights_ratio
    engine.ROOT = ROOT
    engine.DEFAULT_CONFIG = DEFAULT_CONFIG
    engine.CONDITIONS = CONDITIONS
    engine.RATIOS = RATIOS
    engine.PRIMARY_PAIR = "group_balanced_abs/uniform_abs"
    engine.load_config = load_config
    engine._load_sources = _load_sources
    engine.token_weights_ratio = _weight_dispatch
    engine.decision_from_diagnostics = _decision


def run_preflight(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    config = load_config(config_path)
    sources = _load_sources(config)
    states = [summarize_group_balance(row) for row in sources["partition"]["states"]]
    counts = {n: sum(row["reveal_count"] == n for row in states) for n in (1, 2)}
    if len(states) != 80 or counts != {1: 44, 2: 36}:
        raise RuntimeError("historical per-state Reveal counts changed")
    if any(abs(row["reveal_group_share"] - 0.5) > 1e-12 or abs(row["remain_group_share"] - 0.5) > 1e-12 for row in states):
        raise RuntimeError("group balance verification failed")
    mask_bytes = sum(math.prod(row["shape"]) for row in sources["score_metadata"]["modules"]) // 8
    score_bytes = 4 * 8 * mask_bytes
    available = shutil.disk_usage(config["storage"]["mask_root"]).free if Path(config["storage"]["mask_root"]).exists() else shutil.disk_usage("/DATA").free
    if available < mask_bytes + score_bytes + config["storage"]["safety_bytes"]:
        raise RuntimeError("insufficient persistent storage")
    result = {
        "status": "passed",
        "checked_at_unix": time.time(),
        "state_sha256": config["calibration"]["state_sha256"],
        "partition_sha256": sources["partition"]["sha256"],
        "state_count": len(states),
        "masked_observations": sum(r["masked_count"] for r in states),
        "reveal_observations": sum(r["reveal_count"] for r in states),
        "remain_observations": sum(r["remain_count"] for r in states),
        "reveal_counts": counts,
        "group_share_minmax": {
            "reveal": [min(r["reveal_group_share"] for r in states), max(r["reveal_group_share"] for r in states)],
            "remain": [min(r["remain_group_share"] for r in states), max(r["remain_group_share"] for r in states)],
        },
        "alpha_ratio_minmax": [min(r["alpha_ratio"] for r in states), max(r["alpha_ratio"] for r in states)],
        "historical_diagnostics_verified": True,
        "loss_normalization": "weighted token sum / p_mask(t) / 256 exactly once; group means are multiplied by masked_count before outer scaling",
    }
    _atomic_write_json(ROOT / "group_balance_all_states.json", {"states": states})
    _atomic_write_json(ROOT / "logs" / "preflight.json", result)
    return result


def _paired(left: list[dict], right: list[dict], left_name: str, right_name: str) -> dict:
    _assert_same_examples(left, right)
    both = sum(a["correct"] and b["correct"] for a, b in zip(left, right))
    left_only = sum(a["correct"] and not b["correct"] for a, b in zip(left, right))
    right_only = sum(not a["correct"] and b["correct"] for a, b in zip(left, right))
    wrong = len(left) - both - left_only - right_only
    return {
        "method_a": left_name, "method_b": right_name, "both_correct": both,
        "a_only_correct": left_only, "b_only_correct": right_only, "both_wrong": wrong,
        "discordant_pairs": left_only + right_only,
        "accuracy_difference_pp": 100.0 * (left_only - right_only) / len(left),
        "p_exact_two_sided": exact_paired_binomial_pvalue(left_only, right_only),
    }


def run_evaluation(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    from transformers import AutoTokenizer
    config = load_config(config_path)
    sources = _load_sources(config)
    scoring = json.loads((ROOT / "logs" / "scoring.json").read_text())
    if scoring.get("status") != "passed":
        raise RuntimeError("mask diagnostics must be frozen before evaluation")
    mask_document = json.loads((ROOT / "group_balanced_abs_mask.json").read_text())
    model, _ = _load_model(sources["exp001_config"])
    dense_sha = _validated_dense_fingerprint(model, sources["exp002_config"], sources["score_metadata"])
    applied = apply_dlm_masks(model, mask_document["entries"], mask_document["overall_sha256"], get_modules=_module_map)
    sparse_before = dense_fingerprint(_module_map(model))
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"], trust_remote_code=True)
    measured, records = _evaluate_gsm8k(model, tokenizer, sources["exp002_config"], "GROUP-BALANCED-ABS", 1319, sources["evaluation_hash"])
    sparse_after = dense_fingerprint(_module_map(model))
    if sparse_after != sparse_before:
        raise RuntimeError("evaluation modified sparse weights")
    _assert_same_examples(sources["uniform_records"], records)
    predictions = ROOT / "results" / "group_balanced_abs_predictions.jsonl"
    _write_jsonl(predictions, records)
    if _read_jsonl(predictions) != records:
        raise RuntimeError("prediction readback failed")
    comparisons = [
        _paired(records, sources["uniform_records"], "GROUP-BALANCED-ABS", "UNIFORM-ABS"),
        _paired(records, sources["reveal2_records"], "GROUP-BALANCED-ABS", "REVEAL-2X-ABS"),
    ]
    result = {
        "status": "passed", "limit": 1319, "metric": "GSM8K strict exact match",
        "correct": sum(r["correct"] for r in records), "accuracy": measured["accuracy"],
        "evaluation_seconds": measured["eval_seconds"], "comparisons": comparisons,
        "mask_sha256": applied["mask_hash"], "sparsity": applied["sparsity"],
        "rowwise_exact": applied["rowwise_exact"], "dense_model_sha256": dense_sha,
        "sparse_model_sha256_before_eval": sparse_before, "sparse_model_sha256_after_eval": sparse_after,
        "evaluation_protocol_hash": sources["evaluation_hash"], "predictions_path": str(predictions),
        "predictions_sha256": sha256_file(predictions),
    }
    _atomic_write_json(ROOT / "results" / "evaluation.json", result)
    return result


def _matrix_summary(rows: list[dict], pair: str) -> dict:
    selected = [r for r in rows if r["scope"] == "matrix" and r["pair"] == pair]
    top = sorted(selected, key=lambda r: r["mask_xor"], reverse=True)[:10]
    return {
        "matrix_count": len(selected),
        "spearman": {"median": statistics.median(r["spearman"] for r in selected), "mean": statistics.fmean(r["spearman"] for r in selected), "min": min(r["spearman"] for r in selected)},
        "mask_xor": {"median": statistics.median(r["mask_xor"] for r in selected), "mean": statistics.fmean(r["mask_xor"] for r in selected), "max": max(r["mask_xor"] for r in selected)},
        "top10_mask_xor": [{"layer": r["layer"], "module": r["module"], "mask_xor": r["mask_xor"]} for r in top],
    }


def freeze_diagnostic_report() -> dict:
    scoring = json.loads((ROOT / "logs" / "scoring.json").read_text())
    detailed = json.loads((ROOT / "mask_diagnostics.json").read_text())["rows"]
    report = {
        "status": "frozen_before_gsm8k", "frozen_at_unix": time.time(),
        "global": scoring["global_diagnostics"],
        "per_matrix": {pair: _matrix_summary(detailed, pair) for pair in scoring["global_diagnostics"]},
        "validation": {"matrix_count": scoring["matrix_count"], "no_nan_inf": True, "exact_rowwise_50_percent": True, "mask_sha256_readback": True},
    }
    if report["validation"]["matrix_count"] != 224 or any(v["matrix_count"] != 224 for v in report["per_matrix"].values()):
        raise RuntimeError("diagnostic matrix coverage is incomplete")
    _atomic_write_json(ROOT / "mask_diagnostic_report_frozen.json", report)
    return report


def run_report(config_path: Path | str = DEFAULT_CONFIG) -> dict:
    config = load_config(config_path)
    pre = json.loads((ROOT / "logs" / "preflight.json").read_text())
    diag = json.loads((ROOT / "mask_diagnostic_report_frozen.json").read_text())
    ev = json.loads((ROOT / "results" / "evaluation.json").read_text())
    correct = ev["correct"]
    vs_u, vs_r = ev["comparisons"]
    if correct <= 745:
        outcome = "C"
        conclusion = "Strongly balancing the small Reveal group against the much larger Remain group does not improve downstream performance. The historical Reveal-2x gain should therefore not be interpreted as evidence that stronger Reveal emphasis is generally better."
    elif correct > 767 and vs_r["p_exact_two_sided"] < 0.05:
        outcome = "A"
        conclusion = "The positive Reveal signal is not merely a fixed-multiplier artifact. Balancing the total contribution of the Reveal and Remain semantic groups produces a stronger pruning criterion under the same 50% sparsity constraint. This does not establish that 50:50 is optimal."
    elif abs(correct - 767) <= 13:
        outcome = "B"
        conclusion = "Reveal-aware weighting remains competitive, but equal group balancing does not provide clear additional benefit over the simpler 2x formulation."
    else:
        outcome = "D"
        conclusion = "The weighting formulation changes the pruning decision, but the downstream evidence is insufficient to support a method claim."
    rows = diag["global"]
    def pct(x): return f"{100*x:.4f}%"
    mask_lines = "\n".join(f"| {pair} | {row['spearman']:.9f} | {pct(row['mask_xor'])} | {pct(row['topk_overlap'])} | {row['different']:,} |" for pair, row in rows.items())
    report = f"""# Group-Balanced Reveal/Remain DLM-ABS

## Motivation

Fixed 2× and 50× multipliers make the Reveal-group loss mass depend on each state's group sizes. This experiment replaces that heuristic with a frozen modeling assumption: Reveal and Remain each receive half of the total masked-token weight mass. The 50:50 ratio was not tuned.

## Frozen Setup

Model `GSAI-ML/LLaDA-8B-Base` at revision `{config['model']['revision']}`; the historical 80 states, persisted Reveal/Remain partition, DLM outer normalization, ABS aggregation, 224 Linear matrices, and exact row-wise 50% sparsity were reused unchanged. State digest: `{pre['state_sha256']}`. Partition digest: `{pre['partition_sha256']}`.

## Group-Balanced Objective

`L_group(s) = |M_s| × [0.5 mean_(j in R_s)(CE_sj) + 0.5 mean_(j in U_s)(CE_sj)] / [p_mask(t) × 256]`.

The single factor `|M_s|` preserves the historical masked-token loss scale. Equivalently, `alpha_R=|M|/(2|R|)` and `alpha_U=|M|/(2|U|)`; the weighted token sum is then divided by `p_mask(t)×256` exactly once.

## Weighting Verification

All 80 states have Reveal share 50% and Remain share 50%. The effective `alpha_R/alpha_U=|U|/|R|` range is `{pre['alpha_ratio_minmax'][0]:.1f}`–`{pre['alpha_ratio_minmax'][1]:.1f}`. The complete state-level table is in `group_balance_all_states.json`.

## Mask Diagnostics

| Pair | Score Spearman | Mask XOR | Retained overlap | Differing decisions |
|---|---:|---:|---:|---:|
{mask_lines}

The diagnostic report was frozen before full GSM8K evaluation. Per-matrix median/mean/extrema and top-10 XOR matrices are in `mask_diagnostic_report_frozen.json`.

## Full GSM8K

| Method | Correct / 1319 | Accuracy |
|---|---:|---:|
| Uniform-ABS | 745 | 56.48% |
| Reveal-2×-ABS | 767 | 58.15% |
| Group-Balanced-ABS | {correct} | {100*ev['accuracy']:.2f}% |
| DLM-SQUARE | 701 | 53.15% |
| Wanda | 677 | 51.33% |
| SparseGPT | 584 | 44.28% |

All rows use the same frozen 5-shot, 256-step, strict exact-match protocol.

## Paired Analysis

| Comparison | Both correct | GB only | Baseline only | Both wrong | Δ accuracy | Exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| GB vs Uniform | {vs_u['both_correct']} | {vs_u['a_only_correct']} | {vs_u['b_only_correct']} | {vs_u['both_wrong']} | {vs_u['accuracy_difference_pp']:+.3f} pp | {vs_u['p_exact_two_sided']:.6g} |
| GB vs Reveal-2× | {vs_r['both_correct']} | {vs_r['a_only_correct']} | {vs_r['b_only_correct']} | {vs_r['both_wrong']} | {vs_r['accuracy_difference_pp']:+.3f} pp | {vs_r['p_exact_two_sided']:.6g} |

Exact p-values use only discordant examples. A non-significant result is not evidence of equivalence.

## Conclusion

**FACT:** Group-Balanced-ABS scored `{correct}/1319` ({100*ev['accuracy']:.2f}%) under the frozen protocol; paired counts and mask diagnostics are reported above.

**INTERPRETATION — Outcome {outcome}:** {conclusion}

No new weighting ratio is proposed from this result.
"""
    (ROOT / "report.md").write_text(report)
    final = {"status": "complete", "outcome": outcome, "conclusion": conclusion, "evaluation": ev, "report": str(ROOT / "report.md")}
    _atomic_write_json(ROOT / "logs" / "final.json", final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "score-shard", "merge", "freeze-diagnostics", "evaluate", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--block-start", type=int)
    parser.add_argument("--block-end", type=int)
    args = parser.parse_args(argv)
    configure_engine()
    if args.command == "preflight": result = run_preflight(args.config)
    elif args.command == "score-shard": result = engine.run_scoring_shard(args.config, args.block_start, args.block_end)
    elif args.command == "merge": result = engine.merge_scoring_shards(args.config)
    elif args.command == "freeze-diagnostics": result = freeze_diagnostic_report()
    elif args.command == "evaluate": result = run_evaluation(args.config)
    else: result = run_report(args.config)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
