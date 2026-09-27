#!/usr/bin/env python3
"""Read-only validation and summary of the capacity-allocation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

METHODS = ("uniform", "capacity", "reconstruction", "eis_type")
LIMITS = (100, 1319)
EXPECTED_PROJECTIONS = 224
EXPECTED_PRUNED = 4_536_008_704
EXPECTED_WEIGHTS = 6_979_321_856
IDENTITY_FIELDS = ("example_id", "doc_hash", "prompt_hash", "target_hash", "reference_answer")
STRICT_IGNORE_REGEXES = (",", r"\$", r"(?s).*#### ", r"\.$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_prediction_file(path: Path, expected_count: int) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "pending", "verified": False, "path": str(path)}
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "example_id" not in row or not isinstance(row.get("correct"), bool):
            raise ValueError(f"{path}:{line_number}: missing/invalid historical fields")
        missing = [field for field in IDENTITY_FIELDS if field not in row or row[field] is None]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing/non-null identity field: {missing}")
        if row.get("extracted_answer") is None:
            raise ValueError(f"{path}:{line_number}: missing extracted_answer")
        recomputed = strict_exact_match(row["extracted_answer"], row["reference_answer"])
        if recomputed != row["correct"]:
            raise ValueError(f"{path}:{line_number}: saved strict exact-match disagrees with recomputation")
        rows.append(row)
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate example_id")
    if ids != list(range(expected_count)):
        raise ValueError(f"{path}: example_id ordering/count mismatch")
    protocols = {row.get("evaluation_config_hash") for row in rows}
    if None in protocols or len(protocols) != 1:
        raise ValueError(f"{path}: inconsistent evaluation_config_hash")
    return {
        "status": "complete",
        "verified": True,
        "path": str(path),
        "sha256": sha256(path),
        "num_examples": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "recomputed_correct": sum(strict_exact_match(row["extracted_answer"], row["reference_answer"]) for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "evaluation_config_hash": next(iter(protocols)),
        "identities": [[row.get(field) for field in IDENTITY_FIELDS] for row in rows],
        "correctness": [row["correct"] for row in rows],
    }


def strict_exact_match(extracted_answer: str, reference_answer: str) -> bool:
    """Reproduce lm-eval 0.4.8 GSM8K exact_match,strict-match normalization."""
    prediction, reference = str(extracted_answer), str(reference_answer)
    for pattern in STRICT_IGNORE_REGEXES:
        prediction = re.sub(pattern, "", prediction)
        reference = re.sub(pattern, "", reference)
    return prediction.lower() == reference.lower()


def paired_exact_mcnemar(a: Iterable[bool], b: Iterable[bool]) -> dict[str, Any]:
    pairs = list(zip(a, b, strict=True))
    aw = sum(x and not y for x, y in pairs)
    bw = sum(not x and y for x, y in pairs)
    n = aw + bw
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(aw, bw) + 1)) / (2**n)
        p = min(1.0, 2.0 * tail)
    return {
        "both_correct": sum(x and y for x, y in pairs),
        "both_wrong": sum(not x and not y for x, y in pairs),
        "a_correct_b_wrong": aw,
        "a_wrong_b_correct": bw,
        "discordant": n,
        "net_correct_a_minus_b": aw - bw,
        "exact_mcnemar_p": p,
    }


def evaluation_limit(evaluation: dict[str, Any]) -> int:
    """Read either follow-up's top-level limit or the historical nested form."""
    if "limit" in evaluation:
        return int(evaluation["limit"])
    limits = {
        int(item["limit"])
        for key, item in evaluation.items()
        if key in METHODS and isinstance(item, dict) and "limit" in item
    }
    if len(limits) != 1:
        raise ValueError("downstream evaluation has no unambiguous limit")
    return limits.pop()


def validate_audit_invariants(audit: dict[str, Any]) -> None:
    for method, mask in audit["masks"].items():
        checks = (
            (mask["projection_count"] == EXPECTED_PROJECTIONS, "projection count"),
            (mask["module_index_order_verified"], "module ordering"),
            (mask["unique_names_verified"], "unique names"),
            (mask["target_order_and_shapes_verified"], "target ordering/shapes"),
            (mask["budget_verified"], "budget"),
            (mask["declared_totals_verified"], "declared totals"),
            (mask["config_hash_matches"], "config hash"),
        )
        for valid, label in checks:
            if not valid:
                raise ValueError(f"{method}: {label} verification failed")


def _manifest_audit(path: Path, config_path: Path, reference: list[tuple[str, list[int]]] | None) -> tuple[dict[str, Any], list[tuple[str, list[int]]]]:
    data = _read_json(path)
    entries = data.get("entries", [])
    layout = [(entry.get("name"), entry.get("shape")) for entry in entries]
    indices_ok = [entry.get("module_index") for entry in entries] == list(range(EXPECTED_PROJECTIONS))
    unique_ok = len({name for name, _ in layout}) == EXPECTED_PROJECTIONS
    layout_ok = reference is None or layout == reference
    summed_pruned = sum(entry.get("selected_mask", {}).get("pruned", 0) for entry in entries)
    summed_weights = sum(entry.get("weights", 0) for entry in entries)
    result = {
        "path": str(path), "sha256": sha256(path), "projection_count": len(entries),
        "module_index_order_verified": indices_ok, "unique_names_verified": unique_ok,
        "target_order_and_shapes_verified": layout_ok,
        "summed_pruned": summed_pruned, "summed_weights": summed_weights,
        "budget_verified": summed_pruned == EXPECTED_PRUNED and summed_weights == EXPECTED_WEIGHTS,
        "declared_totals_verified": data.get("pruned") == summed_pruned and data.get("weights") == summed_weights,
        "config_sha256_recorded": data.get("config_sha256"),
        "config_sha256_actual": sha256(config_path),
        "config_hash_matches": data.get("config_sha256") == sha256(config_path),
    }
    return result, layout


def run_audit(repo: Path) -> dict[str, Any]:
    allocation = repo / "experiments/projection_capacity_allocation_65"
    followup = repo / "experiments/projection_capacity_followup_65"
    roots = {"uniform": allocation, "capacity": allocation, "reconstruction": followup, "eis_type": followup}
    manifests = {
        "uniform": allocation / "uniform65_mask_manifest.json",
        "capacity": allocation / "capacity65_mask_manifest.json",
        "reconstruction": followup / "reconstruction65_mask_manifest.json",
        "eis_type": followup / "eis_type65_mask_manifest.json",
    }
    mask_results: dict[str, Any] = {}
    reference = None
    for method in METHODS:
        mask_results[method], layout = _manifest_audit(manifests[method], roots[method] / "config.json", reference)
        if reference is None:
            reference = layout

    predictions: dict[str, dict[str, Any]] = {method: {} for method in METHODS}
    paired: dict[str, Any] = {}
    for limit in LIMITS:
        complete = []
        reference_identities = None
        reference_protocol = None
        for method in METHODS:
            result = audit_prediction_file(roots[method] / "gsm8k" / f"{method}_{limit}_predictions.jsonl", limit)
            predictions[method][str(limit)] = result
            if result["status"] == "complete":
                if reference_identities is not None and result["identities"] != reference_identities:
                    raise ValueError(f"{method}/{limit}: prediction example identities mismatch")
                if reference_protocol is not None and result["evaluation_config_hash"] != reference_protocol:
                    raise ValueError(f"{method}/{limit}: evaluation protocol mismatch")
                reference_identities = result["identities"]
                reference_protocol = result["evaluation_config_hash"]
                complete.append(method)
        paired[str(limit)] = {}
        for i, a in enumerate(complete):
            for b in complete[i + 1:]:
                paired[str(limit)][f"{a}_vs_{b}"] = paired_exact_mcnemar(
                    predictions[a][str(limit)]["correctness"], predictions[b][str(limit)]["correctness"]
                )

    heldout = {}
    for method in METHODS:
        path = followup / f"heldout_{method}.json"
        data = _read_json(path)
        heldout[method] = {
            "status": data.get("status"), "mean_kl": data.get("summary", {}).get("mean_kl"),
            "pooled_token_mean_kl": data.get("summary", {}).get("pooled_token_mean_kl"),
            "mask_manifest_sha256": data.get("mask_manifest_sha256"),
            "manifest_hash_matches": data.get("mask_manifest_sha256") == mask_results[method]["sha256"],
        }

    allocation_downstream = _read_json(allocation / "downstream.json")
    followup_downstream = _read_json(followup / "downstream.json")
    if followup_downstream.get("protocol_hash") != allocation_downstream.get("protocol_hash"):
        raise ValueError("saved downstream protocol hashes disagree")
    expected_protocol = allocation_downstream.get("protocol_hash")
    for method in METHODS:
        for limit, result in predictions[method].items():
            if result["status"] == "complete" and result["evaluation_config_hash"] != expected_protocol:
                raise ValueError(f"{method}/{limit}: prediction protocol differs from saved downstream protocol")
    recorded = {}
    for source in (allocation_downstream, followup_downstream):
        for evaluation in source.get("evaluations", []):
            limit = str(evaluation_limit(evaluation))
            for method in METHODS:
                item = evaluation.get(method) or evaluation.get("new_methods", {}).get(method)
                if item:
                    actual = predictions[method][limit]
                    recorded[f"{method}/{limit}"] = {
                        "correct_matches": actual.get("correct") == item.get("correct"),
                        "sha256_matches": actual.get("sha256") == item.get("sha256"),
                    }
    for method in METHODS:
        for result in predictions[method].values():
            result.pop("identities", None)
            result.pop("correctness", None)
    audit = {
        "status": "complete_with_pending" if any(predictions[m]["1319"]["status"] == "pending" for m in METHODS) else "complete",
        "scope": "read-only historical artifact audit",
        "expected_budget": {"pruned": EXPECTED_PRUNED, "weights": EXPECTED_WEIGHTS},
        "masks": mask_results, "heldout": heldout, "predictions": predictions,
        "paired_exact_mcnemar": paired, "saved_downstream_cross_checks": recorded,
        "interpretation_limits": [
            "EIS+type is an oracle-derived, multiset-sorted descriptive control; it is not an official cheap EIS baseline.",
            "A worse reconstruction KL does not imply worse downstream accuracy.",
            "Pending or unavailable full-result records are not reported as verified.",
        ],
    }
    validate_audit_invariants(audit)
    if not all(item["manifest_hash_matches"] for item in heldout.values()):
        raise ValueError("heldout manifest provenance verification failed")
    return audit


def render_report(audit: dict[str, Any]) -> str:
    lines = ["# 기존 결과 감사 보고서", "", f"상태: `{audit['status']}`", "", "## 마스크와 held-out KL", "",
             "| 방법 | 224 순서/shape | 선택/전체 파라미터 | mean KL |", "|---|---:|---:|---:|"]
    for method in METHODS:
        mask, held = audit["masks"][method], audit["heldout"][method]
        valid = mask["projection_count"] == EXPECTED_PROJECTIONS and mask["module_index_order_verified"] and mask["target_order_and_shapes_verified"]
        lines.append(f"| {method} | {'검증' if valid else '실패'} | {mask['summed_pruned']:,}/{mask['summed_weights']:,} | {held['mean_kl']:.6f} |")
    lines += ["", "## GSM8K strict exact match", "", "lm-eval 0.4.8 GSM8K strict-match 정규화(쉼표/$/`#### ` 앞부분/마지막 마침표 제거, 대소문자 무시)를 `extracted_answer`와 `reference_answer`에 다시 적용했다. 재계산 EM이 저장된 `correct`와 전 행 일치하며, example identity와 evaluation config hash도 교차 검증했다.", "",
              "| 방법 | mini100 | full1319 |", "|---|---:|---:|"]
    for method in METHODS:
        cells = []
        for limit in LIMITS:
            item = audit["predictions"][method][str(limit)]
            cells.append(f"{item['correct']}/{limit}" if item["status"] == "complete" else "진행 중(pending)")
        lines.append(f"| {method} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Paired exact McNemar", ""]
    for limit, comparisons in audit["paired_exact_mcnemar"].items():
        lines.append(f"### n={limit}")
        lines.append("")
        for name, value in comparisons.items():
            lines.append(f"- {name}: discordant={value['discordant']}, p={value['exact_mcnemar_p']:.8g}")
        lines.append("")
    lines += ["## 해석 제한", "", "- EIS+type은 oracle에서 유도한 multiset-sorted 설명용 대조군이며 공식 cheap EIS baseline이 아니다.",
              "- Reconstruction의 KL이 더 나쁘다는 사실만으로 downstream 성능도 더 나쁘다고 결론낼 수 없다.",
              "- 존재하지 않거나 미완료인 full 기록은 검증되었다고 표시하지 않았다.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    audit = run_audit(args.repo.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "existing_results_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "existing_results_report.md").write_text(render_report(audit), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
