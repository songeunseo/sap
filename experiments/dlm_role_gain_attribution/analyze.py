"""Offline attribution boundary audit. Never changes masks or runs a model."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.dlm_capacity_predictor.audit_existing import (
    IDENTITY_FIELDS, audit_prediction_file, paired_exact_mcnemar, sha256,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/dlm_role_gain_attribution"
FULL = ROOT / "experiments/dlm_dual_role_full1319"
MINI = ROOT / "experiments/dlm_dual_role_mini100"
SEED = 20260913
RESAMPLES = 20000
STATES = ("correct", "valid_wrong", "strict_invalid")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    # Historical inputs and completed audit artifacts must not be overwritten.
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def outcome(row):
    if row["correct"]:
        return "correct"
    return "strict_invalid" if row["extracted_answer"] == "[invalid]" else "valid_wrong"


def compare(a, r):
    if not a or len(a) != len(r):
        raise ValueError("nonempty aligned predictions required")
    for left, right in zip(a, r, strict=True):
        for field in (*IDENTITY_FIELDS, "evaluation_config_hash"):
            if left[field] != right[field]:
                raise ValueError(f"pair identity mismatch: {field}")
    table = {s: {t: 0 for t in STATES} for s in STATES}
    for left, right in zip(a, r, strict=True):
        table[outcome(left)][outcome(right)] += 1
    paired = paired_exact_mcnemar([x["correct"] for x in r], [x["correct"] for x in a])
    deltas = Counter(int(y["correct"]) - int(x["correct"]) for x, y in zip(a, r))
    probs = np.array([deltas[-1], deltas[0], deltas[1]], dtype=float) / len(a)
    samples = np.random.default_rng(SEED).multinomial(len(a), probs, size=RESAMPLES)
    ci = np.quantile((samples[:, 2] - samples[:, 0]) / len(a), [.025, .975]).tolist()
    return {
        "n": len(a), "aggregate_correct": sum(x["correct"] for x in a),
        "role_correct": sum(x["correct"] for x in r), "paired_role_minus_aggregate": paired,
        "paired_accuracy_difference_bootstrap_ci": ci,
        "transition_aggregate_rows_role_columns": table,
        "net_via_strict_invalid": table["strict_invalid"]["correct"] - table["correct"]["strict_invalid"],
        "net_via_valid_wrong": table["valid_wrong"]["correct"] - table["correct"]["valid_wrong"],
    }


def balanced_bundles(deltas):
    """Repeated smallest zero-sum subset; ties canonical indices, not outcomes.

    Meet-in-the-middle enumerates at most 2**10 entries per side for this audit.
    This is a diagnostic partition, NOT a unique/optimal causal decomposition.
    """
    if sum(deltas.values()) != 0 or any(v == 0 for v in deltas.values()):
        raise ValueError("nonzero changes with exact total zero required")
    if len(deltas) > 24:
        raise ValueError("bounded offline audit supports at most 24 changes")
    remaining = sorted(deltas)
    result = []
    while remaining:
        mid = len(remaining) // 2
        halves = []
        for indices in (remaining[:mid], remaining[mid:]):
            best = defaultdict(list)
            for n in range(len(indices) + 1):
                for sub in itertools.combinations(indices, n):
                    total = sum(deltas[i] for i in sub)
                    # Retain best nonempty subset and empty separately for zero.
                    key = (total, bool(sub))
                    if not best[key]:
                        best[key] = sub
            halves.append(best)
        candidates = []
        for (value, nonempty), sub in halves[0].items():
            for other_nonempty in (False, True):
                other = halves[1].get((-value, other_nonempty))
                if other is not None and (nonempty or other_nonempty):
                    candidates.append(tuple(sub) + tuple(other))
        if not candidates:
            raise ValueError("no balanced decomposition")
        chosen = min(candidates, key=lambda x: (len(x), x))
        assert sum(deltas[i] for i in chosen) == 0
        result.append(list(chosen))
        remaining = [i for i in remaining if i not in chosen]
    return result


def allocation_audit(a, r, raw):
    if len(a["entries"]) != 224 or len(r["entries"]) != 224 or len(raw["projections"]) != 224:
        raise ValueError("224 projections required")
    rows, totals = [], {key: {m: 0.0 for m in ("masked", "unmasked", "aggregate", "max")} for key in ("aggregate", "role")}
    fingerprints = set()
    for idx, (left, right, source) in enumerate(zip(a["entries"], r["entries"], raw["projections"], strict=True)):
        if not (left["name"] == right["name"] == source["name"]):
            raise ValueError("module name mismatch")
        if any(entry["module_index"] != idx for entry in (left, right, source)):
            raise ValueError("module ordering mismatch")
        if left["shape"] != right["shape"] or left["shape"] != source["shape"]:
            raise ValueError("shape mismatch")
        fingerprints.add((source["state_digest"], source["dense_model_sha256"]))
        if sorted(s["state_index"] for s in source["states"]) != list(range(80)):
            raise ValueError("frozen 80-state IDs required")
        if source["grid"] != raw["grid"]:
            raise ValueError("grid mismatch")
        curves = defaultdict(list)
        for level in range(6):
            sums = {k: sum(float(s["levels"][level][k]) for s in source["states"])
                    for k in ("num_masked", "num_unmasked", "den_masked", "den_unmasked")}
            if min(sums["den_masked"], sums["den_unmasked"]) <= 0:
                raise ValueError("positive role denominator required")
            em = sums["num_masked"] / sums["den_masked"]
            eu = sums["num_unmasked"] / sums["den_unmasked"]
            curves["masked"].append(em)
            curves["unmasked"].append(eu)
            curves["aggregate"].append((sums["num_masked"] + sums["num_unmasked"]) / (sums["den_masked"] + sums["den_unmasked"]))
            curves["max"].append(max(em, eu))
        for method, entry in (("aggregate", left), ("role", right)):
            level = entry["level"]
            meta = entry["selected_mask"]
            if source["mask_sha256"][level] != meta["mask_sha256"]:
                raise ValueError("curve/mask identity mismatch")
            if entry["assigned_sparsity"] != raw["grid"][level]:
                raise ValueError("assigned grid mismatch")
            if entry["weights"] != entry["shape"][0] * entry["shape"][1]:
                raise ValueError("weight count mismatch")
            if meta["pruned"] != meta["prune_per_row"] * entry["shape"][0]:
                raise ValueError("row count mismatch")
            for metric in totals[method]:
                totals[method][metric] += curves[metric][level]
        delta = right["selected_mask"]["pruned"] - left["selected_mask"]["pruned"]
        if not delta:
            if left["selected_mask"]["mask_sha256"] != right["selected_mask"]["mask_sha256"]:
                raise ValueError("same count but different mask")
            continue
        la, lr = left["level"], right["level"]
        block, typ = left["name"].split(".")
        rows.append({
            "index": idx, "name": left["name"], "layer": int(block.removeprefix("block_")), "type": typ,
            "aggregate_sparsity": left["assigned_sparsity"], "role_sparsity": right["assigned_sparsity"],
            "delta_pruned": delta, "action": "prune_more" if delta > 0 else "protect",
            "proxy_at_aggregate": {m: curves[m][la] for m in curves},
            "proxy_at_role": {m: curves[m][lr] for m in curves},
            "proxy_delta": {m: curves[m][lr] - curves[m][la] for m in curves},
            "larger_normalized_role_at_aggregate": "masked" if curves["masked"][la] > curves["unmasked"][la] else "unmasked" if curves["masked"][la] < curves["unmasked"][la] else "tie",
        })
    if len(fingerprints) != 1:
        raise ValueError("calibration/model fingerprint mismatch")
    for manifest in (a, r):
        if sum(e["selected_mask"]["pruned"] for e in manifest["entries"]) != manifest["pruned"]:
            raise ValueError("manifest budget mismatch")
        if sum(e["weights"] for e in manifest["entries"]) != manifest["weights"]:
            raise ValueError("manifest parameter mismatch")
    if a["pruned"] != r["pruned"] or a["weights"] != r["weights"]:
        raise ValueError("unequal global budgets")
    deltas = {x["index"]: x["delta_pruned"] for x in rows}
    parts = balanced_bundles(deltas)
    by_type = {}
    for typ in sorted({x["type"] for x in rows}):
        selected = [x for x in rows if x["type"] == typ]
        by_type[typ] = {"changed": len(selected), "delta_pruned": sum(x["delta_pruned"] for x in selected)}
    return {"changed": rows, "by_type": by_type, "global_pruned": a["pruned"], "weights": a["weights"],
            "global_sparsity": a["pruned"] / a["weights"],
            "transferred_weights": sum(max(0, v) for v in deltas.values()),
            "mask_xor_count_assuming_historical_nesting": sum(abs(v) for v in deltas.values()),
            "diagnostic_proxy_sums_not_functional_damage": totals,
            "calibration_fingerprint": list(next(iter(fingerprints))),
            "bundles": [{"id": i, "indices": part, "names": [a["entries"][j]["name"] for j in part],
                         "delta_pruned": sum(deltas[j] for j in part)} for i, part in enumerate(parts)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "config.json").exists():
        raise RuntimeError("existing audit: choose a new --out; never overwrite")
    sources = {}
    def load(path):
        path = Path(path)
        sources[str(path)] = sha256(path)
        return read(path)
    config = load(FULL / "config.json")
    saved = load(FULL / "full1319_results.json")
    manifests = {}
    for method, meta in config["manifests"].items():
        path = ROOT / meta["path"]
        manifests[method] = load(path)
        if sha256(path) != meta["sha256"]:
            raise ValueError("frozen manifest hash mismatch")
    # Bundle construction sees no GSM8K labels.
    raw = load(ROOT / "experiments/dlm_dual_role_allocation/role_reconstruction_raw.json")
    allocation = allocation_audit(manifests["aggregate"], manifests["role"], raw)
    if allocation["global_pruned"] != config["target_pruned"] or allocation["weights"] != config["weights"]:
        raise ValueError("historical global target mismatch")
    predictions = {}
    for method in ("aggregate", "role"):
        path = FULL / f"gsm8k/{method}_1319_predictions.jsonl"
        audit = audit_prediction_file(path, 1319)
        receipt = load(path.with_suffix(".receipt.json"))
        if receipt["status"] != "complete" or receipt["predictions_sha256"] != audit["sha256"]:
            raise ValueError("prediction receipt mismatch")
        fp = receipt["fingerprint"]
        if (fp["config_sha256"] != sha256(FULL / "config.json") or
                fp["manifest_sha256"] != config["manifests"][method]["sha256"] or
                fp["protocol_sha256"] != config["protocol_hash"] or
                audit["evaluation_config_hash"] != config["protocol_hash"]):
            raise ValueError("prediction provenance mismatch")
        sources[str(path)] = audit["sha256"]
        predictions[method] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if audit["correct"] != saved["correct"][method]:
            raise ValueError("historical score mismatch")
        mini = MINI / f"gsm8k/{method}_100_predictions.jsonl"
        audit_prediction_file(mini, 100)
        sources[str(mini)] = sha256(mini)
        small = [json.loads(line) for line in mini.read_text().splitlines() if line.strip()]
        for x, y in zip(small, predictions[method][:100], strict=True):
            for field in (*IDENTITY_FIELDS, "evaluation_config_hash", "correct", "extracted_answer", "generated_text"):
                if x[field] != y[field]:
                    raise ValueError(f"mini/full mismatch: {field}")
    a, r = predictions["aggregate"], predictions["role"]
    results = {"full": compare(a, r), "mini_first100": compare(a[:100], r[:100]),
               "remaining1219_exploratory_not_new_holdout": compare(a[100:], r[100:])}
    for path in (Path(__file__), ROOT / "experiments/dlm_capacity_predictor/audit_existing.py"):
        sources[str(path)] = sha256(path)
    save(args.out / "config.json", {"started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sources,
        "bootstrap": {"seed": SEED, "resamples": RESAMPLES, "unit": "paired GSM8K example", "conditional_on": "fixed masks and fixed generation run"},
        "scope": "offline exploratory, no GPU/model invocation; output invalidity is not a reasoning label",
        "bundle_rule": "repeated minimum-cardinality nonempty zero-pruned-count subset, canonical index tie-break; no labels",
        "mask_payloads": "not reloaded; identities/counts linked to historical receipts, XOR conditional on validated historical nesting"})
    save(args.out / "allocation_audit.json", allocation)
    save(args.out / "outcome_audit.json", results)
    with (args.out / "paired_cases.jsonl").open("x") as f:
        for left, right in zip(a, r):
            f.write(json.dumps({"example_id": left["example_id"], "doc_hash": left["doc_hash"],
                "aggregate_outcome": outcome(left), "role_outcome": outcome(right),
                "aggregate_extracted_answer": left["extracted_answer"], "role_extracted_answer": right["extracted_answer"],
                "aggregate_text": left["generated_text"], "role_text": right["generated_text"],
                "reference_answer": left["reference_answer"]}, ensure_ascii=False) + "\n")
    report = render_report(allocation, results)
    with (args.out / "report.md").open("x") as f:
        f.write(report)
    for path, expected in sources.items():
        if sha256(Path(path)) != expected:
            raise ValueError("input changed during audit")
    save(args.out / "receipt.json", {"status": "complete", "source_hashes_unchanged": True,
         "finished_utc": datetime.now(timezone.utc).isoformat(),
         "outputs": {name: sha256(args.out / name) for name in ("config.json", "allocation_audit.json", "outcome_audit.json", "paired_cases.jsonl", "report.md")}})
    print(report)


def render_report(allocation, results):
    full = results["full"]
    text = ["# 기존 Role-Wanda는 왜 잘됐는가? — 저장 결과 원인 감사", "",
        "## Hypothesis / Setup", "",
        "Role 분리는 유지한다. 추가 성능 이득의 원인은 미확정이다. 기존 Aggregate/Role 65%와 동일 1319개 예측, 80-state role 통계만 재사용했다. 새 생성·GPU·mask 변경 없음.",
        "source/receipt/model config/manifest/문항·prompt·target·protocol identity, strict EM 및 mini100의 full prefix 완전 일치를 검증했다. Mask payload는 다시 읽지 않았으며 XOR는 기존 nested Wanda 검증을 전제한다.", "",
        "## Result: 정확도 및 출력 상태", "",
        "| 범위 | Aggregate | Role | 차이 | paired McNemar p |", "|---|---:|---:|---:|---:|"]
    for name, row in results.items():
        p = row["paired_role_minus_aggregate"]
        text.append(f"| {name} | {row['aggregate_correct']}/{row['n']} | {row['role_correct']}/{row['n']} | {p['net_correct_a_minus_b']:+d} | {p['exact_mcnemar_p']:.6f} |")
    text += ["", "나머지1219도 이미 관측된 결과다. mini와 full을 독립 재현 두 번으로 세지 않는다. 모든 추가 분해는 탐색적이다.",
        f"Full paired accuracy difference 95% bootstrap CI: {full['paired_accuracy_difference_bootstrap_ci']}. 고정 mask/생성 run에서 문항만 재표집한 CI이며 calibration/seed 변동은 포함하지 않는다.", "",
        "행=Aggregate, 열=Role. strict_invalid는 저장된 extractor가 `[invalid]`를 반환한 경우다. 미완성·반복·잘못된 추론으로 자동 해석하지 않는다.", "",
        "| Aggregate → Role | correct | valid_wrong | strict_invalid |", "|---|---:|---:|---:|"]
    for name, values in full["transition_aggregate_rows_role_columns"].items():
        text.append(f"| {name} | " + " | ".join(str(values[s]) for s in STATES) + " |")
    text += ["", f"정답 순증의 산술 분해: strict-invalid↔correct {full['net_via_strict_invalid']:+d}, valid-wrong↔correct {full['net_via_valid_wrong']:+d}. 이 전이 분해는 인과 mediation 분석이 아니다.", "",
        "## Result: 정확한 예산 이동", "",
        f"양 모델 pruned={allocation['global_pruned']:,}/{allocation['weights']:,} ({100*allocation['global_sparsity']:.8f}%). 변경 {len(allocation['changed'])}/224개. 추가 pruning과 복원 각각 {allocation['transferred_weights']:,} weights. Nested-mask XOR {allocation['mask_xor_count_assuming_historical_nesting']:,}.", "",
        "| Projection | Aggregate % | Role % | Δ pruned | Aggregate 지점에서 normalized error가 큰 role |", "|---|---:|---:|---:|---|"]
    for row in allocation["changed"]:
        text.append(f"| {row['name']} | {100*row['aggregate_sparsity']:.0f} | {100*row['role_sparsity']:.0f} | {row['delta_pruned']:+,} | {row['larger_normalized_role_at_aggregate']} |")
    text += ["", "타입별 양수는 추가 pruning, 음수는 보호다. 이는 전체 Role-vs-Uniform 구조가 아니라 작은 Role-vs-Aggregate 차이다.", ""]
    for typ, row in allocation["by_type"].items():
        text.append(f"- {typ}: {row['changed']}개 변경, Δpruned={row['delta_pruned']:+,}")
    text += ["", "## Interpretation: 아직 귀속할 수 없는 것", "",
        "- 20개 allocation 변경이 동시에 적용된 두 모델의 출력만으로 각 projection 효과를 식별할 수 없다. 문항이1319개여도 intervention vector는 두 개뿐이다. 같은 allocation을 문항마다 복제해 회귀해도 모듈별 인과 효과는 식별되지 않는다.",
        "- normalized error가 큰 role은 실제 기능적으로 더 중요한 role과 같지 않다. proxy 변화는 allocation 결정의 설명이지 GSM8K 개선의 원인 증명이 아니다.",
        "- 기존 causal decomposition은 28개 single projection의 DLM KL, decision audit는 Role/Exact-Max/Minimax 교환의 KL을 측정했다. 여기 필요한 Aggregate→Role의 GSM8K 개입 효과를 직접 측정하지 않았다.",
        "- parsing 전이가 정답 차이를 산술적으로 설명해도, 출력 형식만 바뀌었는지 reasoning·completion이 바뀌었는지는 별도 검증이 필요하다. 기존 strict 점수는 변경하지 않는다.", "",
        "## Next Experiment — 설계만, 미실행", "",
        f"문항 정답을 보지 않고 count/canonical order로 변경집합을 {len(allocation['bundles'])}개의 exact-budget bundle로 분해했다. 이것은 유일한 원인 분해나 새 allocation 후보가 아니라 counterfactual 측정 단위다.", ""]
    for b in allocation["bundles"]:
        text.append(f"- B{b['id']}: {', '.join(b['names'])}; Δbudget={b['delta_pruned']}")
    text += ["", "후속 승인 시 각 bundle b에 대해 Aggregate+b(충분성)와 Role−b(필요성)를 둘 다 측정한다. 둘 다 원래 exact global budget을 유지하며 local Wanda masks는 기존 두 모델에서만 가져온다. 동일 문항/seed/5-shot/256-step/strict EM 유지. 기존 mini100 전체를 먼저 사용하고 role-only 정답들만 선별하지 않는다. 결과는 원인 진단이지 후보 selection이 아니다.",
        "두 방향 효과가 다르면 context dependence/interaction의 증거다. full 효과를 주장하려면 전체1319에서도 별도 측정이 필요하다. bundle 내 개별 projection이나 role 경로의 원인으로 더 쪼개려면 후속 role-specific patching이 추가로 필요하며, 이전 KL proxy 최적화를 그대로 반복하지 않는다.", "",
        "## Decision", "", "역할 분리는 유지. 이번에는 기존 성공 결과의 설명 범위만 좁혔다. 새 pruning 방법, GPU 실행, full 평가를 시작하지 않았다.", ""]
    return "\n".join(text)


if __name__ == "__main__":
    main()
