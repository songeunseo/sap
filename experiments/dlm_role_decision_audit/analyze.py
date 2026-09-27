#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_dual_role_allocation.io import atomic_write_json
from experiments.dlm_role_decision_audit.core import exact_sign_flip, holm_adjust, sequence_bootstrap

ROOT = Path("experiments/dlm_role_decision_audit")
RUNTIME = Path("/DATA/tmluser1/sap-dlm-role-decision-audit")


def summarize_delta(rows, baseline, seed):
    rows = sorted(rows, key=lambda x: x["state_index"])
    baseline = sorted(baseline, key=lambda x: x["state_index"])
    delta = np.asarray([x["mean_kl"] - y["mean_kl"] for x, y in zip(rows, baseline)])
    seq = np.asarray([x["sequence_index"] for x in rows])
    time = np.asarray([x["timestep"] for x in rows])
    boot = sequence_bootstrap(delta, seq, seed=seed)
    seq_means = np.asarray(boot["sequence_means"])
    boot.update(median=float(np.median(delta)), improved_states=int((delta < 0).sum()),
                worsened_states=int((delta > 0).sum()), exact_sequence_sign_flip_p=exact_sign_flip(seq_means),
                timestep_means={str(t): float(delta[time == t].mean()) for t in sorted(set(time))},
                state_deltas=delta.tolist())
    return boot


def main():
    bundles = json.loads((ROOT / "bundles.json").read_text())
    analyses, p_refs = {}, []
    for bg_index, background in enumerate(("dense", "role_sparse")):
        path = RUNTIME / f"bundle_{background}.pt"
        if not path.exists(): raise RuntimeError(f"missing {path}")
        rows = torch.load(path, map_location="cpu", weights_only=False)["rows"]
        analyses[background] = {}
        for bi, bundle in enumerate(bundles["bundles"]):
            bid = bundle["bundle_id"]
            group = [x for x in rows if x["bundle_id"] == bid]
            by = defaultdict(list)
            for row in group: by[row["condition"]].append(row)
            expected = 10 if bundle["size"] == 4 else 8
            if len(group) != 40 * expected: raise RuntimeError(f"incomplete {background}/{bid}: {len(group)}")
            baseline = by["baseline"]
            condition = {name: summarize_delta(value, baseline, 1234 + bg_index * 100 + bi)
                         for name, value in by.items() if name != "baseline"}
            bundle_delta = np.asarray(condition["bundle_direct"]["state_deltas"])
            singles = [np.asarray(condition[f"single_{j}"]["state_deltas"]) for j in range(bundle["size"])]
            interaction = bundle_delta - np.sum(singles, axis=0)
            seq = np.asarray([x["sequence_index"] for x in sorted(baseline, key=lambda x:x["state_index"])])
            interaction_stats = sequence_bootstrap(interaction, seq, seed=4321 + bi)
            interaction_stats["exact_sequence_sign_flip_p"] = exact_sign_flip(np.asarray(interaction_stats["sequence_means"]))
            analyses[background][bid] = {"metadata": bundle, "conditions": condition,
                                         "nonadditive_interaction": interaction_stats,
                                         "mixed_direct_max_abs_state_delta": float(np.max(np.abs(
                                             np.asarray(condition["bundle_mixed"]["state_deltas"]) - bundle_delta))),
                                         "sham_max_abs_state_delta": float(np.max(np.abs(condition["sham"]["state_deltas"]))) }
            p_refs.append((background, bid, condition["bundle_direct"]["exact_sequence_sign_flip_p"]))
    adjusted = holm_adjust([x[2] for x in p_refs])
    for (background, bid, _), p in zip(p_refs, adjusted):
        analyses[background][bid]["conditions"]["bundle_direct"]["holm_p_12_primary_tests"] = p

    full_path = RUNTIME / "full_allocations.pt"
    full_rows = torch.load(full_path, map_location="cpu", weights_only=False)["rows"]
    full = {}
    grouped = defaultdict(list)
    for row in full_rows: grouped[row["method"]].append(row)
    for method in ("A", "B", "C"):
        rows = sorted(grouped[method], key=lambda x:x["state_index"])
        if len(rows) != 40: raise RuntimeError(f"incomplete full {method}")
        full[method] = {k: float(np.mean([r[k] for r in rows])) for k in
                        ("mean_kl", "median_kl", "delta_loss", "top1_agreement", "confidence_mae")}
    for method in ("B", "C"):
        full[f"{method}_minus_A"] = summarize_delta(grouped[method], grouped["A"], 9000 + ord(method))

    result = {"status":"complete", "primary_family":"12 bundle-direct tests (6 bundles x 2 backgrounds)",
              "bundle_results":analyses, "full_allocations":full}
    atomic_write_json(ROOT / "analysis.json", result)

    lines = ["# Role Allocation Decision Audit", "", "## 결과", "",
             "이 문서는 downstream 정답률을 사용하지 않고, frozen bundle 교환의 held-out DLM 인과 효과를 측정한다.", "",
             "### Full allocation held-out DLM", "", "| 방법 | Mean KL | Median KL | Top-1 |", "|---|---:|---:|---:|"]
    for method in ("A","B","C"):
        x=full[method]; lines.append(f"| {method} | {x['mean_kl']:.6f} | {x['median_kl']:.6f} | {x['top1_agreement']:.4f} |")
    lines += ["", "### Frozen bundle 교환", "", "| 배경 | Bundle | ΔKL | 95% CI | Holm p | interaction |", "|---|---|---:|---:|---:|---:|"]
    for bg in ("dense","role_sparse"):
        for bid, item in analyses[bg].items():
            x=item["conditions"]["bundle_direct"]; inter=item["nonadditive_interaction"]
            lines.append(f"| {bg} | {bid} | {x['mean']:.6g} | [{x['bootstrap_95_ci'][0]:.6g}, {x['bootstrap_95_ci'][1]:.6g}] | {x['holm_p_12_primary_tests']:.4g} | {inter['mean']:.6g} |")
    lines += ["", "## 해석 원칙", "", "- ΔKL<0이면 A에서 해당 B/C bundle로 바꾼 것이 개선이다.",
              "- dense와 Role-sparse 배경 차이는 composition/context dependence를 뜻한다.",
              "- interaction이 0에서 벗어나면 single-projection damage의 단순 합으로 bundle 효과를 설명할 수 없다.",
              "- 이 감사는 allocation을 새로 고르지 않으며 GSM8K를 사용하지 않는다."]
    (ROOT / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"event":"analysis_complete", "full":full}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
