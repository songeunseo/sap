"""Freeze the oracle-allocation design before constructing any 75% mask."""

import csv
import hashlib
import json
import platform
from pathlib import Path

import torch

from experiments.oracle_allocation75.core import (
    FALLBACK_SCHEDULES,
    GLOBAL_SPARSITY,
    MIDDLE_SPARSITY,
    PROTECTED_COUNT,
    PROTECTED_FRACTION,
    PUBLIC_ALIASES,
    RANDOM_SEEDS,
    REQUESTED_SCHEDULES,
    build_all_plans,
    feasibility_audit,
    load_module_records,
    module_rank,
    sha256_file,
)


ROOT = Path(__file__).parent
STORE = Path("/DATA/tmluser1/sap-oracle-allocation75")
STATS = Path("experiments/wanda_failure_characterization/per_module_statistics.csv")
FAILURE = Path("experiments/wanda_failure_characterization/failure_map.pt")
FAILURE_MANIFEST = Path("experiments/wanda_failure_characterization/failure_map_manifest.json")
DLM_STATS = Path("experiments/cgq_wanda_structured_diagnostic/sufficient_statistics.pt")
MODEL_ID = "GSAI-ML/LLaDA-8B-Base"
MODEL_REVISION = "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap_svg(plan, output):
    types = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]
    values = {(row["layer"], row["module_type"]): row["actual_sparsity"] - 0.75 for row in plan["entries"]}
    cell, left, top = 26, 92, 42
    width, height = left + 32 * cell + 20, top + len(types) * cell + 52
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,sans-serif;fill:#222}.lab{font-size:11px}.title{font-size:15px;font-weight:600}</style>',
        f'<text x="{left}" y="20" class="title">{plan["variant"]} / {plan["schedule"]}: Δ sparsity from 0.75</text>',
    ]
    for layer in range(32):
        if layer % 2 == 0:
            parts.append(f'<text x="{left + layer * cell + cell / 2}" y="{top - 8}" text-anchor="middle" class="lab">{layer}</text>')
    for row_index, module_type in enumerate(types):
        y = top + row_index * cell
        parts.append(f'<text x="{left - 8}" y="{y + 17}" text-anchor="end" class="lab">{module_type}</text>')
        for layer in range(32):
            value = values[(layer, module_type)]
            scale = min(abs(value) / 0.25, 1.0)
            if value < 0:
                rgb = (int(245 - 145 * scale), int(248 - 88 * scale), int(255 - 25 * scale))
            elif value > 0:
                rgb = (int(255 - 25 * scale), int(248 - 105 * scale), int(240 - 125 * scale))
            else:
                rgb = (238, 238, 238)
            parts.append(
                f'<rect x="{left + layer * cell}" y="{y}" width="{cell - 1}" height="{cell - 1}" '
                f'fill="rgb{rgb}"><title>block {layer} {module_type}: {value:+.6f}</title></rect>'
            )
    parts.extend([
        f'<text x="{left}" y="{height - 18}" class="lab">blue: protected (lower sparsity) · orange: donor (higher sparsity) · gray: 0.75</text>',
        '</svg>',
    ])
    Path(output).write_text("\n".join(parts))


def main():
    prereg = ROOT / "preregistered.json"
    if prereg.exists():
        print("Frozen oracle-allocation preregistration already exists; refusing to resample.")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    records = load_module_records(STATS)
    selections, plans = build_all_plans(records)
    dlm = torch.load(DLM_STATS, map_location="cpu", weights_only=False)
    if len(dlm["statistics"]) != 224:
        raise RuntimeError("DLM sufficient statistics do not cover 224 modules")
    source_digest = dlm["source_state_sha256"]
    failure_manifest = json.loads(FAILURE_MANIFEST.read_text())
    if failure_manifest["failure_map_sha256"] != sha256_file(FAILURE):
        raise RuntimeError("failure-map hash mismatch")
    if failure_manifest["module_count"] != 224:
        raise RuntimeError("failure-map module count mismatch")
    audit = feasibility_audit(records, "functional_kl") + feasibility_audit(records, "reconstruction_error")
    write_json(ROOT / "feasibility_audit.json", {"rows": audit})
    write_json(ROOT / "allocation_selections.json", selections)
    write_json(ROOT / "allocation_plans.json", plans)
    heatmap_dir = ROOT / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    for identifier, plan in plans.items():
        if identifier != "V0_uniform":
            heatmap_svg(plan, heatmap_dir / f"{identifier}.svg")
    rank_rows = []
    for identifier, plan in plans.items():
        by_name = {row["module"]: row for row in plan["entries"]}
        for name in ("block_30.ff_out", "block_31.ff_out"):
            row = by_name[name]
            rank_rows.append(
                {
                    "plan": identifier,
                    "module": name,
                    "causal_kl_rank_high_to_low": module_rank(records, "functional_kl", name),
                    "reconstruction_rank_high_to_low": module_rank(records, "reconstruction_error", name),
                    "allocation_status": row["status"],
                    "requested_sparsity": row["requested_sparsity"],
                    "actual_sparsity": row["actual_sparsity"],
                    "retained_parameters": row["retained_parameters"],
                }
            )
    write_csv(ROOT / "terminal_module_allocation.csv", rank_rows)
    document = {
        "status": "frozen_before_any_75pct_mask_or_outcome",
        "research_question": "Does causal functional-KL projection allocation improve 75%-sparse DLM-Wanda downstream behavior?",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "global_sparsity": GLOBAL_SPARSITY,
        "pruning_pattern": "unstructured, exact integer row-wise Wanda ranking",
        "score": "abs(W_ij) * sqrt(A_DLM_j); score unchanged across all variants",
        "dlm_calibration": {
            "source": str(DLM_STATS),
            "source_sha256": sha256_file(DLM_STATS),
            "state_sha256": source_digest,
            "states": 80,
            "semantics": "overall_uniform: mean over 80 corrupted batch-1 states of sum over all 256 token positions X^2",
        },
        "vulnerability": {
            "source": str(STATS),
            "source_sha256": sha256_file(STATS),
            "functional_field": "Y50_KL",
            "reconstruction_negative_control_field": "Erec50",
            "heldout_state_sha256": failure_manifest["heldout_sha"],
            "failure_map_sha256": failure_manifest["failure_map_sha256"],
        },
        "architecture_aliases": PUBLIC_ALIASES,
        "protected_fraction": PROTECTED_FRACTION,
        "protected_group_count": PROTECTED_COUNT,
        "middle_sparsity": MIDDLE_SPARSITY,
        "requested_schedules": REQUESTED_SCHEDULES,
        "fallback_schedules": FALLBACK_SCHEDULES,
        "fallback_reason": "Requested functional-KL schedules require donor sparsity >1.0 at q=10%,15%,20%; protection strength reduced while retaining q=10% and monotonic schedule order.",
        "random_seeds": list(RANDOM_SEEDS),
        "variant_count_stage1": len(plans),
        "stage1": {
            "states": str(Path("experiments/wanda_failure_characterization/heldout_state_manifest.json")),
            "state_count": 40,
            "metrics": ["masked-token KL mean", "masked-token KL p90", "official DLM NLL gap", "timestep KL"],
            "selection": "V0; lowest-mean-KL random; lowest-mean-KL V2; two lowest-mean-KL V3 schedules",
        },
        "stage2": {
            "gsm8k_limit": 100,
            "protocol": "exact EXP-002 GSM8K protocol with only limit=100",
            "no_repeated_tuning": True,
        },
        "stage3_gate": {
            "mini_correct_improvement_min": 2,
            "mean_kl_not_worse": True,
            "p90_kl_not_worse": True,
        },
        "no_new_score_or_recovery": True,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "storage": str(STORE),
    }
    write_json(prereg, document)
    print(json.dumps({"status": "frozen", "plans": len(plans), "state_sha": source_digest}, indent=2))


if __name__ == "__main__":
    main()
