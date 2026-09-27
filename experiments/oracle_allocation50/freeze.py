"""Freeze the 50% re-validation before constructing any new mask."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import torch

from experiments.oracle_allocation50.core import (
    GLOBAL_SPARSITY,
    MIDDLE_SPARSITY,
    MILD_RELATIVE_PROTECTION,
    PROTECTED_SPARSITY,
    RANDOM_SEEDS,
    build_plans,
)
from experiments.oracle_allocation75.core import load_module_records, sha256_file
from experiments.oracle_allocation75.freeze import (
    DLM_STATS,
    FAILURE,
    FAILURE_MANIFEST,
    MODEL_ID,
    MODEL_REVISION,
    STATS,
    write_json,
)


ROOT = Path(__file__).parent
STORE = Path("/DATA/tmluser1/sap-oracle-allocation50")
SELECTIONS_75 = Path("experiments/oracle_allocation75/allocation_selections.json")
PREREG_75 = Path("experiments/oracle_allocation75/preregistered.json")
HELDOUT = Path("experiments/wanda_failure_characterization/heldout_state_manifest.json")


def main():
    prereg = ROOT / "preregistered.json"
    if prereg.exists():
        print("Frozen 50% re-validation already exists; refusing to resample.")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    previous = json.loads(PREREG_75.read_text())
    selections = json.loads(SELECTIONS_75.read_text())
    records = load_module_records(STATS)
    plans = build_plans(records, selections)
    failure_manifest = json.loads(FAILURE_MANIFEST.read_text())
    heldout = json.loads(HELDOUT.read_text())
    dlm = torch.load(DLM_STATS, map_location="cpu", weights_only=False)
    if sha256_file(FAILURE) != failure_manifest["failure_map_sha256"]:
        raise RuntimeError("failure-map hash mismatch")
    if heldout["historical_state_sha256"] != failure_manifest["heldout_sha"]:
        raise RuntimeError("held-out state hash mismatch")
    if dlm["source_state_sha256"] != previous["dlm_calibration"]["state_sha256"]:
        raise RuntimeError("DLM calibration state mismatch")
    write_json(ROOT / "allocation_plans.json", plans)
    plan_summary = {}
    for name, plan in plans.items():
        deviations = [row["sparsity_deviation_from_uniform"] for row in plan["entries"]]
        plan_summary[name] = {
            "protected_sparsity": plan["protected_sparsity"],
            "middle_sparsity": plan["middle_sparsity"],
            "donor_requested_sparsity": plan["donor_requested_sparsity"],
            "global_actual_sparsity": plan["global_actual_sparsity"],
            "minimum_projection_deviation": min(deviations),
            "maximum_projection_deviation": max(deviations),
            "protected_groups": len(plan["protected_groups"]),
            "donor_groups": len(plan["donor_groups"]),
        }
    write_json(ROOT / "budget_summary.json", plan_summary)
    document = {
        "status": "frozen_before_any_50pct_mask_or_outcome",
        "research_question": "Does causal functional-KL allocation become useful when the global budget is relaxed from 75% to 50% sparsity?",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "global_sparsity": GLOBAL_SPARSITY,
        "pruning_pattern": "per-projection unstructured row-wise Wanda",
        "score": previous["score"],
        "dlm_calibration": previous["dlm_calibration"],
        "heldout": {
            "path": str(HELDOUT),
            "sha256": sha256_file(HELDOUT),
            "state_sha256": heldout["historical_state_sha256"],
            "state_count": 40,
        },
        "causal_map": {
            "path": str(STATS),
            "sha256": sha256_file(STATS),
            "field": "Y50_KL",
            "failure_map_sha256": failure_manifest["failure_map_sha256"],
        },
        "selection_reuse": {
            "path": str(SELECTIONS_75),
            "sha256": sha256_file(SELECTIONS_75),
            "note": "Exact 75% functional-KL protected/donor groups are reused. Random groups use the same fixed seeds and stratified sampler, matched to the functional oracle's projection-type/parameter-count composition; 50% masks are newly generated.",
        },
        "relative_protection": MILD_RELATIVE_PROTECTION,
        "protected_sparsity": PROTECTED_SPARSITY,
        "middle_sparsity": MIDDLE_SPARSITY,
        "random_seeds": list(RANDOM_SEEDS),
        "variants": list(plans),
        "metrics": ["mean masked-token KL", "P90 masked-token KL", "P95 masked-token KL", "DLM NLL gap", "top-1 agreement"],
        "no_downstream": True,
        "no_medium_or_aggressive_auto_followup": True,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "storage": str(STORE),
    }
    write_json(prereg, document)
    print(json.dumps({"status": "frozen", "variants": len(plans), "protected_sparsity": PROTECTED_SPARSITY}, indent=2))


if __name__ == "__main__":
    main()
