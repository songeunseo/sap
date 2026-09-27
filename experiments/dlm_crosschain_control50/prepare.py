"""Freeze source identities, full requests, held-out bank, and four allocations on CPU."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from experiments.dlm_multiscale_ac50.artifacts import REPO, DEFAULT_ROOT as OLD, checked, freeze, read, sha, write
from experiments.dlm_multiscale_ac50.core import make_bank, rank_rates
from experiments.dlm_multiscale_ac50.evaluation import freeze_requests
from experiments.dlm_owl65.core import exact_row_counts
from .core import ARMS, graph_invariants, response_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE / "output"
SOURCE = REPO / "experiments/projection_capacity_followup_65/new_heldout_state_manifest.json"
VERIFICATION = REPO / "experiments/projection_capacity_followup_65/new_state_verification.json"
PRIOR_VALIDATION = REPO / "experiments/dlm_multi_validation10050/output"
BETA_EXPECTED = 0.932319907797497
FULL_IDS = tuple(range(1319))
EXPOSED = tuple(sorted(set(range(100)) | {r["example_id"] for r in read(PRIOR_VALIDATION / "requests.json")["development"]}))


def requests_check(rows):
    if [r["example_id"] for r in rows] != list(FULL_IDS):
        raise RuntimeError("Full GSM8K requests not in original order")
    old = read(OLD / "requests.json")
    prior = read(PRIOR_VALIDATION / "requests.json")
    reference = old["development"] + prior["development"]
    if len(EXPOSED) != 200 or {r["example_id"] for r in reference} != set(EXPOSED):
        raise RuntimeError("Prior 200-ID split is not exact")
    keys = ("doc_hash", "prompt_hash", "target_hash", "input_ids_sha256", "reference_answer")
    for row in reference:
        fresh = rows[row["example_id"]]
        if any(fresh[k] != row[k] for k in keys):
            raise RuntimeError(f"Previously seen request changed: {row['example_id']}")
    return {"exposed_ids": list(EXPOSED), "primary_ids": sorted(set(FULL_IDS) - set(EXPOSED)),
            "prior_request_sha256": {str(OLD / "requests.json"): sha(OLD / "requests.json"),
                                     str(PRIOR_VALIDATION / "requests.json"): sha(PRIOR_VALIDATION / "requests.json")}}


def source_bank():
    verification = read(VERIFICATION)
    if (sha(SOURCE) != verification["manifest_sha256"] or verification["all_three_splits_disjoint"] is not True
            or verification["status"] != "verified" or verification["overlaps"]):
        raise RuntimeError("Held-out source identity/disjointness changed")
    old_plan = read(OLD / "config.json")["plan"]
    expected_prior = {str(Path(row["path"]).relative_to(REPO)): row["sha256"]
                      for key,row in old_plan["sources"].items() if key in ("calibration_source", "diagnostic_source")}
    if verification["prior_manifest_sha256"] != expected_prior:
        raise RuntimeError("Held-out verification references different original sources")
    source = read(SOURCE)
    if source["model"] != old_plan["model"] or source["mask_id"] != read(OLD / "bank_calibration.json")["mask_id"]:
        raise RuntimeError("Held-out source model/mask changed")
    if sorted({s["sequence_index"] for s in source["states"]}) != list(range(16, 24)):
        raise RuntimeError("Held-out span indices changed")
    calib = read(OLD / "bank_calibration.json")
    diagnostic = read(OLD / "bank_diagnostic.json")
    settings = copy.deepcopy(calib["settings"])
    settings["diagnostic_seed"] = 20260927
    fresh = make_bank(source, settings, "diagnostic")
    from experiments.dlm_multiscale_ac50.core import clean_sequences
    clean_hashes = lambda bank: {tuple(c["gold"]) for c in bank["chains"]}
    if clean_hashes(fresh) & (clean_hashes(calib) | clean_hashes(diagnostic)):
        raise RuntimeError("Held-out bank gold/query collisions")
    if fresh["states"] != 128 or [c["sequence_index"] for c in fresh["chains"][::2]] != list(range(16, 24)):
        raise RuntimeError("Fresh bank cardinality differs")
    freeze(ROOT / "bank_fresh.json", fresh)
    return fresh


def verify_old_readout(source_path, split, identity):
    row = read(source_path)
    expected = {"config_sha256": sha(OLD / "config.json"),
                "bank_sha256": sha(OLD / f"bank_{split}.json"),
                "mask_identity": identity}
    if row["fingerprint"] != expected or row["complete"] is not True or len(row["values"]) != 128:
        raise RuntimeError(f"Old readout fingerprint/shape differs: {source_path}")
    return row


def reused_readout(source_path, target_path, config_sha, bank_sha, identity):
    split = source_path.stem
    row = verify_old_readout(source_path, split, identity)
    if not row["complete"] or len(row["values"]) != 128:
        raise RuntimeError(f"Incomplete reused readout: {source_path}")
    target = {"fingerprint": {"config_sha256": config_sha, "mask_identity": identity,
                              "bank_sha256": bank_sha}, "values": row["values"], "complete": True,
              "reused_source_sha256": sha(source_path)}
    write(target_path, target)


def allocate(config):
    old = read(OLD / "allocation.json")
    old_config_sha = sha(OLD / "config.json")
    if old["config_sha256"] != old_config_sha:
        raise RuntimeError("Old allocation source changed")
    bank = read(OLD / "bank_calibration.json")
    from experiments.projection_capacity_allocation_65.run import DENSE_SHA
    teacher = verify_old_readout(OLD / "readouts/dense/calibration.json", "calibration", "dense:"+DENSE_SHA)["values"]
    uniform = verify_old_readout(OLD / "readouts/uniform/calibration.json", "calibration", config["legacy_manifests"]["uniform"]["identity"])["values"]
    baseline = response_metrics(uniform, teacher, bank)["mean"]
    if baseline["C_cross"] <= 1e-12 or not np.isfinite(baseline["C_cross"]):
        raise RuntimeError("Invalid Uniform cross C denominator")
    beta = baseline["C_natural"] / baseline["C_cross"]
    if abs(beta - BETA_EXPECTED) > 1e-11:
        raise RuntimeError(f"Uniform beta differs from predeclared value: {beta}")
    refs = read(config["legacy_manifests"]["uniform"]["path"])["entries"]
    costs = []
    for block in range(32):
        probe = read(OLD / "probes" / f"block{block:02d}.json")
        if probe["config_sha256"] != old_config_sha or probe["block"] != block:
            raise RuntimeError("Probe identity changed")
        conditions = [probe["conditions"][str(rate)] for rate in config["pruning"]["probe_rates"]]
        measured = []
        for cond in conditions:
            checked(cond["readout_path"], cond["readout_sha256"])
            probe_row = read(cond["readout_path"])
            if probe_row["fingerprint"] != {"config_sha256": old_config_sha,
                    "bank_sha256": sha(OLD / "bank_calibration.json"),
                    "mask_identity": cond["mask_identity"]} or not probe_row["complete"]:
                raise RuntimeError("Probe readout fingerprint changed")
            m = response_metrics(probe_row["values"], teacher, bank)
            for key in ("A", "Multi"):
                if abs(m["mean"][key] - cond["metrics"]["mean"][key]) > 2e-12:
                    raise RuntimeError("Saved natural probe objective changed")
            measured.append(m["mean"])
        delta = conditions[1]["pruned"] - conditions[0]["pruned"]
        if delta <= 0:
            raise RuntimeError("Probe count order reversed")
        cost = {"A": probe["costs"]["A"], "Multi": probe["costs"]["Multi"]}
        cross = (measured[1]["C_cross"] - measured[0]["C_cross"]) / delta
        cost["Cross"] = cost["A"] + cross
        cost["CrossMatched"] = cost["A"] + beta * cross
        costs.append(cost)
    allocations = {}
    for arm in ARMS:
        scores = [c[arm] for c in costs]
        rates = rank_rates(scores)
        counts, budget = exact_row_counts(refs, rates, config["pruning"]["pruned"])
        allocations[arm] = {"scores": scores, "rates": rates.tolist(), "row_counts": counts, "budget": budget}
        if budget["corrected_pruned"] != 3489660928 or len(counts) != 224:
            raise RuntimeError("Exact target or projection count changed")
    for arm in ("A", "Multi"):
        prior = old["allocations"][arm]
        if allocations[arm]["row_counts"] != prior["row_counts"] or allocations[arm]["rates"] != prior["rates"]:
            raise RuntimeError(f"Original {arm} allocation changed")
    if len({tuple(allocations[a]["row_counts"]) for a in ARMS}) != 4:
        raise RuntimeError("Four allocation controls are not distinct")
    if sum(a != b for a,b in zip(allocations["Cross"]["row_counts"], allocations["CrossMatched"]["row_counts"])) != 14:
        raise RuntimeError("Cross matched allocation differs from CPU preview")
    return {"config_sha256": sha(ROOT / "config.json"), "beta": beta,
            "beta_source": str(OLD / "readouts/uniform/calibration.json"),
            "uniform_C_natural": baseline["C_natural"], "uniform_C_cross": baseline["C_cross"],
            "graph": graph_invariants(), "allocations": allocations,
            "probe_sources": old["probe_sources"], "old_allocation_sha256": sha(OLD / "allocation.json")}


def prepare():
    if (ROOT / "started.json").exists():
        raise RuntimeError("Started experiment is immutable")
    from experiments.dlm_multiscale_ac50.prepare import validate as validate_old
    old = validate_old(OLD)
    source_bank()
    full_plan = copy.deepcopy(old["plan"])
    full_plan["gsm8k"]["development_doc_ids"] = list(FULL_IDS)
    full_plan["gsm8k"]["confirmation_doc_ids"] = []
    _, requests = freeze_requests(ROOT, full_plan, read(old["legacy_config"]))
    request_split = requests_check(requests["development"])
    write(ROOT / "request_split.json", request_split)
    write(ROOT / "cached_development.json", read(OLD / "cached_development.json"))
    config = copy.deepcopy(old)
    config["status"] = "prepared_not_started"
    config["banks"]["fresh"] = {"path": str((ROOT / "bank_fresh.json").resolve()),
                                "sha256": sha(ROOT / "bank_fresh.json"), "states": 128}
    config["requests_sha256"] = sha(ROOT / "requests.json")
    config["cached_development_sha256"] = sha(ROOT / "cached_development.json")
    config["crosschain_control"] = {"arms": list(ARMS), "exposed_ids": list(EXPOSED),
        "primary_ids": request_split["primary_ids"], "bootstrap_seed": 20260927,
        "bootstrap_draws": 10000, "shard_size": 128, "new_diagnostic_seed": 20260927,
        "primary_family": ["Multi-A", "Multi-Cross", "Multi-CrossMatched"],
        "requested_gpus": [0, 1, 2, 3]}
    config["sources"].update({str(SOURCE): sha(SOURCE), str(VERIFICATION): sha(VERIFICATION),
        str(OLD / "allocation.json"): sha(OLD / "allocation.json"),
        str(OLD / "requests.json"): sha(OLD / "requests.json"),
        str(PRIOR_VALIDATION / "requests.json"): sha(PRIOR_VALIDATION / "requests.json")})
    for folder in (HERE,):
        for path in folder.glob("*.py"):
            config["sources"][str(path)] = sha(path)
        for path in folder.glob("*.sh"):
            config["sources"][str(path)] = sha(path)
    for candidate in ("dense", "uniform", "A", "Multi"):
        for split in ("calibration", "diagnostic"):
            path = OLD / "readouts" / candidate / f"{split}.json"
            if path.exists():
                config["sources"][str(path)] = sha(path)
    for block in range(32):
        path = OLD / "probes" / f"block{block:02d}.json"
        config["sources"][str(path)] = sha(path)
    for arm in ("A", "Multi"):
        for filename in ("mask_manifest.json", "model_identity.json"):
            path = OLD / "candidates" / arm / filename
            config["sources"][str(path)] = sha(path)
    write(ROOT / "config.json", config)
    ch = sha(ROOT / "config.json")
    from experiments.projection_capacity_allocation_65.run import DENSE_SHA
    for split in ("calibration", "diagnostic"):
        for candidate in ("dense", "uniform", "A", "Multi"):
            path = OLD / "readouts" / candidate / f"{split}.json"
            if not path.exists():
                continue
            identity = "dense:" + DENSE_SHA if candidate == "dense" else (
                config["legacy_manifests"]["uniform"]["identity"] if candidate == "uniform" else
                read(OLD / "candidates" / candidate / "model_identity.json")["mask_identity"])
            reused_readout(path, ROOT / "readouts" / candidate / f"{split}.json",
                           ch, config["banks"][split]["sha256"], identity)
    result = allocate(config)
    write(ROOT / "allocation.json", result)
    validate()
    return {"config_sha256": ch, "allocation_sha256": sha(ROOT / "allocation.json"),
            "beta": result["beta"], "full_requests": len(requests["development"]),
            "primary_questions": len(request_split["primary_ids"]), "gpu_used": False}


def validate():
    from experiments.dlm_multiscale_ac50.prepare import validate as base_validate
    c = base_validate(ROOT)
    if c["crosschain_control"]["arms"] != list(ARMS) or c["crosschain_control"]["requested_gpus"] != [0,1,2,3]:
        raise RuntimeError("Frozen study configuration changed")
    req = read(ROOT / "requests.json")
    split = requests_check(req["development"])
    if split != read(ROOT / "request_split.json"):
        raise RuntimeError("Primary split changed")
    allocation = read(ROOT / "allocation.json")
    if allocation["config_sha256"] != sha(ROOT / "config.json") or abs(allocation["beta"] - BETA_EXPECTED) > 1e-11:
        raise RuntimeError("Allocation/config/beta mismatch")
    for path, expected in allocation["probe_sources"].items():
        checked(path, expected)
    if any(allocation["allocations"][a]["budget"]["corrected_pruned"] != 3489660928 for a in ARMS):
        raise RuntimeError("Budget changed")
    return c
