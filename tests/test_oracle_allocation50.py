import json

from experiments.oracle_allocation50.core import (
    GLOBAL_SPARSITY,
    PROTECTED_SPARSITY,
    build_plans,
)
from experiments.oracle_allocation75.core import load_module_records


def _plans():
    records = load_module_records("experiments/wanda_failure_characterization/per_module_statistics.csv")
    selections = json.load(open("experiments/oracle_allocation75/allocation_selections.json"))
    return records, build_plans(records, selections)


def test_expected_variants_and_exact_budget():
    records, plans = _plans()
    assert set(plans) == {
        "V0_50_uniform",
        "V1_50_random_20260907",
        "V1_50_random_20260908",
        "V1_50_random_20260909",
        "V3_50_functional_kl_mild",
    }
    total = sum(row["nweight"] for row in records)
    for plan in plans.values():
        assert plan["total_pruned"] == total // 2
        assert plan["global_actual_sparsity"] == GLOBAL_SPARSITY


def test_relative_mild_strength_is_frozen():
    _, plans = _plans()
    oracle = plans["V3_50_functional_kl_mild"]
    assert PROTECTED_SPARSITY == 0.48
    assert oracle["protected_sparsity"] == 0.48
    assert len(oracle["protected_groups"]) == 22
    assert len(oracle["donor_groups"]) == 22
    assert 0.5 < oracle["donor_requested_sparsity"] < 1.0


def test_random_group_assignments_match_oracle_type_and_parameter_counts():
    _, plans = _plans()
    oracle = plans["V3_50_functional_kl_mild"]
    oracle_by_module = {row["module"]: row for row in oracle["entries"]}
    oracle_protected_types = sorted(oracle_by_module[name]["module_type"] for name in oracle["protected_groups"])
    oracle_donor_types = sorted(oracle_by_module[name]["module_type"] for name in oracle["donor_groups"])
    for seed in (20260907, 20260908, 20260909):
        plan = plans[f"V1_50_random_{seed}"]
        by_module = {row["module"]: row for row in plan["entries"]}
        assert sorted(by_module[name]["module_type"] for name in plan["protected_groups"]) == oracle_protected_types
        assert sorted(by_module[name]["module_type"] for name in plan["donor_groups"]) == oracle_donor_types
        assert plan["donor_requested_sparsity"] == oracle["donor_requested_sparsity"]


def test_projection_deviations_are_nonuniform_only_in_expected_groups():
    _, plans = _plans()
    uniform = plans["V0_50_uniform"]
    assert all(row["sparsity_deviation_from_uniform"] == 0 for row in uniform["entries"])
    for name, plan in plans.items():
        if name == "V0_50_uniform":
            continue
        status = {row["status"] for row in plan["entries"]}
        assert status == {"protected", "middle", "donor"}
        assert min(row["sparsity_deviation_from_uniform"] for row in plan["entries"]) < 0
        assert max(row["sparsity_deviation_from_uniform"] for row in plan["entries"]) > 0
