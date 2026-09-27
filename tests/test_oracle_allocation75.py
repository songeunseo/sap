import math

from experiments.oracle_allocation75.core import (
    FALLBACK_SCHEDULES,
    GLOBAL_SPARSITY,
    build_all_plans,
    feasibility_audit,
    load_module_records,
)


STATS = "experiments/wanda_failure_characterization/per_module_statistics.csv"


def test_requested_functional_schedules_are_infeasible_at_registered_q_values():
    records = load_module_records(STATS)
    audit = feasibility_audit(records, "functional_kl")
    assert len(audit) == 9
    assert all(not row["feasible"] for row in audit)


def test_all_fallback_plans_have_exact_global_budget():
    records = load_module_records(STATS)
    selections, plans = build_all_plans(records)
    assert len(plans) == 16
    assert len(selections["functional_kl"]["protected"]) == 22
    for plan in plans.values():
        assert plan["global_actual_sparsity"] == GLOBAL_SPARSITY
        assert plan["total_pruned"] == int(GLOBAL_SPARSITY * plan["total_parameters"])
        assert len(plan["entries"]) == 224
        assert all(0 <= row["actual_sparsity"] <= 1 for row in plan["entries"])


def test_functional_fallback_strength_and_budget_order():
    records = load_module_records(STATS)
    _, plans = build_all_plans(records)
    donors = []
    for schedule in ("mild", "medium", "aggressive"):
        plan = plans[f"V3_functional_kl_{schedule}"]
        assert math.isclose(plan["protected_sparsity"], FALLBACK_SCHEDULES[schedule])
        donors.append(plan["donor_requested_sparsity"])
    assert donors[0] < donors[1] < donors[2] < 1.0


def test_random_groups_match_reconstruction_type_counts():
    records = load_module_records(STATS)
    lookup = {row["module"]: row["module_type"] for row in records}
    selections, _ = build_all_plans(records)
    reference = selections["reconstruction"]
    for seed, groups in selections["random"].items():
        for side in ("protected", "donors"):
            expected = sorted(lookup[name] for name in reference[side])
            actual = sorted(lookup[name] for name in groups[side])
            assert actual == expected, (seed, side)
