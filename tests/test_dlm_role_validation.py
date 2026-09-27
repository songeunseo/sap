"""Unit tests for the preregistered Role validation pipeline."""
import numpy as np
import pytest
import torch

from experiments.dlm_role_validation.core import (
    allocate_grid, deterministic_random_role, partition_digest, partition_sums,
    pool_partition, random_mini_gate, sparsity_mini_gate,
)
from experiments.projection_capacity_allocation_65.core import allocate
from experiments.dlm_role_validation.status import (
    calculate_serial_eta, generation_progress, parse_clock,
)


def test_random_role_is_deterministic_cardinality_matched_and_seeded():
    real = torch.tensor([True, False, True, False, False, True, False, False])
    first = deterministic_random_role(real, 101, 3)
    assert torch.equal(first, deterministic_random_role(real, 101, 3))
    assert int(first.sum()) == int(real.sum())
    assert partition_digest([first]) == partition_digest([first.clone()])
    assert not torch.equal(first, deterministic_random_role(real, 202, 3))


def test_partition_sums_reconstruct_aggregate_independent_of_partition():
    dense = torch.arange(1, 13, dtype=torch.float32).reshape(1, 6, 2)
    outputs = torch.cat([dense, (dense + 1).repeat(6, 1, 1)])
    left = partition_sums(outputs, torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool))
    right = partition_sums(outputs, torch.tensor([1, 0, 1, 0, 0, 0], dtype=torch.bool))
    for level in range(6):
        assert left["num_a"][level] + left["num_b"][level] == right["num_a"][level] + right["num_b"][level]
        assert left["den_a"] + left["den_b"] == right["den_a"] + right["den_b"]


def test_pool_partition_uses_ratio_of_pooled_sums():
    records = []
    for state, scale in enumerate((1.0, 3.0)):
        records.append({"state_index": state, "groups": {"x": [{
            "num_a": scale, "den_a": 2 * scale, "num_b": 2 * scale,
            "den_b": 8 * scale,
        }]}})
    result = pool_partition(records, "x")
    assert result["a"][0] == pytest.approx(.5)
    assert result["b"][0] == pytest.approx(.25)
    assert result["role"][0] == pytest.approx(.5)
    assert result["aggregate"][0] == pytest.approx(.3)


def test_general_allocator_reproduces_original_and_shifted_exact_budgets():
    rng = np.random.default_rng(7)
    shapes = [(4, 20), (8, 20), (4, 40), (8, 40)]
    curves = np.cumsum(rng.uniform(0, 1, size=(4, 6)), axis=1)
    original = allocate(curves, shapes)
    generalized = allocate_grid(curves, shapes, (.50, .55, .60, .65, .70, .75))
    assert generalized["levels"] == original["levels"]
    assert generalized["pruned"] == original["pruned"]
    for grid in ((.35, .40, .45, .50, .55, .60), (.60, .65, .70, .75, .80, .85)):
        result = allocate_grid(curves, shapes, grid)
        expected = sum(rows * int(cols * grid[3]) for rows, cols in shapes)
        assert result["pruned"] == expected
        assert result["budget_error"] == 0


def test_preregistered_mini_gates_are_strict():
    assert random_mini_gate(24, [20, 23, 25])["passed"]
    assert not random_mini_gate(24, [24, 23, 25])["passed"]
    assert sparsity_mini_gate(12, 19, 20)["passed"]
    assert not sparsity_mini_gate(20, 19, 20)["passed"]


def test_status_parses_tqdm_eta_and_serial_queue():
    text = "\rGenerating...:  56%|x| 56/100 [19:33<15:44, 21.46s/it]"
    progress = generation_progress(text)
    assert progress["completed"] == 56
    assert progress["remaining_seconds"] == 15 * 60 + 44
    assert parse_clock("1:02:03") == 3723
    rows = calculate_serial_eta([
        {"stage": "active", "intrinsic_remaining": 100.0},
        {"stage": "queue 대기", "intrinsic_remaining": 200.0},
    ])
    assert rows[0]["complete_in"] == 100
    assert rows[1]["start_in"] == 100
    assert rows[1]["complete_in"] == 300
