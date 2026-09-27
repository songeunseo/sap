"""Tests for the frozen dual-role reconstruction allocation diagnostic."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.dlm_dual_role_allocation.core import (
    allocation_mask_xor,
    bootstrap_mean_ci,
    partition_role_sums,
    pool_role_curves,
    role_contrast,
)
from experiments.dlm_dual_role_allocation.io import (
    validate_checkpoint,
    validate_frozen_payloads,
)
from experiments.dlm_dual_role_allocation.collect import collect_projection_from_block
from experiments.dlm_dual_role_allocation.analyze import (
    describe_roles,
    evaluate_crossfit,
    validate_reconstruction,
)
from experiments.dlm_dual_role_allocation.models import (
    apply_diagnostic_gate,
    fit_nested_ols,
    predict_nested_ols,
)
from experiments.dlm_dual_role_allocation import run


def test_partition_role_sums_reconstructs_aggregate():
    dense = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    sparse = dense + torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]])
    outputs = torch.cat([dense, sparse.repeat(6, 1, 1)], dim=0)
    result = partition_role_sums(outputs, torch.tensor([True, False, True]))
    assert result["num_masked"][0] + result["num_unmasked"][0] == 14.0
    assert result["den_masked"] + result["den_unmasked"] == 91.0


def test_partition_role_sums_rejects_empty_group_and_zero_denominator():
    outputs = torch.ones(7, 3, 2)
    with pytest.raises(ValueError, match="empty token role"):
        partition_role_sums(outputs, torch.ones(3, dtype=torch.bool))
    outputs[:, 0] = 0
    with pytest.raises(ValueError, match="zero role denominator"):
        partition_role_sums(outputs, torch.tensor([True, False, False]))


def test_partition_role_sums_rejects_wrong_variant_shape():
    with pytest.raises(ValueError, match="seven variants"):
        partition_role_sums(torch.ones(6, 3, 2), torch.tensor([True, False, False]))


def test_role_contrast_has_no_epsilon_and_zero_zero_is_zero():
    got = role_contrast(np.array([0.0, 3.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(got, [0.0, 0.5, 0.0])


def test_pool_role_curves_uses_ratio_of_sums_not_mean_of_ratios():
    raw = [
        {
            "state_index": 0,
            "levels": [
                {"num_masked": 1.0, "den_masked": 2.0,
                 "num_unmasked": 3.0, "den_unmasked": 6.0},
                {"num_masked": 2.0, "den_masked": 2.0,
                 "num_unmasked": 6.0, "den_unmasked": 6.0},
            ],
        },
        {
            "state_index": 1,
            "levels": [
                {"num_masked": 9.0, "den_masked": 18.0,
                 "num_unmasked": 1.0, "den_unmasked": 2.0},
                {"num_masked": 18.0, "den_masked": 18.0,
                 "num_unmasked": 2.0, "den_unmasked": 2.0},
            ],
        },
    ]
    pooled = pool_role_curves(raw, state_ids=[0, 1])
    assert pooled["masked"][0] == pytest.approx((1 + 9) / (2 + 18))
    assert pooled["unmasked"][0] == pytest.approx((3 + 1) / (6 + 2))
    assert pooled["aggregate"][1] == pytest.approx((2 + 6 + 18 + 2) / (2 + 6 + 18 + 2))
    assert pooled["role"][0] == pytest.approx(0.5)


def test_pool_role_curves_rejects_missing_and_duplicate_states():
    raw = [{"state_index": 0, "levels": [{"num_masked": 1.0, "den_masked": 1.0,
                                            "num_unmasked": 1.0, "den_unmasked": 1.0}]}]
    with pytest.raises(ValueError, match="requested states"):
        pool_role_curves(raw, [0, 1])
    with pytest.raises(ValueError, match="duplicate"):
        pool_role_curves(raw + raw, [0])


def test_nested_mask_xor_uses_selected_pruned_counts():
    manifest = [
        {"weights": 20, "masks": [{"pruned": 4}, {"pruned": 6}, {"pruned": 8}]},
        {"weights": 20, "masks": [{"pruned": 2}, {"pruned": 4}, {"pruned": 6}]},
    ]
    result = allocation_mask_xor([0, 2], [1, 1], manifest)
    assert result["xor_pruned_weights"] == 4
    assert result["xor_fraction_of_prunable_weights"] == pytest.approx(4 / 40)
    assert result["changed_projection_count"] == 2


def test_bootstrap_mean_ci_is_deterministic():
    first = bootstrap_mean_ci([-2.0, -1.0, 0.0, 1.0], resamples=1000, seed=7)
    second = bootstrap_mean_ci([-2.0, -1.0, 0.0, 1.0], resamples=1000, seed=7)
    assert first == second
    assert first["count"] == 4


def _frozen_payloads():
    grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    config = {
        "model": {"id": "GSAI-ML/LLaDA-8B-Base",
                  "revision": "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"},
        "dense_model_sha256": "dense",
        "grid": grid,
        "projection_count": 2,
        "budget": {"total_pruned": 52, "total_weights": 80},
    }
    verification = {"status": "verified", "disjoint": True}
    entries = []
    projections = []
    for index, (name, shape) in enumerate((("block_00.attn_out", [2, 20]),
                                            ("block_00.q_proj", [2, 20]))):
        masks = [{"nominal_sparsity": level, "pruned": shape[0] * int(shape[1] * level),
                  "mask_sha256": f"mask-{index}-{level}", "path": f"mask-{index}-{level}.pt",
                  "file_sha256": f"file-{index}-{level}"} for level in grid]
        entries.append({"module_index": index, "name": name, "shape": shape,
                        "weights": int(np.prod(shape)), "masks": masks})
        projections.append({"module_index": index, "name": name, "shape": shape,
                            "curves": [{"sparsity": level, "per_state": []} for level in grid]})
    candidate = {"entries": entries}
    curves = {"grid": grid, "projections": projections}
    state_pairs = [(0, 0.05), (1, 0.05), (0, 0.15), (1, 0.15)]
    states = {"historical_state_sha256": "state", "states": [
        {"state_index": index, "sequence_index": sequence,
         "timestep": timestep, "noisy_ids": [[1, 2]]}
        for index, (sequence, timestep) in enumerate(state_pairs)
    ]}
    return config, verification, candidate, curves, states


def test_validate_frozen_inputs_rejects_module_order_mismatch():
    config, verification, candidate, curves, states = _frozen_payloads()
    curves["projections"].reverse()
    with pytest.raises(RuntimeError, match="module ordering"):
        validate_frozen_payloads(config, verification, candidate, curves, states,
                                 expected_projection_count=2,
                                 expected_state_count=4,
                                 expected_sequences={0, 1},
                                 expected_timesteps={0.05, 0.15})


def test_validate_frozen_inputs_checks_budget_and_returns_metadata():
    config, verification, candidate, curves, states = _frozen_payloads()
    result = validate_frozen_payloads(config, verification, candidate, curves, states,
                                      expected_projection_count=2,
                                      expected_state_count=4,
                                      expected_sequences={0, 1},
                                      expected_timesteps={0.05, 0.15})
    assert result["module_names"] == ["block_00.attn_out", "block_00.q_proj"]
    assert result["uniform65_pruned"] == 52
    config["budget"]["total_pruned"] += 1
    with pytest.raises(RuntimeError, match="uniform row-floor budget"):
        validate_frozen_payloads(config, verification, candidate, curves, states,
                                 expected_projection_count=2,
                                 expected_state_count=4,
                                 expected_sequences={0, 1},
                                 expected_timesteps={0.05, 0.15})


def test_validate_checkpoint_rejects_state_digest_mismatch():
    expected = {
        "source_receipt_sha256": "receipt",
        "dense_model_sha256": "dense",
        "state_digest": "good",
        "module_index": 3,
        "name": "block_00.q_proj",
        "shape": [2, 2],
        "grid": [0.5, 0.55],
        "mask_sha256": ["a", "b"],
        "state_count": 4,
        "level_count": 2,
    }
    payload = {**expected, "state_digest": "bad", "states": [
        {"levels": [{}, {}]} for _ in range(4)
    ]}
    with pytest.raises(RuntimeError, match="state digest"):
        validate_checkpoint(payload, expected)


def test_validate_checkpoint_accepts_complete_identity():
    expected = {
        "source_receipt_sha256": "receipt", "dense_model_sha256": "dense",
        "state_digest": "good", "module_index": 3, "name": "block_00.q_proj",
        "shape": [2, 2], "grid": [0.5, 0.55], "mask_sha256": ["a", "b"],
        "state_count": 2, "level_count": 2,
    }
    payload = {**expected, "states": [{"levels": [{}, {}]}, {"levels": [{}, {}]}]}
    validate_checkpoint(payload, expected)


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target = torch.nn.Linear(2, 2, bias=False)
        self.target.weight.data.copy_(torch.eye(2))
        self.suffix_calls = 0

    def forward(self, value):
        value = self.target(value)
        self.suffix_calls += 1
        return value + 100


def test_target_stop_collects_output_without_running_suffix():
    block = _ToyBlock()
    prefixes = [torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])]
    states = [{"mask": [[True, False, True]], "sequence_index": 0,
               "timestep": 0.05, "p_mask": 0.1}]
    mask = torch.tensor([[True, False], [False, False]])
    records = collect_projection_from_block(block, block.target, prefixes, states,
                                            [mask] * 6)
    assert len(records) == 1
    assert block.suffix_calls == 0
    assert records[0]["levels"][0]["num_masked"] == pytest.approx(26.0)
    assert records[0]["levels"][0]["num_unmasked"] == pytest.approx(9.0)


def test_target_stop_rejects_nonidentical_variant_inputs():
    class VariantChangingBlock(_ToyBlock):
        def forward(self, value):
            value = value.clone()
            value[1] += 1
            return super().forward(value)

    block = VariantChangingBlock()
    prefixes = [torch.ones(1, 3, 2)]
    states = [{"mask": [[True, False, True]], "sequence_index": 0,
               "timestep": 0.05, "p_mask": 0.1}]
    with pytest.raises(RuntimeError, match="variant inputs differ"):
        collect_projection_from_block(block, block.target, prefixes, states,
                                      [torch.zeros(2, 2, dtype=torch.bool)] * 6)


def test_reconstruction_sanity_enforces_absolute_threshold():
    historical = np.ones((2, 3, 6))
    reconstructed = historical.copy()
    reconstructed[0, 0, 0] += 2e-7
    with pytest.raises(RuntimeError, match="absolute reconstruction"):
        validate_reconstruction(reconstructed, historical)


def test_reconstruction_sanity_enforces_relative_threshold_and_reports_location():
    historical = np.full((2, 3, 6), 1e-6)
    reconstructed = historical.copy()
    reconstructed[1, 2, 5] += 2e-11
    with pytest.raises(RuntimeError, match="relative reconstruction"):
        validate_reconstruction(reconstructed, historical)
    result = validate_reconstruction(historical.copy(), historical)
    assert result["passed"] is True
    assert result["max_absolute_error"] == 0


def test_describe_roles_counts_dominance_and_negative_marginals():
    masked = np.array([[1, 2, 1, 3, 4, 5], [2, 2, 3, 4, 5, 6]], dtype=float)
    unmasked = np.array([[0, 1, 2, 2, 3, 4], [3, 2, 2, 3, 4, 5]], dtype=float)
    report = describe_roles(masked, unmasked,
                            names=["block_00.attn_out", "block_01.q_proj"])
    assert sum(report["dominance_counts"].values()) == 12
    assert report["role_negative_marginal_count"] == 1
    assert set(report["by_type"]) == {"attn_out", "q_proj"}


def _nested_fixture():
    layers = np.array([0, 1, 2])
    types = np.array(["attn_out", "q_proj", "v_proj"])
    aggregate = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5) / 100 + 1
    delta = np.sin(np.arange(3 * 4 * 5, dtype=float)).reshape(3, 4, 5) / 10
    role = aggregate + delta
    functional = 2 * aggregate + 3 * delta
    return functional, aggregate, role, layers, types


def test_nested_ols_uses_fixed_reference_categories_and_train_scaling():
    functional, aggregate, role, layers, types = _nested_fixture()
    fit = fit_nested_ols(functional, aggregate, role, layers, types)
    assert fit["baseline_columns"][:3] == ["intercept", "layer_1", "layer_2"]
    assert "layer_0" not in fit["extended_columns"]
    assert "type_attn_out" not in fit["extended_columns"]
    assert fit["baseline_columns"][-1] == "aggregate_marginal_per_parameter"
    assert fit["extended_columns"][-1] == "role_minus_aggregate_marginal_per_parameter"
    prediction = predict_nested_ols(fit, aggregate, role, layers, types)
    assert prediction["baseline"].shape == functional.shape
    np.testing.assert_allclose(prediction["extended"], functional, atol=1e-10)


def test_nested_ols_rejects_zero_variance_feature():
    functional, aggregate, role, layers, types = _nested_fixture()
    with pytest.raises(ValueError, match="zero-variance"):
        fit_nested_ols(functional, np.ones_like(aggregate), role, layers, types)


def _passing_gate_fixture():
    return {
        "reconstruction_sanity_passed": True,
        "dense_hash_passed": True,
        "role_better_both_folds": True,
        "damage_difference_ci": [-2.0, -1.0],
        "mask_xor_fraction": 0.02,
        "extended_better_both_folds": True,
        "rmse_difference_ci": [-0.2, -0.1],
    }


def test_gate_passes_only_when_all_five_criteria_hold():
    result = apply_diagnostic_gate(_passing_gate_fixture())
    assert result["decision"] == "DIAGNOSTIC SUPPORTED"
    assert all(result["criteria"].values())


def test_gate_fails_when_only_mask_xor_is_below_one_percent():
    evidence = _passing_gate_fixture()
    evidence["mask_xor_fraction"] = 0.0099
    result = apply_diagnostic_gate(evidence)
    assert result["decision"] == "NOT SUPPORTED"
    assert result["criteria"]["mask_xor_at_least_one_percent"] is False


def test_crossfit_hits_exact_budget_and_returns_eight_sequence_differences():
    modules, states, levels = 2, 8, 6
    level_axis = np.arange(levels, dtype=float)[None, None, :]
    module_axis = np.arange(modules, dtype=float)[:, None, None]
    state_axis = np.arange(states, dtype=float)[None, :, None]
    masked = 1.0 + level_axis * (0.2 + 0.1 * module_axis) + state_axis * 0.001
    unmasked = 1.0 + level_axis * (0.3 - 0.05 * module_axis) + state_axis * 0.002
    den_masked = np.ones((modules, states, levels))
    den_unmasked = np.full((modules, states, levels), 2.0)
    fields = {
        "num_masked": masked * den_masked,
        "den_masked": den_masked,
        "num_unmasked": unmasked * den_unmasked,
        "den_unmasked": den_unmasked,
    }
    functional = (masked + unmasked) * 0.01 + module_axis * 0.001
    shapes = [(2, 20), (2, 20)]
    manifest = [
        {"weights": 40, "masks": [
            {"pruned": 2 * int(20 * sparsity)}
            for sparsity in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        ]} for _ in range(2)
    ]
    result = evaluate_crossfit(functional, fields, shapes, manifest,
                               np.arange(8), np.array([0, 1]),
                               np.array(["attn_out", "q_proj"]))
    assert len(result["folds"]) == 2
    assert result["sequence_damage_difference"]["count"] == 8
    assert result["sequence_rmse_difference"]["count"] == 8
    for fold in result["folds"]:
        assert fold["aggregate_allocation"]["pruned"] == 52
        assert fold["role_allocation"]["pruned"] == 52


def test_freeze_config_contains_pre_registered_gate(tmp_path, monkeypatch):
    config, verification, candidate, curves, states = _frozen_payloads()
    frozen = type("Frozen", (), {
        "config": config,
        "verification": verification,
        "candidate": candidate,
        "curves": curves,
        "states": states,
        "metadata": {
            "state_count": 4,
            "uniform65_pruned": 52,
            "total_weights": 80,
            "state_digest": "state",
            "dense_model_sha256": "dense",
            "module_names": ["block_00.attn_out", "block_00.q_proj"],
        },
        "receipt": {"receipt_sha256": "receipt", "files": {"a": "b"}},
    })()
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "load_frozen_inputs", lambda: frozen)
    result = run.freeze()
    assert result["role_curve"] == "max(E_masked,E_unmasked)"
    assert result["mask_xor_gate"] == 0.01
    assert result["full_model_evaluation"] is False
    assert result["source_receipt_sha256"] == "receipt"
    assert (tmp_path / "state_verification.json").exists()


def test_freeze_refuses_to_change_existing_config(tmp_path, monkeypatch):
    test_freeze_config_contains_pre_registered_gate(tmp_path, monkeypatch)
    payload = __import__("json").loads((tmp_path / "config.json").read_text())
    payload["mask_xor_gate"] = 0.0
    (tmp_path / "config.json").write_text(__import__("json").dumps(payload))
    with pytest.raises(RuntimeError, match="frozen experiment config"):
        run.freeze()
