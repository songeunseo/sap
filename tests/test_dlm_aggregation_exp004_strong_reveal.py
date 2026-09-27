import json
from pathlib import Path

import pytest
import torch

from experiments.dlm_loss_aggregation.exp004.run import token_weights
from experiments.dlm_loss_aggregation.exp004_strong_reveal.run import (
    DeviceAbsAccumulator,
    decision_from_diagnostics,
    load_config,
    summarize_token_weight_ratio,
    token_weights_ratio,
)


def test_strong_reveal_weights_preserve_raw_ratio_and_masked_mean_one():
    mask = torch.tensor([[True, True, True, True, False]])
    reveal = torch.tensor([[True, False, False, False, False]])
    alpha = token_weights_ratio(mask, reveal, 50.0)
    assert alpha[reveal].item() / alpha[mask & ~reveal][0].item() == pytest.approx(50.0)
    assert alpha[mask].mean().item() == pytest.approx(1.0)
    assert alpha[~mask].eq(0).all().item()


def test_ratio_two_matches_exp004_reveal_weighting():
    mask = torch.tensor([[True, True, False, True]])
    reveal = torch.tensor([[False, True, False, False]])
    torch.testing.assert_close(
        token_weights_ratio(mask, reveal, 2.0),
        token_weights(mask, reveal, "reveal"),
    )


def test_one_of_133_reveal_tokens_gets_expected_50x_mass_share():
    summary = summarize_token_weight_ratio(
        {"masked_count": 133, "reveal_count": 1, "remain_count": 132}, 50.0
    )
    assert summary["normalized_mean"] == pytest.approx(1.0)
    assert summary["reveal_normalized_weight_mass_share"] == pytest.approx(50 / 182)


def test_device_abs_accumulator_matches_direct_fp32_definition():
    accumulator = DeviceAbsAccumulator((2, 2), torch.device("cpu"))
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32)
    gradients = [
        torch.tensor([[0.5, -1.0], [-2.0, 0.25]]),
        torch.tensor([[-0.5, 2.0], [1.0, -0.75]]),
    ]
    for gradient in gradients:
        accumulator.add_gradient(weight, gradient)
    expected = torch.stack([(weight * gradient).abs() for gradient in gradients]).mean(0)
    torch.testing.assert_close(accumulator.finalize(), expected)


@pytest.mark.parametrize("ratio", [0.5, float("inf"), float("nan")])
def test_invalid_reveal_ratios_are_rejected(ratio):
    mask = torch.tensor([[True, True]])
    reveal = torch.tensor([[True, False]])
    with pytest.raises(ValueError):
        token_weights_ratio(mask, reveal, ratio)


def test_preregistered_decision_rules():
    redundant = decision_from_diagnostics({"mask_xor": 0.009, "spearman": 0.995})
    assert redundant == {
        "interpretation": "ranking_redundant",
        "run_downstream": False,
        "gate": "mask_xor >= 0.01 or spearman <= 0.99",
    }
    strong = decision_from_diagnostics({"mask_xor": 0.025, "spearman": 0.98})
    assert strong["interpretation"] == "weighting_strength_supported"
    assert strong["run_downstream"] is True
    intermediate = decision_from_diagnostics({"mask_xor": 0.015, "spearman": 0.995})
    assert intermediate["interpretation"] == "intermediate_mask_sensitivity"
    assert intermediate["run_downstream"] is True


def test_preregistration_freezes_only_the_three_requested_conditions():
    config = load_config()
    assert [row["name"] for row in config["conditions"]] == [
        "uniform_abs",
        "reveal2_abs",
        "reveal50_abs",
    ]
    assert [row["reveal_raw"] for row in config["conditions"]] == [1.0, 2.0, 50.0]
    assert config["aggregation"] == "mean_state(abs(-weight_fp32 * gradient_fp32))"
    assert config["mask"]["sparsity"] == 0.5


def test_frozen_partition_counts_match_preregistration():
    config = json.loads(
        Path("experiments/dlm_loss_aggregation/exp004_strong_reveal/preregistered.json").read_text()
    )
    partition = json.loads(Path(config["calibration"]["partition"]["path"]).read_text())
    assert partition["summary"]["masked_tokens"] == 10267
    assert partition["summary"]["reveal_tokens"] == 116
    assert partition["summary"]["remain_tokens"] == 10151
