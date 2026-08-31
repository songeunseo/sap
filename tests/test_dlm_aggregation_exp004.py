import math

import pytest
import torch

from experiments.dlm_loss_aggregation.exp004.run import (
    MagnitudeAccumulator,
    compare_score_masks,
    holm_adjust,
    independent_weight_effects,
    paired_comparison,
    freeze_token_partition,
    require_disk_capacity,
    validate_uniform_reuse,
    token_weights,
    validate_uniform_reproduction,
    weighted_dlm_loss,
)


def test_freeze_partition_uses_remaining_schedule_and_partitions_every_mask():
    logits = torch.full((1, 6, 4), -10.0)
    logits[0, 0, 1] = math.log(9)
    logits[0, 1, 2] = math.log(8)
    logits[0, 2, 3] = math.log(7)
    logits[0, 3, 1] = math.log(6)
    noisy = torch.tensor([[99, 99, 99, 99, 4, 5]])

    partition = freeze_token_partition(
        logits, noisy, mask_id=99, p_mask=0.25, denoising_steps=8
    )

    assert partition["steps_remaining"] == 2
    assert partition["masked_indices"] == [0, 1, 2, 3]
    assert partition["reveal_indices"] == [0, 1]
    assert partition["remain_indices"] == [2, 3]
    assert set(partition["reveal_indices"]).isdisjoint(partition["remain_indices"])
    assert sorted(partition["reveal_indices"] + partition["remain_indices"]) == partition["masked_indices"]


def test_token_weights_normalize_each_condition_to_masked_mean_one():
    mask = torch.tensor([[True, True, True, True, False]])
    reveal = torch.tensor([[True, True, False, False, False]])

    uniform = token_weights(mask, reveal, "uniform")
    reveal_up = token_weights(mask, reveal, "reveal")
    remain_up = token_weights(mask, reveal, "remain")

    assert torch.equal(uniform[mask], torch.ones(4))
    assert torch.equal(reveal_up[mask], torch.tensor([4 / 3, 4 / 3, 2 / 3, 2 / 3]))
    assert torch.equal(remain_up[mask], torch.tensor([2 / 3, 2 / 3, 4 / 3, 4 / 3]))
    assert reveal_up[mask].mean().item() == 1.0
    assert remain_up[mask].mean().item() == 1.0
    assert not uniform[~mask].any().item()


def test_weighted_loss_changes_only_token_terms_not_dlm_normalization():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    clean = torch.tensor([[0, 0, 1]])
    mask = torch.tensor([[True, True, False]])
    alpha = torch.tensor([[1.5, 0.5, 0.0]])

    loss = weighted_dlm_loss(logits, clean, mask, p_mask=0.5, alpha=alpha)
    token_ce = torch.nn.functional.cross_entropy(
        logits[mask].float(), clean[mask], reduction="none"
    )

    assert loss.item() == pytest.approx((1.5 * token_ce[0] + 0.5 * token_ce[1]).item() / 0.5 / 3)


def test_condition_effects_share_graph_without_accumulating_parameter_gradients():
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -3.0]]))
    output = layer(torch.tensor([[4.0, 5.0]])).sum()

    effects = independent_weight_effects(
        {
            "uniform": output,
            "reveal": 2 * output,
            "remain": -output,
        },
        {"linear": layer.weight},
    )

    assert torch.equal(effects["uniform"]["linear"], torch.tensor([[-8.0, 15.0]]))
    assert torch.equal(effects["reveal"]["linear"], torch.tensor([[-16.0, 30.0]]))
    assert torch.equal(effects["remain"]["linear"], torch.tensor([[8.0, -15.0]]))
    assert layer.weight.grad is None


def test_uniform_reuse_requires_identical_protocol_and_paired_examples():
    uniform = {
        "UNIFORM-ABS": [
            {
                "example_id": 0,
                "doc_hash": "d0",
                "prompt_hash": "p0",
                "target_hash": "t0",
                "evaluation_config_hash": "cfg",
            }
        ],
        "UNIFORM-SQUARE": [
            {
                "example_id": 0,
                "doc_hash": "d0",
                "prompt_hash": "p0",
                "target_hash": "t0",
                "evaluation_config_hash": "cfg",
            }
        ],
    }

    assert validate_uniform_reuse(uniform, "cfg", "cfg", expected_count=1)["passed"]

    with pytest.raises(ValueError, match="config hash"):
        validate_uniform_reuse(uniform, "cfg", "different", expected_count=1)
    uniform["UNIFORM-SQUARE"][0]["prompt_hash"] = "different"
    with pytest.raises(ValueError, match="paired examples"):
        validate_uniform_reuse(uniform, "cfg", "cfg", expected_count=1)


def test_disk_preflight_fails_before_writing_when_final_masks_do_not_fit(tmp_path):
    result = require_disk_capacity(
        tmp_path, mask_bytes_per_method=100, method_count=4, safety_bytes=50,
        free_bytes=450,
    )
    assert result["required_bytes"] == 450

    with pytest.raises(RuntimeError, match="insufficient disk"):
        require_disk_capacity(
            tmp_path, mask_bytes_per_method=100, method_count=4,
            safety_bytes=51, free_bytes=450,
        )


def test_magnitude_accumulator_keeps_normalized_scores_and_raw_scale_diagnostics():
    accumulator = MagnitudeAccumulator((1, 2))
    accumulator.add(torch.tensor([[1.0, -2.0]]), raw_scale=1.5)
    accumulator.add(torch.tensor([[-3.0, 4.0]]), raw_scale=2.0)

    scores, diagnostics = accumulator.finalize()

    assert torch.equal(scores["abs"], torch.tensor([[2.0, 3.0]]))
    assert torch.equal(scores["square"], torch.tensor([[5.0, 10.0]]))
    assert diagnostics["normalized_abs_mean"] == pytest.approx(2.5)
    assert diagnostics["raw_abs_mean"] == pytest.approx(4.625)
    assert diagnostics["normalized_square_mean"] == pytest.approx(7.5)
    assert diagnostics["raw_square_mean"] == pytest.approx(27.8125)


def test_uniform_reproduction_reports_small_mismatch_and_aborts_above_threshold():
    rows = [
        {"method": "abs", "num_weights": 1_000_000, "different": 1},
        {"method": "square", "num_weights": 1_000_000, "different": 0},
    ]

    result = validate_uniform_reproduction(
        rows, global_xor_threshold=1e-6, module_xor_threshold=1e-5
    )
    assert result["exact"] is False
    assert result["by_method"]["abs"]["xor"] == 1e-6

    rows[0]["different"] = 11
    with pytest.raises(RuntimeError, match="UNIFORM mask reproduction"):
        validate_uniform_reproduction(
            rows, global_xor_threshold=1e-6, module_xor_threshold=1e-5
        )


def test_score_mask_comparison_reports_exact_overlap_diagnostics():
    left_score = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    right_score = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
    left_mask = torch.tensor([[True, True, False, False]])
    right_mask = torch.tensor([[True, False, True, False]])

    result = compare_score_masks(left_score, right_score, left_mask, right_mask)

    assert result["spearman"] == pytest.approx(0.8)
    assert result["mask_xor"] == 0.5
    assert result["mask_iou"] == pytest.approx(1 / 3)
    assert result["topk_overlap"] == 0.5


def test_paired_comparison_uses_exact_two_sided_binomial_and_holm_adjustment():
    left = [True, True, True, False, True, False]
    right = [True, False, False, True, False, False]
    records = {
        "A": [{"correct": value} for value in left],
        "B": [{"correct": value} for value in right],
    }

    row = paired_comparison("A", "B", records)

    assert row["both_correct"] == 1
    assert row["a_only_correct"] == 3
    assert row["b_only_correct"] == 1
    assert row["both_wrong"] == 1
    assert row["accuracy_difference_pp"] == pytest.approx(100 * (4 / 6 - 2 / 6))
    assert row["p_exact_two_sided"] == 0.625

    adjusted = holm_adjust([{"p": 0.01}, {"p": 0.04}, {"p": 0.03}], "p")
    assert [item["p_holm"] for item in adjusted] == pytest.approx([0.03, 0.06, 0.06])
    compare_score_masks,
    holm_adjust,
    paired_comparison,
