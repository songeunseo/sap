import pytest
import torch

from lib.channel_constraint import (
    apply_delta_to_mask,
    build_channel_delta,
    protected_axis,
)


def _baseline_mask():
    return torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 1],
        ],
        dtype=torch.bool,
    )


@pytest.mark.parametrize(
    ("module_name", "axis"),
    [
        ("q_proj", "column"),
        ("k_proj", "column"),
        ("v_proj", "column"),
        ("ff_proj", "column"),
        ("up_proj", "column"),
        ("attn_out", "row"),
        ("ff_out", "row"),
    ],
)
def test_protected_axis_follows_residual_data_flow(module_name, axis):
    assert protected_axis(module_name) == axis


def test_input_column_restores_and_compensates_in_same_rows():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1
    baseline = _baseline_mask()

    delta = build_channel_delta("q_proj", weight, weight.clone(), baseline, channel=0)
    constrained = apply_delta_to_mask(baseline, delta)

    assert not constrained[:, 0].any()
    assert torch.equal(constrained.sum(1), baseline.sum(1))
    assert delta["restore_indices"].tolist() == [0, 6, 12, 18]
    assert delta["compensation_indices"].tolist() == [3, 7, 14, 19]
    assert delta["restore_values"].tolist() == [1, 7, 13, 19]


def test_output_row_restores_and_compensates_only_in_other_rows():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1
    score = weight.flip(0).clone()
    baseline = _baseline_mask()

    delta = build_channel_delta("attn_out", weight, score, baseline, channel=1)
    constrained = apply_delta_to_mask(baseline, delta)

    assert not constrained[1].any()
    assert constrained.sum() == baseline.sum()
    assert delta["restore_indices"].tolist() == [6, 8, 9]
    assert delta["compensation_indices"].tolist() == [19, 20, 22]


def test_stats_count_only_baseline_pruned_protected_weights_as_intervention():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1
    baseline = _baseline_mask()

    delta = build_channel_delta("ff_out", weight, weight.clone(), baseline, channel=1)
    stats = delta["stats"]

    assert stats == {
        "total_weights": 24,
        "baseline_pruned": 12,
        "constrained_pruned": 12,
        "baseline_sparsity": 0.5,
        "constrained_sparsity": 0.5,
        "protected_total": 6,
        "protected_already_survived_in_baseline": 3,
        "protected_already_survived_fraction": 0.5,
        "protected_restored": 3,
        "protected_restored_fraction": 0.5,
        "compensation_pruned": 3,
        "mask_difference_count": 6,
        "mask_difference_fraction": 0.25,
    }


def test_constraint_fails_instead_of_expanding_compensation_scope():
    weight = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 1
    baseline = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)

    with pytest.raises(ValueError, match="compensation"):
        build_channel_delta("attn_out", weight, weight.clone(), baseline, channel=0)
