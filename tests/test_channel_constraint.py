import copy
import json

import pytest
import torch
from torch import nn

from lib.channel_constraint import (
    apply_channel_delta,
    apply_delta_to_mask,
    build_channel_delta,
    new_delta_bundles,
    protected_axis,
    record_module_deltas,
    save_delta_bundles,
    select_experiment_channels,
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


def test_seeded_channels_are_distinct_and_exclude_protected_channel():
    selected = select_experiment_channels(6, protected=4, random_count=3, seed=7)

    assert selected == select_experiment_channels(6, protected=4, random_count=3, seed=7)
    assert selected[0] == 4
    assert len(selected) == len(set(selected)) == 4
    assert 4 not in selected[1:]


def test_variants_are_independent_branches_from_baseline():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1
    score = weight.clone()
    baseline = _baseline_mask()
    module_name = "model.transformer.blocks.0.q_proj"
    channels = select_experiment_channels(6, protected=4, random_count=2, seed=7)
    bundles = new_delta_bundles(channels)

    record_module_deltas(bundles, module_name, weight, score, baseline)

    constrained = {
        channel: apply_delta_to_mask(baseline, bundles[channel]["modules"][module_name])
        for channel in channels
    }
    for channel, mask in constrained.items():
        assert not mask[:, channel].any()
        assert torch.equal(mask.sum(1), baseline.sum(1))
    accumulated = apply_delta_to_mask(constrained[channels[0]], bundles[channels[1]]["modules"][module_name])
    assert not torch.equal(constrained[channels[1]], accumulated)


class _ToyModel(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.blocks = nn.ModuleList([nn.Module()])
        self.model.transformer.blocks[0].q_proj = nn.Linear(6, 4, bias=False)
        self.model.transformer.blocks[0].q_proj.weight.data.copy_(weight)


def test_delta_artifact_round_trip_applies_to_fresh_baseline(tmp_path):
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) + 1
    score = weight.clone()
    baseline_mask = _baseline_mask()
    module_name = "model.transformer.blocks.0.q_proj"
    bundles = new_delta_bundles([0, 4])
    record_module_deltas(bundles, module_name, weight, score, baseline_mask)
    paths = save_delta_bundles(bundles, tmp_path, {"method": "wanda"})
    baseline_model = _ToyModel(weight)
    baseline_model.model.transformer.blocks[0].q_proj.weight.data[baseline_mask] = 0

    variant = copy.deepcopy(baseline_model)
    report = apply_channel_delta(variant, tmp_path / "channel-0.pt")
    variant_weight = variant.model.transformer.blocks[0].q_proj.weight.data.flatten()
    delta = bundles[0]["modules"][module_name]

    assert {path.name for path in paths} == {"channel-0.pt", "channel-4.pt"}
    assert json.loads((tmp_path / "channel-0.json").read_text())["method"] == "wanda"
    assert torch.equal(variant_weight[delta["restore_indices"]], delta["restore_values"])
    assert not variant_weight[delta["compensation_indices"]].any()
    assert report["actual_global_sparsity"] == 0.5

    other_variant = copy.deepcopy(baseline_model)
    apply_channel_delta(other_variant, tmp_path / "channel-4.pt")
    other_weight = other_variant.model.transformer.blocks[0].q_proj.weight.data
    assert not other_weight[:, 4].eq(0).any()
    assert torch.equal(other_weight[:, 0].eq(0), apply_delta_to_mask(baseline_mask, bundles[4]["modules"][module_name])[:, 0])
