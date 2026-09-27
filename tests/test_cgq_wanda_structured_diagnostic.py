import pytest
import torch

from experiments.cgq_wanda_structured_diagnostic.core import (
    aggregate_by_timestep,
    attention_head_scores,
    mlp_neuron_scores,
    ratio_deviation_attribution,
    select_components,
    structured_statistics,
)


def test_mlp_neuron_score_uses_two_incoming_rows_and_outgoing_column():
    weights = {
        "ff_proj": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "up_proj": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        "ff_out": torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
    }
    uniform = {name: torch.ones(weight.shape[1]) for name, weight in weights.items()}
    cgq = {
        "ff_proj": torch.tensor([4.0, 9.0]),
        "up_proj": torch.tensor([4.0, 9.0]),
        "ff_out": torch.tensor([16.0, 25.0]),
    }

    result = mlp_neuron_scores(weights, uniform, cgq)

    assert torch.equal(result["uniform"]["ff_proj"], torch.tensor([3.0, 7.0]))
    assert torch.equal(result["uniform"]["up_proj"], torch.tensor([11.0, 15.0]))
    assert torch.equal(result["uniform"]["ff_out"], torch.tensor([20.0, 22.0]))
    assert torch.equal(result["uniform"]["total"], torch.tensor([34.0, 44.0]))
    assert torch.equal(result["cgq"]["total"], torch.tensor([116.0, 166.0]))


def test_attention_head_score_uses_qkv_rows_and_output_columns():
    weights = {
        "q_proj": torch.ones(4, 4),
        "k_proj": torch.ones(4, 4) * 2,
        "v_proj": torch.ones(4, 4) * 3,
        "attn_out": torch.ones(4, 4) * 4,
    }
    uniform = {name: torch.ones(4) for name in weights}
    cgq = {name: torch.ones(4) * 4 for name in weights}

    result = attention_head_scores(weights, uniform, cgq, num_heads=2, head_dim=2)

    assert torch.equal(result["uniform"]["q_proj"], torch.tensor([8.0, 8.0]))
    assert torch.equal(result["uniform"]["k_proj"], torch.tensor([16.0, 16.0]))
    assert torch.equal(result["uniform"]["v_proj"], torch.tensor([24.0, 24.0]))
    assert torch.equal(result["uniform"]["attn_out"], torch.tensor([32.0, 32.0]))
    assert torch.equal(result["uniform"]["total"], torch.tensor([80.0, 80.0]))
    assert torch.equal(result["cgq"]["total"], torch.tensor([160.0, 160.0]))


def test_structured_statistics_reports_rank_movement_and_boundary_stability():
    uniform = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    cgq = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])

    result = structured_statistics(uniform, cgq, fractions=(0.2, 0.4))

    assert result["rank_displacement"]["mean_absolute"] == pytest.approx(1.2)
    assert result["rank_displacement"]["mean_absolute_normalized"] == pytest.approx(0.24)
    assert result["rank_sets"]["top_20"]["intersection"] == 0
    assert result["rank_sets"]["top_20"]["crossed_boundary"] == 2
    assert result["rank_sets"]["bottom_20"]["intersection"] == 1


def test_aggregate_by_timestep_preserves_additive_sufficient_statistics():
    state_values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
    timestep_indices = torch.tensor([0, 0, 1])

    result = aggregate_by_timestep(state_values, timestep_indices, timestep_count=2)

    assert torch.equal(result, torch.tensor([[4.0, 6.0], [10.0, 20.0]]))


def test_select_components_excludes_unrelated_layer_statistics():
    values = {"q_proj": 1, "ff_proj": 2, "up_proj": 3, "ff_out": 4}

    result = select_components(values, ("ff_proj", "up_proj", "ff_out"))

    assert result == {"ff_proj": 2, "up_proj": 3, "ff_out": 4}


def test_ratio_deviation_attribution_removes_common_layer_scaling():
    groups = {
        "uniform": {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([1.0, 1.0]), "total": torch.tensor([2.0, 2.0])},
        "cgq": {"a": torch.tensor([3.0, 1.0]), "b": torch.tensor([1.0, 1.0]), "total": torch.tensor([4.0, 2.0])},
    }

    result = ratio_deviation_attribution(groups)

    assert torch.equal(result["a"], torch.tensor([0.5, -0.5]))
    assert torch.equal(result["b"], torch.tensor([0.0, 0.0]))
    assert torch.equal(result["a"] + result["b"], torch.tensor([0.5, -0.5]))
