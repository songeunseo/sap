import math

import pytest
import torch

from experiments.mlp_token_contrast_diagnostic.core import (
    abs_scores,
    contrast_scores,
    gate_directional_derivative,
    mirror_from_uniform_contrast,
    rank_comparison,
    ratio_summary,
    symmetry_error,
    stability_summary,
    timestep_scores,
)


def test_gate_directional_derivative_sums_positions_in_fp32():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, -4.0]]], dtype=torch.bfloat16)
    gradient = torch.tensor([[[0.5, -1.0], [2.0, 0.25]]], dtype=torch.bfloat16)

    result = gate_directional_derivative(hidden, gradient)

    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([-6.5, 3.0]))


def test_symmetry_error_measures_independently_collected_derivatives():
    uniform = torch.tensor([1.0, -2.0])
    reveal = torch.tensor([1.25, -1.5])
    remain = torch.tensor([0.75, -2.5])
    exact = symmetry_error(uniform, reveal, remain)
    perturbed = symmetry_error(uniform, reveal, remain + torch.tensor([0.01, 0.0]))

    assert exact["max_absolute"] == 0.0
    assert perturbed["max_absolute"] == pytest.approx(0.01)
    assert perturbed["max_relative"] > 0


def test_mirror_from_uniform_contrast_preserves_exact_linear_identity():
    uniform = torch.tensor([1.0, -2.0, 3.0])
    contrast = torch.tensor([0.25, 0.5, -1.0])

    reveal, remain = mirror_from_uniform_contrast(uniform, contrast)

    assert torch.equal(reveal, torch.tensor([1.25, -1.5, 2.0]))
    assert torch.equal(remain, torch.tensor([0.75, -2.5, 4.0]))
    assert symmetry_error(uniform, reveal, remain)["max_absolute"] == 0.0


def test_abs_and_contrast_scores_use_abs_across_states():
    uniform = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])
    reveal = torch.tensor([[3.0, -3.0], [-1.0, 5.0]])
    remain = torch.tensor([[1.0, 1.0], [-3.0, 1.0]])

    scores = abs_scores(uniform, reveal, remain)
    contrast = contrast_scores(uniform, reveal, remain)

    assert torch.equal(scores["uniform"], torch.tensor([2.0, 2.0]))
    assert torch.equal(scores["reveal"], torch.tensor([2.0, 4.0]))
    assert torch.equal(scores["remain"], torch.tensor([2.0, 1.0]))
    assert torch.equal(contrast["signed"], torch.tensor([[1.0, -2.0], [1.0, 2.0]]))
    assert torch.equal(contrast["score"], torch.tensor([1.0, 2.0]))


def test_ratio_summary_excludes_near_zero_baselines_and_reports_tail_share():
    baseline = torch.tensor([0.0, 1.0, 2.0, 4.0])
    contrast = torch.tensor([100.0, 0.5, 2.0, 8.0])

    result = ratio_summary(contrast, baseline, relative_floor=0.25)

    assert result["valid_count"] == 3
    assert result["excluded_count"] == 1
    assert result["median"] == pytest.approx(1.0)
    assert result["tail_fraction_from_excluded"] == pytest.approx(100 / 110.5)


def test_rank_comparison_reports_displacement_and_boundary_stability():
    left = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    right = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])

    result = rank_comparison(left, right, fractions=(0.2, 0.4))

    assert result["spearman"] == pytest.approx(0.8)
    assert result["rank_displacement"]["mean_absolute"] == pytest.approx(0.8)
    assert result["sets"]["top_20"]["crossed_boundary"] == 2
    assert result["sets"]["bottom_20"]["crossed_boundary"] == 0


def test_timestep_scores_group_eight_states_per_timestep():
    values = torch.arange(20.0).reshape(10, 2)
    timestep_index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    result = timestep_scores(values, timestep_index)

    assert set(result) == {0, 1, 2, 3, 4}
    assert torch.equal(result[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(result[4], torch.tensor([17.0, 18.0]))


def test_stability_summary_compares_each_group_with_global_ranking():
    global_score = torch.tensor([1.0, 2.0, 3.0, 4.0])
    grouped = torch.stack([global_score, torch.tensor([1.0, 2.0, 4.0, 3.0])])

    result = stability_summary(grouped, global_score, top_fraction=0.25)

    assert result["spearman"]["mean"] == pytest.approx(0.9)
    assert result["top_overlap"]["mean"] == pytest.approx(0.5)


def test_ratio_summary_rejects_nonpositive_floor():
    with pytest.raises(ValueError):
        ratio_summary(torch.ones(2), torch.ones(2), relative_floor=0.0)
