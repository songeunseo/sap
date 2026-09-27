import pytest
import torch

from experiments.cgq_wanda_diagnostic.core import (
    ActivationAccumulator,
    cgq_factors,
    feature_comparison,
    masked_confidence_deciles,
)


def test_cgq_factors_apply_masked_and_unmasked_bases():
    confidence = torch.tensor([[0.0, 0.25, 1.0]])
    masked = torch.tensor([[True, False, True]])

    actual = cgq_factors(confidence, masked)

    assert torch.allclose(actual, torch.tensor([[1.0, 1.2, 2.0]]))


def test_activation_accumulator_matches_wrapped_gpt_batch_normalization():
    accumulator = ActivationAccumulator(columns=2)
    first = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    second = torch.tensor([[[2.0, 1.0], [0.0, 3.0]]])
    first_r = torch.tensor([[1.0, 2.0]])
    second_r = torch.tensor([[0.5, 1.5]])

    accumulator.add_batch(first, first_r)
    accumulator.add_batch(second, second_r)

    assert accumulator.nsamples == 2
    assert torch.allclose(accumulator.uniform, torch.tensor([7.0, 15.0]))
    assert torch.allclose(accumulator.cgq, torch.tensor([19.0, 44.25]))
    assert torch.allclose(
        accumulator.token_norms,
        torch.tensor([[5.0**0.5, 5.0], [5.0**0.5, 3.0]]),
    )


def test_feature_comparison_detects_uniform_scaling():
    uniform = torch.tensor([1.0, 4.0, 9.0, 16.0])
    weighted = uniform * 3.0

    result = feature_comparison(uniform, weighted)

    assert result["spearman"] == pytest.approx(1.0)
    assert result["cosine_similarity"] == pytest.approx(1.0)
    assert abs(result["ratio"]["mean"] - 3.0) < 1e-12
    assert result["ratio"]["cv"] == 0.0
    assert abs(result["relative_l2_difference"] - 2.0) < 1e-12


def test_masked_confidence_deciles_partition_ties_deterministically():
    confidence = torch.tensor([0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    energy = torch.arange(1, 11, dtype=torch.float64)
    weighted_energy = energy * 2
    activation_norm = torch.arange(11, 21, dtype=torch.float64)

    rows = masked_confidence_deciles(confidence, energy, weighted_energy, activation_norm)

    assert [row["token_count"] for row in rows] == [1] * 10
    assert [row["decile"] for row in rows] == list(range(1, 11))
    assert rows[0]["mean_confidence"] == pytest.approx(0.1)
    assert rows[0]["mean_activation_norm"] == 11.0
    assert abs(sum(row["uniform_energy_fraction"] for row in rows) - 1.0) < 1e-12
    assert abs(sum(row["cgq_energy_fraction"] for row in rows) - 1.0) < 1e-12
