import pytest
import torch

from experiments.mlp_neuron_redundancy.core import (
    ablate_variant_rows,
    distribution_summary,
    fixed_candidates,
    masked_losses_and_kl,
    pairwise_stability,
    rank_predictability,
    sham_corrected_losses,
)


def test_fixed_candidates_are_reproducible_unique_and_layer_stratified():
    first = fixed_candidates([0, 4], hidden_size=20, per_layer=4, seed=17)
    second = fixed_candidates([0, 4], hidden_size=20, per_layer=4, seed=17)
    assert first == second
    assert set(first) == {0, 4}
    assert all(len(values) == len(set(values)) == 4 for values in first.values())


def test_ablate_variant_rows_preserves_sham_and_zeros_one_neuron_per_variant():
    hidden = torch.arange(2 * 3 * 4.0).reshape(2, 3, 4)
    original = hidden.clone()
    result = ablate_variant_rows(hidden, [None, 2])
    assert torch.equal(result[0], original[0])
    assert torch.equal(result[1, :, 2], torch.zeros(3))
    assert torch.equal(result[1, :, [0, 1, 3]], original[1, :, [0, 1, 3]])
    assert torch.equal(hidden, original)


def test_masked_losses_and_kl_match_dense_reference():
    dense = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    variants = torch.cat([dense, torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])])
    targets = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])
    losses, kl = masked_losses_and_kl(variants, dense, targets, mask, p_mask=0.5)
    assert losses[0].item() == pytest.approx(torch.log1p(torch.exp(torch.tensor(-2.0))).item())
    assert kl[0].item() == pytest.approx(0.0, abs=1e-8)
    assert losses[1] > losses[0]
    assert kl[1] > 0


def test_sham_corrected_losses_remove_common_batch_drift():
    losses = torch.tensor([10.2, 10.5, 10.1])
    delta = sham_corrected_losses(losses)
    assert torch.allclose(delta, torch.tensor([0.3, -0.1]), atol=1e-6)


def test_distribution_summary_reports_tail_ratios_and_gini():
    result = distribution_summary(torch.tensor([1.0, 1.0, 1.0, 9.0]))
    assert result["mean"] == pytest.approx(3.0)
    assert result["gini"] == pytest.approx(0.5)
    assert result["p90_over_p10"] > 1


def test_pairwise_stability_reports_rank_and_bottom_sets():
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 4.0, 3.0]])
    result = pairwise_stability(scores, fractions=(0.25, 0.5))
    assert result["spearman"]["median"] == pytest.approx(0.8)
    assert result["bottom_25"]["jaccard"]["median"] == 1.0


def test_rank_predictability_uses_within_layer_ranks():
    baseline = torch.tensor([1.0, 2.0, 10.0, 20.0], requires_grad=True)
    exact = torch.tensor([2.0, 1.0, 20.0, 10.0])
    layers = torch.tensor([0, 0, 1, 1])
    result = rank_predictability(baseline, exact, layers, fractions=(0.5,))
    assert result["spearman"] == pytest.approx(-1.0)
    assert result["bottom_50"]["overlap"] == 0.0
