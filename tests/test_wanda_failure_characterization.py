import pytest
import torch

from experiments.wanda_failure_characterization.core import (
    rowwise_wanda_mask,
    masked_linear_variants,
    reconstruction_metrics,
    threshold_geometry,
)
from experiments.wanda_failure_characterization.propagation_core import (
    tensor_pair_metrics, propagation_gain, matched_perturbation, factorial_contrasts,
    cyclic_donor_map, same_timestep_donor_map, factorize_token_feature,
    token_feature_factorial_contrasts, token_row_pairing_contrasts,
    class_preserving_permutation,
)


def test_rowwise_wanda_mask_prunes_exact_fraction_per_row():
    scores = torch.tensor([[4., 1., 3., 2.], [1., 4., 2., 3.]])
    mask = rowwise_wanda_mask(scores, 0.5)
    assert torch.equal(mask, torch.tensor([[False, True, False, True], [True, False, True, False]]))


def test_masked_linear_variants_preserves_sham_and_applies_each_mask():
    x = torch.tensor([[[1., 2.]], [[1., 2.]], [[1., 2.]]])
    w = torch.tensor([[3., 4.]])
    masks = [None, torch.tensor([[True, False]]), torch.tensor([[False, True]])]
    out = masked_linear_variants(x, w, None, masks)
    assert torch.equal(out[:, 0, 0], torch.tensor([11., 8., 3.]))


def test_reconstruction_metrics_are_zero_for_identical_outputs():
    y = torch.tensor([[1., 2.], [3., 4.]])
    result = reconstruction_metrics(y, y)
    assert result["relative_squared_error"] == 0
    assert result["relative_l2_error"] == 0
    assert result["cosine"] == pytest.approx(1)


def test_threshold_geometry_reports_gap_and_near_threshold_mass():
    scores = torch.tensor([[1., 2., 3., 4.]])
    result = threshold_geometry(scores, 0.5)
    assert result["threshold_mean"] == 2
    assert result["gap_mean"] == 1
    assert result["normalized_gap_mean"] == 0.5


def test_tensor_pair_metrics_uses_sham_denominator_and_mask_localization():
    sham = torch.ones(2, 2)
    variant = sham.clone()
    variant[0] += 1
    result = tensor_pair_metrics(sham, variant, torch.tensor([True, False]))
    assert result["abs_energy"] == pytest.approx(0.5)
    assert result["relative_energy"] == pytest.approx(0.5)
    assert result["relative_l2"] == pytest.approx(2 ** -0.5)
    assert result["masked_energy_fraction"] == pytest.approx(1)
    assert result["masked_enrichment"] == pytest.approx(2)
    assert result["aligned_energy_fraction"] == pytest.approx(0.5)
    assert result["orthogonal_energy_fraction"] == pytest.approx(0.5)


def test_propagation_gain_is_safe_for_zero_baseline():
    gain, floored = propagation_gain(torch.tensor(0.2), torch.tensor(0.0), eps=1e-6)
    assert gain.item() == pytest.approx(200000)
    assert floored.item() is True


def test_matched_perturbation_hits_relative_geometric_mean_norm():
    delta30 = torch.tensor([[3.0, 4.0]])
    delta31 = torch.tensor([[0.0, 12.0]])
    h30 = torch.tensor([[6.0, 8.0]])
    h31 = torch.tensor([[0.0, 30.0]])
    tau = (0.5 * 0.4) ** 0.5
    injected, achieved = matched_perturbation(delta30, h31, tau)
    assert achieved.item() == pytest.approx(tau)
    assert torch.linalg.vector_norm(injected).item() == pytest.approx(tau * 30)


def test_factorial_contrasts_match_preregistered_difference_in_differences():
    cells = torch.tensor([[1.0, 3.0], [2.0, 8.0]])  # direction x location
    result = factorial_contrasts(cells)
    assert result["location_D30"].item() == 2
    assert result["location_D31"].item() == 6
    assert result["direction_L30"].item() == 1
    assert result["direction_L31"].item() == 5
    assert result["location_main"].item() == 4
    assert result["direction_main"].item() == 3
    assert result["interaction"].item() == 4
    batched = factorial_contrasts(torch.stack((cells, cells * 2)))
    assert torch.equal(batched["interaction"], torch.tensor([4.0, 8.0]))


def test_state_swap_donor_maps_are_deterministic_and_same_timestep():
    sequence = [8, 8, 9, 9]
    timestep = [0, 1, 0, 1]
    assert cyclic_donor_map(4) == [1, 2, 3, 0]
    assert same_timestep_donor_map(sequence, timestep) == [2, 3, 0, 1]


def test_token_feature_factorization_reconstructs_direction():
    delta = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    allocation, direction = factorize_token_feature(delta, eps=1e-12)
    reconstructed = allocation[:, None] * direction
    assert torch.allclose(allocation.norm(), torch.tensor(1.0))
    assert torch.allclose(reconstructed, delta / delta.norm())


def test_token_feature_factorial_contrasts():
    cells = torch.tensor([10.0, 6.0, 4.0, 2.0])  # NN, NF, FN, FF
    effects = token_feature_factorial_contrasts(cells)
    assert torch.equal(effects["token_main"], torch.tensor(5.0))
    assert torch.equal(effects["feature_main"], torch.tensor(3.0))
    assert torch.equal(effects["interaction"], torch.tensor(2.0))
    assert torch.equal(effects["native_excess"], torch.tensor(4.0))


def test_token_row_pairing_contrasts():
    cells = torch.tensor([10.0, 4.0, 7.0, 6.0])  # NN, NS, SN, SS
    effects = token_row_pairing_contrasts(cells)
    assert torch.equal(effects["pairing"], torch.tensor(2.5))
    assert torch.equal(effects["position"], torch.tensor(4.0))
    assert torch.equal(effects["feature_placement"], torch.tensor(3.0))


def test_class_preserving_permutation_rotates_each_class_independently():
    mask = torch.tensor([True, False, True, False, False])
    permutation = class_preserving_permutation(mask, 1)
    assert torch.equal(permutation, torch.tensor([2, 3, 0, 4, 1]))
    assert torch.equal(mask[permutation], mask)
