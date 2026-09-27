import numpy as np
import torch

from experiments.dlm_role_proxy_aggregation_65.core import (
    aggregate_role_curves, allocation_difference, diagonal_reconstruction_curve,
    selected_signature,
)


def test_max_level_and_max_marginal_are_distinct():
    masked = np.array([[1., 5., 6., 10., 11., 15.]])
    unmasked = np.array([[4., 4.5, 8., 8.5, 12., 12.5]])
    level = aggregate_role_curves(masked, unmasked, "max_level")
    marginal = aggregate_role_curves(masked, unmasked, "max_marginal")
    assert np.allclose(level, [[4., 5., 8., 10., 12., 15.]])
    assert np.allclose(np.diff(marginal), np.maximum(np.diff(masked), np.diff(unmasked)))
    assert not np.allclose(np.diff(level), np.diff(marginal))


def test_balanced_is_role_equal_not_token_count_weighted():
    masked = np.ones((2, 6))
    unmasked = np.full((2, 6), 3.)
    assert np.allclose(aggregate_role_curves(masked, unmasked, "balanced"), 2.)


def test_diagonal_proxy_matches_explicit_for_one_hot_inputs():
    weight = torch.tensor([[1., 2.], [3., 4.]])
    mask = torch.tensor([[True, False], [False, True]])
    x = torch.eye(2)
    delta = weight * mask
    exact = (x @ delta.T).square().sum().item()
    curve = diagonal_reconstruction_curve(weight, [mask], x.square().sum(0), 2.)
    assert np.allclose(curve, [exact / 2.])


def test_manifest_signature_uses_selected_masks():
    def manifest(digest):
        return {"entries": [{"name": "m", "selected_mask": {"mask_sha256": digest, "pruned": 2}}]}
    assert selected_signature(manifest("a")) == selected_signature(manifest("a"))
    assert selected_signature(manifest("a")) != selected_signature(manifest("b"))


def test_allocation_difference_uses_exact_persisted_row_floor_counts():
    def manifest(level, pruned):
        return {"entries": [{"name": "m", "level": level, "weights": 101,
                             "selected_mask": {"mask_sha256": str(level), "pruned": pruned}}]}
    result = allocation_difference(manifest(0, 50), manifest(1, 55))
    assert result["mask_xor_weights"] == 5
    assert result["mask_xor_fraction_of_prunable_weights"] == 5 / 101
