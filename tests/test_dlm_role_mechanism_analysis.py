import numpy as np

from experiments.dlm_role_mechanism_analysis.core import (
    allocation_difference, marginal_cost, pair_order_summary, percentile_contrast, pooled_curves,
)


def test_pooling_preserves_role_ratios_and_energy_weighted_aggregate():
    fields = {
        "num_masked": np.array([[[2., 4.], [6., 8.]]]),
        "den_masked": np.array([[[2., 2.], [2., 2.]]]),
        "num_unmasked": np.array([[[9., 12.], [15., 18.]]]),
        "den_unmasked": np.array([[[3., 3.], [3., 3.]]]),
    }
    result = pooled_curves(fields, [0, 1])
    np.testing.assert_allclose(result["masked"], [[2., 3.]])
    np.testing.assert_allclose(result["unmasked"], [[4., 5.]])
    np.testing.assert_allclose(result["aggregate"], [[3.2, 4.2]])
    np.testing.assert_allclose(result["alpha_masked"], [[.4, .4]])


def test_rank_contrast_detects_reversal_without_scale_dependence():
    masked = np.array([[1., 3.], [2., 2.], [3., 1.]])
    unmasked = np.array([[30., 10.], [20., 20.], [10., 30.]])
    contrast = percentile_contrast(masked, unmasked)
    np.testing.assert_allclose(contrast[:, 0], [-1., 0., 1.])
    assert pair_order_summary(masked[:, 0], unmasked[:, 0])["rank_reversal_fraction"] == 1.0


def test_marginal_and_nested_mask_xor_use_parameter_count():
    curves = np.array([[1., 3., 7.], [2., 4., 8.]])
    costs = marginal_cost(curves, [(2, 10), (1, 20)], [.5, .6, .8])
    np.testing.assert_allclose(costs, [[1., 1.], [1., 1.]])
    left = {"levels": [0, 2]}
    right = {"levels": [1, 1]}
    result = allocation_difference(left, right, [(2, 10), (1, 20)], [.5, .6, .8])
    assert result["changed_projections"] == 2
    assert result["xor_pruned_weights"] == 6
