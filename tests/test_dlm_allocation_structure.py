"""Tests for DLM projection-allocation structure diagnostics."""
import unittest

import numpy as np

from experiments.dlm_allocation_structure.core import (
    allocation_from_marginals,
    allocation_from_per_parameter_costs,
    apply_temporal_models,
    bootstrap_mean_ci,
    compute_marginals,
    fit_reconstruction_residual,
    fit_static_models,
    fit_temporal_models,
    nested_mask_xor,
    prediction_metrics,
    selected_additive_damage,
)


class AllocationStructureTest(unittest.TestCase):
    def test_marginals_preserve_negative_cost_and_parameter_normalization(self):
        damage = np.array([[[1.0, 3.0, 2.0]], [[2.0, 6.0, 10.0]]])
        reconstruction = damage * 2
        out = compute_marginals(damage, reconstruction, [(2, 20), (4, 20)],
                                grid=(.50, .55, .60))
        np.testing.assert_allclose(out["functional"], [[[2.0, -1.0]], [[4.0, 4.0]]])
        np.testing.assert_allclose(out["reconstruction"], out["functional"] * 2)
        np.testing.assert_allclose(out["functional_per_parameter"],
                                   [[[1.0, -.5]], [[1.0, 1.0]]])

    def test_static_models_recover_balanced_depth_and_type_effects(self):
        layers = np.array([0, 0, 1, 1])
        types = np.array(["q", "v", "q", "v"])
        depth = np.array([1.0, 3.0])
        kind = {"q": 10.0, "v": 20.0}
        mean = np.array([[depth[l] + kind[t]] for l, t in zip(layers, types)])
        values = np.repeat(mean[:, None, :], 3, axis=1)
        models = fit_static_models(values, layers, types)
        np.testing.assert_allclose(models["depth_type"], mean)
        np.testing.assert_allclose(models["projection"], mean)
        self.assertFalse(np.allclose(models["depth"], mean))
        self.assertFalse(np.allclose(models["type"], mean))

    def test_temporal_type_interaction_is_recovered_without_test_states(self):
        layers = np.array([0, 0, 1, 1])
        types = np.array(["q", "v", "q", "v"])
        timesteps = np.array([.1, .9, .1, .9])
        values = np.empty((4, 4, 1))
        for m, typ in enumerate(types):
            for s, timestep in enumerate(timesteps):
                values[m, s, 0] = 1.0 + (4.0 if typ == "v" and timestep == .9 else 0.0)
        fit = fit_temporal_models(values, timesteps, layers, types)
        predicted = apply_temporal_models(fit, timesteps)
        structured_error = np.mean((predicted["structured_interactions"] -
                                    fit["normalized_train"]) ** 2)
        static_error = np.mean((predicted["static_projection_time"] -
                                fit["normalized_train"]) ** 2)
        self.assertLess(structured_error, static_error)
        np.testing.assert_allclose(predicted["projection_time"], fit["normalized_train"])

    def test_temporal_normalization_is_frozen_from_construction_states(self):
        layers = np.array([0, 1])
        types = np.array(["q", "q"])
        train = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        timesteps = np.array([.1, .9])
        fit = fit_temporal_models(train, timesteps, layers, types)
        before = fit["rms_by_timestep"].copy()
        test = train * 1000
        _ = apply_temporal_models(fit, timesteps, values=test)
        np.testing.assert_array_equal(fit["rms_by_timestep"], before)

    def test_temporal_fit_can_preserve_pre_normalized_residual_scale(self):
        layers = np.array([0, 1])
        types = np.array(["q", "q"])
        values = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
        fit = fit_temporal_models(values, np.array([.1, .9]), layers, types,
                                  normalize=False)
        np.testing.assert_array_equal(fit["normalized_train"], values)
        np.testing.assert_array_equal(fit["rms_by_timestep"], np.ones((2, 1)))

    def test_reconstruction_residual_removes_train_fitted_linear_signal(self):
        reconstruction_train = np.arange(1, 13, dtype=float).reshape(2, 3, 2)
        functional_train = 3.0 * reconstruction_train + np.array([2.0, -1.0])
        reconstruction_test = reconstruction_train + 20.0
        functional_test = 3.0 * reconstruction_test + np.array([2.0, -1.0])
        out = fit_reconstruction_residual(functional_train, reconstruction_train,
                                          functional_test, reconstruction_test)
        np.testing.assert_allclose(out["train_residual"], 0.0, atol=1e-12)
        np.testing.assert_allclose(out["test_residual"], 0.0, atol=1e-12)
        np.testing.assert_allclose(out["slopes"], [3.0, 3.0])

    def test_allocation_from_marginals_hits_exact_row_floor_uniform_budget(self):
        marginals = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=float)
        result = allocation_from_marginals(marginals, [(2, 20), (4, 20)])
        self.assertEqual(result["budget_error"], 0)
        self.assertEqual(result["pruned"], result["uniform_pruned"])

    def test_per_parameter_cost_is_converted_back_to_raw_marginal_for_allocator(self):
        costs = np.ones((2, 5), dtype=float)
        result = allocation_from_per_parameter_costs(costs, [(2, 20), (4, 20)])
        self.assertEqual(result["budget_error"], 0)
        self.assertTrue(all(abs(row["cost_per_nominal_parameter"] - 1.0) < 1e-12
                            for row in result["trace"]))

    def test_bootstrap_mean_ci_is_reproducible(self):
        a = bootstrap_mean_ci(np.array([-2.0, -1.0, 0.0, 1.0]), 1000, 7)
        b = bootstrap_mean_ci(np.array([-2.0, -1.0, 0.0, 1.0]), 1000, 7)
        self.assertEqual(a, b)
        self.assertEqual(a["count"], 4)

    def test_prediction_metrics_and_selected_damage_use_all_increments(self):
        truth = np.array([[1.0, 2.0], [3.0, 4.0]])
        predicted = truth.copy()
        metrics = prediction_metrics(predicted, truth)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertEqual(selected_additive_damage(truth, [0, 2]), 7.0)

    def test_nested_mask_xor_uses_exact_row_floor_counts(self):
        entries = [
            {"weights": 40, "masks": [{"pruned": 20}, {"pruned": 22}]},
            {"weights": 80, "masks": [{"pruned": 40}, {"pruned": 44}]},
        ]
        out = nested_mask_xor([0, 1], [1, 0], entries)
        self.assertEqual(out["xor_weights"], 6)
        self.assertEqual(out["total_weights"], 120)
        self.assertEqual(out["xor_fraction"], .05)


if __name__ == "__main__":
    unittest.main()
