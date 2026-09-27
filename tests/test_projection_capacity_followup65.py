"""Regression tests for the post-oracle allocation diagnostics."""
import unittest

import numpy as np

from experiments.projection_capacity_followup_65.queue_control import exact_session_exists

from experiments.projection_capacity_followup_65.core import (
    additive_damage,
    build_selected_manifest,
    eis_type_control,
    fit_anchor_curve,
    interval_overlaps,
    paired_comparison,
    paired_binary_comparison,
    summarize_curve_records,
)


class FollowupAllocationTest(unittest.TestCase):
    def test_tmux_session_match_is_exact_not_prefix(self):
        names = ["projection_capacity_followup65_downstream"]
        self.assertFalse(exact_session_exists(names, "projection_capacity_followup65"))
        self.assertTrue(exact_session_exists(names, "projection_capacity_followup65_downstream"))

    def test_curve_summary_averages_each_state_once(self):
        records = [
            {
                "name": "block_00.q_proj",
                "shape": [2, 20],
                "curves": [
                    {"sparsity": 0.50, "per_state": [
                        {"sequence_index": 0, "timestep": 0.05, "mean_kl": 1.0,
                         "local_reconstruction_error": 2.0},
                        {"sequence_index": 1, "timestep": 0.05, "mean_kl": 3.0,
                         "local_reconstruction_error": 6.0},
                    ]},
                    {"sparsity": 0.55, "per_state": [
                        {"sequence_index": 0, "timestep": 0.05, "mean_kl": 5.0,
                         "local_reconstruction_error": 10.0},
                        {"sequence_index": 1, "timestep": 0.05, "mean_kl": 7.0,
                         "local_reconstruction_error": 14.0},
                    ]},
                ],
            }
        ]
        out = summarize_curve_records(records, grid=(0.50, 0.55))
        np.testing.assert_allclose(out["damage"], [[2.0, 6.0]])
        np.testing.assert_allclose(out["reconstruction"], [[4.0, 12.0]])
        self.assertEqual(out["state_keys"], [(0, 0.05), (1, 0.05)])

    def test_anchor_fit_uses_only_construction_states(self):
        # State 2 has a radically different ratio and must not leak into fitting.
        damage = np.array([[[2.0, 4.0], [4.0, 8.0], [1000.0, 1000.0]]])
        reconstruction = np.array([[[1.0, 2.0], [2.0, 4.0], [1.0, 1.0]]])
        predicted, alpha = fit_anchor_curve(
            damage, reconstruction, construction_state_indices=[0, 1], anchor_index=1
        )
        np.testing.assert_allclose(alpha, [2.0])
        np.testing.assert_allclose(predicted, [[3.0, 6.0]])

    def test_anchor_fit_rejects_zero_denominator(self):
        damage = np.ones((1, 2, 2))
        reconstruction = np.zeros((1, 2, 2))
        with self.assertRaisesRegex(ValueError, "zero anchor reconstruction"):
            fit_anchor_curve(damage, reconstruction, [0, 1], anchor_index=1)

    def test_eis_type_control_preserves_type_multiset_and_budget(self):
        names = [
            "block_01.q_proj", "block_00.q_proj",
            "block_01.ff_out", "block_00.ff_out",
        ]
        shapes = [(2, 20)] * 4
        oracle = [0.50, 0.75, 0.55, 0.70]
        control = eis_type_control(names, shapes, oracle)
        self.assertEqual(control, [0.50, 0.75, 0.55, 0.70])
        # Reorder the input to prove assignment follows layer, not repository order.
        names2 = [names[1], names[0], names[3], names[2]]
        oracle2 = [0.50, 0.75, 0.55, 0.70]
        control2 = eis_type_control(names2, shapes, oracle2)
        self.assertEqual(control2, [0.75, 0.50, 0.70, 0.55])

    def test_eis_type_control_rejects_shape_mismatch_within_type(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            eis_type_control(
                ["block_00.q_proj", "block_01.q_proj"],
                [(2, 20), (3, 20)],
                [0.50, 0.75],
            )

    def test_additive_damage_is_relative_to_each_projection_baseline(self):
        curves = np.array([[10.0, 11.0, 15.0], [20.0, 25.0, 40.0]])
        self.assertEqual(additive_damage(curves, [2, 1]), 10.0)

    def test_selected_manifest_uses_requested_grid_masks_and_sums_budget(self):
        entries = [
            {"name": "block_00.q_proj", "shape": [2, 20], "weights": 40,
             "masks": [{"sparsity": 0.50, "pruned": 20, "path": "a"},
                       {"sparsity": 0.55, "pruned": 22, "path": "b"}]},
            {"name": "block_01.q_proj", "shape": [2, 20], "weights": 40,
             "masks": [{"sparsity": 0.50, "pruned": 20, "path": "c"},
                       {"sparsity": 0.55, "pruned": 22, "path": "d"}]},
        ]
        manifest = build_selected_manifest("diagnostic", entries, [0.55, 0.50],
                                           grid=(0.50, 0.55))
        self.assertEqual(manifest["pruned"], 42)
        self.assertEqual(manifest["weights"], 80)
        self.assertEqual([row["selected_mask"]["path"] for row in manifest["entries"]],
                         ["b", "c"])
        self.assertNotIn("masks", manifest["entries"][0])

    def test_selected_manifest_rejects_off_grid_sparsity(self):
        with self.assertRaisesRegex(ValueError, "not on grid"):
            build_selected_manifest(
                "bad",
                [{"name": "block_00.q_proj", "shape": [1, 20], "weights": 20,
                  "masks": [{"sparsity": 0.50, "pruned": 10}]}],
                [0.60], grid=(0.50,),
            )

    def test_interval_overlap_detects_only_nonempty_intersection(self):
        existing = [{"start": 0, "end_exclusive": 10}, {"start": 20, "end_exclusive": 30}]
        candidates = [{"start": 10, "end_exclusive": 20},
                      {"start": 29, "end_exclusive": 40}]
        self.assertEqual(interval_overlaps(existing, candidates),
                         [{"existing": existing[1], "candidate": candidates[1]}])

    def test_paired_comparison_reports_state_sequence_and_timestep_consistency(self):
        reference = [
            {"sequence_index": s, "timestep": t, "mean_kl": 2.0}
            for s in range(2) for t in (0.1, 0.3)
        ]
        candidate = [dict(row, mean_kl=1.5 if row["sequence_index"] == 0 else 2.25)
                     for row in reference]
        result = paired_comparison(reference, candidate, resamples=1000, seed=7)
        self.assertEqual(result["mean_difference"], -0.125)
        self.assertEqual(result["states_improved"], 2)
        self.assertEqual(result["states_worsened"], 2)
        self.assertEqual(result["sequence_means_improved"], 1)
        self.assertEqual(result["timestep_means_improved"], 2)
        self.assertEqual(len(result["state_bootstrap_95_ci"]), 2)
        self.assertEqual(len(result["sequence_cluster_bootstrap_95_ci"]), 2)

    def test_paired_comparison_rejects_misaligned_states(self):
        reference = [{"sequence_index": 0, "timestep": 0.1, "mean_kl": 2.0}]
        candidate = [{"sequence_index": 1, "timestep": 0.1, "mean_kl": 1.0}]
        with self.assertRaisesRegex(ValueError, "pairing mismatch"):
            paired_comparison(reference, candidate, resamples=10, seed=0)

    def test_paired_binary_comparison_counts_transitions_and_exact_mcnemar(self):
        reference = [False] * 10 + [True, True]
        candidate = [True] * 10 + [False, True]
        result = paired_binary_comparison(reference, candidate)
        self.assertEqual(result["reference_wrong_candidate_correct"], 10)
        self.assertEqual(result["reference_correct_candidate_wrong"], 1)
        self.assertEqual(result["net_correct"], 9)
        self.assertAlmostEqual(result["exact_mcnemar_p"], 0.01171875)


if __name__ == "__main__":
    unittest.main()
