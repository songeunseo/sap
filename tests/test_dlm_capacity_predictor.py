import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

from experiments.dlm_capacity_predictor import core


class CapacityPredictorTests(unittest.TestCase):
    def test_candidate_index_transitions_from_awaiting_to_frozen_once(self):
        from experiments.dlm_capacity_predictor import analyze
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidates.json'
            preliminary = {'status': 'awaiting_dense_statistics', 'config_sha256': 'cfg',
                           'candidates': [{'method': 'probe16'}]}
            final = {'status': 'frozen', 'config_sha256': 'cfg',
                     'candidates': [{'method': 'probe16'}]}
            path.write_text(json.dumps(preliminary, indent=2, sort_keys=True) + '\n')
            analyze.freeze_candidate_index(path, final)
            self.assertEqual(json.loads(path.read_text()), final)
            analyze.freeze_candidate_index(path, final)
            with self.assertRaises(RuntimeError):
                analyze.freeze_candidate_index(path, {**final, 'candidates': []})

    def test_training_standardization_does_not_include_extreme_test_value(self):
        fit = core.fit_log_alpha(np.array([0., 2.]), np.exp([1., 3.]))
        self.assertEqual(fit['feature_mean'], 1.)
        self.assertEqual(fit['feature_std'], 1.)
        np.testing.assert_allclose(core.predict_alpha(fit, [4.]), [np.exp(5.)])

    def test_common_shape_uses_only_training_modules_and_states(self):
        d = np.ones((3, 4, 6))
        d[0, :2] = [1, 2, 3, 4, 5, 6]
        d[1:, :] = 999
        shape = core.common_shape(d, [0], [0, 1])
        np.testing.assert_allclose(shape, np.array([1, 2, 3, 4, 5, 6]) / 4)

    def test_probe_anchor_rejects_zero_local_error(self):
        with self.assertRaises(ValueError):
            core.anchor_prediction(np.ones((2, 3, 6)), np.zeros((2, 3, 6)), [0])

    def test_state_selection_is_exact_and_disjoint(self):
        keys = [(s, t) for t in [.05, .15, .85] for s in range(8)]
        a = core.state_indices(keys, [0, 1, 2, 3], [.15, .85])
        b = core.state_indices(keys, [4, 5, 6, 7], [.15, .85])
        self.assertEqual(len(a), 8)
        self.assertFalse(set(a) & set(b))
        with self.assertRaises(ValueError):
            core.state_indices(keys + [keys[0]], [0], [.05])

    def test_temporal_statistics_preserve_token_pairing(self):
        # Same temporal inputs: token-to-token variation is NOT state variation.
        x = np.array([[[1., 0.], [0., 3.]], [[1., 0.], [0., 3.]]])
        result = core.temporal_statistics(x, np.eye(2))
        self.assertAlmostEqual(result['variation_ratio'], 0.)
        self.assertAlmostEqual(result['feature_use_variability'], 0.)
        self.assertIsNone(result['pr_variation'])

    def test_temporal_statistics_known_variance(self):
        x = np.array([[[1., 0.]], [[3., 0.]]])
        result = core.temporal_statistics(x, np.eye(2))
        self.assertAlmostEqual(result['variation_ratio'], .2)
        self.assertAlmostEqual(result['pr_variation'], 1.)

    def test_covariance_participation_ratio_is_centered(self):
        x = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        self.assertAlmostEqual(core.participation_ratio(x), 2.)
        self.assertAlmostEqual(core.participation_ratio(x + 100), 2.)

    def test_sketch_is_repeatable_and_preserves_rank_one(self):
        a = core.sketch_matrix(7, 64)
        np.testing.assert_array_equal(a, core.sketch_matrix(7, 64))
        x = np.arange(10)[:, None] * np.arange(7)[None, :]
        self.assertAlmostEqual(core.participation_ratio(x @ a), 1.)

    def test_gate_rejects_single_sequence_driven_result(self):
        comparison = {'mean_difference': -.1, 'state_bootstrap_95_ci': [-.2, -.01],
                      'sequence_means_improved': 1, 'timestep_means_improved': 5}
        self.assertFalse(core.passes_gate(comparison))

    def test_disjoint_spans_reject_internal_and_cross_split_overlap(self):
        core.require_disjoint_intervals([[{'start': 0, 'end_exclusive': 10}],
                                         [{'start': 10, 'end_exclusive': 20}]])
        for groups in ([[{'start': 0, 'end_exclusive': 10}, {'start': 9, 'end_exclusive': 12}]],
                       [[{'start': 0, 'end_exclusive': 10}], [{'start': 9, 'end_exclusive': 12}]]):
            with self.assertRaises(ValueError):
                core.require_disjoint_intervals(groups)

    def test_dependency_success_requires_full_results(self):
        self.assertFalse(core.dependency_complete({'status': 'running', 'evaluations': []}))
        self.assertFalse(core.dependency_complete({'status': 'complete', 'evaluations': [{'limit': 100}]}))
        self.assertTrue(core.dependency_complete({'status': 'complete',
                         'evaluations': [{'limit': 100}, {'limit': 1319}]}))

    def test_crossed_fold_excludes_test_layers_and_test_sequences(self):
        from experiments.dlm_capacity_predictor import analyze
        keys = [(s, t/10 + .05) for s in range(8) for t in range(10)]
        local = np.broadcast_to(np.array([1., 2., 4., 8., 16., 32.]), (32, 80, 6)).copy()
        x = np.linspace(0., 1., 32)
        damage = np.exp(1. + .2*x)[:, None, None] * local
        data = dict(names=[f'block_{i:02d}.q_proj' for i in range(32)], shapes=[(1, 20)]*32,
                    state_keys=keys, damage_states=damage, reconstruction_states=local)
        feature_matrix = np.stack([x, 2*x+1])
        originals = analyze.crossed_fold_predictions(data, feature_matrix)
        self.assertEqual(len(originals), 8)
        for index, original in enumerate(originals):
            with self.subTest(fold=index):
                altered = dict(data, damage_states=damage.copy())
                altered['damage_states'][original['test_modules']] *= 100
                test_states = core.state_indices(keys, original['evaluation_sequences'])
                altered['damage_states'][:, test_states] *= 100
                changed_features = feature_matrix.copy()
                changed_features[1-index//4] += 1000
                changed = analyze.crossed_fold_predictions(altered, changed_features)[index]
                self.assertEqual(original['fit'], changed['fit'])
                self.assertEqual(original['predicted']['allocation'], changed['predicted']['allocation'])
                self.assertNotEqual(original['predicted']['heldout_additive_damage'],
                                    changed['predicted']['heldout_additive_damage'])
                self.assertEqual(original['predicted']['allocation']['pruned'], 104)
                self.assertEqual(original['predicted']['allocation']['budget_error'], 0)
                self.assertFalse(set(original['train_modules']) & set(original['test_modules']))


if __name__ == '__main__':
    unittest.main()
