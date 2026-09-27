import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.dlm_capacity_predictor.collect_dense import (
    _collect_one,
    _fingerprint,
    _require_candidate_order,
    _validate_sequence_payload,
    _valid_payload,
    aggregate_sequences,
    load_validated_statistics,
    sequence_record,
)
from experiments.dlm_capacity_predictor import core
from experiments.dlm_capacity_predictor.core import sketch_matrix, temporal_statistics


class DenseStatisticsTest(unittest.TestCase):
    def test_sequence_validation_accepts_complete_hook_execution_order(self):
        """Hook call order may differ from canonical mapping order; names remain keyed."""
        moment = {"sum": np.zeros(64), "outer": np.zeros((64, 64)), "count": 10}
        record = {"timesteps": 10, "feature_width": 1, "energy_sum": 1.,
                  "energy_count": 10, "variation_energy_sum": 0.,
                  "feature_sq_sum": np.ones((10, 1)), "feature_count": np.ones(10),
                  "mean_sketch": moment, "variation_sketch": moment}
        fingerprint = {"module_names": ["canonical_first", "canonical_second"],
                       "sketch_size": 64}
        payload = {"sequence_index": 0, "names": fingerprint["module_names"],
                   "state_order": [{"timestep_index": i} for i in range(10)],
                   "records": {"canonical_second": record, "canonical_first": record}}
        _validate_sequence_payload(payload, fingerprint, 0)

    def test_candidate_order_enforces_frozen_source_sha256(self):
        config = {"source_sha256": {
            "experiments/projection_capacity_allocation_65/candidate_mask_manifest.json": "wrong"}}
        with self.assertRaisesRegex(RuntimeError, "candidate manifest hash mismatch"):
            _require_candidate_order({}, config)

    def test_standalone_loader_rejects_stale_fingerprint_and_truncated_payload(self):
        config = json.loads((core.ROOT / "config.json").read_text())
        manifest = json.loads(core.CALIBRATION.read_text())
        candidates = json.loads((core.SOURCE / "candidate_mask_manifest.json").read_text())
        names = [row["name"] for row in candidates["entries"]]
        fingerprint = _fingerprint(config, manifest, names)
        fingerprint["candidate_manifest_sha256"] = core.sha(
            core.SOURCE / "candidate_mask_manifest.json")
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "stale.pt"
            torch.save({"fingerprint": {"stale": True}}, stale)
            with self.assertRaisesRegex(RuntimeError, "different fingerprint"):
                load_validated_statistics(stale)
            truncated = Path(directory) / "truncated.pt"
            torch.save({"fingerprint": fingerprint,
                        "config_sha256": fingerprint["config_sha256"]}, truncated)
            with self.assertRaisesRegex(RuntimeError, "names or sequences"):
                load_validated_statistics(truncated)

    def test_streaming_collector_matches_inputs_seen_by_real_linear_hook(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2, bias=False)

            def activations(self, ids):
                position = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
                return torch.stack((ids.float() + 1, position.float() + 2,
                                    torch.full_like(ids, 5).float()), dim=-1)

            def forward(self, ids):
                return self.linear(self.activations(ids))

        model = TinyModel()
        states = []
        noisy = ([0, 1, 2], [0, 2, 2], [2, 2, 0])
        for timestep, ids in enumerate(noisy):
            states.append({"noisy_ids": [ids], "clean_ids": [ids],
                           "mask": [[False] * 3], "timestep_index": timestep,
                           "timestep": timestep / 10})
        records, order = _collect_one(model, {"tiny.linear": model.linear}, states)
        actual = np.stack([model.activations(torch.tensor([ids])).detach().numpy()[0]
                           for ids in noisy])
        reference = sequence_record(actual, sketch_matrix(3))
        got = aggregate_sequences([records["tiny.linear"]])
        want = aggregate_sequences([reference])
        self.assertEqual([row["timestep_index"] for row in order], [0, 1, 2])
        for key in want:
            if want[key] is None:
                self.assertIsNone(got[key])
            else:
                self.assertAlmostEqual(got[key], want[key], places=9)
        self.assertGreater(got["variation_ratio"], 0)
        self.assertEqual(actual[:, :, 2].var(), 0)
        self.assertGreater(actual[0, :, 1].var(), 0)

    def test_resume_rejects_truncated_sequence_even_with_matching_fingerprint(self):
        fingerprint = {"module_names": ["a", "b"], "sketch_size": 64}
        payload = {"fingerprint": fingerprint, "sequence_index": 0,
                   "names": ["a", "b"], "state_order": [], "records": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.pt"
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                _valid_payload(path, fingerprint, sequence=0)

    def test_single_sequence_matches_direct_reference(self):
        x = np.arange(3 * 5 * 4, dtype=np.float64).reshape(3, 5, 4) / 13 + 1
        sketch = sketch_matrix(4, size=3)
        got = aggregate_sequences([sequence_record(x, sketch)])
        ref = temporal_statistics(x, sketch)
        for key in ("variation_ratio", "feature_use_variability", "pr_mean",
                    "pr_variation", "log_pr_ratio", "activation_energy"):
            self.assertAlmostEqual(got[key], ref[key], places=11)
        second = np.mean(x * x, axis=(0, 1))
        self.assertEqual(got["activation_outlier_ratio"],
                         float(np.mean(second > 7 * second.mean())))

    def test_pooled_statistics_equal_direct_concatenated_reference(self):
        rng = np.random.default_rng(19)
        xs = [rng.normal(size=(4, 3, 5)), rng.normal(size=(4, 7, 5)) + 1]
        sketch = sketch_matrix(5, size=4)
        got = aggregate_sequences([sequence_record(x, sketch) for x in xs])

        energy = sum(np.square(x).sum() for x in xs)
        count = sum(x.shape[0] * x.shape[1] for x in xs)
        residual = [x - x.mean(0, keepdims=True) for x in xs]
        variation = sum(np.square(x).sum() for x in residual)
        per_t = np.stack([sum(np.square(x[t]).sum(0) for x in xs) /
                          sum(x.shape[1] for x in xs) for t in range(4)])
        norm = per_t / per_t.sum(1, keepdims=True)
        average = norm.mean(0)
        mean_samples = np.concatenate([x.mean(0) @ sketch for x in xs])
        var_samples = np.concatenate([(r @ sketch).reshape(-1, 4) for r in residual])
        feature_second = sum(np.square(x).sum(axis=(0, 1)) for x in xs) / count

        def pr(a):
            cov = (a - a.mean(0)).T @ (a - a.mean(0)) / len(a)
            return np.trace(cov) ** 2 / np.square(cov).sum()

        self.assertAlmostEqual(got["activation_energy"], energy / count)
        self.assertAlmostEqual(got["variation_ratio"], variation / energy)
        self.assertAlmostEqual(got["feature_use_variability"],
                               np.square(norm - average).sum(1).mean() /
                               np.square(average).sum())
        self.assertAlmostEqual(got["pr_mean"], pr(mean_samples), places=10)
        self.assertAlmostEqual(got["pr_variation"], pr(var_samples), places=10)
        self.assertAlmostEqual(got["activation_outlier_ratio"],
                               np.mean(feature_second > 7 * feature_second.mean()))

    def test_temporal_residual_preserves_token_pairing(self):
        x = np.array([[[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        shuffled = x.copy()
        shuffled[1] = shuffled[1, ::-1]
        sketch = np.eye(2)
        self.assertEqual(aggregate_sequences([sequence_record(x, sketch)])["variation_ratio"], 0)
        self.assertGreater(aggregate_sequences([sequence_record(shuffled, sketch)])["variation_ratio"], 0)

    def test_constant_and_varying_inputs(self):
        sketch = np.eye(3)
        constant = np.ones((3, 4, 3))
        varying = constant.copy(); varying[1, :, 0] = 4
        c = aggregate_sequences([sequence_record(torch.tensor(constant), torch.tensor(sketch))])
        v = aggregate_sequences([sequence_record(varying, sketch)])
        self.assertEqual(c["variation_ratio"], 0)
        self.assertIsNone(c["pr_variation"])
        self.assertGreater(v["variation_ratio"], 0)
        self.assertIsNotNone(v["pr_variation"])

    def test_energy_pooling_is_token_weighted_not_sequence_averaged(self):
        sketch = np.eye(2)
        small = np.ones((2, 1, 2))
        large = np.full((2, 9, 2), 3.0)
        got = aggregate_sequences([sequence_record(small, sketch), sequence_record(large, sketch)])
        self.assertAlmostEqual(got["activation_energy"], (4 + 324) / 20)
        self.assertNotAlmostEqual(got["activation_energy"], (2 + 18) / 2)

    def test_fixed_sketch_is_repeatable_for_numpy_and_torch_records(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(3, 4, 7))
        a = sketch_matrix(7)
        b = sketch_matrix(7)
        np.testing.assert_array_equal(a, b)
        one = aggregate_sequences([sequence_record(x, a)])
        two = aggregate_sequences([sequence_record(torch.tensor(x), torch.tensor(b))])
        self.assertEqual(one, two)
        self.assertTrue(math.isfinite(one["log_pr_ratio"]))


if __name__ == "__main__":
    unittest.main()
