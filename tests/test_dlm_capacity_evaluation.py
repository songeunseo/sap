import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.dlm_capacity_predictor import evaluate, freeze_states


class EvaluationSafetyTests(unittest.TestCase):
    def test_legacy_full_downstream_without_status_is_complete(self):
        document = {"evaluations": [
            {"uniform": {"limit": 100}, "capacity": {"limit": 100}},
            {"uniform": {"limit": 1319}, "capacity": {"limit": 1319}},
        ]}
        self.assertTrue(evaluate._full_downstream(document))

    def test_historical_pending_fails_before_model_load(self):
        pending = {"status": "running", "evaluations": [{"limit": 100}]}
        with patch.object(evaluate, "_load_json", return_value=pending), \
             patch.object(evaluate, "load_dense") as load_dense:
            with self.assertRaisesRegex(RuntimeError, "historical full downstream"):
                evaluate.require_historical_complete()
            load_dense.assert_not_called()

    def _heldout(self):
        # build_calibration_manifest order: timestep outer, sequence inner.
        rows = [{"state_index": i, "sequence_index": 24 + i % 8,
                 "timestep": [.1, .3, .5, .7, .9][i // 8], "mean_kl": float(i + 1)}
                for i in range(40)]
        return {"status": "complete", "method": "candidate", "fingerprint": {"x": "y"},
                "model_sha256": "model", "per_state": rows,
                "summary": {"mean_kl": float(np.mean([r["mean_kl"] for r in rows]))},
                "pruned": evaluate.TARGET, "weights": 6_979_321_856}

    def _expected_states(self):
        return [{"sequence_index": 24 + i % 8,
                 "timestep": [.1, .3, .5, .7, .9][i // 8]} for i in range(40)]

    def test_heldout_resume_rejects_reordered_states(self):
        result = self._heldout()
        result["per_state"][0], result["per_state"][1] = result["per_state"][1], result["per_state"][0]
        with self.assertRaisesRegex(RuntimeError, "state identity/order"):
            evaluate.validate_heldout_result(result, "candidate", {"x": "y"}, "model", self._expected_states())

    def test_heldout_resume_rejects_nonfinite_or_bad_summary(self):
        result = self._heldout()
        result["per_state"][3]["mean_kl"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            evaluate.validate_heldout_result(result, "candidate", {"x": "y"}, "model", self._expected_states())
        result = self._heldout()
        result["summary"]["mean_kl"] += 1
        with self.assertRaisesRegex(RuntimeError, "summary"):
            evaluate.validate_heldout_result(result, "candidate", {"x": "y"}, "model", self._expected_states())

    def test_heldout_resume_rejects_method_model_and_budget_mismatch(self):
        for key, value, message in (("method", "other", "method"),
                                    ("model_sha256", "other", "model"),
                                    ("pruned", 1, "budget")):
            result = self._heldout()
            result[key] = value
            with self.assertRaisesRegex(RuntimeError, message):
                evaluate.validate_heldout_result(result, "candidate", {"x": "y"}, "model", self._expected_states())

    def test_tampered_result_fails_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heldout.json"
            path.write_text(json.dumps(self._heldout()))
            receipt = {"result_sha256": "wrong", "fingerprint": {"x": "y"},
                       "model_sha256": "model"}
            with self.assertRaisesRegex(RuntimeError, "receipt"):
                evaluate.load_verified_heldout(path, receipt, "candidate", {"x": "y"}, self._expected_states())

    def test_aggregate_decision_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            value = {"status": "complete", "fingerprint": {"x": "y"},
                     "methods": {"candidate": self._heldout()}, "decisions": {"candidate": "PASS"}}
            path.write_text(json.dumps(value))
            receipt = {"fingerprint": {"x": "y"}, "aggregate_sha256": evaluate.sha(path)}
            value["decisions"]["candidate"] = "FAIL"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "aggregate receipt"):
                evaluate.load_verified_aggregate(path, receipt, {"x": "y"})

    def test_preflight_cache_requires_verified_status_and_mask_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "mask.pt"
            payload.write_bytes(b"payload")
            entry = {"shape": [2, 2], "selected_mask": {"path": str(payload), "pruned": 1}}
            paths = {"uniform": root / "manifest.json"}
            paths["uniform"].write_text("manifest")
            manifests = {"uniform": {"entries": [entry]}}
            fingerprint = {"uniform": {"manifest": evaluate.sha(paths["uniform"]),
                                        "payloads": [evaluate.sha(payload)]}}
            (root / "mask_preflight_receipt.json").write_text(json.dumps(
                {"status": "not_verified", "fingerprint": fingerprint}))
            with patch.object(evaluate, "ROOT", root), \
                 patch.object(evaluate, "selected_mask", return_value=np.zeros((1, 1), dtype=bool)):
                with self.assertRaisesRegex(RuntimeError, "shape"):
                    evaluate.preflight_mask_payloads(paths, manifests, [])

    def test_frozen_state_rejects_candidate_config_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "candidate/config"):
            freeze_states.require_candidate_config(
                {"status": "frozen", "config_sha256": "old"}, "new")

    def test_resumed_prediction_recomputes_strict_correctness(self):
        row = {"example_id": 0, "doc_hash": "d", "prompt_hash": "p", "target_hash": "t",
               "evaluation_config_hash": "protocol", "reference_answer": "work\n#### 18",
               "extracted_answer": "18", "correct": False}
        with self.assertRaisesRegex(RuntimeError, "strict exact-match"):
            evaluate.validate_prediction_rows([row], 1, "protocol")


if __name__ == "__main__":
    unittest.main()
