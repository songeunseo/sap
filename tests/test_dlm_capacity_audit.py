import json
import tempfile
import unittest
from pathlib import Path

from experiments.dlm_capacity_predictor.audit_existing import (
    audit_prediction_file,
    evaluation_limit,
    paired_exact_mcnemar,
    validate_audit_invariants,
)


class PredictionAuditTests(unittest.TestCase):
    def _row(self, example_id, correct=False):
        return {"example_id": example_id, "correct": correct,
                "evaluation_config_hash": "p", "doc_hash": f"d{example_id}",
                "prompt_hash": f"q{example_id}", "target_hash": f"t{example_id}",
                "reference_answer": "work\n#### 1", "extracted_answer": "1" if correct else "2"}

    def _write(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with tmp:
            for row in rows:
                tmp.write(json.dumps(row) + "\n")
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        return Path(tmp.name)

    def test_missing_prediction_is_pending(self):
        result = audit_prediction_file(Path("/definitely/missing.jsonl"), 100)
        self.assertEqual(result["status"], "pending")
        self.assertFalse(result["verified"])

    def test_duplicate_example_is_rejected(self):
        path = self._write([
            self._row(0, True), self._row(0, False),
        ])
        with self.assertRaisesRegex(ValueError, "duplicate example_id"):
            audit_prediction_file(path, 2)

    def test_mismatched_example_order_is_rejected(self):
        path = self._write([
            self._row(1, True), self._row(0, False),
        ])
        with self.assertRaisesRegex(ValueError, "example_id ordering"):
            audit_prediction_file(path, 2)

    def test_missing_identity_field_is_rejected(self):
        path = self._write([{
            "example_id": 0, "correct": False, "evaluation_config_hash": "p",
            "doc_hash": "d", "prompt_hash": "q", "target_hash": "t",
            # reference_answer intentionally absent
            "extracted_answer": "1",
        }])
        with self.assertRaisesRegex(ValueError, "identity field"):
            audit_prediction_file(path, 1)

    def test_inconsistent_saved_exact_match_is_rejected(self):
        path = self._write([{
            "example_id": 0, "correct": False, "evaluation_config_hash": "p",
            "doc_hash": "d", "prompt_hash": "q", "target_hash": "t",
            "reference_answer": "work\n#### 18", "extracted_answer": "18",
        }])
        with self.assertRaisesRegex(ValueError, "strict exact-match"):
            audit_prediction_file(path, 1)

    def test_exact_paired_result(self):
        # A wins 3 discordant pairs, B wins 0: two-sided exact p = 2/2^3.
        result = paired_exact_mcnemar(
            [True, True, True, False], [False, False, False, False]
        )
        self.assertEqual(result["a_correct_b_wrong"], 3)
        self.assertEqual(result["a_wrong_b_correct"], 0)
        self.assertEqual(result["discordant"], 3)
        self.assertEqual(result["exact_mcnemar_p"], 0.25)

    def test_limit_is_inferred_from_historical_nested_method(self):
        self.assertEqual(evaluation_limit({"uniform": {"limit": 1319}}), 1319)

    def test_bad_config_or_budget_fails_audit(self):
        base = {m: {"projection_count": 224, "module_index_order_verified": True,
                    "unique_names_verified": True, "target_order_and_shapes_verified": True,
                    "budget_verified": True, "declared_totals_verified": True,
                    "config_hash_matches": True} for m in ("uniform", "capacity", "reconstruction", "eis_type")}
        base["capacity"]["budget_verified"] = False
        with self.assertRaisesRegex(ValueError, "capacity.*budget"):
            validate_audit_invariants({"masks": base})
        base["capacity"]["budget_verified"] = True
        base["eis_type"]["config_hash_matches"] = False
        with self.assertRaisesRegex(ValueError, "eis_type.*config"):
            validate_audit_invariants({"masks": base})


if __name__ == "__main__":
    unittest.main()
