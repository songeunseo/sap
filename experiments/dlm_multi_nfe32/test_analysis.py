"""Focused paired-statistics and coverage checks without model loading."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from .analysis import CELLS, collect_costs, exact_mcnemar, paired_bootstrap, summarize, validate_attempts


class AnalysisTests(unittest.TestCase):
    def test_exact_discordances(self):
        result = exact_mcnemar([True, True, False, False], [False, False, True, False])
        self.assertEqual((result["gain"], result["loss"], result["both_wrong"]), (2, 1, 1))
        self.assertEqual(result["exact_p_two_sided"], 1.0)
        self.assertEqual(exact_mcnemar([True] * 6, [False] * 6)["exact_p_two_sided"], 0.03125)

    def test_paired_interaction_and_seed(self):
        ids = list(range(200))
        rows = {cell: {i: {"correct": False} for i in ids} for cell in CELLS}
        rows["Multi_32"][0]["correct"] = True
        rows["A_32"][1]["correct"] = True
        rows["Multi_256"][0]["correct"] = True
        result = summarize(ids, rows, draws=100, seed=12)
        self.assertEqual(result["primary_multi32_minus_A32"]["gain"], 1)
        self.assertEqual(result["primary_multi32_minus_A32"]["loss"], 1)
        self.assertEqual(result["secondary_interaction_32_minus_256"]["sum"], -1)
        self.assertEqual(result["per_question"][0]["interaction"], 0)
        self.assertEqual(result["per_question"][1]["interaction"], -1)
        self.assertEqual(paired_bootstrap([0, 1, -1], 100, 12),
                         paired_bootstrap([0, 1, -1], 100, 12))

    def test_success_coverage_and_failed_attempt_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "attempts"
            folder.mkdir()
            complete = {"status": "complete", "arm": "Multi", "step": 32,
                        "example_id": 3, "forwards": 32, "forward_tokens": 640,
                        "seconds": 1.0}
            failed = {**complete, "status": "failed", "forwards": 7,
                      "forward_tokens": 70, "seconds": 0.2}
            (folder / "Multi_3_success.json").write_text(json.dumps(complete))
            (folder / "Multi_3_failure.json").write_text(json.dumps(failed))
            self.assertEqual(validate_attempts(root, [3])["successful_questions"], 1)
            self.assertEqual(collect_costs(root)["forwards"], 39)
            complete["forwards"] = 63
            (folder / "Multi_3_success.json").write_text(json.dumps(complete))
            with self.assertRaisesRegex(RuntimeError, "forward count"):
                validate_attempts(root, [3])

    def test_missing_or_duplicate_questions_rejected(self):
        ids = list(range(200))
        rows = {cell: {i: {"correct": False} for i in ids} for cell in CELLS}
        with self.assertRaisesRegex(ValueError, "paired ID coverage"):
            summarize(ids, {**rows, "A_32": {i: rows["A_32"][i] for i in ids[:-1]}}, 10)
        with self.assertRaisesRegex(ValueError, "distinct"):
            summarize(ids[:-1] + [0], rows, 10)


if __name__ == "__main__":
    unittest.main()
