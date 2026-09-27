"""Adversarial tests for the evidence gates, independent of GPU availability."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import closeout as c


class Gates(unittest.TestCase):
    def test_known_exact_pairs(self):
        for gain, loss, expected in [(6, 5, 1), (13, 6, .1670684814453125), (13, 7, .26317596435546875), (0, 0, 1), (8, 0, .0078125)]:
            result = c.statistics([False]*gain + [True]*loss + [False], [True]*gain + [False]*loss + [False])
            self.assertAlmostEqual(result["exact_mcnemar_p"], expected)

    def test_holm_monotonicity(self):
        family = {"a": {"exact_mcnemar_p": .04}, "b": {"exact_mcnemar_p": .01}, "c": {"exact_mcnemar_p": .02}}
        c.adjust(family)
        self.assertEqual([family[x]["holm_p"] for x in ("b", "c", "a")], [.03, .04, .04])

    def test_alias_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"0000.json").write_text("{}")
            (root/"0.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Noncanonical"):
                c.strict_paths(root, [0])

    def test_missing_or_extra_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"0000.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Missing/unexpected"):
                c.strict_paths(root, [0, 1])
            with self.assertRaisesRegex(RuntimeError, "Missing/unexpected"):
                c.strict_paths(root, [])

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"source.json"
            p.write_text("{}"); expected = c.sha(p); p.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError, "Changed source"):
                c.checked(p, expected)

    def test_same_grade_different_text_is_not_reproduction(self):
        fields = ("doc_hash", "prompt_hash", "target_hash", "reference_answer", "evaluation_config_hash", "generated_text", "extracted_answer")
        old = {k: "x" for k in fields}; old.update(example_id=0, correct=True)
        new = {**old, "generated_text": "different reasoning"}
        with self.assertRaisesRegex(RuntimeError, "Historical reproduction"):
            c.exact_history([old], {0: new})

    def test_missing_cost_is_lower_bound(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/"attempts").mkdir(); (root/"costs").mkdir()
            (root/"attempts/a.json").write_text(json.dumps({"job":"teacher"}))
            result = c.cost_accounting({"run": root})
            self.assertTrue(result["counts_are_lower_bounds"])
            self.assertFalse(result["complete_accounting"])

    def test_preserved_attempt_counted(self):
        with tempfile.TemporaryDirectory() as d:
            roots = {}
            for label, count in [("current", 300), ("first", 128)]:
                root = Path(d)/label; (root/"attempts").mkdir(parents=True); (root/"costs").mkdir()
                (root/"attempts/a.json").write_text(json.dumps({"job":"teacher"}))
                (root/"costs/a.json").write_text(json.dumps({"job":"teacher", "forward_calls":count, "wall_seconds":1, "status":"complete"}))
                roots[label]=root
            result = c.cost_accounting(roots)
            self.assertEqual(result["recorded_forwards_total"], 428)
            self.assertTrue(result["complete_accounting"])

    def test_incomplete_run_cannot_finalize(self):
        with patch.object(c, "read", return_value={"status":"running"}), patch.object(c, "write") as write:
            with self.assertRaisesRegex(RuntimeError, "before completion"):
                c.finalize()
            write.assert_not_called()

    def test_pending_and_completed_render(self):
        # Synthetic values are in-memory fixtures only; no manuscript is written.
        mini = {"scores": {a: {"correct": 0} for a in ("Multi", "A", "Uniform")},
                "development": {a: {"correct": 0} for a in ("Multi", "A", "Uniform")}, "paired": {}}
        comparison = {"gain": 0, "loss": 0, "net": 0, "difference_pp": 0,
                      "exact_mcnemar_p": 1, "holm_p": 1, "paired_bootstrap_95pp_unadjusted": [0, 0]}
        full = {"scores": {g: {a: {"correct": 0, "total": n} for a in c.ARMS}
                 for g,n in [("full_1319", 1319), ("previously_seen_200", 200), ("primary_remaining_1119", 1119)]},
                "comparisons": {"primary_remaining_1119": {k: comparison for k in c.FAMILY}},
                "fresh_diagnostics": {a: {k: 0 for k in ("A", "C_natural", "C_cross", "query_CE", "response_sign_flip_rate")} for a in c.ARMS},
                "costs": {"recorded_forwards_total": 0, "counts_are_lower_bounds": True}, "wall_hours_current_run": 0}
        with patch.object(c, "read", return_value={"beta": .5}):
            pending = c.render(mini)
            complete = c.render(mini, full)
        self.assertIn("PENDING", pending)
        self.assertNotIn("PENDING", complete)
        self.assertNotIn("{{", complete)
        self.assertIn("lower bound", complete)
        self.assertIn("does not pass", complete)

    def test_unpaired_arrays_fail(self):
        with self.assertRaisesRegex(RuntimeError, "Unpaired"):
            c.statistics([True], [])


if __name__ == "__main__":
    unittest.main()
