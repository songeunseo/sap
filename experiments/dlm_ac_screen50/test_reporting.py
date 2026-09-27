"""Synthetic CPU coverage for the standalone report builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .core import ARMS, CONTRASTS
from .reporting import REFERENCES, build_report


def _rows(n=100, correct_every=2, doc_prefix="doc"):
    return [
        {
            "example_id": i,
            "doc_hash": f"{doc_prefix}-{i}",
            "prompt_hash": f"prompt-{i}",
            "target_hash": f"target-{i}",
            "reference_answer": "1",
            "correct": bool(i % correct_every == 0),
        }
        for i in range(n)
    ]


def _manifest():
    arms = {}
    for label in ARMS:
        family = "Square" if label.startswith("Square") else "Vector" if label.startswith("Vector") else "Exchange" if label.startswith("Exchange") else "Multi"
        arms[label] = {
            "family": family,
            "bank": {"id": f"{family}-fixture", "path": f"banks/{family}.json"},
            "readout": f"{family}-readout",
            "allocator": "fixture-allocator",
            "objective_version": "fixture-v1",
            "reduction": "equal-pair",
            "control": [],
        }
    return {
        "arms": arms,
        "deferred": ["DKD", "A-floor"],
        "costs": {"Square": 11360, "Vector": 11360},
        "source_ledger": "design.md",
        "source_ledger_sha256": "ledger-sha",
        "source_hashes": {"core.py": "core-sha"},
        "exchange": {"d": 41, "proposals": 8, "rounds": 3},
    }


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ReportingTests(unittest.TestCase):
    def _fixture(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name) / "screen"
        multi = Path(td.name) / "multi"
        root.mkdir()
        multi.mkdir()
        _write(root / "manifest.json", _manifest())
        rows = {}
        for label in (*ARMS, *REFERENCES):
            rows[label] = {
                "rows": _rows(doc_prefix="shared"),
                "fingerprint": {"mask_identity": f"mask-{label}"},
                "provenance": "fixture",
            }
        return td, root, multi, rows

    def test_complete_report_has_identity_pairs_holm_and_diagnostics(self):
        td, root, multi, rows = self._fixture()
        self.addCleanup(td.cleanup)
        _write(multi / "diagnostics" / "Multi" / "calibration.json", {"mean": {"A": 1, "C1": 2, "C2": 3, "C4": 4, "C_path": 5, "C_all": 6}})
        _write(root / "metrics" / "Square" / "Square-A" / "calibration.json", {"mean": {"A": 1, "C": 2, "common2": 3, "main1_2": 4, "main2_2": 5, "interaction2": 6, "mixed2": 7}})
        _write(root / "metrics" / "Vector" / "Vector-A" / "calibration.json", {"mean": {"A": 1, "C": 2, "AC": 3}})
        _write(root / "costs" / "Square-A.json", {"job": "Square-A", "forward_calls": 10, "wall_seconds": 2, "gpu_seconds": 1})
        _write(root / "stage_costs" / "Vector-teacher.json", {"job": "Vector_teacher_calibration", "stage": "teacher_vectors", "forward_calls": 20, "wall_seconds": 3})
        _write(root / "exchange" / "state.json", {"loss": 1.2, "accepted": [{"round": 0, "donor": 1, "receiver": 2, "measured_gain": 0.1, "predicted_gain": 0.2}], "termination": "no_improvement_among_offered"})

        report, markdown = build_report(root, multi, rows)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["holm_applied"])
        self.assertEqual(len(report["comparisons"]), 7)
        self.assertTrue(all("holm_p" in value for value in report["comparisons"].values()))
        self.assertEqual(report["scores"]["Square-A"]["mask_identity"], "mask-Square-A")
        self.assertEqual(report["diagnostics"]["Multi"]["components"]["C4"], 4.0)
        self.assertEqual(report["diagnostics"]["Vector"]["vocabulary_normalization"], "1/V over all emitted vocabulary coordinates")
        self.assertEqual(report["diagnostics"]["Exchange"]["termination"], "no_improvement_among_offered")
        self.assertEqual(report["costs"]["stages"]["teacher_setup"]["forward_calls"], 20.0)
        self.assertIn("Per-question output links", markdown)
        self.assertIn("Holm p", markdown)

    def test_partial_identity_and_common_ids_do_not_get_holm(self):
        td, root, multi, rows = self._fixture()
        self.addCleanup(td.cleanup)
        rows["Square-A"] = {"rows": _rows(3, doc_prefix="shared"), "provenance": "fixture"}
        rows["Square-AC"] = {"rows": _rows(3, doc_prefix="other"), "fingerprint": {"mask_identity": "mask-square-ac"}}
        rows["legacy_AC"] = {"rows": _rows(3, doc_prefix="shared"), "fingerprint": {"mask_identity": "mask-legacy-ac"}, "alias_of": "legacy_AC"}
        report, markdown = build_report(root, multi, rows)
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["holm_applied"])
        self.assertFalse(report["scores"]["Square-A"]["complete"])
        self.assertEqual(report["scores"]["Square-A"]["status"], "invalid_identity")
        comparison = report["comparisons"]["Square-AC_vs_Square-A"]
        self.assertEqual(comparison["common_id_count"], 0)
        self.assertNotIn("holm_p", comparison)
        self.assertIn("missing/invalid", markdown)

    def test_identity_mismatch_excludes_question_from_paired_set(self):
        td, root, multi, rows = self._fixture()
        self.addCleanup(td.cleanup)
        rows["Square-A"]["rows"][0]["prompt_hash"] = "changed"
        report, _ = build_report(root, multi, rows)
        pair = report["comparisons"]["Square-AC_vs_Square-A"]
        self.assertEqual(pair["common_id_count"], 99)
        self.assertEqual(pair["total"], 99)


if __name__ == "__main__":
    unittest.main()
