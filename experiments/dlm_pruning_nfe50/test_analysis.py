"""Known-vector tests for paired interaction and strict ID alignment."""
from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from experiments.dlm_multiscale_ac50.artifacts import digest, write
from .analysis import CELLS, exact_mcnemar, interaction_rows, read_cell, summarize


def rows_for(ids, vectors):
    return {cell: {i: {"example_id": i, "correct": bool(vectors[cell][j])}
                   for j, i in enumerate(ids)} for cell in CELLS}


class PairedAnalysisTest(unittest.TestCase):
    def test_positive_and_negative_interactions(self):
        ids = [7, 3]
        rows = rows_for(ids, {
            "Dense_256": [1, 0], "A_256": [1, 0],
            "Dense_64": [1, 0], "A_64": [0, 1],
            "Dense_32": [1, 0], "A_32": [0, 1],
        })
        paired = interaction_rows(ids, rows)
        self.assertEqual([r["example_id"] for r in paired], ids)
        self.assertEqual([r["interaction_32"] for r in paired], [1, -1])
        self.assertEqual([r["interaction_64"] for r in paired], [1, -1])
        result = summarize(ids, rows, draws=100, seed=9)
        self.assertEqual(result["interaction_primary_32"]["estimate"], 0)
        self.assertEqual(result["paired_dense_minus_A"]["32"]["dense_only"], 1)
        self.assertEqual(result["paired_dense_minus_A"]["32"]["A_only"], 1)
        self.assertEqual(result["paired_dense_minus_A"]["32"]["exact_p"], 1)

    def test_interaction_subtracts_baseline_gap(self):
        ids = [5]
        rows = rows_for(ids, {cell: [int(cell.startswith("Dense"))] for cell in CELLS})
        self.assertEqual(interaction_rows(ids, rows)[0]["interaction_32"], 0)
        rows["A_256"][5]["correct"] = True
        self.assertEqual(interaction_rows(ids, rows)[0]["interaction_32"], 1)

    def test_exact_mcnemar_zero_discordance(self):
        self.assertEqual(exact_mcnemar([True, False], [True, False])["exact_p"], 1)

    def test_reject_missing_id(self):
        rows = rows_for([1, 2], {cell: [1, 0] for cell in CELLS})
        del rows["A_32"][2]
        with self.assertRaises(ValueError):
            interaction_rows([1, 2], rows)

    def test_reject_extra_id(self):
        rows = rows_for([1], {cell: [1] for cell in CELLS})
        rows["A_64"][2] = {"example_id": 2, "correct": True}
        with self.assertRaises(ValueError):
            interaction_rows([1], rows)

    def test_reject_duplicate_requested_id(self):
        rows = rows_for([1], {cell: [1] for cell in CELLS})
        with self.assertRaises(ValueError):
            interaction_rows([1, 1], rows)


    def test_reject_altered_checkpoint_identity(self):
        request = {"example_id": 7, "doc_hash": "original", "prompt_hash": "p",
                   "target_hash": "t", "reference_answer": "#### 1"}
        row = {**request, "doc_hash": "altered", "method": "Dense",
               "evaluation_config_hash": "protocol", "generated_text": "#### 1",
               "correct": True, "extracted_answer": "1", "denoising_steps": 32}
        fp = {"cell": "Dense_32"}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "gsm8k/Dense_32"
            write(folder / "identity.json", {"fingerprint": fp, "document_ids": [7]})
            write(folder / "examples/0007.json", {"fingerprint": fp, "row": row, "row_sha256": digest(row)})
            with patch("experiments.dlm_pruning_nfe50.prepare.cell_info",
                       return_value={"fingerprint": fp, "protocol_hash": "protocol"}):
                with self.assertRaisesRegex(RuntimeError, "identity differs"):
                    read_cell(Path(temp), {}, [request], "Dense_32", None)

    def test_reject_duplicate_checkpoint_name_for_one_id(self):
        request = {"example_id": 7}
        fp = {"cell": "Dense_32"}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "gsm8k/Dense_32"
            write(folder / "identity.json", {"fingerprint": fp, "document_ids": [7]})
            write(folder / "examples/0007.json", {})
            write(folder / "examples/7.json", {})
            with patch("experiments.dlm_pruning_nfe50.prepare.cell_info",
                       return_value={"fingerprint": fp, "protocol_hash": "protocol"}):
                with self.assertRaisesRegex(RuntimeError, "ambiguous checkpoint"):
                    read_cell(Path(temp), {}, [request], "Dense_32", None)


if __name__ == "__main__":
    unittest.main()
