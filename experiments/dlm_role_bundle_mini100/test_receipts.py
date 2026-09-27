import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.dlm_role_bundle_mini100 import run


class ReceiptTests(unittest.TestCase):
    def fixture(self, root):
        run.write(root / "config.json", {"test": True})
        run.write(root / "add_b0/mask_manifest.json", {"test": "mask"})
        data = [dict(example_id=i, doc_hash=str(i), prompt_hash=str(i), target_hash=str(i),
                reference_answer="#### 1", evaluation_config_hash="protocol", extracted_answer="1", correct=True)
                for i in range(100)]
        self.writer(root / "baseline.jsonl", data)
        return dict(manifests={"add_b0": str(root / "add_b0/mask_manifest.json")},
            baselines={"aggregate": str(root / "baseline.jsonl")}, protocol_hash="protocol", target_pruned=10), data

    @staticmethod
    def writer(path, data):
        path.write_text("".join(json.dumps(x) + "\n" for x in data))

    def test_string_and_path_digest_match(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "value.json"
            run.write(path, {"x": 1})
            self.assertEqual(run.sha256(str(path)), run.sha256(path))

    def test_full_save_and_completed_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run, "ROOT", Path(temp)):
            c, data = self.fixture(Path(temp))
            result = run.persist_evaluation("add_b0", c, data, "modelhash", {"eval_seconds": 2}, "0", self.writer)
            self.assertEqual(run.completed("add_b0", c), data)
            self.assertEqual(result["correct"], 100)
            pending = run.read(Path(temp) / "add_b0/pending_eval.json")
            self.assertEqual(pending["metadata"]["sparse_model_sha256"], "modelhash")

    def test_metadata_failure_precedes_prediction_write(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run, "ROOT", Path(temp)):
            c, data = self.fixture(Path(temp))
            c["manifests"]["add_b0"] += ".missing"
            with self.assertRaises(FileNotFoundError):
                run.persist_evaluation("add_b0", c, data, "hash", {}, "0", self.writer)
            self.assertFalse((Path(temp) / "add_b0/predictions.jsonl").exists())

    def test_prediction_failure_keeps_verified_metadata(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(run, "ROOT", Path(temp)):
            c, data = self.fixture(Path(temp))
            def fail(path, rows):
                raise OSError("simulated storage failure")
            with self.assertRaises(OSError):
                run.persist_evaluation("add_b0", c, data, "hash", {}, "0", fail)
            self.assertTrue((Path(temp) / "add_b0/pending_eval.json").exists())
            self.assertFalse((Path(temp) / "add_b0/results.json").exists())


if __name__ == "__main__": unittest.main()
