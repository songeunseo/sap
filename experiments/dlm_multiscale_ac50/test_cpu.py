"""CPU regressions for scientific controls, checkpointing, and launch isolation."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .artifacts import PLAN, REPO, ReadoutStore, digest, freeze, read, sha, write
from .core import aliases, clean_sequences, make_bank, metrics, paired, rank_rates, validate_bank
from .evaluation import evaluate_requests


def real_bank():
    plan = read(PLAN)
    source = read(plan["sources"]["calibration_source"]["path"])
    return make_bank(source, plan["bank"], "calibration")


def request(index):
    return dict(example_id=index, doc_hash=f"doc{index}", prompt_hash=f"prompt{index}",
                target_hash=f"target{index}", reference_answer="#### 2", prompt=f"Question {index}")


class BanksAndObjectives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = real_bank()

    def test_reproducibility_monotonicity_and_queries(self):
        self.assertEqual(self.bank, real_bank())
        self.assertEqual(self.bank["states"], 128)
        validate_bank(self.bank)
        for index in range(0, 16, 2):
            a, b = self.bank["chains"][index:index+2]
            self.assertEqual(a["query"], b["query"])
            self.assertNotEqual(a["uniforms"], b["uniforms"])
            self.assertEqual(len(a["query"]), 8)

    def test_unchanged_pairs_are_not_dropped_or_forced_to_reveal(self):
        bank = json.loads(json.dumps(self.bank))
        for chain in bank["chains"]:
            chain["uniforms"] = [.999]*len(chain["eligible"])
            for node in chain["nodes"]:
                node.update(visible=[], input_ids=[bank["mask_id"]]*256, masked_count=256, actual_mask_fraction=1.0)
        validate_bank(bank)
        self.assertEqual(bank["states"], 128)

    def test_query_reveal_rejected(self):
        bank = json.loads(json.dumps(self.bank))
        chain = bank["chains"][0]
        chain["nodes"][0]["input_ids"][chain["query"][0]] = 1
        with self.assertRaises(ValueError):
            validate_bank(bank)

    def test_multiscale_detects_short_matching_blind_spot(self):
        dense = np.zeros((128, 8))
        constant = np.ones((128, 8))
        pattern = np.repeat(np.tile([1, 1, -1, -1, 1, 1, -1, -1], 16)[:, None], 8, axis=1)
        a, b = metrics(constant, dense, self.bank)["mean"], metrics(pattern, dense, self.bank)["mean"]
        self.assertEqual(a["A"], b["A"])
        self.assertEqual(a["C1"], b["C1"])
        self.assertEqual(b["C2"], 4)
        self.assertAlmostEqual(b["Multi"]-a["Multi"], 4/3)
        self.assertGreater(b["Path"], a["Path"])

    def test_all_pair_identity_and_matched_total_weights(self):
        rng = np.random.default_rng(8)
        pred, dense = rng.normal(size=(128, 8)), rng.normal(size=(128, 8))
        row = metrics(pred, dense, self.bank)["mean"]
        error = (pred-dense).reshape(16, 8, 8)
        self.assertAlmostEqual(row["C_all"], 16/7*np.var(error, axis=1).mean())
        self.assertAlmostEqual(row["Multi"], row["A"]+(row["C1"]+row["C2"]+row["C4"])/3)
        self.assertLessEqual(row["Multi"]-row["A"], 4*row["A"]+1e-12)

    def test_nonfinite_and_wrong_shape_rejected(self):
        for bad in (np.zeros((127, 8)), np.full((128, 8), np.nan)):
            with self.assertRaises(ValueError):
                metrics(bad, np.zeros((128, 8)), self.bank)

    def test_signed_costs_and_ties(self):
        self.assertTrue(np.all(rank_rates([0]*32) == .5))
        rates = rank_rates(list(range(-16, 16)))
        self.assertAlmostEqual(rates.mean(), .5)
        self.assertAlmostEqual(rates[0], .55)
        self.assertAlmostEqual(rates[-1], .45)

    def test_aliases_follow_exact_row_counts(self):
        rows = {m:dict(row_counts=[1, 2]) for m in ("A", "Short", "Path", "All", "Multi")}
        rows["Path"]["row_counts"] = [2, 1]
        result = aliases(rows)
        self.assertEqual(result["Short"], "Multi")
        self.assertEqual(result["Path"], "Path")

    def test_historical_paired_counts(self):
        result = paired([False]*9+[True]*3+[True]*88, [True]*9+[False]*3+[True]*88)
        self.assertEqual((result["gained"], result["lost"], result["net"]), (9, 3, 6))
        self.assertAlmostEqual(result["exact_mcnemar_p"], .14599609375)


class Checkpoints(unittest.TestCase):
    def test_node_resume_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node.json"
            a = ReadoutStore(path, {"mask":"a"}, 2, 2)
            a.append([1., 2.])
            b = ReadoutStore(path, {"mask":"a"}, 2, 2)
            self.assertEqual(b.values, [[1., 2.]])
            b.append([3., 4.])
            self.assertEqual(b.complete(), [[1., 2.], [3., 4.]])
            with self.assertRaises(RuntimeError):
                ReadoutStore(path, {"mask":"different"}, 2, 2)

    def test_interrupted_document_resume_skips_completed_and_rejects_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            requests = [request(i) for i in (3, 12, 90)]
            called = []
            def generate(req):
                called.append(req["example_id"])
                if req["example_id"] == 12:
                    raise RuntimeError("simulated interruption")
                return "#### 2"
            grade = lambda req, text:dict(extracted_answer="2", correct=True)
            progress = lambda *a, **kw:None
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                evaluate_requests(folder, requests, {"mask":"a"}, "p", "Multi", generate, grade, progress)
            self.assertEqual(len(list((folder/"examples").glob("*.json"))), 1)
            called.clear()
            def resumed(req):
                called.append(req["example_id"])
                return "#### 2"
            result = evaluate_requests(folder, requests, {"mask":"a"}, "p", "Multi", resumed, grade, progress)
            self.assertEqual(called, [12, 90])
            self.assertEqual(result["correct"], 3)
            called.clear()
            self.assertEqual(evaluate_requests(folder, requests, {"mask":"a"}, "p", "Multi", resumed, grade, progress), result)
            self.assertEqual(called, [])
            with self.assertRaises(RuntimeError):
                evaluate_requests(folder, requests, {"mask":"b"}, "p", "Multi", resumed, grade, progress)
            saved = read(folder / "examples/0012.json")
            saved["row"]["correct"] = False
            write(folder / "examples/0012.json", saved)
            with self.assertRaises(RuntimeError):
                evaluate_requests(folder, requests, {"mask":"a"}, "p", "Multi", resumed, grade, progress)

    def test_freeze_refuses_changed_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            freeze(path, {"value":1})
            with self.assertRaises(RuntimeError):
                freeze(path, {"value":2})


class CpuIntegration(unittest.TestCase):
    def test_dry_run_never_queries_gpu_or_launches(self):
        from .run import launch
        with patch("subprocess.check_output", side_effect=AssertionError("subprocess forbidden")), \
             patch("subprocess.run", side_effect=AssertionError("subprocess forbidden")), \
             contextlib.redirect_stdout(io.StringIO()) as captured:
            launch(Path("/tmp/not-created-multiscale"), ["0", "3"], "development", True)
        result = json.loads(captured.getvalue())
        self.assertFalse(result["gpu_queried"])
        self.assertIn("PYTHONPATH=", result["command"])
        self.assertIn("HF_HUB_OFFLINE=1", result["command"])

    def test_real_readout_loop_with_small_cpu_model(self):
        import torch
        from .gpu import Runtime
        from types import SimpleNamespace
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32))
                self.calls = 0
            def forward(self, ids):
                self.calls += 1
                value = self.weight[None, None, :] + (ids.float().sum() % 11)/12
                return SimpleNamespace(logits=value.expand(1, ids.shape[1], 12))
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            root = Path(directory)
            settings = dict(read(PLAN)["bank"], sequences_per_split=1)
            source = dict(mask_id=11, states=[dict(sequence_index=0, clean_ids=[[i%10 for i in range(256)]])])
            bank = make_bank(source, settings, "calibration")
            rt = Runtime.__new__(Runtime)
            rt.root, rt.model, rt.banks = root, Tiny(), {"calibration":bank}
            rt.config_hash = "test"
            rt.config = dict(banks={"calibration":dict(sha256=digest(bank))}, evaluation=dict(max_cuda_gib=30))
            rt.progress = lambda *a, **kw:None
            first = rt.readouts("tiny", "mask1", "calibration")
            self.assertEqual(rt.model.calls, 16)
            self.assertEqual(rt.readouts("tiny", "mask1", "calibration"), first)
            self.assertEqual(rt.model.calls, 16)
            self.assertFalse(torch.cuda.is_initialized())

    def test_native_wanda_mask_and_restore_preserve_dense_survivors(self):
        import torch
        from .gpu import Runtime, save_tensor
        from experiments.dlm_ppl50.sequential import wanda_mask
        from experiments.dlm_loss_aggregation.core import pack_mask, mask_sha256
        with tempfile.TemporaryDirectory() as directory, patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            root = Path(directory)
            model = torch.nn.ModuleList([torch.nn.Linear(8, 4, bias=False) for _ in range(7)])
            mapping = {f"block_00.p{i}":module for i, module in enumerate(model)}
            originals = {name:module.weight.detach().clone() for name, module in mapping.items()}
            activation = {name:torch.arange(8, dtype=torch.float32) for name in mapping}
            activation_path = root / "activation.pt"
            save_tensor(activation_path, activation)
            refs, expected = [], {}
            for name, module in mapping.items():
                mask = wanda_mask(originals[name], activation[name], 4)
                packed = pack_mask(mask)
                path = root / (name+".pt")
                save_tensor(path, packed)
                refs.append(dict(name=name, shape=[4, 8], weights=32,
                    selected_mask=dict(path=str(path), file_sha256=sha(path), mask_sha256=mask_sha256(packed),
                                       prune_per_row=4, pruned=16)))
                expected[name] = originals[name].masked_fill(mask, 0)
            rt = Runtime.__new__(Runtime)
            rt.model, rt.mapping, rt.refs = model, mapping, refs
            rt.config = dict(activations=[str(activation_path)], sources={str(activation_path):sha(activation_path)},
                             evaluation=dict(max_cuda_gib=30))
            rt.mask_block(0, [4]*7, check_uniform=True)
            for name, module in mapping.items():
                self.assertTrue(torch.equal(module.weight, expected[name]))
            rt.mask_block(0, [5]*7, originals=originals)
            for name, module in mapping.items():
                mask = wanda_mask(originals[name], activation[name], 5)
                self.assertTrue(torch.equal(module.weight, originals[name].masked_fill(mask, 0)))
            rt.restore_block(0, originals)
            for name, module in mapping.items():
                self.assertTrue(torch.equal(module.weight, expected[name]))
            self.assertFalse(torch.cuda.is_initialized())

    def test_official_grading_on_fresh_task_without_building_all_requests(self):
        from .artifacts import LEGACY
        from .evaluation import task_and_protocol, grade
        task, _, _ = task_and_protocol(read(LEGACY / "config.json"))
        doc = task.eval_docs[0]
        req = dict(example_id=0, doc=doc, prompt=task.doc_to_text(doc),
                   generation_kwargs=task.get_config("generation_kwargs"))
        result = grade(task, req, task.doc_to_target(doc))
        self.assertTrue(result["correct"])
        wrong = grade(task, req, "No strict answer marker")
        self.assertFalse(wrong["correct"])

    def test_64_probe_replay_to_exact_50_allocation(self):
        from .analysis import allocate
        bank = real_bank()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(pruning={"probe_rates":[.48, .52], "pruned":2240},
                          banks={"calibration":{"path":str(root/"bank.json")}},
                          legacy_manifests={"uniform":{"path":str(root/"uniform.json")}})
            write(root/"config.json", config)
            write(root/"bank.json", bank)
            refs = [dict(name=f"block_{b:02d}.p{p}", shape=[2, 10], weights=20) for b in range(32) for p in range(7)]
            write(root/"uniform.json", {"entries":refs})
            teacher = np.zeros((128, 8))
            write(root/"readouts/dense/calibration.json", {"values":teacher.tolist()})
            for b in range(32):
                conditions = {}
                for rate, count in ((.48, 56), (.52, 70)):
                    rng = np.random.default_rng(b*100+int(rate*100))
                    values = rng.normal(size=(128, 8))*.1 + .5 + (rate-.5)*(b-16)
                    path = root / f"fake_{b}_{rate}.json"
                    write(path, {"values":values.tolist()})
                    conditions[str(rate)] = dict(metrics=metrics(values, teacher, bank), pruned=count,
                        readout_path=str(path), readout_sha256=sha(path))
                from .core import marginal_cost
                cost = marginal_cost(conditions['0.48']['metrics'], conditions['0.52']['metrics'], 56, 70)
                write(root/"probes"/f"block{b:02d}.json", dict(config_sha256=sha(root/"config.json"), block=b,
                    conditions=conditions, costs=cost))
            with patch("experiments.dlm_multiscale_ac50.analysis.validate", return_value=config):
                result = allocate(root)
                self.assertEqual(result, allocate(root))
            for arm in result["allocations"].values():
                self.assertEqual(sum(2*k for k in arm["row_counts"]), 2240)
                self.assertEqual(arm["budget"]["budget_error"], 0)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    unittest.main()
