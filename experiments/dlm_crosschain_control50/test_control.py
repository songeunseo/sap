"""CPU gates for objective, source, budget, and result identities."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.dlm_multiscale_ac50.artifacts import DEFAULT_ROOT as OLD, ReadoutStore, read, sha, write
from experiments.dlm_multiscale_ac50.evaluation import evaluate_requests, grade, read_predictions, task_and_protocol
from experiments.dlm_multiscale_ac50.core import metrics
from experiments.dlm_crosschain_control50.core import ARMS, EDGES, edge_graph, graph_invariants, response_metrics
from experiments.dlm_crosschain_control50.prepare import BETA_EXPECTED, EXPOSED, FULL_IDS, ROOT, SOURCE, VERIFICATION, requests_check
from experiments.dlm_crosschain_control50.run import SHARDS, job_graph, paired_stats, span_bootstrap_difference, queue_deadlocked, select_ready


class CrosschainControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = read(OLD / "bank_calibration.json")
        cls.teacher = read(OLD / "readouts/dense/calibration.json")["values"]

    def test_zero_residual_is_zero_for_both_pairings(self):
        row = response_metrics(self.teacher, self.teacher, self.bank)["mean"]
        for key in ("A", "C_natural", "C_cross", "Multi", "Cross"):
            self.assertEqual(row[key], 0)

    def test_manual_cross_edges_and_both_chain_orientations(self):
        dense = np.zeros((128, 8))
        residual = np.zeros((8, 2, 8, 8))
        residual[0, 0, 0, :] = 2
        residual[0, 1, 1, :] = 3
        observed = response_metrics(residual.reshape(128, 8), dense, self.bank)
        manual = np.mean([(residual[0, 1, j] - residual[0, 0, i])**2 for i,j in EDGES["C1"]] +
                         [(residual[0, 0, j] - residual[0, 1, i])**2 for i,j in EDGES["C1"]])
        self.assertAlmostEqual(observed["sequences"][0]["C1_cross"], manual)

    def test_chain_swap_and_residual_response_identity(self):
        rng = np.random.default_rng(22)
        dense = rng.normal(size=(8, 2, 8, 8))
        sparse = dense + rng.normal(size=dense.shape)
        a = response_metrics(sparse.reshape(128,8), dense.reshape(128,8), self.bank)
        b = response_metrics(sparse[:, ::-1].reshape(128,8), dense[:, ::-1].reshape(128,8), self.bank)
        for key in ("A", "C_natural", "C_cross", "Multi", "Cross"):
            self.assertAlmostEqual(a["mean"][key], b["mean"][key])
        e = sparse - dense
        i,j = EDGES["C2"][0]
        self.assertTrue(np.allclose(e[0,1,j]-e[0,0,i],
            (sparse[0,1,j]-sparse[0,0,i])-(dense[0,1,j]-dense[0,0,i])))

    def test_same_state_support_and_graph_spectrum(self):
        self.assertEqual(len(edge_graph("natural")), 24)
        self.assertEqual(len(edge_graph("cross")), 24)
        for name in ("C1", "C2", "C4"):
            self.assertEqual(len(EDGES[name]), 4)
        self.assertLess(graph_invariants()["max_eigenvalue_difference"], 1e-12)

    def test_natural_reproduces_original_saved_metrics(self):
        saved = read(OLD / "readouts/uniform/calibration.json")["values"]
        new = response_metrics(saved, self.teacher, self.bank)["mean"]
        old = metrics(saved, self.teacher, self.bank)["mean"]
        for name in ("A", "Multi", "C1", "C2", "C4"):
            key = name if name in ("A", "Multi") else name + "_natural"
            self.assertAlmostEqual(new[key], old[name], places=12)

    def test_uniform_beta_only_and_frozen(self):
        uniform = read(OLD / "readouts/uniform/calibration.json")["values"]
        row = response_metrics(uniform, self.teacher, self.bank)["mean"]
        beta = row["C_natural"] / row["C_cross"]
        self.assertAlmostEqual(beta, BETA_EXPECTED, places=11)
        self.assertAlmostEqual(beta*row["C_cross"], row["C_natural"], places=12)
        allocation = read(ROOT / "allocation.json")
        self.assertEqual(beta, allocation["beta"])
        self.assertEqual(allocation["beta_source"], str(OLD / "readouts/uniform/calibration.json"))

    def test_four_allocations_exact_and_reference_counts(self):
        current = read(ROOT / "allocation.json")["allocations"]
        old = read(OLD / "allocation.json")["allocations"]
        self.assertEqual(set(current), set(ARMS))
        self.assertEqual(len({tuple(current[a]["row_counts"]) for a in ARMS}), 4)
        for arm in ARMS:
            self.assertEqual(current[arm]["budget"]["corrected_pruned"], 3489660928)
        for arm in ("A", "Multi"):
            self.assertEqual(current[arm]["row_counts"], old[arm]["row_counts"])
            self.assertEqual(current[arm]["rates"], old[arm]["rates"])
        self.assertEqual(sum(x!=y for x,y in zip(current["Cross"]["row_counts"], current["CrossMatched"]["row_counts"])), 14)

    def test_full_request_freeze_and_primary_split(self):
        rows = read(ROOT / "requests.json")["development"]
        split = requests_check(rows)
        self.assertEqual(len(rows), 1319)
        self.assertEqual(len(EXPOSED), 200)
        self.assertEqual(len(split["primary_ids"]), 1119)
        self.assertEqual(set(split["primary_ids"]) & set(EXPOSED), set())

    def test_fresh_source_identity(self):
        verification = read(VERIFICATION)
        self.assertEqual(sha(SOURCE), verification["manifest_sha256"])
        self.assertTrue(verification["all_three_splits_disjoint"])
        bank = read(ROOT / "bank_fresh.json")
        self.assertEqual(bank["seed"], 20260927)
        self.assertEqual(bank["states"], 128)
        self.assertEqual([c["sequence_index"] for c in bank["chains"][::2]], list(range(16,24)))

    def test_shards_and_job_graph_exact_coverage(self):
        self.assertEqual(len(SHARDS), 11)
        self.assertEqual(len(SHARDS[-1]), 39)
        self.assertEqual([i for shard in SHARDS for i in shard], list(FULL_IDS))
        jobs = job_graph()
        self.assertEqual(len(jobs), 49)
        self.assertEqual(len({j["id"] for j in jobs}), 49)
        self.assertEqual(sum(j["kind"]=="shard" for j in jobs), 44)

    def test_teacher_completion_unlocks_initializers(self):
        jobs = job_graph()
        done = {"teacher"}
        self.assertEqual(select_ready(jobs, done, set())["id"], "init_A")
        self.assertFalse(queue_deadlocked(jobs, done, {}))

    def test_checkpoint_wrong_identity_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            write(folder / "examples/0000.json", {"fingerprint": {"wrong": True}, "row_sha256": "x", "row": {}})
            with self.assertRaises(RuntimeError):
                read_predictions(folder, [{"example_id": 0}], {"correct": True}, "protocol")

    def test_readout_resume_completes_only_missing_nodes(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "readout.json"
            fp = {"bank": "fixed", "model": "fixed"}
            first = ReadoutStore(path, fp, 3, 2)
            first.append([1.0, 2.0])
            resumed = ReadoutStore(path, fp, 3, 2)
            self.assertEqual(len(resumed.values), 1)
            resumed.append([3.0, 4.0])
            resumed.append([5.0, 6.0])
            self.assertEqual(len(resumed.complete()), 3)
            self.assertEqual(len(ReadoutStore(path, fp, 3, 2).values), 3)

    def test_document_checkpoint_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            requests = read(ROOT / "requests.json")["development"][:2]
            config = read(ROOT / "config.json")
            task, _, _ = task_and_protocol(read(config["legacy_config"]))
            counter = [0]
            def generate(_):
                counter[0] += 1
                return "#### 0"
            fp = {"fixed": True}
            for _ in range(2):
                evaluate_requests(folder, requests, fp, config["protocol_hash"], "A", generate,
                    lambda req, text: grade(task, req, text), lambda *args, **kwargs: None)
            self.assertEqual(counter[0], 2)
            path = folder / "examples/0000.json"
            damaged = read(path)
            damaged["row_sha256"] = "tampered"
            write(path, damaged)
            with self.assertRaises(RuntimeError):
                read_predictions(folder, requests, fp, config["protocol_hash"])

    def test_span_bootstrap_uses_paired_span_units(self):
        a = [0.0]*8
        b = [1.0]*8
        result = span_bootstrap_difference(a,b,draws=1000)
        self.assertEqual(result["mean_difference"],1.)
        self.assertEqual(result["paired_bootstrap_95_unadjusted"],[1.,1.])
        self.assertEqual(result["span_count"],8)

    def test_paired_statistics_direction_and_exact_p(self):
        result = paired_stats([False, False, True, True], [True, True, True, True], draws=1000)
        self.assertEqual((result["gain"], result["loss"], result["net"]), (2,0,2))
        self.assertEqual(result["exact_mcnemar_p"], .5)
        self.assertEqual(result["difference_pp"], 50.)


if __name__ == "__main__":
    unittest.main()
