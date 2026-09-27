import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.dlm_multiscale_ac50.artifacts import read, sha, write

from .integrity import freeze_checked, read_checked


def _refs():
    # Deliberately vary block 1 so a whole-model quota calculation cannot pass.
    return [
        {"name": f"r{i:03d}", "shape": [1, 10 if not 7 <= i < 14 else 20]}
        for i in range(224)
    ]

def _metrics(mask, a):
    rows=[{"A":a, "C":a / 2, "AC":a * 1.5}]
    return dict(fingerprint={"config_sha256":"cfg", "mask_identity":mask}, mean=rows[0], rows=rows)


class ProbeIntegrityTests(unittest.TestCase):
    def test_probe_uses_block_local_quota_and_sealed_metrics(self):
        from . import allocation
        refs=_refs()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); paths={}; conditions={}
            for rate, value in ((0.48, 2.0), (0.52, 3.0)):
                key=f"{rate:.2f}"; mask=f"mask-{key}"
                metrics=_metrics(mask, value)
                mp=root/f"metrics-{key}.json"; freeze_checked(mp, metrics)
                conditions[key]=dict(
                    pruned=sum(int(r["shape"][1] * rate) * r["shape"][0] for r in refs[7:14]),
                    mask_identity=mask, metrics={k:metrics[k] for k in ("mean", "rows")},
                    metrics_path=str(mp), metrics_sha256=sha(mp),
                )
                paths[key]=mp
            delta=conditions["0.52"]["pruned"]-conditions["0.48"]["pruned"]
            costs={k:(conditions["0.52"]["metrics"]["mean"][k]-conditions["0.48"]["metrics"]["mean"][k])/delta for k in ("A", "AC")}
            probe=root/"block01.json"
            freeze_checked(probe, dict(config_sha256="cfg", family="Square", block=1, conditions=conditions, costs=costs))
            row=allocation._probe(probe, "Square", 1, "cfg", refs)
            self.assertEqual(row["conditions"]["0.48"]["pruned"], 7 * 9)
            # Finite payload corruption is rejected before allocation can consume it.
            bad=read(paths["0.48"]);bad["mean"]["A"]=99.0;write(paths["0.48"], bad)
            with self.assertRaisesRegex(ValueError, "metric source changed"):
                allocation._probe(probe, "Square", 1, "cfg", refs)


class AllocationResumeTests(unittest.TestCase):
    def test_allocation_rejects_changed_probe_source(self):
        from . import allocation
        refs=_refs()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);multi=root/"multi";multi.mkdir()
            refs_path=multi/"refs.json";write(refs_path, {"entries":refs})
            write(multi/"config.json", {"legacy_manifests":{"uniform":{"path":str(refs_path)}}})
            manifest_path=root/"manifest.json";write(manifest_path, {"placeholder":True})
            allocations=root/"allocations";allocations.mkdir()
            probes=root/"probes"/"Square";probes.mkdir(parents=True)
            metrics_root=root/"metrics";metrics_root.mkdir()
            bank_sha="bank-sha";manifest={"target":sum(int(r["shape"][1] * 0.5) * r["shape"][0] for r in refs), "banks":{"Square_calibration":{"sha256":bank_sha}}}
            # Match the collector's sealed metric/probe schema for every block.
            for block in range(32):
                conditions={}
                for rate, value in ((0.48, float(block + 1)), (0.52, float(block + 2))):
                    key=f"{rate:.2f}";mask=f"mask-{block}-{key}"
                    metrics=_metrics(mask, value)
                    metrics["fingerprint"].update(config_sha256=sha(manifest_path), bank_sha256=bank_sha, readout="Square-v1")
                    mp=metrics_root/f"{block:02d}-{key}.json";freeze_checked(mp, metrics)
                    block_refs=refs[block*7:block*7+7]
                    pruned=sum(int(r["shape"][1]*rate)*r["shape"][0] for r in block_refs)
                    conditions[key]=dict(pruned=pruned, mask_identity=mask, metrics={k:metrics[k] for k in ("mean", "rows")}, metrics_path=str(mp), metrics_sha256=sha(mp))
                delta=conditions["0.52"]["pruned"]-conditions["0.48"]["pruned"]
                costs={k:(conditions["0.52"]["metrics"]["mean"][k]-conditions["0.48"]["metrics"]["mean"][k])/delta for k in ("A", "AC")}
                freeze_checked(probes/f"block{block:02d}.json", dict(config_sha256=sha(manifest_path), family="Square", block=block, conditions=conditions, costs=costs))
            import numpy as np
            def fake_rank(scores): return np.full(32, 0.5, dtype=float)
            def fake_exact(entries, rates, target): return ([int(r["shape"][1] * 0.5) for r in entries], 0)
            with patch.object(allocation, "ROOT", root), patch.object(allocation, "MULTI", multi), patch.object(allocation, "validate", return_value=manifest), patch.object(allocation, "exact_row_counts", fake_exact), patch("experiments.dlm_multiscale_ac50.core.rank_rates", fake_rank):
                allocation.allocate("Square")
                self.assertEqual(allocation.load_allocation("Square")["family"], "Square")
                probe=probes/"block00.json";probe.write_text(probe.read_text()+"\n")
                with self.assertRaisesRegex(ValueError, "probe changed"):
                    allocation.load_allocation("Square")


class TeacherCacheTests(unittest.TestCase):
    def test_teacher_vectors_interrupt_resume_and_hash_mismatch(self):
        import numpy as np
        from . import worker
        pairs=[dict(before=[[1]], after=[[2]], query=[0], gold=[1]) for _ in range(80)]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);w=worker.Worker.__new__(worker.Worker)
            w.m={"memory":{"vocab":3}};w.calls=0;w.label="fake";w.checked_teachers=set();w.progress=lambda *a, **k: None
            w.bank=lambda family, split: {"pairs":pairs}
            w.fp=lambda family, split, identity: {"config_sha256":"cfg", "bank_sha256":"bank", "mask_identity":identity, "readout":family+"-v1"}
            attempts=[0]
            def interrupted_logits(ids, query):
                attempts[0]+=1
                if attempts[0]==5:
                    raise RuntimeError("simulated interruption")
                return np.asarray([[1., 0., -1.]], dtype=np.float32)
            w.logits=interrupted_logits
            with patch.object(worker, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    w.teacher_vectors("calibration")
                w.logits=lambda ids, query: np.asarray([[1., 0., -1.]], dtype=np.float32)
                w.teacher_vectors("calibration")
                complete=read_checked(root/"teachers/Vector/calibration/complete.json", {"fingerprint":w.fp("Vector", "calibration", "dense")})
                self.assertEqual(len(complete["files"]), 160)
                first=Path(complete["files"][0]["path"]);first.write_bytes(first.read_bytes()+b"corrupt")
                with self.assertRaisesRegex(RuntimeError, "SHA256 changed"):
                    w.teacher_vectors("calibration")


if __name__ == "__main__":
    unittest.main()
