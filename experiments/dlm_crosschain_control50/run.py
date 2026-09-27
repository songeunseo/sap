"""Audited, restartable four-arm full GSM8K controller."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import numpy as np

from experiments.dlm_multiscale_ac50.artifacts import REPO, checked, freeze, lock, mask_identity, read, sha, write
from experiments.dlm_multiscale_ac50.core import holm
from experiments.dlm_multiscale_ac50.evaluation import evaluate_requests, generator, grade, read_predictions, task_and_protocol
from experiments.dlm_multiscale_ac50.run import idle_devices
from experiments.dlm_ac_screen50.run import _owned_group, _proc_info, proc_identity, terminate_owned
from .core import ARMS, response_metrics
from .prepare import ROOT, OLD, EXPOSED, FULL_IDS, prepare, validate

SESSION = "dlm_crosschain_control50"
MODULE = "experiments.dlm_crosschain_control50.run"
SHARD_SIZE = 128
SHARDS = tuple(tuple(range(i, min(i + SHARD_SIZE, 1319))) for i in range(0, 1319, SHARD_SIZE))


def code_sources():
    return {str(p): sha(p) for p in sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(__file__).parent.glob("*.sh"))}


def launch_gates():
    validate()
    receipt = read(ROOT / "cpu_validation.json")
    audit = read(ROOT / "audit_pass.json")
    if receipt.get("status") != "passed" or receipt.get("config_sha256") != sha(ROOT / "config.json"):
        raise RuntimeError("CPU validation receipt missing/stale")
    if receipt.get("code_sources") != code_sources() or receipt.get("allocation_sha256") != sha(ROOT / "allocation.json"):
        raise RuntimeError("Validated code or allocation changed")
    if audit.get("status") != "passed" or audit.get("config_sha256") != receipt["config_sha256"] or audit.get("cpu_validation_sha256") != sha(ROOT / "cpu_validation.json"):
        raise RuntimeError("Independent code/config audit missing/stale")
    if audit.get("code_sources") != receipt["code_sources"]:
        raise RuntimeError("Audited code changed")


def checks():
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun < 8:
        raise RuntimeError("CPU test gate failed")
    validate()
    receipt = {"status": "passed", "tests_run": result.testsRun,
               "config_sha256": sha(ROOT / "config.json"),
               "allocation_sha256": sha(ROOT / "allocation.json"),
               "code_sources": code_sources(), "time": time.time()}
    write(ROOT / "cpu_validation.json", receipt)
    return receipt


def job_graph():
    jobs = [{"id": "teacher", "kind": "teacher", "dependencies": []}]
    for arm in ARMS:
        jobs.append({"id": f"init_{arm}", "kind": "initialize", "arm": arm, "dependencies": ["teacher"]})
    for shard in range(len(SHARDS)):
        for arm in ARMS:
            jobs.append({"id": f"{arm}_{shard:02d}", "kind": "shard", "arm": arm,
                         "shard": shard, "dependencies": [f"init_{arm}"]})
    return jobs


def model_identity_for(rt, arm, manifest):
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    ident = mask_identity(manifest)
    model_hash = model_sha(rt.model)
    if arm in ("A", "Multi"):
        prior = read(OLD / "candidates" / arm / "model_identity.json")
        if prior["mask_identity"] != ident or prior["sparse_model_sha256"] != model_hash:
            raise RuntimeError(f"Original {arm} physical model did not reproduce")
        old_manifest = read(OLD / "candidates" / arm / "mask_manifest.json")
        if [e["selected_mask"]["mask_sha256"] for e in old_manifest["entries"]] != [e["selected_mask"]["mask_sha256"] for e in manifest["entries"]]:
            raise RuntimeError(f"Original {arm} row mask hashes differ")
    saved = {"config_sha256": rt.config_hash, "mask_identity": ident,
             "sparse_model_sha256": model_hash}
    freeze(ROOT / "candidates" / arm / "model_identity.json", saved)
    return saved


def verified_manifest(rt, arm):
    allocation = read(ROOT / "allocation.json")
    manifest = rt.build(arm)
    if manifest["pruned"] != 3489660928 or [e["selected_mask"]["prune_per_row"] for e in manifest["entries"]] != allocation["allocations"][arm]["row_counts"]:
        raise RuntimeError("Physical mask quota differs from allocation")
    for entry in manifest["entries"]:
        checked(entry["selected_mask"]["path"], entry["selected_mask"]["file_sha256"])
    return manifest


def worker(job_id):
    launch_gates()
    if not os.environ.get("TMUX") or os.environ.get("CUDA_VISIBLE_DEVICES") not in tuple(str(i) for i in range(4)):
        raise RuntimeError("Run worker on one authorized GPU within tmux")
    job = next((x for x in job_graph() if x["id"] == job_id), None)
    if job is None:
        raise ValueError("Unknown job")
    from experiments.dlm_multiscale_ac50.gpu import runtime
    from experiments.projection_capacity_allocation_65.run import DENSE_SHA
    from experiments.wanda_failure_characterization.run_failure_map import model_sha
    from transformers import AutoTokenizer
    import torch
    import resource
    started = time.monotonic()
    attempt = str(time.time_ns())
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    write(ROOT / "attempts" / f"{job_id}_{attempt}.json", {"job": job_id, "attempt": attempt, "gpu": gpu, "started": time.time(), "pid": os.getpid()})
    outcome = "failed"
    error = None
    calls = [0]
    try:
        rt = runtime(ROOT, job_id)
        def tick(*_):
            calls[0] += 1
            rt.memory_check()
        hook = rt.model.register_forward_hook(tick)
        try:
            if job["kind"] == "teacher":
                if model_sha(rt.model) != DENSE_SHA:
                    raise RuntimeError("Dense model SHA differs")
                target = ROOT / "readouts/dense/fresh.json"
                cached = len(read(target)["values"]) if target.exists() else 0
                before = calls[0]
                rt.readouts("dense", "dense:" + DENSE_SHA, "fresh")
                if calls[0] - before != 128 - cached:
                    raise RuntimeError("Fresh teacher resumed forward count changed")
                freeze(ROOT / "done/teacher.json", {"config_sha256": rt.config_hash,
                    "readout_sha256": sha(ROOT / "readouts/dense/fresh.json"),
                    "dense_model_sha256": DENSE_SHA})
            else:
                arm = job["arm"]
                manifest = verified_manifest(rt, arm)
                identity = model_identity_for(rt, arm, manifest)
                if job["kind"] == "initialize":
                    for split in ("calibration", "diagnostic", "fresh"):
                        target = ROOT / "readouts" / arm / f"{split}.json"
                        cached = len(read(target)["values"]) if target.exists() else 0
                        before = calls[0]
                        rt.distortion(arm, identity["mask_identity"], split)
                        if calls[0] - before != 128 - cached:
                            raise RuntimeError(f"Unexpected resumed readout forwards: {arm}/{split}")
                    if model_sha(rt.model) != identity["sparse_model_sha256"]:
                        raise RuntimeError("Model changed during readout")
                    freeze(ROOT / "done" / f"init_{arm}.json", {"arm": arm, "config_sha256": rt.config_hash,
                        "mask_identity": identity["mask_identity"], "model_sha256": identity["sparse_model_sha256"],
                        "readout_sha256": {s: sha(ROOT / "readouts" / arm / f"{s}.json") for s in ("calibration", "diagnostic", "fresh")}})
                else:
                    shard = job["shard"]
                    ids = SHARDS[shard]
                    requests = read(ROOT / "requests.json")["development"]
                    if [requests[i]["example_id"] for i in ids] != list(ids):
                        raise RuntimeError("Shard request order changed")
                    fp = {"config_sha256": rt.config_hash,
                          "requests_sha256": sha(ROOT / "requests.json"),
                          "mask_identity": identity["mask_identity"],
                          "sparse_model_sha256": identity["sparse_model_sha256"],
                          "split": "full", "shard": shard,
                          "protocol_hash": rt.config["protocol_hash"]}
                    task, _, _ = task_and_protocol(read(rt.config["legacy_config"]))
                    tokenizer = AutoTokenizer.from_pretrained(rt.config["model"]["id"], revision=rt.config["model"]["revision"], trust_remote_code=True, local_files_only=True)
                    generate = generator(rt.model, tokenizer, rt.config["evaluation"], rt.banks["calibration"]["mask_id"])
                    def counted(req):
                        before = calls[0]
                        text = generate(req)
                        if calls[0] - before != 256:
                            raise RuntimeError(f"Generation forward count differs for ID {req['example_id']}")
                        return text
                    result = evaluate_requests(ROOT / "gsm8k" / arm / f"shard{shard:02d}",
                        [requests[i] for i in ids], fp, rt.config["protocol_hash"], arm,
                        counted, lambda req, text: grade(task, req, text), rt.progress)
                    if result["total"] != len(ids) or model_sha(rt.model) != identity["sparse_model_sha256"]:
                        raise RuntimeError("Shard count or model SHA changed")
                    freeze(ROOT / "done" / f"{job_id}.json", {"job": job_id, "model_sha256": identity["sparse_model_sha256"],
                        "result_sha256": sha(ROOT / "gsm8k" / arm / f"shard{shard:02d}" / "results.json")})
            outcome = "complete"
        finally:
            hook.remove()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write(ROOT / "costs" / f"{job_id}_{attempt}.json", {"job": job_id, "status": outcome, "error": error,
              "gpu": gpu, "wall_seconds": time.monotonic() - started,
              "forward_calls": calls[0], "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})


def command(*args):
    return [sys.executable, "-B", "-u", "-m", MODULE, *args]


def select_ready(jobs, done, active):
    return next((j for j in jobs if j["id"] not in done and j["id"] not in active and set(j["dependencies"]) <= done), None)


def queue_deadlocked(jobs, done, active):
    return len(done) != len(jobs) and not active and select_ready(jobs, done, set()) is None


def completion(job):
    path = ROOT / "done" / f"{job['id']}.json"
    row = read(path)
    if job["kind"] == "shard":
        arm, index = job["arm"], job["shard"]
        result_path = ROOT / "gsm8k" / arm / f"shard{index:02d}" / "results.json"
        if row["result_sha256"] != sha(result_path) or read(result_path)["total"] != len(SHARDS[index]):
            raise RuntimeError("Completed shard result changed")
    elif job["kind"] == "initialize":
        arm = job["arm"]
        identity = read(ROOT / "candidates" / arm / "model_identity.json")
        if row["config_sha256"] != sha(ROOT / "config.json") or row["mask_identity"] != identity["mask_identity"] or row["model_sha256"] != identity["sparse_model_sha256"]:
            raise RuntimeError("Initialized model receipt changed")
        for split, expected in row["readout_sha256"].items():
            checked(ROOT / "readouts" / arm / f"{split}.json", expected)
    elif job["kind"] == "teacher":
        from experiments.projection_capacity_allocation_65.run import DENSE_SHA
        if row["config_sha256"] != sha(ROOT / "config.json") or row["dense_model_sha256"] != DENSE_SHA:
            raise RuntimeError("Dense teacher receipt changed")
        checked(ROOT / "readouts/dense/fresh.json", row["readout_sha256"])
    return row


def pipeline(gpus):
    if not os.environ.get("TMUX") or not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= set("0123"):
        raise RuntimeError("Expected tmux and distinct GPUs from 0,1,2,3")
    jobs = job_graph()
    active = {}
    def interrupted(*_):
        raise KeyboardInterrupt("Controller interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with lock(ROOT / "pipeline.lock"):
        launch_gates()
        freeze(ROOT / "started.json", {"config_sha256": sha(ROOT / "config.json")})
        base = {"pid": os.getpid(), "start_ticks": proc_identity(os.getpid()),
                "gpus": gpus, "started": time.time(), "config_sha256": sha(ROOT / "config.json")}
        done = set()
        for job in jobs:
            if (ROOT / "done" / f"{job['id']}.json").exists():
                completion(job)
                done.add(job["id"])
        try:
            while len(done) != len(jobs):
                for gpu in gpus:
                    if any(x["gpu"] == gpu for x in active.values()):
                        continue
                    job = select_ready(jobs, done, set(active))
                    if job is None:
                        break
                    idle_devices([gpu])
                    path = ROOT / "logs" / f"{job['id']}.log"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    stream = path.open("a")
                    p = subprocess.Popen(command("worker", "--job", job["id"]), cwd=REPO,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    info = _proc_info(p.pid)
                    active[job["id"]] = {"proc": p, "pid": p.pid, "pgid": info["pgid"],
                        "start_ticks": info["start_ticks"], "session": info["session"],
                        "gpu": gpu, "job": job["id"], "log": str(path), "stream": stream,
                        "started": time.time()}
                for name, row in list(active.items()):
                    code = row["proc"].poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"{name} exited {code}; see {row['log']}")
                    completion(next(j for j in jobs if j["id"] == name))
                    row["stream"].close()
                    del active[name]
                    done.add(name)
                write(ROOT / "execution.json", {**base, "status": "running", "completed": sorted(done),
                    "workers": [{k:v for k,v in row.items() if k not in ("proc", "stream")} for row in active.values()]})
                if active:
                    time.sleep(2)
                elif queue_deadlocked(jobs, done, active):
                    raise RuntimeError("No ready job and no active worker")
            report()
            write(ROOT / "execution.json", {**base, "status": "complete", "completed": sorted(done),
                "workers": [], "ended": time.time()})
        except BaseException as exc:
            terminate_owned(list(active.values()))
            write(ROOT / "execution.json", {**base, "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "completed": sorted(done), "workers": [], "error": repr(exc), "ended": time.time()})
            raise
        finally:
            for row in active.values():
                row["stream"].close()


def launch(gpus):
    if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= set("0123"):
        raise RuntimeError("Use distinct authorized GPUs 0,1,2,3")
    launch_gates()
    with lock(ROOT / "launch.lock"):
        with lock(ROOT / "pipeline.lock"):
            pass
        if subprocess.run(["tmux", "has-session", "-t", SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError("Experiment tmux session already exists")
        state = read(ROOT / "execution.json") if (ROOT / "execution.json").exists() else {}
        if any(_owned_group(x) for x in state.get("workers", [])):
            raise RuntimeError("Prior owned worker still active")
        idle_devices(gpus)
        env_keys = ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
        args = ["env", "CUDA_VISIBLE_DEVICES=", *[f"{k}={os.environ.get(k, '')}" for k in env_keys],
                *command("pipeline", "--gpus", ",".join(gpus))]
        log = ROOT / "logs/controller.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        tmux_cmd = shlex.join(args) + " >> " + shlex.quote(str(log)) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "-c", str(REPO), tmux_cmd], check=True)
    print(f"Launched {SESSION} on GPUs {','.join(gpus)}")


def stop():
    path = ROOT / "execution.json"
    if not path.exists():
        return
    state = read(path)
    pid = state.get("pid")
    if pid and proc_identity(pid) == state.get("start_ticks"):
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if MODULE not in cmdline or "pipeline" not in cmdline:
            raise RuntimeError("Recorded controller identity points to unrelated process")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and proc_identity(pid) == state["start_ticks"]:
            time.sleep(.2)
        if proc_identity(pid) == state["start_ticks"]:
            raise RuntimeError("Controller did not exit")
    terminate_owned(state.get("workers", []))
    state = read(path)
    if state.get("status") == "running":
        write(path, {**state, "status": "interrupted", "workers": [], "ended": time.time()})


def status():
    state = read(ROOT / "execution.json") if (ROOT / "execution.json").exists() else {"status": "prepared"}
    print("Cross-chain C control |", state["status"])
    print("Configuration:", sha(ROOT / "config.json") if (ROOT / "config.json").exists() else "missing")
    for arm in ARMS:
        correct = count = 0
        elapsed = []
        for path in (ROOT / "gsm8k" / arm).glob("shard*/examples/*.json"):
            row = read(path)["row"]
            count += 1
            correct += int(row["correct"])
            elapsed.append(row["eval_seconds"])
        workers = [x for x in state.get("workers", []) if x["job"].startswith(arm + "_") or x["job"] == f"init_{arm}"]
        details = ", ".join(f"{w['job']} GPU{w['gpu']}" for w in workers)
        eta = f" ETA~{np.mean(elapsed)*(1319-count)/max(1,len(workers))/3600:.1f}h" if elapsed and count < 1319 else ""
        print(f"{arm:13} {count:4}/1319 correct {correct:4}{eta} {details}")
    if state.get("error"):
        print("Failure:", state["error"])


def paired_stats(a, b, draws=10000, seed=20260927):
    from scipy.stats import binomtest
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError("Unpaired outcomes")
    gain = int(np.sum(bb & ~aa))
    loss = int(np.sum(aa & ~bb))
    n = gain + loss
    p = float(binomtest(min(gain, loss), n, .5, alternative="two-sided").pvalue) if n else 1.0
    delta = bb.astype(float) - aa.astype(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for start in range(0, draws, 500):
        take = min(500, draws-start)
        ids = rng.integers(0, len(delta), size=(take, len(delta)))
        samples[start:start+take] = delta[ids].mean(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return {"gain": gain, "loss": loss, "net": gain-loss,
            "difference_pp": float(100*delta.mean()), "exact_mcnemar_p": p,
            "paired_bootstrap_95pp_unadjusted": [float(100*lo), float(100*hi)],
            "bootstrap_draws": draws, "bootstrap_seed": seed}


def span_bootstrap_difference(reference, candidate, seed=20260927, draws=10000):
    """Paired bootstrap over the eight source spans, never over edges or tokens."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != (8,) or b.shape != (8,):
        raise ValueError("Expected eight paired spans")
    differences = b-a
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 8, size=(draws, 8))
    lo, hi = np.quantile(differences[ids].mean(axis=1), [.025, .975])
    return {"mean_difference": float(differences.mean()), "paired_bootstrap_95_unadjusted": [float(lo), float(hi)],
            "span_count": 8, "bootstrap_draws": draws, "seed": seed}


def report():
    launch_gates()
    if not all((ROOT / "done" / f"{j['id']}.json").exists() for j in job_graph()):
        raise RuntimeError("Full 4-arm run incomplete")
    requests = read(ROOT / "requests.json")["development"]
    c = read(ROOT / "config.json")
    from experiments.projection_capacity_allocation_65.run import DENSE_SHA
    completion({"id": "teacher", "kind": "teacher"})
    for arm in ARMS:
        completion({"id": f"init_{arm}", "kind": "initialize", "arm": arm})
        manifest = read(ROOT / "candidates" / arm / "mask_manifest.json")
        identity = read(ROOT / "candidates" / arm / "model_identity.json")
        allocation = read(ROOT / "allocation.json")["allocations"][arm]
        if (manifest["config_sha256"] != sha(ROOT / "config.json")
                or manifest["allocation_sha256"] != sha(ROOT / "allocation.json")
                or manifest["pruned"] != 3489660928
                or [e["selected_mask"]["prune_per_row"] for e in manifest["entries"]] != allocation["row_counts"]
                or mask_identity(manifest) != identity["mask_identity"]):
            raise RuntimeError("Final manifest identity or quota changed")
        for entry in manifest["entries"]:
            checked(entry["selected_mask"]["path"], entry["selected_mask"]["file_sha256"])
    all_rows = {}
    for arm in ARMS:
        identity = read(ROOT / "candidates" / arm / "model_identity.json")
        rows = []
        for index, ids in enumerate(SHARDS):
            completion({"id": f"{arm}_{index:02d}", "kind": "shard", "arm": arm, "shard": index})
            fp = {"config_sha256": sha(ROOT / "config.json"), "requests_sha256": sha(ROOT / "requests.json"),
                  "mask_identity": identity["mask_identity"], "sparse_model_sha256": identity["sparse_model_sha256"],
                  "split": "full", "shard": index, "protocol_hash": c["protocol_hash"]}
            folder = ROOT / "gsm8k" / arm / f"shard{index:02d}"
            subset = [requests[i] for i in ids]
            rr = read_predictions(folder, subset, fp, c["protocol_hash"])
            result = read(folder / "results.json")
            if result["fingerprint"] != fp or result["total"] != len(ids) or result["correct"] != sum(r["correct"] for r in rr):
                raise RuntimeError("Shard result identity/count changed")
            if result["predictions_sha256"] != sha(folder / "predictions.json") or rr != read(folder / "predictions.json"):
                raise RuntimeError("Shard checkpoint and aggregate differ")
            rows.extend(rr)
        if [r["example_id"] for r in rows] != list(FULL_IDS):
            raise RuntimeError("Missing/duplicate/out-of-order full results")
        task, _, _ = task_and_protocol(read(c["legacy_config"]))
        for req, row in zip(requests, rows, strict=True):
            graded = grade(task, req, row["generated_text"])
            if row["correct"] != graded["correct"] or row["extracted_answer"] != graded["extracted_answer"]:
                raise RuntimeError(f"Official strict-match regrade differs for {arm}/{req['example_id']}")
        all_rows[arm] = rows
        write(ROOT / "gsm8k" / arm / "predictions.json", rows)
    slices = {"full_1319": list(FULL_IDS), "previously_seen_200": list(EXPOSED),
              "primary_remaining_1119": sorted(set(FULL_IDS) - set(EXPOSED))}
    scores = {group: {arm: {"correct": sum(all_rows[arm][i]["correct"] for i in ids), "total": len(ids)} for arm in ARMS}
              for group, ids in slices.items()}
    comparisons = {}
    for group, ids in slices.items():
        baseline = [all_rows["Multi"][i]["correct"] for i in ids]
        comparisons[group] = {}
        for arm in ("A", "Cross", "CrossMatched"):
            control = [all_rows[arm][i]["correct"] for i in ids]
            comparisons[group][f"Multi-{arm}"] = paired_stats(control, baseline)
        if group == "primary_remaining_1119":
            holm(comparisons[group])
    diagnostics = {}
    for split in ("calibration", "diagnostic", "fresh"):
        bank = read(c["banks"][split]["path"])
        dense_path = ROOT / "readouts/dense" / f"{split}.json"
        dense_row = read(dense_path)
        if dense_row["fingerprint"] != {"config_sha256": sha(ROOT / "config.json"),
            "mask_identity": "dense:" + DENSE_SHA, "bank_sha256": c["banks"][split]["sha256"]} or not dense_row["complete"]:
            raise RuntimeError("Dense diagnostic source identity changed")
        if split != "fresh":
            dense_source = OLD / "readouts/dense" / f"{split}.json"
            checked(dense_source, dense_row["reused_source_sha256"])
            if dense_row["values"] != read(dense_source)["values"]:
                raise RuntimeError("Reused dense wrapper differs from frozen source")
        teacher = dense_row["values"]
        diagnostics[split] = {}
        for arm in ARMS:
            path = ROOT / "readouts" / arm / f"{split}.json"
            identity = read(ROOT / "candidates" / arm / "model_identity.json")
            row = read(path)
            if row["fingerprint"] != {"config_sha256": sha(ROOT / "config.json"),
                                      "mask_identity": identity["mask_identity"],
                                      "bank_sha256": c["banks"][split]["sha256"]} or not row["complete"]:
                raise RuntimeError("Diagnostic readout identity differs")
            if row.get("reused_source_sha256"):
                reused_source = OLD / "readouts" / arm / f"{split}.json"
                checked(reused_source, row["reused_source_sha256"])
                if row["values"] != read(reused_source)["values"]:
                    raise RuntimeError("Reused candidate wrapper differs from frozen source")
            diagnostics[split][arm] = response_metrics(row["values"], teacher, bank)
    diagnostic_comparisons = {}
    diagnostic_keys = ("A", "C1_natural", "C2_natural", "C4_natural", "C_natural",
                       "C1_cross", "C2_cross", "C4_cross", "C_cross",
                       "query_CE", "response_sign_flip_rate")
    for arm in ("A", "Cross", "CrossMatched"):
        diagnostic_comparisons[f"Multi-{arm}"] = {}
        for key in diagnostic_keys:
            control = [r[key] for r in diagnostics["fresh"][arm]["sequences"]]
            natural = [r[key] for r in diagnostics["fresh"]["Multi"]["sequences"]]
            diagnostic_comparisons[f"Multi-{arm}"][key] = span_bootstrap_difference(control, natural)
    result = {"status": "complete", "config_sha256": sha(ROOT / "config.json"),
        "allocation_sha256": sha(ROOT / "allocation.json"), "beta": read(ROOT / "allocation.json")["beta"],
        "scores": scores, "comparisons": comparisons, "diagnostics": diagnostics,
        "fresh_diagnostic_span_comparisons": diagnostic_comparisons,
        "primary_family": ["Multi-A", "Multi-Cross", "Multi-CrossMatched"],
        "primary_sample": "remaining 1119; previously examined 200 removed by frozen IDs",
        "limitations": ["one fixed calibration bank and allocation", "past project exposure of remaining 1119 not ruled out",
                        "bootstrap intervals are unadjusted", "no layer-causal attribution"],
        "costs": {"attempts": len(list((ROOT / "costs").glob("*.json"))),
                  "forward_calls_recorded": sum(read(p).get("forward_calls", 0) for p in (ROOT / "costs").glob("*.json")),
                  "nominal_generation_forwards": 5276*256, "nominal_state_forwards": 1152}}
    write(ROOT / "report.json", result)
    lines = ["# Full GSM8K natural/cross-chain C control", "", "| Arm | Full 1319 | Seen 200 | Primary 1119 |",
             "|---|---:|---:|---:|"]
    for arm in ARMS:
        vals = [scores[g][arm] for g in slices]
        lines.append(f"| {arm} | " + " | ".join(f"{v['correct']}/{v['total']}" for v in vals) + " |")
    lines += ["", "Primary comparisons: exact McNemar, Holm over three preregistered contrasts; paired bootstrap 95% intervals unadjusted.", "",
              "| Comparison | Net | Gain/loss | Exact p | Holm p | 95% CI (pp) |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in comparisons["primary_remaining_1119"].items():
        lines.append(f"| {name} | {row['net']:+d} | {row['gain']}/{row['loss']} | {row['exact_mcnemar_p']:.5g} | {row['holm_p']:.5g} | {row['paired_bootstrap_95pp_unadjusted']} |")
    lines += ["", "Diagnostics, per-span values, raw predictions, masks, and cost records are in report.json and output/."]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "validate", "status", "report", "stop"):
        sub.add_parser(name)
    for name in ("launch", "pipeline"):
        p = sub.add_parser(name)
        p.add_argument("--gpus", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--job", required=True)
    args = parser.parse_args(argv)
    if args.cmd != "worker":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.cmd == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.cmd == "validate":
        print(json.dumps(checks(), indent=2))
    elif args.cmd == "launch":
        launch(args.gpus.split(","))
    elif args.cmd == "pipeline":
        pipeline(args.gpus.split(","))
    elif args.cmd == "worker":
        with lock(ROOT / "locks" / f"{args.job}.lock"):
            worker(args.job)
    elif args.cmd == "status":
        status()
    elif args.cmd == "report":
        print(json.dumps(report(), indent=2))
    else:
        stop()


if __name__ == "__main__":
    main()
