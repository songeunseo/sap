"""CPU preparation/status and explicit tmux launch for the frozen experiment."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from .artifacts import DEFAULT_ROOT, PLAN, REPO, Progress, freeze, lock, read, sha, write


def cpu_only():
    # Called before numpy/lm_eval/torch imports. Preparing inputs cannot initialize CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def command(root, *args):
    return [sys.executable, "-u", "-m", "experiments.dlm_multiscale_ac50.run", "--root", str(root), *args]


def devices(value):
    result = value.split(",")
    if not 1 <= len(result) <= 2 or len(set(result)) != len(result) or any(not s.isdigit() for s in result):
        raise argparse.ArgumentTypeError("Use one or two distinct GPU indices, e.g. 0 or 0,3")
    return result


def idle_devices(gpus):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    info = {}
    for line in output.splitlines():
        index, uuid, memory = [s.strip() for s in line.split(",")]
        info[index] = (uuid, int(memory))
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    active_uuids = {line.split(",")[0].strip() for line in active.splitlines() if line.strip()}
    for gpu in gpus:
        if gpu not in info:
            raise RuntimeError(f"GPU {gpu} does not exist")
        uuid, memory = info[gpu]
        if uuid in active_uuids or memory > 512:
            raise RuntimeError(f"GPU {gpu} is occupied ({memory} MiB or active compute process); no job launched")


def launch(root, gpus, phase, dry_run):
    # dry-run is deliberately torch-free and does not even query nvidia-smi.
    session = f"dlm_multiscale_ac50_{phase}"
    pipeline = command(root, "pipeline", "--gpus", ",".join(gpus), "--phase", phase)
    log = root / "logs" / f"pipeline_{phase}.log"
    # An existing tmux server does not inherit arbitrary client environment variables.
    environment = {"PYTHONPATH":os.environ.get("PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES":"",
                   "OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS", "2"),
                   "OPENBLAS_NUM_THREADS":os.environ.get("OPENBLAS_NUM_THREADS", "2"),
                   "MKL_NUM_THREADS":os.environ.get("MKL_NUM_THREADS", "2"),
                   "TOKENIZERS_PARALLELISM":"false", "HF_HUB_OFFLINE":"1", "HF_DATASETS_OFFLINE":"1"}
    wrapped = ["env", *[f"{key}={value}" for key, value in environment.items()], *pipeline]
    shell = shlex.join(wrapped) + " >> " + shlex.quote(str(log)) + " 2>&1"
    tmux_command = ["tmux", "new-session", "-d", "-s", session, "-c", str(REPO), shell]
    if dry_run:
        print(json.dumps(dict(dry_run=True, gpu_queried=False, gpu_used=False, phase=phase,
                              tmux_session=session, command=shlex.join(tmux_command)), indent=2))
        return
    from .prepare import validate
    validate(root)
    if subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")
    # The pipeline lock also prevents a development and confirmation launch from overlapping.
    with lock(root / "launch.lock"):
        with lock(root / "pipeline.lock"):
            pass
        for other_phase in ("development", "confirmation"):
            other = f"dlm_multiscale_ac50_{other_phase}"
            if subprocess.run(["tmux", "has-session", "-t", other], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                raise RuntimeError(f"Experiment tmux session already exists: {other}")
        if phase == "confirmation":
            from .analysis import select_confirmation
            select_confirmation(root)
        idle_devices(gpus)
        log.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(tmux_command, check=True)
    print(json.dumps(dict(launched=True, tmux_session=session, log=str(log), phase=phase), indent=2))


def experiment_note(root, phase, status, started):
    value = dict(status=status, phase=phase, start_time=started, updated_at=time.time(),
                 config_sha256=sha(root / "config.json"), obsidian_sync="pending")
    write(root / "experiment_state.json", value)
    text = f"""---
status: {status}
phase: {phase}
obsidian_sync: pending
---
# Multiscale A+C 50% — small GSM8K evaluation

## Objective
Test multiscale context-response preservation against same-bank A/Short/Path/All.
## Hypothesis
Preserving medium/long reveal responses improves allocation and small-sample generation.
## Planned Setup
See {PLAN}. Global sparsity 50%; no full GSM8K/PPL; maximum 200 distinct GSM8K documents.
## Actual Setup
Config: {root / 'config.json'}; SHA256: {value['config_sha256']}; phase: {phase}.
Started (Unix seconds): {started}.
## Progress / Notes
Status: {status}. Per-worker progress and logs are in this directory.
## Results
Read the phase summary if present; no result is inferred from launch.
## Interpretation
Development and confirmation remain separate.
## Decision
Run only the explicitly launched phase. Confirmation is never launched automatically.
## Next Experiment
None scheduled.
## Related Notes
Research/DLM-Pruning/Hypotheses/2026-09-22-Multiscale-AC-Monotone-Reveal-Design.md

This is a local record. Sync to Obsidian Experiments/ when MCP is callable; it is not an Obsidian write receipt.
"""
    (root / "experiment_note.md").write_text(text)


def run_jobs(root, jobs, gpus):
    waiting, active = list(jobs), []
    try:
        while waiting or active:
            used = {row["gpu"] for row in active}
            for gpu in gpus:
                if not waiting or gpu in used:
                    continue
                label, args = waiting.pop(0)
                path = root / "logs" / f"{label}.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = path.open("a")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                proc = subprocess.Popen(command(root, "worker", *args), cwd=REPO, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active.append(dict(process=proc, gpu=gpu, label=label, stream=stream, log=path))
            for row in list(active):
                code = row["process"].poll()
                if code is None:
                    continue
                row["stream"].close()
                active.remove(row)
                if code:
                    raise RuntimeError(f"Worker {row['label']} failed ({code}); see {row['log']}")
            if active:
                time.sleep(1)
    finally:
        for row in active:
            proc = row["process"]
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            row["stream"].close()


def pipeline(root, gpus, phase):
    if not os.environ.get("TMUX"):
        raise RuntimeError("Pipeline must run inside tmux; use launch")
    cpu_only()
    from .prepare import validate
    from .analysis import allocate, select_confirmation, summarize
    progress = Progress(root, "pipeline")
    started = time.time()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    with lock(root / "pipeline.lock"):
        validate(root)
        if not (root / "started.json").exists():
            freeze(root / "started.json", dict(start_time=started, config_sha256=sha(root / "config.json")))
        experiment_note(root, phase, "running", started)
        try:
            if phase == "development":
                progress("reference")
                run_jobs(root, [("reference", ["--job", "reference"])], gpus[:1])
                shards = [list(range(32))] if len(gpus) == 1 else [list(range(16)), list(range(16, 32))]
                progress("probes")
                run_jobs(root, [(f"probes_{i}", ["--job", "probes", "--blocks", ",".join(map(str, shard))])
                                for i, shard in enumerate(shards)], gpus)
                progress("allocation")
                allocation = allocate(root)
                methods = list(dict.fromkeys(allocation["aliases"][m] for m in ("Multi", "Short", "Path", "All", "A")))
                # Historical A+C needs fresh response diagnostics, but reuses its mini100 generations.
                methods.append("legacy_AC")
            else:
                methods = select_confirmation(root)["canonical_methods"]
            progress("gsm8k", total=len(methods))
            run_jobs(root, [(f"{phase}_{m}", ["--job", "candidate", "--method", m, "--split", phase]) for m in methods], gpus)
            result = summarize(root, phase)
            experiment_note(root, phase, result["status"], started)
            progress("complete", completed=len(methods), total=len(methods), scores=result["scores"])
        except BaseException as exc:
            experiment_note(root, phase, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", started)
            progress("failed", error=str(exc))
            raise


def status(root):
    print("DLM Multiscale A+C 50% | " + time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if not (root / "config.json").exists():
        print("Not prepared. Run: bash experiments/dlm_multiscale_ac50/run.sh prepare")
        return
    print("State: " + (read(root / "experiment_state.json")["status"] if (root / "experiment_state.json").exists() else "prepared; not started"))
    print(f"{'Worker':28} {'Stage':20} {'Progress':12} {'GPU':5} {'ETA':12} {'Process'}")
    for path in sorted((root / "progress").glob("*.json")):
        row = read(path)
        try:
            os.kill(row["pid"], 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        finished = row["stage"] in ("complete", "failed")
        state = "finished" if finished else "running" if alive else "stopped/stale"
        eta = f"{row['eta_seconds']/60:.1f} min" if row.get("eta_seconds") is not None and alive and not finished else "-"
        print(f"{row['worker']:28} {row['stage']:20} {str(row['completed'])+'/'+str(row['total']):12} {row.get('gpu','-'):5} {eta:12} {state}")
    for phase in ("development", "confirmation"):
        path = root / f"{phase}_summary.json"
        if path.exists():
            result = read(path)
            print(phase + ": " + json.dumps(result["scores"]))
    print("Logs: " + str(root / "logs"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    actions = parser.add_subparsers(dest="action", required=True)
    p = actions.add_parser("prepare", help="CPU only: freeze banks, requests, baseline receipts")
    p.add_argument("--plan", type=Path, default=PLAN)
    actions.add_parser("validate", help="CPU only: verify frozen artifact hashes")
    actions.add_parser("status", help="CPU only: read progress files; does not query GPUs")
    for name in ("launch", "pipeline"):
        p = actions.add_parser(name)
        p.add_argument("--gpus", type=devices, required=True)
        p.add_argument("--phase", choices=("development", "confirmation"), default="development")
        if name == "launch":
            p.add_argument("--dry-run", action="store_true", help="Print only; no GPU query, tmux or subprocess")
    actions.add_parser("allocate", help="CPU only: combine completed block probes")
    p = actions.add_parser("summarize")
    p.add_argument("--split", choices=("development", "confirmation"), default="development")
    p = actions.add_parser("worker", help="Internal GPU worker; requires tmux and one visible GPU")
    p.add_argument("--job", choices=("reference", "probes", "candidate"), required=True)
    p.add_argument("--blocks")
    p.add_argument("--method", choices=("A", "Short", "Path", "All", "Multi", "legacy_AC"))
    p.add_argument("--split", choices=("development", "confirmation"), default="development")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.action != "worker":
        cpu_only()
    if args.action == "prepare":
        from .prepare import prepare
        with lock(root / "preparation.lock"):
            print(json.dumps(prepare(root, args.plan), indent=2))
    elif args.action == "validate":
        from .prepare import validate
        validate(root)
        print("Frozen inputs verified; GPU not used.")
    elif args.action == "status":
        status(root)
    elif args.action == "launch":
        launch(root, args.gpus, args.phase, args.dry_run)
    elif args.action == "pipeline":
        pipeline(root, args.gpus, args.phase)
    elif args.action == "allocate":
        from .analysis import allocate
        allocate(root)
    elif args.action == "summarize":
        from .analysis import summarize
        print(json.dumps(summarize(root, args.split), indent=2))
    else:
        # Check environment BEFORE importing the module that imports torch.
        if not os.environ.get("TMUX") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise RuntimeError("Use launch: GPU workers require tmux and an explicit visible device")
        from . import gpu
        label = f"worker_{args.job}_{args.method or args.blocks or 'reference'}"
        with lock(root / "locks" / f"{label}.lock"):
            try:
                if args.job == "reference":
                    gpu.reference(root)
                elif args.job == "probes":
                    blocks = [int(s) for s in (args.blocks or "").split(",")]
                    if not blocks or len(set(blocks)) != len(blocks) or any(not 0 <= b < 32 for b in blocks):
                        raise ValueError("Expected unique block indices within 0..31")
                    gpu.probes(root, blocks)
                else:
                    if args.method is None:
                        raise ValueError("Candidate worker requires --method")
                    gpu.candidate(root, args.method, args.split)
            except BaseException as exc:
                Progress(root, label)("failed", error=str(exc))
                raise


if __name__ == "__main__":
    main()
