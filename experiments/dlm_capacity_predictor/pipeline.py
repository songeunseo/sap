#!/usr/bin/env python3
"""Resource-safe tmux queue. Default boundary is raw diagnostics/candidate freezing."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

# Capture controller/core bytes before importing local helpers; check again before
# snapshotting and dispatch. Stage implementations execute in fresh subprocesses.
_BOOT_PATHS = [Path(__file__).resolve(), Path(__file__).resolve().with_name('core.py')]
_BOOT_DIGESTS = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in _BOOT_PATHS}
from experiments.dlm_capacity_predictor.core import dependency_complete, write_json

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
DEPENDENCY = 'projection_capacity_followup65_downstream'
DEPENDENCY_RESULT = REPO/'experiments/projection_capacity_followup_65/downstream.json'


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_snapshot(snapshot):
    for path, digest in snapshot.items():
        if not Path(path).is_file() or file_digest(path) != digest:
            raise RuntimeError(f'code/config changed after queue snapshot: {path}')


def dependency_state(session_names, document, gpu_pids):
    if DEPENDENCY in session_names:
        return 'waiting_existing_gsm8k'
    if not dependency_complete(document):
        raise RuntimeError('historical GSM8K session ended without complete mini/full results')
    if gpu_pids:
        return 'waiting_gpu0'
    return 'ready'


def observe_dependency():
    sessions = subprocess.run(['tmux', 'list-sessions', '-F', '#{session_name}'],
                              text=True, capture_output=True, check=False)
    if sessions.returncode and 'no server running' not in sessions.stderr and 'no sessions' not in sessions.stderr:
        raise RuntimeError(f'cannot query tmux: {sessions.stderr}')
    if DEPENDENCY in sessions.stdout.splitlines():
        return 'waiting_existing_gsm8k'
    gpu = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                          '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    pids = [int(line.strip()) for line in gpu.stdout.splitlines() if line.strip()]
    document = json.loads(DEPENDENCY_RESULT.read_text())
    return dependency_state(sessions.stdout.splitlines(), document, pids)


def snapshot_paths(repo=REPO, root=ROOT):
    result = subprocess.run(['rg', '--files', '-g', '*.py'], cwd=repo,
                            text=True, capture_output=True, check=True)
    return sorted(set([Path(repo)/p for p in result.stdout.splitlines()] +
                      [Path(root)/p for p in ('config.json', 'PLAN.md', 'run_tmux.sh')]))


def wait_for_ready(through):
    previous = None
    while True:
        state = observe_dependency()
        if state != previous:
            status(state, through=through)
            previous = state
        if state == 'ready':
            return
        time.sleep(30)


def stage_commands(through):
    prefix = [sys.executable, '-m', 'experiments.dlm_capacity_predictor.']
    def command(module, *args):
        return prefix[:2] + [prefix[2]+module] + list(args)
    stages = [('audit', command('audit_existing', '--output-dir', str(ROOT))),
              ('dense_collection', command('collect_dense')),
              ('dense_analysis', command('analyze', '--dense'))]
    if through in ('heldout', 'downstream'):
        stages += [('freeze_states', command('freeze_states')), ('heldout', command('evaluate'))]
    if through == 'downstream':
        stages += [('downstream', command('evaluate', '--downstream'))]
    return stages


def run_stage(name, command, repo=REPO, output_root=ROOT):
    output_root = Path(output_root)
    logs = output_root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log = logs/f'{name}.log'
    started = time.monotonic()
    with log.open('a') as handle:
        result = subprocess.run(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT, check=False)
    receipt = dict(stage=name, command=command, returncode=result.returncode,
                   status='complete' if result.returncode == 0 else 'failed',
                   wall_seconds=time.monotonic()-started, log=str(log),
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output_root/f'stage_{name}.json', receipt)
    result.check_returncode()
    return receipt


def status(state, **kwargs):
    row = dict(status=state, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    write_json(ROOT/'status.json', row)
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--through', choices=['diagnostics', 'heldout', 'downstream'], default='diagnostics')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(dict(dependency=observe_dependency(), through=args.through,
                              stages=[n for n, _ in stage_commands(args.through)])))
        return
    (ROOT/'runtime').mkdir(parents=True, exist_ok=True)
    with (ROOT/'runtime/pipeline.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_snapshot(_BOOT_DIGESTS)
        paths = snapshot_paths()
        snapshot = {str(path): file_digest(path) for path in paths}
        # Immutable per launch, not a mutable promise that new code was reviewed.
        snapshot_path = ROOT/'runtime'/f'execution_snapshot_{os.getpid()}.json'
        write_json(snapshot_path, snapshot, frozen=True)
        try:
            wait_for_ready(args.through)
            verify_snapshot(_BOOT_DIGESTS)
            verify_snapshot(snapshot)
            # Audit is mandatory before dense collection, not only a final report.
            from experiments.dlm_capacity_predictor.audit_existing import run_audit
            audit = run_audit(REPO)
            if audit['status'] != 'complete':
                raise RuntimeError('historical audit has pending/unverified results')
            for name, command in stage_commands(args.through):
                if name in ('dense_collection', 'heldout', 'downstream'):
                    wait_for_ready(args.through)
                verify_snapshot(_BOOT_DIGESTS)
                verify_snapshot(snapshot)
                status('running', stage=name, through=args.through)
                run_stage(name, command)
            status('diagnostics_complete' if args.through == 'diagnostics' else 'complete',
                   through=args.through,
                   next_step='review raw statistics and candidates before further research conclusions')
        except BaseException as error:
            status('failed', error=str(error), traceback=traceback.format_exc(), through=args.through)
            raise


if __name__ == '__main__':
    main()
