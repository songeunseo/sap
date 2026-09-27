"""Small atomic artifacts, content identities, locks, and resumable scalar output."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(__file__).resolve().parent / "output"
PLAN = REPO / "research/ac_multiscale_experiment_plan_2026-09-22.json"
LEGACY = REPO / "experiments/dlm_context_response50"
ARMS = ("A", "Short", "Path", "All", "Multi")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Frozen artifact mismatch: {path}")
    else:
        write(path, value)
    return value


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 changed: {path}; expected {expected}, got {actual}")


def mask_identity(manifest):
    return digest([[r["name"], r["shape"], r["selected_mask"]["mask_sha256"]]
                   for r in manifest["entries"]])


class Progress:
    def __init__(self, root, worker):
        self.path = Path(root) / "progress" / f"{worker}.json"
        self.worker = worker
        self.started = time.time()
        self.stage_key = None
        self.stage_start = self.started
        self.stage_completed = 0

    def __call__(self, stage, completed=0, total=0, **values):
        now = time.time()
        key = (stage, values.get("candidate"), values.get("split"))
        if self.stage_key != key:
            self.stage_key, self.stage_start, self.stage_completed = key, now, completed
        delta = completed - self.stage_completed
        seconds = now - self.stage_start
        eta = seconds / delta * (total - completed) if delta > 0 and total > completed else None
        row = dict(worker=self.worker, stage=stage, completed=completed, total=total,
                   elapsed_seconds=now-self.started, eta_seconds=eta, time=now, pid=os.getpid(),
                   gpu=os.environ.get("CUDA_VISIBLE_DEVICES", ""), **values)
        write(self.path, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


class ReadoutStore:
    """Checkpoint each node before advancing; identity binds bank, code and mask."""
    def __init__(self, path, fingerprint, count, width):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.count, self.width = count, width
        self.values = []
        if self.path.exists():
            row = read(self.path)
            if row["fingerprint"] != fingerprint:
                raise RuntimeError(f"Readout fingerprint mismatch: {path}")
            self.values = row["values"]
        self._validate()

    def _validate(self):
        import math
        if len(self.values) > self.count:
            raise RuntimeError("Too many cached readout nodes")
        for row in self.values:
            if len(row) != self.width or any(not math.isfinite(x) for x in row):
                raise RuntimeError("Invalid cached readout shape/value")

    def append(self, values):
        if len(self.values) >= self.count:
            raise RuntimeError("Readout already complete")
        self.values.append([float(x) for x in values])
        self._validate()
        write(self.path, dict(fingerprint=self.fingerprint, values=self.values,
                             complete=len(self.values) == self.count))

    def complete(self):
        if len(self.values) != self.count:
            raise RuntimeError(f"Incomplete readout: {self.path}")
        return self.values
