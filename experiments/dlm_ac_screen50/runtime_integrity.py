"""Controller-side identity checks for resumable screen jobs.

This module is deliberately CPU-only.  A scheduler receipt is accepted only
when it names the exact frozen job and every expected terminal artifact still
has the recorded content hash.
"""

from __future__ import annotations

from pathlib import Path

from experiments.dlm_multiscale_ac50.artifacts import digest, freeze, read, sha


def _candidate_name(job: dict) -> str:
    if job["kind"] != "legacy":
        return job["id"]
    return "A" if job["id"] == "MS-A" else job["id"]


def expected_outputs(root: Path, multi: Path, job: dict) -> list[Path]:
    """Return terminal artifacts that prove a successful job completed."""
    root, multi = Path(root), Path(multi)
    kind = job["kind"]
    family = job.get("family")
    if kind == "smoke":
        return [root / "smoke.json"]
    if kind == "legacy":
        name = _candidate_name(job)
        folder = multi / "gsm8k" / "development" / name
        return [folder / "results.json", folder / "identity.json", folder / "predictions.json", multi / "candidates" / name / "model_identity.json"]
    if kind == "teacher":
        if family == "Square":
            return [root / "readouts" / "Square" / "dense" / f"{job['split']}.json"]
        return [root / "teachers" / "Vector" / job["split"] / "complete.json"]
    if kind == "uniform":
        paths = [root / "metrics" / family / "uniform" / "calibration.json"]
        if family == "Square":
            paths.append(root / "readouts" / "Square" / "uniform" / "calibration.json")
        else:
            paths.extend(sorted((root / "pair_metrics" / "Vector" / "uniform" / "calibration").glob("*.json")))
        return paths
    if kind == "probe":
        paths = [root / "probes" / family / f"block{b:02d}.json" for b in job["blocks"]]
        for b in job["blocks"]:
            paths.extend(sorted((root / "metrics" / family).glob(f"probe_{b:02d}_*/calibration.json")))
            if family == "Vector":
                paths.extend(sorted((root / "pair_metrics" / "Vector").glob(f"probe_{b:02d}_*/calibration/*.json")))
            else:
                paths.extend(sorted((root / "readouts" / "Square").glob(f"probe_{b:02d}_*/calibration.json")))
        return paths
    if kind == "allocation":
        return [root / "allocations" / f"{family}.json"]
    if kind == "candidate":
        label = job["id"]
        paths = [
            root / "candidates" / label / "mask_manifest.json",
            root / "candidates" / label / "model_identity.json",
            root / "metrics" / family / label / "calibration.json",
            root / "metrics" / family / label / "diagnostic.json",
            root / "gsm8k" / label / "identity.json",
            root / "gsm8k" / label / "predictions.json",
            root / "gsm8k" / label / "results.json",
        ]
        paths.extend(sorted((root / "readouts" / family / label).rglob("*.json")))
        paths.extend(sorted((root / "pair_metrics" / family / label).rglob("*.json")))
        return paths
    if kind == "exchange":
        folder = root / "gsm8k" / "Exchange-AC"
        state = root / "exchange" / "state.json"
        paths = [state, folder / "identity.json", folder / "predictions.json", folder / "results.json"]
        paths.extend(sorted((root / "exchange" / "readouts").rglob("*.json")))
        paths.extend(sorted((root / "exchange" / "cache").glob("*.json")))
        if state.is_file():
            try:
                round_id = read(state).get('round')
                if isinstance(round_id, int):
                    paths.append(root / "exchange" / f"round{round_id}.json")
            except (OSError, ValueError, TypeError):
                pass
        return paths
    raise ValueError(f"Unknown job kind: {kind}")


def _referenced_paths(path: Path) -> list[Path]:
    """Expand declared child hashes in sealed JSON index artifacts."""
    if path.suffix.lower() != ".json":
        return []
    try:
        row = read(path)
    except (OSError, UnicodeError, ValueError, TypeError):
        return []
    def add(child: Path, expected: str | None) -> Path:
        if expected is not None and child.is_file() and sha(child) != expected:
            raise RuntimeError(f"Declared child hash mismatch: {child}")
        return child
    out: list[Path] = []
    if not isinstance(row, dict):
        return out
    if isinstance(row.get("files"), list):
        for item in row["files"]:
            if isinstance(item, dict) and item.get("path"):
                out.append(add(Path(item["path"]), item.get("sha256")))
    entries = row.get("entries")
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            selected = item.get("selected_mask")
            if isinstance(selected, dict) and selected.get("path"):
                out.append(add(Path(selected["path"]), selected.get("file_sha256")))
    if row.get("predictions_sha256") is not None:
        candidate = path.with_name("predictions.json")
        out.append(add(candidate, row.get("predictions_sha256")))
    # Probe/index artifacts nest metric_path + metric_sha256 under conditions.
    def walk(value):
        if isinstance(value, dict):
            child = value.get("metrics_path") or value.get("path")
            expected = value.get("metrics_sha256") or value.get("sha256")
            if child and expected and isinstance(child, str) and child != str(path):
                child_path = Path(child)
                if child_path.suffix.lower() == ".json" and child_path not in out:
                    out.append(add(child_path, expected))
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)
    walk(row)
    return out


def _artifacts(paths: list[Path]) -> list[dict]:
    expanded: list[Path] = []
    seen: set[str] = set()
    pending = list(paths)
    while pending:
        path = Path(pending.pop(0))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.is_file():
            raise RuntimeError("Missing terminal job artifact: " + key)
        expanded.append(path)
        pending.extend(_referenced_paths(path))
    return [dict(path=str(p), sha256=sha(p)) for p in expanded]


def write_job_receipt(path: Path, root: Path, multi: Path, job: dict, config_sha256: str, jobs_sha256: str) -> dict:
    payload = dict(
        status="complete",
        job_id=job["id"],
        job_sha256=digest(job),
        config_sha256=config_sha256,
        jobs_sha256=jobs_sha256,
        outputs=_artifacts(expected_outputs(root, multi, job)),
    )
    freeze(path, payload)
    return payload


def validate_job_receipt(path: Path, root: Path, multi: Path, job: dict, config_sha256: str, jobs_sha256: str) -> dict:
    row = read(path)
    if row.get("status") != "complete":
        raise RuntimeError(f"Job receipt is not complete: {path}")
    expected = {
        "job_id": job["id"],
        "job_sha256": digest(job),
        "config_sha256": config_sha256,
        "jobs_sha256": jobs_sha256,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"Job receipt identity mismatch ({key}): {path}")
    current = _artifacts(expected_outputs(root, multi, job))
    if row.get("outputs") != current:
        raise RuntimeError(f"Job output identity mismatch: {path}")
    return row
