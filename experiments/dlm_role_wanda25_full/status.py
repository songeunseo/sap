#!/usr/bin/env python3
"""Show progress and a whole-pipeline ETA for the target25 full comparison."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import re
from pathlib import Path

def _parse_clock(value: str) -> float:
    if "?" in value:
        return FULL_SECONDS
    fields = [int(part) for part in value.split(":")]
    return float(fields[-1] + 60 * fields[-2] + (3600 * fields[-3] if len(fields) == 3 else 0))


def generation_progress(text: str) -> dict | None:
    matches = list(re.finditer(r"Generating\.\.\.:\s*\d+%\|[^\r\n]*?\|\s*(\d+)\/(\d+)\s*\[([^<\],]+)<([^,\]]+),\s*([^\]]+)\]", text))
    if not matches:
        return None
    match = matches[-1]
    return {"completed": int(match.group(1)), "total": int(match.group(2)), "elapsed_seconds": _parse_clock(match.group(3).strip()), "remaining_seconds": _parse_clock(match.group(4).strip()), "rate": match.group(5).strip()}



ROOT = Path("experiments/dlm_role_wanda25_full")
LOG = ROOT / "logs/pipeline.log"
UNIFORM_LOG = ROOT / "logs/uniform_gpu0.log"
FULL_SECONDS = 28_300.0
PREPARE_SECONDS = 600.0


def events(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def fmt(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}시간 {minutes:02d}분" if hours else f"{minutes}분 {secs:02d}초"


def main() -> None:
    now = datetime.now()
    if (ROOT / "full_results.json").exists():
        result = json.loads((ROOT / "full_results.json").read_text())
        print("complete")
        print(f"Uniform-Wanda-25: {result['methods']['uniform']['correct']}/1319")
        print(f"Role-Wanda-25: {result['methods']['role']['correct']}/1319")
        print(result["paired_role_vs_uniform"])
        return
    text = (LOG.read_text(errors="replace") if LOG.exists() else "") + "\n" + (UNIFORM_LOG.read_text(errors="replace") if UNIFORM_LOG.exists() else "")
    rows = events(text)
    completed = [row.get("method") for row in rows if row.get("event") == "full_worker_complete"]
    progress = generation_progress(text)
    prepared = any(row.get("event") == "prepare_complete" for row in rows)
    projections = [row for row in rows if row.get("event") in {"projection_complete", "projection_skipped"}]
    materialized = [row for row in rows if row.get("event") == "mask_materialized"]

    if progress is not None and progress["completed"] < progress["total"]:
        active = "role" if "role" not in completed else "uniform"
        remaining = progress["remaining_seconds"]
        stage = f"parallel full ({active} 기준): {progress['completed']}/{progress['total']}"
    elif prepared:
        active = "uniform" if "role" in completed else "role"
        remaining = FULL_SECONDS
        stage = f"{active} full 모델 준비"
    elif materialized:
        remaining = 120.0 + FULL_SECONDS
        stage = f"mask materialization {materialized[-1].get('module', len(materialized))}/224"
    elif projections:
        done = int(projections[-1].get("module", len(projections)))
        elapsed = float(projections[-1].get("elapsed_seconds", 0.0))
        collection_left = elapsed / max(done, 1) * (224 - done) if elapsed else PREPARE_SECONDS
        remaining = collection_left + 300.0 + FULL_SECONDS
        stage = f"role curve collection {done}/224"
    else:
        remaining = PREPARE_SECONDS + FULL_SECONDS
        stage = "시작/모델 로드"
    print(f"stage: {stage}")
    print(f"completed full methods: {completed}")
    print(f"전체 ETA: {fmt(remaining)}")
    print(f"예상 완료: {(now + timedelta(seconds=remaining)):%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
