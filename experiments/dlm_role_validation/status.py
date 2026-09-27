#!/usr/bin/env python3
"""Human-readable stage and ETA display for the serialized validation queue."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import re


ROOT = Path("experiments/dlm_role_validation")
SUITES = ("random65", "target50", "target75")
MINI_METHODS = {"random65": 3, "target50": 2, "target75": 2}

# Historical fixed-protocol mini runs take about 2,106--2,165 seconds of
# generation. The extra time covers model load, mask application, and hashing.
MINI_METHOD_SECONDS = 2_200.0
FULL_METHOD_SECONDS = 28_300.0
COLLECTION_SECONDS = 300.0
TARGET_MATERIALIZATION_SECONDS = 240.0
RANDOM_MATERIALIZATION_SECONDS = 30.0


def _events(text: str) -> list[dict]:
    result = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def parse_clock(value: str) -> float:
    """Parse tqdm H:MM:SS or MM:SS text."""
    fields = [int(part) for part in value.split(":")]
    if len(fields) == 2:
        return float(fields[0] * 60 + fields[1])
    if len(fields) == 3:
        return float(fields[0] * 3600 + fields[1] * 60 + fields[2])
    raise ValueError(f"invalid clock: {value}")


def generation_progress(text: str) -> dict | None:
    """Read the last tqdm generation counter, including carriage-return logs."""
    matches = list(re.finditer(
        r"Generating\.\.\.:\s*\d+%\|[^\r\n]*?\|\s*(\d+)/(\d+)\s*"
        r"\[([^<\],]+)<([^,\]]+),\s*([^\]]+)\]", text,
    ))
    if not matches:
        return None
    match = matches[-1]
    return {
        "completed": int(match.group(1)), "total": int(match.group(2)),
        "elapsed_seconds": parse_clock(match.group(3).strip()),
        "remaining_seconds": parse_clock(match.group(4).strip()),
        "rate": match.group(5).strip(),
    }


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "계산 중"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분"
    if minutes:
        return f"{minutes}분 {secs:02d}초"
    return f"{secs}초"


def _suite_status(suite: str, text: str) -> dict:
    rows = _events(text)
    mini_path = ROOT / f"mini_{suite}.json"
    if mini_path.exists():
        result = json.loads(mini_path.read_text())
        return {"suite": suite, "stage": "mini 완료", "intrinsic_remaining": 0.0,
                "stage_remaining": 0.0, "gate": result.get("gate"), "progress": None}

    completed_methods = [row for row in rows
                         if row.get("event") == "gsm8k_method_complete"
                         and int(row.get("limit", 0)) == 100]
    progress = generation_progress(text)
    collection = next((row for row in reversed(rows)
                       if row.get("event") == "collection_complete"
                       and row.get("status") == "complete"), None)
    projection_rows = [row for row in rows
                       if row.get("event") in {"projection_complete", "projection_skipped"}]
    analyzed = any(row.get("event") == "analysis_complete" for row in rows)
    materialized = any(row.get("event") == "materialize_complete" for row in rows)
    method_total = MINI_METHODS[suite]

    if progress is not None and progress["completed"] < progress["total"]:
        active_index = min(len(completed_methods) + 1, method_total)
        stage = f"mini 모델 {active_index}/{method_total}: {progress['completed']}/{progress['total']}"
        stage_remaining = progress["remaining_seconds"]
        intrinsic = stage_remaining + max(0, method_total - active_index) * MINI_METHOD_SECONDS
    elif completed_methods:
        done = len(completed_methods)
        stage = f"mini 모델 {done}/{method_total} 완료; 다음 모델 준비"
        stage_remaining = 45.0
        intrinsic = stage_remaining + max(0, method_total - done) * MINI_METHOD_SECONDS
    elif materialized:
        stage = "첫 mini 모델 준비"
        stage_remaining = 45.0
        intrinsic = stage_remaining + method_total * MINI_METHOD_SECONDS
    elif analyzed:
        stage = "mask materialization"
        stage_remaining = (RANDOM_MATERIALIZATION_SECONDS if suite == "random65"
                           else TARGET_MATERIALIZATION_SECONDS)
        intrinsic = stage_remaining + method_total * MINI_METHOD_SECONDS
    elif collection:
        stage = "allocation 분석"
        materialize = (RANDOM_MATERIALIZATION_SECONDS if suite == "random65"
                       else TARGET_MATERIALIZATION_SECONDS)
        stage_remaining = 60.0
        intrinsic = stage_remaining + materialize + method_total * MINI_METHOD_SECONDS
    elif projection_rows:
        last = projection_rows[-1]
        completed = int(last.get("module", len(projection_rows)))
        elapsed = float(last.get("elapsed_seconds", 0.0))
        if elapsed > 0 and completed > 1:
            collection_remaining = elapsed / completed * (224 - completed) + 40.0
        else:
            collection_remaining = COLLECTION_SECONDS
        materialize = (RANDOM_MATERIALIZATION_SECONDS if suite == "random65"
                       else TARGET_MATERIALIZATION_SECONDS)
        stage = f"curve collection {completed}/224"
        stage_remaining = collection_remaining
        intrinsic = collection_remaining + 60.0 + materialize + method_total * MINI_METHOD_SECONDS
    else:
        stage = "queue 대기"
        materialize = (RANDOM_MATERIALIZATION_SECONDS if suite == "random65"
                       else TARGET_MATERIALIZATION_SECONDS)
        stage_remaining = None
        intrinsic = COLLECTION_SECONDS + 60.0 + materialize + method_total * MINI_METHOD_SECONDS
    return {"suite": suite, "stage": stage, "intrinsic_remaining": intrinsic,
            "stage_remaining": stage_remaining, "gate": None, "progress": progress}


def _full_status(suite: str, text: str) -> dict:
    result_path = ROOT / f"full_{suite}.json"
    if result_path.exists():
        return {"suite": suite, "stage": "full 완료", "intrinsic_remaining": 0.0,
                "stage_remaining": 0.0, "gate": None, "progress": None}
    rows = _events(text)
    completed = [row for row in rows if row.get("event") == "gsm8k_method_complete"
                 and int(row.get("limit", 0)) == 1319]
    progress = generation_progress(text)
    # Current target50/75 full runs are user-directed Role-only evaluations.
    total_methods = 1
    if progress is not None and progress["completed"] < progress["total"]:
        active = min(len(completed) + 1, total_methods)
        stage = f"full 모델 {active}/{total_methods}: {progress['completed']}/{progress['total']}"
        stage_remaining = progress["remaining_seconds"]
        remaining = stage_remaining + max(0, total_methods - active) * FULL_METHOD_SECONDS
    elif completed:
        done = len(completed)
        stage = f"full 모델 {done}/{total_methods} 완료; 다음 모델 준비"
        stage_remaining = 60.0
        remaining = stage_remaining + max(0, total_methods - done) * FULL_METHOD_SECONDS
    else:
        stage = "첫 full 모델 준비"
        stage_remaining = 60.0
        remaining = stage_remaining + total_methods * FULL_METHOD_SECONDS
    return {"suite": suite, "stage": stage, "intrinsic_remaining": remaining,
            "stage_remaining": stage_remaining, "gate": None, "progress": progress}


def calculate_serial_eta(statuses: list[dict]) -> list[dict]:
    """Attach delays, respecting queue order unless a suite is already active."""
    previous_completion = 0.0
    result = []
    for status in statuses:
        row = dict(status)
        if status["stage"] == "queue 대기":
            row["start_in"] = previous_completion
            row["complete_in"] = previous_completion + float(status["intrinsic_remaining"])
        else:
            row["start_in"] = 0.0
            row["complete_in"] = float(status["intrinsic_remaining"])
        previous_completion = row["complete_in"]
        result.append(row)
    return result


def main():
    now = datetime.now()
    statuses = []
    for suite in SUITES:
        full_path = ROOT / "logs" / f"{suite}_full.log"
        if full_path.exists() and suite in {"target50", "target75"}:
            statuses.append(_full_status(suite, full_path.read_text(errors="replace")))
        else:
            path = ROOT / "logs" / f"{suite}.log"
            statuses.append(_suite_status(suite, path.read_text(errors="replace") if path.exists() else ""))
    statuses = calculate_serial_eta(statuses)
    print(f"기준 시각: {now:%Y-%m-%d %H:%M:%S}")
    for row in statuses:
        fields = [f"{row['suite']:<8}", row["stage"]]
        if row["stage"] == "queue 대기":
            fields.append(f"시작까지 {_format_seconds(row['start_in'])}")
        elif row["stage_remaining"] is not None and row["stage_remaining"] > 0:
            fields.append(f"현재 단계 ETA {_format_seconds(row['stage_remaining'])}")
        fields.append(f"완료까지 {_format_seconds(row['complete_in'])}")
        fields.append(f"예상 완료 {(now + timedelta(seconds=row['complete_in'])):%m-%d %H:%M}")
        if row["gate"] is not None:
            fields.append(f"gate={row['gate'].get('decision', row['gate'].get('passed'))}")
        print(" | ".join(fields))
    overall = max((row["complete_in"] for row in statuses), default=0.0)
    label = "전체 target full ETA" if any("full" in row["stage"] for row in statuses) else "전체 mini screen ETA"
    print(f"{label}: {_format_seconds(overall)} (예상 완료 {(now + timedelta(seconds=overall)):%Y-%m-%d %H:%M})")
    print("주의: 현재 tqdm 속도와 같은 protocol의 과거 runtime 기반 추정치이며, 생성 길이에 따라 변동됩니다.")


if __name__ == "__main__":
    main()
