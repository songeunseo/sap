"""Standard-library status and total queue ETA; never initializes CUDA."""
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def duration(seconds):
    return f"{int(seconds)//3600}시간 {int(seconds)%3600//60}분"


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def generation_count(log):
    if not log.exists(): return 0
    with log.open("rb") as f:
        f.seek(max(0, log.stat().st_size - 64000))
        text = f.read().decode(errors="replace")
    matches = re.findall(r"Generating\.\.\.[^\r\n]*?(\d+)/100", text)
    return int(matches[-1]) if matches else 0


def main():
    c = read(ROOT / "config.json")
    if not c:
        print("설정 준비 중: 아직 실행 전")
        return
    times = [read(ROOT / m / "results.json").get("metrics", {}).get("eval_seconds") for m in c["methods"]]
    times = [t for t in times if t is not None]
    typical = sum(times)/len(times) if times else c["historical_eval_seconds_per100"]
    prep = c["preparation_allowance_seconds"]
    all_done = sum(bool(read(ROOT / m / "results.json")) for m in c["methods"])
    print(f"전체 모델 완료: {all_done}/12 (각각 동일 100문제)")
    etas, blocked = [], False
    for gpu, methods in c["queues"].items():
        w = read(ROOT / f"worker{gpu}/progress.json")
        queue_eta = 0
        print(f"\nGPU {gpu}:")
        for method in methods:
            result = read(ROOT / method / "results.json")
            if result:
                print(f"  {method}: 완료, {result['correct']}/100 정답")
                continue
            progress = read(ROOT / method / "progress.json")
            stage = progress.get("stage", "queued")
            n = 0
            if stage == "gsm8k":
                n = generation_count(ROOT / "logs" / f"{method}.log")
                elapsed = max(0, time.time() - progress["timestamp"])
                rate = elapsed / n if n >= 3 else typical / 100
                remaining = max(0, 100-n)*rate + 30
                print(f"  {method}: 생성 {n}/100, 현재 모델 ETA 약 {duration(remaining)}")
            elif stage == "verifying":
                remaining = 30
                print(f"  {method}: 100/100 생성 완료, 검증 중")
            else:
                remaining = typical + prep
                print(f"  {method}: {stage}")
            if stage == "failed": blocked = True
            queue_eta += remaining
        if queue_eta and (w.get("stage") == "failed" or not alive(w.get("pid"))):
            blocked = True
            print("  대기열 실행이 확인되지 않음/중단됨: ETA 확정 불가")
        print(f"  이 GPU 전체 대기열 ETA: 약 {duration(queue_eta)}" if queue_eta else "  대기열 완료")
        etas.append(queue_eta)
    print("\n전체 ETA: 중단된 대기열 확인 필요" if blocked else f"\n전체 남은 시간: 약 {duration(max(etas))}")
    print("ETA는 완료 속도/현재 진행률 추정이며 아직 없는 결과를 의미하지 않습니다.")


if __name__ == "__main__": main()
