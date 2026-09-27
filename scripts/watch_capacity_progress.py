#!/usr/bin/env python3
"""Live, read-only progress display for capacity evaluation; standard library only."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1] / 'experiments/dlm_capacity_predictor'
# Measured from this run's saved dense_statistics.pt (2026-09-10).
# This is an initial timing proxy, not a guaranteed bound: collection includes
# CPU statistics, and preparation includes several dense/sparse weight hashes.
MODEL_LOAD_SECONDS = 34.64282421208918
COLLECTION_SECONDS_PER_STATE = 194.41555012296885 / 80


def duration(seconds):
    if seconds is None:
        return '계산 중'
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:d}:{minutes:02d}:{seconds:02d}'


class Progress:
    def __init__(self):
        self.key = None
        self.count = 0
        self.total = 0
        self.label = '시작 대기'
        self.baseline = None
        self.last = None
        self.message = ''
        self.finished = False

    def consume(self, row, now):
        kind = row.get('event')
        if kind == 'heldout_complete':
            self.finished = True
            self.message = '평가 완료: ' + json.dumps(row.get('decisions', {}), ensure_ascii=False)
            return
        if kind in ('mask_preflight_started', 'mask_preflight_complete', 'mask_preflight_reused'):
            self.key, self.count, self.total = None, 0, 0
            self.baseline = self.last = None
            self.label = ('마스크 파일 확인 중' if kind == 'mask_preflight_started'
                          else '마스크 확인 완료 · 모델 준비 중')
            return
        if kind == 'mask_preflight_progress':
            key = ('mask', row['method'])
            label = f"마스크 검증 [{row['method']}]"
            count, total = int(row['checked']), int(row['total'])
        elif kind == 'dense_reference':
            key, label = ('dense',), 'Dense 기준 출력'
            count, total = int(row['state']), int(row['total'])
        elif kind == 'heldout_state':
            key, label = ('heldout', row['method']), f"KL 평가 [{row['method']}]"
            count, total = int(row['state']), 40
        else:
            return
        if self.key != key or count < self.count:
            self.key, self.baseline, self.last = key, (now, count), (now, count)
        elif count > self.count:
            self.last = (now, count)
        self.count, self.total, self.label = count, total, label

    def render(self, now):
        if self.finished:
            return self.message
        if not self.total:
            return self.label + ' | ETA 계산 대기'
        seconds_per_item = None
        if self.baseline and self.last:
            elapsed = self.last[0] - self.baseline[0]
            completed = self.last[1] - self.baseline[1]
            if elapsed > 0 and completed > 0:
                seconds_per_item = elapsed / completed
        fraction = self.count / self.total
        width = 22
        bars = int(width*fraction)
        bar = '█'*bars + '░'*(width-bars)
        if self.count == self.total:
            return f'{self.label}: 100% |{bar}| {self.count}/{self.total} · 다음 단계 준비 중'
        eta = (self.total-self.count)*seconds_per_item if seconds_per_item is not None else None
        speed = f'{seconds_per_item:.2f}s/개' if seconds_per_item is not None else '속도 측정 중'
        return (f'{self.label}: {fraction:4.0%} |{bar}| {self.count}/{self.total} '
                f'[이 단계 ETA {duration(eta)}, {speed}]')


class OverallProgress(Progress):
    """Estimate the complete heldout job, including unobserved future methods."""
    def __init__(self, methods=None, load_seconds=MODEL_LOAD_SECONDS,
                 state_seconds=COLLECTION_SECONDS_PER_STATE):
        super().__init__()
        self.methods = list(methods or ['uniform', 'eis_type', 'probe16'])
        self.load_seconds = load_seconds
        self.state_seconds = state_seconds
        self.mask_seconds = 2*load_seconds/224
        self.preparation_seconds = 3*load_seconds
        self.phase = 'preflight'
        self.phase_started = None
        self.changed_at = None
        self.observed_rate = False
        self.completed_methods = set()

    def consume(self, row, now):
        kind = row.get('event')
        if kind == 'mask_preflight_started':
            self.methods = list(row.get('methods', self.methods))
            self.phase = 'preflight'
            self.phase_started = now
            self.changed_at = now
            self.completed_methods.clear()
            self.finished = False
        elif kind in ('mask_preflight_complete', 'mask_preflight_reused'):
            self.phase, self.phase_started, self.changed_at = 'load', now, now
        elif kind == 'dense_reference':
            self.phase = 'dense'
        elif kind == 'heldout_state':
            self.phase = 'heldout'
        old_key, old_count = self.key, self.count
        super().consume(row, now)
        if self.key != old_key or self.count != old_count:
            self.changed_at = now
        if self.baseline and self.last and self.last[0] > self.baseline[0]:
            n = self.last[1]-self.baseline[1]
            if n > 0:
                rate = (self.last[0]-self.baseline[0])/n
                if self.key[0] == 'heldout':
                    self.state_seconds, self.observed_rate = rate, True
                elif self.key[0] == 'dense':
                    # Dense reference evaluates the identical input twice for sham.
                    self.state_seconds, self.observed_rate = rate/2, True
                elif self.key[0] == 'mask':
                    self.mask_seconds = rate

    def remaining_seconds(self, now):
        if self.finished:
            return 0.
        gap = max(0., now-(self.changed_at if self.changed_at is not None else now))
        # Preparation and final hashing cannot be inferred from forward speed alone.
        per_method = self.preparation_seconds + 40*self.state_seconds
        if self.phase == 'heldout' and self.key:
            method = self.key[1]
            index = self.methods.index(method)
            future = sum(m not in self.completed_methods for m in self.methods[index+1:])
            if method in self.completed_methods:
                return max(5., future*per_method-gap)
            forward = (40-self.count)*self.state_seconds
            # A result is persisted only after the final sparse-model hash.
            tail = max(5., self.load_seconds-gap) if self.count == 40 else self.load_seconds
            return forward + tail + future*per_method
        methods_left = sum(m not in self.completed_methods for m in self.methods)
        rest = methods_left*per_method
        if self.phase == 'dense':
            rest += (40-self.count)*2*self.state_seconds
            if self.count == 40:
                rest -= min(gap, max(0., self.preparation_seconds-5))
            return max(5., rest)
        if self.phase == 'load':
            return max(5., self.load_seconds-gap) + 80*self.state_seconds + rest
        checked = 0
        if self.key and self.key[0] == 'mask':
            checked = self.methods.index(self.key[1])*224 + self.count
        return ((224*len(self.methods)-checked)*self.mask_seconds
                + self.load_seconds + 80*self.state_seconds + rest)

    def render(self, now):
        if self.finished:
            return self.message
        eta = self.remaining_seconds(now)
        quality = '실측 속도 반영·준비시간 추정' if self.observed_rate else '초기 추정'
        detail = self.label
        if self.total:
            width = 18
            filled = int(width*self.count/self.total)
            detail += f" |{'█'*filled}{'░'*(width-filled)}| {self.count}/{self.total}"
        return f'전체 ETA 약 {duration(eta)} ({quality}) · {detail}'


def latest_events(path, offset):
    if not path.exists():
        return [], offset
    with path.open('rb') as handle:
        if path.stat().st_size < offset:
            offset = 0
        handle.seek(offset)
        data = handle.read()
    # Leave incomplete trailing writes for the next refresh.
    boundary = data.rfind(b'\n')
    if boundary < 0:
        return [], offset
    events = []
    for line in data[:boundary+1].splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and 'event' in value:
                events.append(value)
        except (ValueError, UnicodeDecodeError):
            pass
    return events, offset+boundary+1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true', help='print current phase and exit')
    args = parser.parse_args()
    progress, offset = OverallProgress(), 0
    print('전체 held-out 평가 종료까지의 추정 ETA입니다. 모델 준비·남은 모든 방법 평가를 포함합니다. Ctrl+C: 모니터 종료', flush=True)
    try:
        while True:
            now = time.monotonic()
            events, offset = latest_events(ROOT/'logs/heldout.log', offset)
            for event in events:
                progress.consume(event, now)
            # Persisted completed results let the monitor recognize hash/write completion,
            # even though older workers emit only per-state progress events.
            for method in progress.methods:
                result_path = ROOT/f'heldout_{method}.json'
                if method not in progress.completed_methods and result_path.exists():
                    try:
                        if json.loads(result_path.read_text()).get('status') == 'complete':
                            progress.completed_methods.add(method)
                    except (OSError, ValueError):
                        pass
            line = progress.render(now)
            done = progress.finished
            try:
                status = json.loads((ROOT/'status.json').read_text())
                if status.get('status') == 'failed':
                    line, done = '실험 중단/오류 · status.json 및 heldout.log 확인', True
                elif status.get('status') == 'complete':
                    line, done = '실험 정상 완료' + (' · '+progress.message if progress.finished else ''), True
                elif status.get('stage') != 'heldout':
                    line = f"준비 단계: {status.get('stage', status.get('status', '대기'))} | ETA 계산 대기"
                elif status.get('pid') and not Path(f"/proc/{status['pid']}").exists() and not done:
                    line, done = '실행 프로세스 없음 · 종료 로그 확인 필요', True
            except (OSError, ValueError):
                pass
            if sys.stdout.isatty() and not args.once:
                print('\r\033[K'+line, end='', flush=True)
            else:
                print(line, flush=True)
            if done or args.once:
                if sys.stdout.isatty():
                    print()
                return
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n모니터 종료. 실험 실행에는 영향이 없습니다.')


if __name__ == '__main__':
    main()
