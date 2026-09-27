"""Standard-library progress display; no CUDA/PyTorch initialization."""
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for method in ('sparse_context', 'dense_target'):
    p = ROOT / method / 'progress.json'
    if not p.exists():
        print(f'{method}: not started (freeze/preflight may be running)')
        continue
    r = json.loads(p.read_text())
    stage = r['stage']
    detail = f"{r.get('completed', '?')}/{r.get('total', '?')}" if 'total' in r else ''
    if stage == 'gsm8k':
        log = ROOT / 'logs' / f'{method}.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0, log.stat().st_size - 24000))
                text = f.read().decode(errors='replace')
            matches = re.findall(r'(\d+)/100\s*\[([^\]]+)\]', text)
            if matches:
                n, timing = matches[-1]
                detail = f'{n}/100 [{timing}]'
        # progress.json marks phase start, not generation heartbeat; tqdm log carries progress.
    if stage == 'collecting' and r.get('eta_seconds') is not None:
        detail += f"; collection ETA {r['eta_seconds']/60:.1f} min, GSM8K additional"
    if stage == 'complete':
        detail = f"{r['correct']}/100 correct"
    if stage == 'failed':
        detail = r['error']
    print(f'{method}: {stage} {detail}')
