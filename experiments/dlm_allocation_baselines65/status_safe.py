"""Live PID-aware read-only status; historical frozen status.py is kept unchanged."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
for m in ('dlp','dsa','alpha','lsa'):
    p=ROOT/m;r=read(p/'results.json')
    if r and r.get('status')=='complete':print(f'{m}: complete {r["correct"]}/100');continue
    active=read(p/'active_attempt.json');progress=read(p/'progress.json')
    pid=(active or progress or {}).get('pid');proc=Path(f'/proc/{pid}/cmdline')
    try:cmd=proc.read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):cmd=''
    live='dlm_allocation_baselines65' in cmd and m in cmd.split()
    checkpoint=read(p/'checkpoints/progress.json');count=checkpoint['completed'] if checkpoint else 0
    if not live:
        print(f'{m}: STOPPED / incomplete; checkpoints {count}/100; exit={read(p/"last_exit.json")}')
    else:
        print(f'{m}: running; stage={(progress or {}).get("stage","preparing")}; checkpoints {count}/100; pid={pid}')
        if active:print(f'  log: {active["log"]}')
