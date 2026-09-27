"""Read-only progress display; uses only Python standard library."""
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for method in ('dlp','dsa','alpha','lsa'):
    path=ROOT/method/'progress.json'
    if not path.exists():print(f'{method}: queued');continue
    p=json.loads(path.read_text());stage=p['stage']
    if stage=='complete':print(f'{method}: complete {p["correct"]}/100');continue
    if stage=='failed':print(f'{method}: FAILED {p["error"]}');continue
    suffix=f' {p["completed"]}/{p["total"]}' if 'completed' in p else ''
    if stage=='gsm8k':
        log=ROOT/'logs'/f'{method}.log'
        matches=re.findall(r'[^\r\n]*\d+/100[^\r\n]*',log.read_text(errors='replace')) if log.exists() else []
        if matches:suffix=' '+matches[-1].strip()
    print(f'{method}: {stage}{suffix}')
