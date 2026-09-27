"""Lightweight, read-only progress without loading torch."""
import json
import re
from pathlib import Path
root=Path(__file__).resolve().parent
p=root/'progress.json'
if not p.exists(): print('OWL: not started')
else:
    r=json.loads(p.read_text());print('OWL:',r['stage'])
    if r['stage']=='complete':print(f"GSM8K: {r['correct']}/100")
    elif r['stage']=='failed':print(r['error'])
    elif r['stage']=='gsm8k':
        log=root/'logs/run.log'
        lines=re.split(r'[\r\n]',log.read_text(errors='replace')) if log.exists() else []
        progress=[x for x in lines if re.search(r'\d+/100',x) and '<' in x]
        print(progress[-1] if progress else 'Preparing generation; ETA after first completed examples')
    else:print(f"{r.get('completed',0)}/{r.get('total','?')} — preparation; generation ETA about 35–40 min after start")
