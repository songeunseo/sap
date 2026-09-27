"""Lightweight progress, no torch import."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for method in ('uniform','owl','dlp','alpha','lsa','dsa'):
    folder=ROOT/method
    if (folder/'results.json').exists():
        r=json.loads((folder/'results.json').read_text()); print(f"{method}: complete {r['correct']}/{r['total']}")
    elif (folder/'progress.json').exists():
        p=json.loads((folder/'progress.json').read_text())
        detail=f" block {p['block']+1}/32" if 'block' in p else ''
        detail+=f" {p['completed']}/{p['total']}" if 'completed' in p and 'total' in p else ''
        print(f"{method}: {p['stage']}{detail} {p.get('error','')}")
    else: print(f'{method}: pending')
