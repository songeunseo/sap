"""Dependency-free status for the bounded DSA DLM search."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
if (root/'results.json').exists():
    r=json.loads((root/'results.json').read_text())
    print(f"complete: {r['correct']}/100; selected graph={r['graph']}")
elif (root/'progress.json').exists():
    print(json.loads((root/'progress.json').read_text()))
    if (root/'search.json').exists():
        h=json.loads((root/'search.json').read_text())
        valid=[r for r in h['candidates'].values() if r['status']=='valid']
        print(f"candidates completed={len(h['candidates'])}; valid={len(valid)}; generations={len(h['generations'])}/4")
        if valid:
            print(f"best search masked CE={min(r['fitness']['mean_ce'] for r in valid):.6f}")
else:
    print('not started')
