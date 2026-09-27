"""Works with system Python: no torch, no expensive checkpoint loading."""
import json
import time
from pathlib import Path
root=Path(__file__).resolve().parent
remaining=[]
for split in ['development','final']:
    path=root/f'progress_{split}.json'
    if not path.exists():print(f'{split}: preparing/smoke, ETA not yet measured');continue
    p=json.loads(path.read_text());eta=p.get('eta_seconds',0)
    remaining.append(eta);age=time.time()-p['updated']
    print(f"{split}: {p['completed']}/{p['total']} ({p['status']}), ETA {eta/60:.1f} min, update {age:.0f}s ago")
complete=(root/'analysis.json').exists()
print('analysis:','complete' if complete else 'pending')
if not complete and remaining: print(f'Collection ETA: {max(remaining)/60:.1f} min; analysis time additional')
