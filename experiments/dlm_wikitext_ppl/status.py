"""Torch-free pilot progress and runtime estimates."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if (ROOT/'pilot_results.json').exists():
    r=json.loads((ROOT/'pilot_results.json').read_text())
    print('timing pilot: complete')
    print(f"evaluation: {r['evaluation_seconds']/60:.2f} min; preparation: {r['preparation_seconds']/60:.2f} min")
    for split,v in r['projected_full_split_times'].items():
        print(f"{split}: {v['blocks']} blocks; projected evaluation {v['evaluation_seconds_upper_length_estimate']/60:.1f} min/candidate")
elif (ROOT/'progress.json').exists():
    p=json.loads((ROOT/'progress.json').read_text());print(p['stage'],f"{p.get('completed',0)}/{p.get('total','?')}",p.get('error',''))
else: print('pending')
