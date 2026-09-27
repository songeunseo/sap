"""Torch-free full validation progress; pilot status stays immutable."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'validation'
if not ROOT.exists():print('validation: pending')
else:
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir():continue
        if (folder/'results.json').exists():
            r=json.loads((folder/'results.json').read_text())
            print(folder.name,'complete',f"NELBO={r['summary']['token_nelbo']:.6f}",f"PPL bound estimate={r['summary']['ppl_upper_bound_estimate']:.4f}")
        elif (folder/'progress.json').exists():
            p=json.loads((folder/'progress.json').read_text())
            eta=f"ETA {p['remaining_seconds']/60:.1f} min" if 'remaining_seconds' in p else 'ETA pending'
            print(folder.name,p['stage'],f"{p.get('completed',0)}/{p.get('total','?')}",eta,p.get('error',''))
        else:print(folder.name,'pending')
