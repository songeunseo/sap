"""No torch needed. GSM8K progress is read from the evaluator's tqdm log."""
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
folder=ROOT/'lsac'
if (folder/'results.json').exists():
    r=json.loads((folder/'results.json').read_text())
    print(f"LSA projection: complete {r['correct']}/{r['total']}")
elif (folder/'progress.json').exists():
    p=json.loads((folder/'progress.json').read_text())
    print('LSA projection:',p['stage'],p.get('error',''))
    if 'block' in p:print(f"Block {p['block']+1}/32; states {p.get('completed',0)}/{p.get('total',80)}")
    log=ROOT/'logs/run.log'
    if p['stage']=='gsm8k' and log.exists():
        matches=re.findall(r'[^\r\n]*\d+/100\s*\[[^\r\n]*',log.read_text(errors='replace'))
        if matches:print(matches[-1])
else:print('LSA projection: pending')
