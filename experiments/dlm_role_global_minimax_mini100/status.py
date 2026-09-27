import json
import re
import subprocess
import time
from pathlib import Path

root=Path(__file__).resolve().parent
result=root/'mini100_results.json'
if result.exists():
    x=json.loads(result.read_text()); print(f"complete: {x['correct']}/100; {x['decision']}")
else:
    log=root/'logs/run.log'
    complete=0
    elapsed=None
    seconds_per_example=None
    if log.exists():
        raw=log.read_text(errors='replace').replace('\r','\n')
        for line in raw.splitlines():
            try:
                row=json.loads(line)
                if row.get('event') in ('example_complete','gsm8k_example'): complete=max(complete,int(row.get('example',row.get('completed',0))))
            except Exception: pass
        matches=re.findall(r'Generating\.\.\.:\s*\d+%\|[^|]*\|\s*(\d+)/100\s*\[([^<]+)<[^,]+,\s*([0-9.]+)s/it\]',raw)
        if matches:
            done, elapsed_text, rate=matches[-1]
            complete=max(complete,int(done)); seconds_per_example=float(rate)
            pieces=[float(x) for x in elapsed_text.split(':')]
            elapsed=sum(value*60**i for i,value in enumerate(reversed(pieces)))
    running=subprocess.run(['pgrep','-f','experiments/dlm_role_global_minimax_mini100/run.py'],
                           capture_output=True,text=True).returncode==0
    label='running' if running else 'not running / interrupted'
    print(f'{label}: progress {complete}/100')
    if complete and seconds_per_example:
        remaining=(100-complete)*seconds_per_example
        print(f'ETA: about {remaining/60:.1f} min; observed {seconds_per_example:.2f} s/example')
    print('log:',log)
