"""Read-only experiment status; no GPU imports."""
import datetime, json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    print('Support coverage50 | '+datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
    p=ROOT/'progress.json'
    if p.exists():
        r=json.loads(p.read_text())
        print(f"Stage: {r['stage']} | updated {time.time()-r['time']:.0f}s ago | PID {r['pid']}")
        print(' | '.join(f'{k}: {v}' for k,v in r.items() if k not in ('stage','time','pid')))
    for m in ('uniform','pooled','coverage'):
        p=ROOT/m/'results.json'; q=ROOT/m/'nelbo'/'results.json'
        label=f"{json.loads(p.read_text())['correct']}/100" if p.exists() else 'pending'
        if q.exists(): label+=f" | mini NELBO {json.loads(q.read_text())['mean_nelbo']:.6f}"
        print(f'{m:10s} {label}')
    p=ROOT/'exit_code.txt'
    if p.exists(): print('Pipeline exit:',p.read_text().strip())
    print('Log:',ROOT/'pipeline.log')

if __name__=='__main__': main()
