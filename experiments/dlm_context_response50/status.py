"""Read-only progress, standard library only."""
import datetime,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    print('Context response50 | '+datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
    p=ROOT/'progress.json'
    if p.exists():
        r=json.loads(p.read_text());age=time.time()-r['time']
        print(f"Stage: {r['stage']} | last update {age:.0f}s ago | PID {r['pid']}")
        print(' | '.join(f'{k}: {v}' for k,v in r.items() if k not in ('stage','time','pid')))
    print('Completed probe blocks:',len(list((ROOT/'probes').glob('block??.json'))),'/32')
    for m in ('uniform','A','AC'):
        p=ROOT/m/'results.json'
        print(m, f"{json.loads(p.read_text())['correct']}/100 complete" if p.exists() else 'pending')
    p=ROOT/'exit_code.txt'
    if p.exists():print('Pipeline exit:',p.read_text().strip())
    print('Log:',ROOT/'pipeline.log')
if __name__=='__main__':main()
