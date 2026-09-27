"""No torch or CUDA startup; display shared collection and both mini runs."""
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}


def main():
    pipeline=read(ROOT/'pipeline/progress.json')
    print('Pipeline:',pipeline.get('stage','not started'))
    if pipeline.get('error'):print(pipeline['error'])
    r=read(ROOT/'collection/progress.json')
    print('Shared collection:',r.get('stage','queued'),f"{r.get('completed',0)}/{r.get('total',80)}")
    if r.get('eta_seconds') is not None:
        print(f"수집 ETA 약 {r['eta_seconds']/60:.1f}분 + mini 평가 약35분(병렬), 준비/검증시간 별도")
    for method in ('token_relative','token_angular'):
        p=read(ROOT/method/'progress.json');result=read(ROOT/method/'results.json');detail=''
        if result:detail=f"{result['correct']}/100 정답"
        elif p.get('stage')=='gsm8k':
            path=ROOT/'logs'/f'{method}.log'
            if path.exists():
                with path.open('rb') as f:
                    f.seek(max(0,path.stat().st_size-32000));text=f.read().decode(errors='replace')
                matches=re.findall(r'Generating\.\.\.[^\r\n]*?(\d+)/100\s*\[([^\]]+)\]',text)
                if matches:detail=f'{matches[-1][0]}/100 [{matches[-1][1]}]'
        print(method+':',p.get('stage','queued'),detail,p.get('error',''))
    for phase in ('collect','prepare'):
        r=read(ROOT/phase/'progress.json')
        if r.get('stage')=='failed':print(phase,'FAILED:',r.get('error'))


if __name__=='__main__':main()
