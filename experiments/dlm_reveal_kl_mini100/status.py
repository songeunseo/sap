"""Small CPU-only experiment progress reader."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def main():
    out={'root':str(ROOT)}
    for filename in ['progress.json','results.json']:
        p=ROOT/filename
        if p.exists():out[filename]=read(p)
    out['trajectory_prompts']=len(list((ROOT/'trajectory').glob('prompt*.json')))
    out['probe_blocks']=len(list((ROOT/'probe').glob('block*.json')))
    for mode in ['reveal','all_masked']:
        p=ROOT/mode/'progress.json';out[mode]=read(p) if p.exists() else {'stage':'queued'}
        p=ROOT/mode/'results.json'
        if p.exists():out[mode]=read(p)
    p=ROOT/'exit_code.txt'
    if p.exists():out['pipeline_exit_code']=p.read_text().strip()
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
