import json
from pathlib import Path
from experiments.dlm_ppl50.run import METHODS, ROOT, read

def snapshot():
    output={}
    for method in METHODS:
        path=ROOT/method/'progress.json'
        row=read(path) if path.exists() else {'stage':'queued'}
        result=ROOT/method/'validation/results.json'
        if result.exists():row=dict(row,summary=read(result)['summary'])
        output[method]=row
    return output

if __name__=='__main__':
    output=snapshot()
    from experiments.dlm_dual_role_allocation.io import atomic_write_json
    atomic_write_json(ROOT/'status.json',output)
    print(json.dumps(output,ensure_ascii=False,indent=2))
