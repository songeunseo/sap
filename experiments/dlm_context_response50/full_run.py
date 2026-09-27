"""Full GSM8K evaluation for frozen context-response50 masks."""
from __future__ import annotations
import argparse, gc, json, os, time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from experiments.dlm_context_response50 import run as mini
from experiments.dlm_loss_aggregation.exp002.run import (
    _evaluate_gsm8k, _evaluation_config_hash, _read_jsonl, _write_jsonl,
    load_config,
)
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
from experiments.wanda_failure_characterization.run_failure_map import model_sha
from experiments.dlm_dual_role_full1319.run import _validate_rows
from experiments.projection_capacity_followup_65.core import paired_binary_comparison

ROOT=Path(__file__).resolve().parent
LIMIT=1319
REF=Path("experiments/dlm_loss_aggregation/exp002/results/predictions/dense.jsonl")
METHODS=("uniform","A","AC")

def sha(p): return mini.sha(Path(p))
def event(stage,**kw):
    row=dict(stage=stage,time=time.time(),pid=os.getpid(),method=kw.pop('method',None),**kw)
    mini.write(ROOT / f"full_progress_{row['method'] or 'summary'}.json", row);print(json.dumps(row),flush=True)

def config():
    c=mini.validate(); cfg=load_config(mini.old.EVAL); ph,protocol=_evaluation_config_hash(cfg)
    if ph!=c['protocol_hash']: raise RuntimeError('protocol hash mismatch')
    path=ROOT/'full_config.json'; value=dict(status='frozen_before_full',limit=LIMIT,
      model=c['model'],protocol_hash=ph,protocol=protocol,methods=list(METHODS),
      manifests={m:sha(ROOT/m/'mask_manifest.json') for m in METHODS},
      source_config_sha256=sha(ROOT/'config.json'),no_retuning=True)
    if path.exists() and json.loads(path.read_text())!=value: raise RuntimeError('full config changed')
    if not path.exists(): mini.write(path,value)
    return value

def evaluate(method):
    c=config(); out=ROOT/'full'/f'{method}_1319_predictions.jsonl'; receipt=out.with_suffix('.receipt.json')
    fingerprint=dict(config_sha256=sha(ROOT/'full_config.json'),manifest_sha256=c['manifests'][method],protocol_sha256=c['protocol_hash'],limit=LIMIT,method=method)
    if out.exists() and receipt.exists():
        r=json.loads(receipt.read_text())
        if r.get('fingerprint')!=fingerprint or r.get('predictions_sha256')!=sha(out): raise RuntimeError('receipt mismatch')
        rows=_read_jsonl(out);_validate_rows(rows,_read_jsonl(REF),c['protocol_hash'])
        event('reused',method=method,correct=sum(x['correct'] for x in rows));return
    event('loading',method=method);model,mapping=mini.load_dense();model.eval()
    manifest=json.loads((ROOT/method/'mask_manifest.json').read_text())
    if apply_manifest(model,mapping,manifest)!=mini.PRUNED: raise RuntimeError('budget mismatch')
    sparse_sha=model_sha(model)
    tok=AutoTokenizer.from_pretrained(c['model']['id'],revision=c['model']['revision'],trust_remote_code=True)
    event('generating',method=method,completed=0,total=LIMIT)
    metrics,rows=_evaluate_gsm8k(model,tok,load_config(mini.old.EVAL),f'context_response50_full_{method}',LIMIT,c['protocol_hash'])
    if model_sha(model)!=sparse_sha: raise RuntimeError('model changed during eval')
    _validate_rows(rows,_read_jsonl(REF),c['protocol_hash'])
    out.parent.mkdir(parents=True,exist_ok=True);_write_jsonl(out,rows)
    mini.write(receipt,dict(status='complete',fingerprint=fingerprint,predictions_sha256=sha(out),sparse_model_sha256=sparse_sha,metrics=metrics))
    event('complete',method=method,correct=sum(x['correct'] for x in rows),seconds=metrics['eval_seconds'])
    del model;gc.collect();torch.cuda.empty_cache()

def finalize():
    c=config(); ref=_read_jsonl(REF); rows={}
    for m in METHODS:
        p=ROOT/'full'/f'{m}_1319_predictions.jsonl'; rows[m]=_read_jsonl(p);_validate_rows(rows[m],ref,c['protocol_hash'])
    result=dict(status='complete',limit=LIMIT,correct={m:sum(x['correct'] for x in rows[m]) for m in METHODS},
      accuracy={m:sum(x['correct'] for x in rows[m])/LIMIT for m in METHODS},
      paired={'AC_vs_A':paired_binary_comparison([x['correct'] for x in rows['A']],[x['correct'] for x in rows['AC']]),
              'AC_vs_uniform':paired_binary_comparison([x['correct'] for x in rows['uniform']],[x['correct'] for x in rows['AC']]),
              'A_vs_uniform':paired_binary_comparison([x['correct'] for x in rows['uniform']],[x['correct'] for x in rows['A']])},
      config_sha256=sha(ROOT/'full_config.json'))
    mini.write(ROOT/'full_results.json',result);event('full_complete',**result['correct'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','evaluate','finalize']);p.add_argument('--method',choices=METHODS);a=p.parse_args()
    if a.action=='freeze': config()
    elif a.action=='evaluate': evaluate(a.method)
    else: finalize()
