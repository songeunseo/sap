"""Full DEVELOPMENT validation scoring; preserve frozen pilot and test split."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import statistics
import time

import torch

from experiments.dlm_wikitext_ppl import run as protocol
from experiments.dlm_wikitext_ppl.core import digest, mask_digest, masks, normalized_nelbo, summarize
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write, file_sha256 as sha

METHODS=('uniform','owl','dlp','alpha','lsa','dsa')


def synchronize(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def seal(row):
    return dict(row, row_sha256=digest(row))


def verify_row(row, source, cfg, config_sha):
    unsigned={k:v for k,v in row.items() if k!='row_sha256'}
    if row.get('row_sha256') != digest(unsigned):
        raise RuntimeError('block checkpoint digest mismatch')
    for key, expected in (('block_id',source['block_id']),('tokens',len(source['clean_ids'])),
                          ('mc_samples',cfg['mc_samples']),('mask_sha256',source['mask_sha256']),
                          ('config_sha256',config_sha)):
        if row.get(key)!=expected:
            raise RuntimeError('block checkpoint identity mismatch: '+key)
    values=row['sample_token_nelbo']
    if len(values)!=cfg['mc_samples'] or not all(math.isfinite(x) for x in values):
        raise RuntimeError('invalid per-draw losses')
    if row['token_nelbo']!=statistics.mean(values) or row['mc_variance']!=statistics.variance(values):
        raise RuntimeError('block statistics mismatch')


@torch.inference_mode()
def evaluate_block(model, source, cfg, progress=None):
    length=len(source['clean_ids'])
    if mask_digest(length,cfg['mc_samples'],source['mask_seed'])!=source['mask_sha256']:
        raise RuntimeError('frozen mask draws changed')
    device=next(model.parameters()).device
    clean=torch.tensor(source['clean_ids'],device=device)
    values=[]; synchronize(device); started=time.monotonic()
    for index,cpu_mask in enumerate(masks(length,cfg['mc_samples'],source['mask_seed'])):
        mask=cpu_mask.to(device)
        logits=model(clean.masked_fill(mask,cfg['mask_id']).unsqueeze(0)).logits[0]
        values.append(float(normalized_nelbo(logits,clean,mask)))
        del logits
        if progress and (index+1)%16==0:
            progress(index+1)
    synchronize(device)
    return dict(block_id=source['block_id'],tokens=length,mc_samples=len(values),
                token_nelbo=statistics.mean(values),mc_variance=statistics.variance(values),
                sample_token_nelbo=values,mask_sha256=source['mask_sha256'],
                evaluation_seconds=time.monotonic()-started,origin='full_validation')


def candidate_config(method, folder):
    cfg=protocol.validate()
    manifest=protocol.REPO/'experiments/dlm_allocation_sequential65'/method/'mask_manifest.json'
    if not manifest.exists() or protocol.read(manifest)['pruned']!=4536008704:
        raise RuntimeError('completed exact-budget mask manifest required')
    run_cfg=dict(method=method,split='validation',phase='development',
                 scope='full frozen validation; no test access, no GSM8K, no automatic hyperparameter search',
                 protocol_config_sha256=sha(protocol.ROOT/'config.json'),
                 mask_manifest=str(manifest),mask_manifest_sha256=sha(manifest),
                 sources={str(p):sha(p) for p in (Path(__file__),protocol.ROOT/'test_evaluate.py')})
    protocol.frozen(folder/'config.json',run_cfg)
    return cfg,run_cfg


@torch.inference_mode()
def run(method):
    folder=protocol.ROOT/'validation'/method
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg,run_cfg=candidate_config(method,folder)
        config_sha=sha(folder/'config.json')
        if (folder/'results.json').exists():
            r=protocol.read(folder/'results.json')
            if r['config_sha256']!=config_sha:
                raise RuntimeError('result config changed')
            print('already complete',r['summary'],flush=True)
            return
        progress_path=folder/'progress.json'
        invocation_started=time.monotonic()
        def event(stage,**kw):
            value=dict(stage=stage,method=method,split='validation',pid=os.getpid(),time=time.time(),**kw)
            write(progress_path,value); print(value,flush=True)
        try:
            corpus=protocol.read(protocol.ROOT/'corpus_manifest.json')['splits']['validation']
            rows=[]; missing=[]
            pilot_rows={}
            if method=='uniform':
                pilot=protocol.read(protocol.ROOT/'pilot_results.json')
                if pilot['config_sha256']!=run_cfg['protocol_config_sha256'] or pilot['manifest_sha256']!=run_cfg['mask_manifest_sha256']:
                    raise RuntimeError('pilot model/protocol differs')
                pilot_rows={r['block_id']:r for r in pilot['rows']}
            for index,source in enumerate(corpus):
                path=folder/'blocks'/f"{digest(source['block_id'])[:20]}.json"
                if path.exists():
                    row=protocol.read(path);verify_row(row,source,cfg,config_sha)
                    rows.append(row)
                elif source['block_id'] in pilot_rows:
                    original=pilot_rows[source['block_id']]
                    row=seal(dict(original,config_sha256=config_sha,origin='exact_uniform_pilot_reuse'))
                    verify_row(row,source,cfg,config_sha)
                    protocol.frozen(path,row);rows.append(row)
                else:
                    missing.append((index,source,path))
            total=len(corpus)*cfg['mc_samples']
            baseline_completed=len(rows)*cfg['mc_samples']
            event('loading_dense',completed=baseline_completed,total=total,blocks_completed=len(rows),blocks_total=len(corpus))
            preparation_seconds=0.
            if missing:
                from experiments.projection_capacity_allocation_65.run import load_dense
                from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
                started=time.monotonic(); model,mapping=load_dense()
                apply_manifest(model,mapping,protocol.read(run_cfg['mask_manifest']))
                device=next(model.parameters()).device;synchronize(device)
                preparation_seconds=time.monotonic()-started
                measured_draws=0; eval_started=time.monotonic()
                for index,source,path in missing:
                    completed_before=len(rows)*cfg['mc_samples']
                    def tick(current):
                        fresh=measured_draws+current
                        elapsed=time.monotonic()-eval_started
                        completed=completed_before+current
                        event('evaluating_validation',completed=completed,total=total,
                              blocks_completed=len(rows),blocks_total=len(corpus),
                              block_id=source['block_id'],remaining_seconds=(total-completed)*elapsed/fresh)
                    row=evaluate_block(model,source,cfg,tick)
                    row=seal(dict(row,config_sha256=config_sha));verify_row(row,source,cfg,config_sha)
                    protocol.frozen(path,row);rows.append(row)
                    measured_draws+=cfg['mc_samples']
            ordered={r['block_id']:r for r in rows}
            if set(ordered)!=set(s['block_id'] for s in corpus) or len(rows)!=len(corpus):
                raise RuntimeError('full corpus coverage mismatch')
            rows=[ordered[s['block_id']] for s in corpus]
            result=dict(status='complete',method=method,split='validation',phase='development',
                        summary=summarize(rows),config_sha256=config_sha,
                        protocol_config_sha256=run_cfg['protocol_config_sha256'],
                        mask_manifest_sha256=run_cfg['mask_manifest_sha256'],
                        preparation_seconds_this_invocation=preparation_seconds,
                        wall_seconds_this_invocation=time.monotonic()-invocation_started,
                        evaluation_seconds=sum(r['evaluation_seconds'] for r in rows),
                        reused_pilot_blocks=sum(r['origin']=='exact_uniform_pilot_reuse' for r in rows),
                        block_checkpoint_digests={r['block_id']:r['row_sha256'] for r in rows},
                        note='Full development score, not final test; no hyperparameters selected')
            protocol.validate()
            for path,value in run_cfg['sources'].items():
                if sha(path)!=value:raise RuntimeError('evaluation source changed')
            if sha(run_cfg['mask_manifest'])!=run_cfg['mask_manifest_sha256']:
                raise RuntimeError('mask manifest changed')
            protocol.frozen(folder/'results.json',result)
            event('complete',completed=total,total=total,summary=result['summary'],remaining_seconds=0)
        except Exception as exc:
            event('failed',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=METHODS)
    run(parser.parse_args().method)
