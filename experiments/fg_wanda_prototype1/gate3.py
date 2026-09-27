"""Gate 3 for the one frozen Fisher-Geometry Wanda prototype.

Jobs are intentionally separable so GSM8K and WinoGrande can run on distinct
GPUs without ever constructing another prototype.
"""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from scipy.stats import pearsonr, rankdata
from transformers import AutoTokenizer

from cgq_sparsegpt_downstream import _evaluate_winogrande, assert_same_examples as assert_same_wino
from experiments.dlm_loss_aggregation.core import mask_sha256, pack_mask, unpack_mask
from experiments.dlm_loss_aggregation.exp002.run import (
    _assert_same_examples,
    _evaluate_gsm8k,
    _evaluation_config_hash,
    _overall_mask_hash,
    _read_jsonl,
    _validated_dense_fingerprint,
    _write_jsonl,
    _zero_mask_summary,
    load_config as load_exp002_config,
)
from experiments.dlm_loss_aggregation.run import _load_model, load_config as load_base_config
from experiments.fg_wanda_prototype1.freeze import ROOT, STORE, sha, write_json
from experiments.fg_wanda_prototype1.gate3_core import (
    apply_masks,
    build_standard_and_fg_masks,
    paired_summary,
    raw_mask_sha256,
)
from experiments.wanda_failure_characterization.run_failure_map import model_sha, modules
from lib.prune_llada import WrappedGPT, prune_sparsegpt, prune_wanda


MASK_ROOT=STORE/'gate3_masks'
GATE3_ROOT=ROOT/'gate3'
EXP002=Path('experiments/dlm_loss_aggregation/exp002')
TARGET='block_31.ff_out'


def _require_gate2():
    gate=json.loads((ROOT/'gate2.json').read_text())
    if gate['status']!='pass':raise RuntimeError('Gate 2 did not pass')
    return gate


def _atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(data);tmp.replace(path)


def _load_packed_masks(method):
    manifest=json.loads((ROOT/'gate3_mask_manifest.json').read_text())
    rows=[x for x in manifest['entries'] if x['method']==method]
    result={}
    for row in rows:
        bits=Path(row['runtime_path']).read_bytes()
        payload={'shape':row['shape'],'bits':bits}
        if mask_sha256(payload)!=row['sha256']:raise RuntimeError('packed mask hash mismatch')
        mask=unpack_mask(payload)
        if raw_mask_sha256(mask)!=row['raw_sha256']:raise RuntimeError('raw mask hash mismatch')
        result[row['canonical_name']]=mask
    if len(result)!=224:raise RuntimeError('full mask set is incomplete')
    if _overall_mask_hash(rows,method)!=manifest['overall_sha256'][method]:
        raise RuntimeError('overall mask hash mismatch')
    return result,manifest


@torch.no_grad()
def prepare_masks():
    _require_gate2();GATE3_ROOT.mkdir(parents=True,exist_ok=True);MASK_ROOT.mkdir(parents=True,exist_ok=True)
    cfg=load_base_config('experiments/dlm_loss_aggregation/config.yaml')
    model,_=_load_model(cfg);dense=model_sha(model);mapping=modules(model)
    fg=torch.load(STORE/'mask_FG.pt',map_location='cpu',weights_only=True)
    target=mapping[TARGET];target_dense_weight=target.weight.detach().float().cpu();captured={}
    original_add_batch=WrappedGPT.add_batch
    def instrument_add_batch(wrapper,inp,out):
        original_add_batch(wrapper,inp,out)
        if wrapper.layer is target and wrapper.nsamples==8:
            captured['A']=wrapper.scaler_row.detach().cpu().clone()
    WrappedGPT.add_batch=instrument_add_batch
    tokenizer=AutoTokenizer.from_pretrained(cfg['model']['id'],revision=cfg['model']['revision'],trust_remote_code=True)
    args=SimpleNamespace(nsamples=8,seed=0,sparsity_ratio=.5,use_variant=False)
    started=time.monotonic()
    try:prune_wanda(args,model,tokenizer,device=torch.device('cuda:0'))
    finally:WrappedGPT.add_batch=original_add_batch
    pruning_seconds=time.monotonic()-started
    if 'A' not in captured:raise RuntimeError('repository target scaler was not captured')
    standard={name:module.weight.eq(0).detach().cpu() for name,module in mapping.items()}
    seq_a=captured['A'];seq_score=target_dense_weight.abs()*seq_a.sqrt()[None,:]
    if not torch.equal(standard[TARGET],__import__('experiments.wanda_failure_characterization.core',fromlist=['rowwise_wanda_mask']).rowwise_wanda_mask(seq_score,.5)):
        raise RuntimeError('captured sequential target score does not reproduce repository Wanda mask')
    torch.save(seq_score,STORE/'score_WandaSequential.pt');torch.save(seq_a,STORE/'A_WandaSequential.pt')
    ours=dict(standard);ours[TARGET]=fg
    if raw_mask_sha256(ours[TARGET])!=json.loads((ROOT/'gate1.json').read_text())['mask_raw_hashes']['FG']:
        raise RuntimeError('FG target mask differs from Gate 1')
    changed=[name for name in mapping if not torch.equal(standard[name],ours[name])]
    if changed!=[TARGET]:raise RuntimeError('prototype changes modules other than target')
    entries=[];overall={}
    for method,maskset in [('Wanda',standard),('Ours-FG',ours)]:
        method_rows=[]
        for canonical_name,mask in sorted(maskset.items()):
            block=int(canonical_name[6:8]);name=canonical_name.split('.',1)[1]
            payload=pack_mask(mask);path=MASK_ROOT/method/f'{canonical_name}.bin';_atomic_bytes(path,payload['bits'])
            row={'method':method,'block_index':block,'module':name,'canonical_name':canonical_name,
                 'shape':list(mask.shape),'sha256':mask_sha256(payload),'raw_sha256':raw_mask_sha256(mask),
                 'runtime_path':str(path),'pruned':int(mask.sum()),'weights':mask.numel()}
            method_rows.append(row);entries.append(row)
        overall[method]=_overall_mask_hash(method_rows,method)
    exp002_expected=json.loads((EXP002/'masks/baseline_hashes.json').read_text())['hashes']['Wanda']
    if overall['Wanda']!=exp002_expected:raise RuntimeError('full standard Wanda mask differs from EXP-002')
    # Re-run Gate 1 against the exact sequential full-model Wanda target mask.
    score_sets={'FG':torch.load(STORE/'score_FG.pt',map_location='cpu',weights_only=True),
                'DLMW':torch.load(STORE/'score_DLMW.pt',map_location='cpu',weights_only=True),
                'WandaSequential':seq_score}
    mask_sets={'FG':fg,'DLMW':torch.load(STORE/'mask_DLMW.pt',map_location='cpu',weights_only=True),
               'WandaSequential':standard[TARGET]}
    arrays={k:v.numpy() for k,v in score_sets.items()};full={k:rankdata(v.ravel()) for k,v in arrays.items()}
    rowranks={k:rankdata(v,axis=1) for k,v in arrays.items()};comparisons=[]
    for left,right in [('FG','WandaSequential'),('FG','DLMW'),('DLMW','WandaSequential')]:
        xor=(mask_sets[left]!=mask_sets[right]).float().mean(1).numpy();a=rowranks[left];b=rowranks[right]
        a=a-a.mean(1,keepdims=True);b=b-b.mean(1,keepdims=True);rho=(a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1))
        comparisons.append({'pair':left+'_vs_'+right,'full_score_spearman':float(pearsonr(full[left],full[right]).statistic),
            'row_spearman_mean':float(rho.mean()),'row_spearman_median':float(np.median(rho)),
            'global_mask_xor':float(xor.mean()),'median_row_xor':float(np.median(xor)),
            'p10_row_xor':float(np.quantile(xor,.1)),'p90_row_xor':float(np.quantile(xor,.9))})
    primary=comparisons[0];passed=primary['global_mask_xor']>=.02 and primary['median_row_xor']>=.01
    write_json(ROOT/'gate1_repository_exact.json',{'status':'pass' if passed else 'failed','comparisons':comparisons,
        'standard_wanda':'repository sequential layer-wise calibration','target_scaler_nsamples':8})
    if not passed:raise RuntimeError('Gate 1 fails against exact repository full-model Wanda; branch terminated')
    manifest={'status':'complete','model_revision':cfg['model']['revision'],'dense_model_sha256':dense,
              'methods':['Wanda','Ours-FG'],'matrix_count_per_method':224,'changed_modules':[TARGET],
              'overall_sha256':overall,'exp002_standard_wanda_sha256':exp002_expected,'entries':entries,
              'repository_wanda_pruning_seconds':pruning_seconds,'repository_wanda_pruned_model_sha256':model_sha(model),
              'gate1_repository_exact':primary}
    write_json(ROOT/'gate3_mask_manifest.json',manifest)
    print(json.dumps({'event':'gate3_masks_prepared','overall':overall,'changed':changed}),flush=True)


def _load_model_and_apply(method):
    cfg=load_base_config('experiments/dlm_loss_aggregation/config.yaml')
    model,_=_load_model(cfg);before=model_sha(model)
    expected=json.loads((ROOT/'gate3_mask_manifest.json').read_text())['dense_model_sha256']
    if before!=expected:raise RuntimeError('dense model SHA differs before Gate 3')
    pruning_seconds=0.0
    if method in ('Wanda','Ours-FG'):
        masks,manifest=_load_packed_masks(method);started=time.monotonic()
        applied=apply_masks(modules(model),masks);pruning_seconds=time.monotonic()-started
        zero=_zero_mask_summary(model,method,require_exact_rowwise=True)
        if abs(applied['sparsity']-.5)>0 or abs(zero['sparsity']-.5)>0:raise RuntimeError('rowwise sparsity mismatch')
        if zero['mask_hash']!=manifest['overall_sha256'][method]:raise RuntimeError('applied full mask hash mismatch')
    elif method=='SparseGPT':
        tok=AutoTokenizer.from_pretrained(cfg['model']['id'],revision=cfg['model']['revision'],trust_remote_code=True)
        args=SimpleNamespace(nsamples=8,seed=0,sparsity_ratio=.5)
        started=time.monotonic();prune_sparsegpt(args,model,tok,dev=torch.device('cuda:0'))
        pruning_seconds=time.monotonic()-started
        zero=_zero_mask_summary(model,method,require_exact_rowwise=False)
    elif method=='Dense':
        zero={'mask_hash':'','sparsity':0.0,'matrix_count':224,'rowwise_exact':None}
    else:raise ValueError(method)
    return model,before,zero,pruning_seconds


def _load_gsm_references(config_hash):
    names={'Dense':'dense','Wanda':'wanda','SparseGPT':'sparsegpt'};rows={};reference=None
    with (EXP002/'results/gsm8k.csv').open() as handle:
        summary={r['method']:r for r in csv.DictReader(handle)}
    for method,stem in names.items():
        rec=_read_jsonl(EXP002/f'results/predictions/{stem}.jsonl')
        if len(rec)!=1319 or any(x['evaluation_config_hash']!=config_hash for x in rec):
            raise RuntimeError(f'EXP-002 {method} records do not match exact protocol')
        if reference is None:reference=rec
        else:_assert_same_examples(reference,rec)
        correct=sum(x['correct'] for x in rec)
        if correct!=round(float(summary[method]['accuracy'])*1319):raise RuntimeError('GSM reference aggregate mismatch')
        rows[method]={'accuracy':correct/1319,'correct':correct,'records':rec,
                      'source':str(EXP002/f'results/predictions/{stem}.jsonl'),
                      'source_sha256':sha(EXP002/f'results/predictions/{stem}.jsonl')}
    return rows


def run_gsm():
    _require_gate2();GATE3_ROOT.mkdir(parents=True,exist_ok=True)
    fixed=json.loads((ROOT/'gate3_preregistered.json').read_text());config=load_exp002_config(EXP002/'config.yaml')
    config_hash,doc=_evaluation_config_hash(config)
    expected=json.loads((EXP002/'logs/evaluation_config.json').read_text())
    if config_hash!=expected['sha256'] or doc['gsm8k_task_sha256']!=expected['gsm8k_task_sha256']:
        raise RuntimeError('current GSM evaluator differs from exact EXP-002 protocol')
    refs=_load_gsm_references(config_hash)
    score_meta=json.loads(Path(config['source_exp001']['score_metadata']).read_text())
    model,before,mask,pruning_seconds=_load_model_and_apply('Ours-FG')
    dense_fingerprint=_validated_dense_fingerprint(model,config,score_meta)
    # _validated_dense_fingerprint hashes values, so after pruning it must not be compared to Dense.
    pruned_before=model_sha(model)
    tokenizer=AutoTokenizer.from_pretrained(config['model']['id'],revision=config['model']['revision'],trust_remote_code=True)
    measured,records=_evaluate_gsm8k(model,tokenizer,config,'Ours-FG',None,config_hash)
    if measured['num_examples']!=fixed['gsm8k']['sample_count']:raise RuntimeError('wrong GSM sample count')
    _assert_same_examples(refs['Dense']['records'],records)
    pruned_after=model_sha(model)
    if pruned_before!=pruned_after:raise RuntimeError('GSM evaluation changed pruned weights')
    out=GATE3_ROOT/'gsm8k_ours_predictions.jsonl';_write_jsonl(out,records)
    paired={name:paired_summary(refs[name]['records'],records) for name in ['Wanda','SparseGPT']}
    result={'status':'complete','protocol_hash':config_hash,'protocol':doc,'model_revision':config['model']['revision'],
            'dense_model_sha256':before,'pruned_model_sha256_before_eval':pruned_before,
            'pruned_model_sha256_after_eval':pruned_after,'mask':mask,'pruning_seconds':pruning_seconds,
            'ours':{**measured,'correct':sum(x['correct'] for x in records),'prediction_sha256':sha(out)},
            'references':{k:{a:b for a,b in v.items() if a!='records'} for k,v in refs.items()},'paired':paired,
            'note':'dense_fingerprint below is intentionally measured after applying Ours-FG and identifies the evaluated prunable tensors',
            'evaluated_prunable_fingerprint':dense_fingerprint,
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'lm_eval':importlib.metadata.version('lm_eval')}}
    write_json(GATE3_ROOT/'gsm8k.json',result);print(json.dumps({'event':'gsm_complete','ours':result['ours'],'paired':paired}),flush=True)


def _save_wino_method(method,result,records):
    path=GATE3_ROOT/f'winogrande_{method.lower().replace("-","_")}_predictions.jsonl';_write_jsonl(path,records)
    result=dict(result,prediction_path=str(path),prediction_sha256=sha(path))
    write_json(GATE3_ROOT/f'winogrande_{method.lower().replace("-","_")}.json',result)
    return result


def run_wino():
    _require_gate2();GATE3_ROOT.mkdir(parents=True,exist_ok=True)
    fixed=json.loads((ROOT/'gate3_preregistered.json').read_text());base=load_base_config('experiments/dlm_loss_aggregation/config.yaml')
    tokenizer=AutoTokenizer.from_pretrained(base['model']['id'],revision=base['model']['revision'],trust_remote_code=True)
    methods=['Dense','Wanda','SparseGPT','Ours-FG'];all_results={};all_records={};reference=None
    for method in methods:
        stem=method.lower().replace('-','_');summary_path=GATE3_ROOT/f'winogrande_{stem}.json'
        prediction_path=GATE3_ROOT/f'winogrande_{stem}_predictions.jsonl'
        if summary_path.exists() and prediction_path.exists():
            result=json.loads(summary_path.read_text());records=_read_jsonl(prediction_path)
            if result['status']!='complete' or len(records)!=fixed['winogrande']['sample_count']:
                raise RuntimeError('invalid partial Wino result')
            print(json.dumps({'event':'wino_reuse_completed','method':method}),flush=True)
        else:
            model,before,mask,pruning_seconds=_load_model_and_apply(method);pruned_before=model_sha(model)
            measured,records=_evaluate_winogrande(model,tokenizer,torch.device('cuda:0'))
            if measured['sample_count']!=fixed['winogrande']['sample_count']:raise RuntimeError('wrong Wino sample count')
            pruned_after=model_sha(model)
            if pruned_before!=pruned_after:raise RuntimeError('Wino evaluation changed model weights')
            result={'status':'complete','method':method,'model_revision':base['model']['revision'],'dense_model_sha256':before,
                    'pruned_model_sha256_before_eval':pruned_before,'pruned_model_sha256_after_eval':pruned_after,
                    'mask':mask,'pruning_seconds':pruning_seconds,'evaluation':measured,
                    'environment':{'python':platform.python_version(),'torch':torch.__version__,'lm_eval':importlib.metadata.version('lm_eval')}}
            result=_save_wino_method(method,result,records);del model;gc.collect();torch.cuda.empty_cache()
            print(json.dumps({'event':'wino_method_complete','method':method,'evaluation':measured}),flush=True)
        if reference is None:reference=records
        else:assert_same_wino(reference,records)
        all_results[method]=result;all_records[method]=records
    paired={name:paired_summary(all_records[name],all_records['Ours-FG']) for name in ['Wanda','SparseGPT']}
    result={'status':'complete','model_revision':base['model']['revision'],'protocol':fixed['winogrande'],
            'methods':all_results,'paired':paired}
    write_json(GATE3_ROOT/'winogrande.json',result);print(json.dumps({'event':'wino_complete','paired':paired}),flush=True)


def finalize():
    gsm=json.loads((GATE3_ROOT/'gsm8k.json').read_text());wino=json.loads((GATE3_ROOT/'winogrande.json').read_text())
    primary=gsm['paired']['Wanda'];success=primary['fg_correct']>primary['wanda_correct']
    verdict='SUCCESS' if success else 'FAILED DOWNSTREAM'
    result={'verdict':verdict,'gate1':'pass','gate2':'pass','primary_rule':json.loads((ROOT/'gate3_preregistered.json').read_text())['downstream_success_rule'],
            'gsm8k_ours_vs_wanda':primary,'gsm8k_ours_vs_sparsegpt':gsm['paired']['SparseGPT'],
            'winogrande_ours_vs_wanda':wino['paired']['Wanda'],'winogrande_ours_vs_sparsegpt':wino['paired']['SparseGPT'],
            'no_rescue_experiment':True}
    write_json(ROOT/'verdict.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',choices=['prepare','gsm','wino','finalize']);job=parser.parse_args().job
    {'prepare':prepare_masks,'gsm':run_gsm,'wino':run_wino,'finalize':finalize}[job]()


if __name__=='__main__':main()
