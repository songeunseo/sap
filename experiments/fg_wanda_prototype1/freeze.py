import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from experiments.dlm_loss_aggregation.run import load_config, historical_state_digest, build_calibration_manifest

ROOT=Path(__file__).parent
STORE=Path('/DATA/tmluser1/sap-fg-wanda-prototype1')


def sha(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for b in iter(lambda:handle.read(8<<20),b''):digest.update(b)
    return digest.hexdigest()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def main():
    if (ROOT/'preregistered.json').exists():
        print('Frozen manifests already exist; will not resample.');return
    ROOT.mkdir(exist_ok=True);STORE.mkdir(exist_ok=True)
    cfg=load_config('experiments/dlm_loss_aggregation/config.yaml')
    historical=Path('experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json')
    calibration=json.loads(historical.read_text())
    assert historical_state_digest(calibration)==cfg['calibration']['expected_state_sha256']
    write_json(ROOT/'geometry_calibration_manifest.json',calibration)
    previous=json.loads(Path('experiments/wanda_failure_characterization/heldout_state_manifest.json').read_text())
    tok=AutoTokenizer.from_pretrained(cfg['model']['id'],revision=cfg['model']['revision'],trust_remote_code=True)
    data=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train')
    corpus=tok(' '.join(data['text']),return_tensors='pt').input_ids[0].numpy()
    # Recover every occurrence, so disjointness is not just inequality of full-span hashes.
    excluded=[]; unique=[]
    for source,manifest in [('geometry_and_clean_wanda',calibration),('mechanism',previous)]:
        for state in manifest['states']:
            ids=state['clean_ids'][0]
            if ids in unique:continue
            unique.append(ids)
            matches=[]
            for start in np.flatnonzero(corpus==ids[0]):
                if start+256<=len(corpus) and np.array_equal(corpus[start:start+256],ids):matches.append(int(start))
            if not matches:raise RuntimeError('Cannot locate historical span in exact tokenized corpus')
            excluded.extend({'start':s,'stop':s+256,'source':source} for s in matches)
    # Also verify the standard seed-0 clean loader's exact spans are in the historical set.
    std_rng=random.Random(0)
    for _ in range(8):
        start=std_rng.randint(0,len(corpus)-256-1)
        if corpus[start:start+256].tolist() not in unique:
            raise RuntimeError('clean Wanda calibration not covered by exclusion manifest')
    rng=random.Random(20260906);spans=[];rejected=0
    while len(spans)<8:
        start=rng.randint(0,len(corpus)-256-1);stop=start+256
        if any(start<x['stop'] and stop>x['start'] for x in excluded+spans):
            rejected+=1;continue
        spans.append({'sequence_index':16+len(spans),'start':start,'stop':stop})
    new_cfg=copy.deepcopy(cfg)
    new_cfg['calibration'].update(seed=20260906,sequence_indices=list(range(16,24)),timesteps=[.1,.3,.5,.7,.9])
    clean=[torch.tensor(corpus[s['start']:s['stop']].copy()).unsqueeze(0) for s in spans]
    evaluation=build_calibration_manifest(clean,126336,new_cfg)
    evaluation.update(mask_id=126336,historical_state_sha256=historical_state_digest(evaluation),
                      frozen_before_scoring=True,span_ids=spans,excluded_intervals=excluded,
                      tokenized_corpus_sha256=hashlib.sha256(corpus.tobytes()).hexdigest(),
                      selection_seed=20260906,rejections_for_interval_overlap_only=rejected)
    for state in evaluation['states']:
        state['state_sha256']=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    write_json(ROOT/'independent_evaluation_manifest.json',evaluation)
    write_json(ROOT/'preregistered.json',{
        'status':'frozen_before_scores_and_outcomes','prototype':1,'target':'model.transformer.blocks.31.ff_out',
        'storage':str(STORE),'sparsity':.5,'geometry_manifest_sha256':sha(ROOT/'geometry_calibration_manifest.json'),
        'evaluation_manifest_sha256':sha(ROOT/'independent_evaluation_manifest.json'),
        'score':'abs(W)*sqrt(K.T @ X_squared / total_masked_tokens)',
        'kappa_validation_states':[0,32,64],'kappa_validation_tokens':'first 2 masked positions',
        'kappa_validation_output_indices':[0,1,63,127,511,1023,2047,4095],
        'kappa_rtol':.001,'kappa_atol':1e-10,'negative_tolerance':'1e-12 + 1e-9*sum(abs(three variance terms))',
        'fp_precision':'FP32 continuous logits; FP64 softmax, moments, G accumulation; FP32 persisted scores',
        'gate1':{'global_xor_min':.02,'median_row_xor_min':.01},
        'gate2':{'W_minus_FG_mean_positive':True,'W_minus_FG_cluster_CI_lower_gt_zero':True,
                 'DLMW_minus_FG_mean_positive':True,'W_improvement_positive_timesteps_min':4,
                 'W_improvement_positive_sequences_min':6},
        'bootstrap_seed':20260905,'bootstrap_replicates':10000,'failure_map_claim':'target only, option B',
        'only_three_scores':['Wanda','DLMW','FG'],'no_tuning_or_second_prototype':True})
    print('Geometry digest:',historical_state_digest(calibration))
    print('Independent evaluation digest:',evaluation['historical_state_sha256'])
    print('Excluded historical intervals:',len(excluded),'new disjoint intervals:',spans)


if __name__=='__main__':main()
