#!/usr/bin/env python3
"""Freeze fourth disjoint split; refuse overlapping or unverifiable historical spans."""
import copy
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from model import LLaDAConfig
from lib.data import get_loaders
from experiments.dlm_loss_aggregation.run import build_calibration_manifest, historical_state_digest, load_config
from experiments.projection_capacity_followup_65.freeze_new_states import intervals_for
from experiments.dlm_capacity_predictor.core import (
    ROOT, FOLLOWUP, CALIBRATION, sha, write_json, require_disjoint_intervals, state_indices)


def require_candidate_config(candidates, config_sha256):
    if candidates.get('status') != 'frozen' or candidates.get('config_sha256') != config_sha256:
        raise RuntimeError('candidate/config freeze mismatch')


def main():
    candidates = json.loads((ROOT/'candidates.json').read_text())
    config_path = ROOT/'config.json'
    require_candidate_config(candidates, sha(config_path))
    for c in candidates['candidates']:
        if sha(c['mask_manifest']) != c['mask_manifest_sha256']:
            raise RuntimeError('candidate manifest changed')
    config = json.loads(config_path.read_text())
    split = config['fresh_split']
    prior_paths = [CALIBRATION,
                   FOLLOWUP.parent/'wanda_failure_characterization/heldout_state_manifest.json',
                   FOLLOWUP/'new_heldout_state_manifest.json']
    prior = [json.loads(p.read_text()) for p in prior_paths]
    for document in prior:
        if historical_state_digest(document) != document['historical_state_sha256']:
            raise RuntimeError('historical state digest mismatch')
    base = copy.deepcopy(load_config('experiments/dlm_loss_aggregation/config.yaml'))
    if base['model']['id'] != config['model'] or base['model']['revision'] != config['revision']:
        raise RuntimeError('model revision mismatch')
    base['calibration'].update(seed=split['seed'], sequence_count=8,
        sequence_indices=split['sequences'], timesteps=split['timesteps'])
    model_config = LLaDAConfig.from_pretrained(config['model'], revision=config['revision'])
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], trust_remote_code=True)
    loader, _ = get_loaders('wikitext2', nsamples=8, seed=split['seed'], seqlen=256, tokenizer=tokenizer)
    document = build_calibration_manifest([sample[0] for sample in loader], model_config.mask_token_id, base)
    document.update(mask_id=model_config.mask_token_id,
                    historical_state_sha256=historical_state_digest(document),
                    purpose='two-path capacity predictor confirmatory evaluation')
    corpus = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    tokens = tokenizer(' '.join(corpus['text']), return_tensors='pt').input_ids[0]
    groups = [intervals_for(d, tokens) for d in prior + [document]]
    require_disjoint_intervals(groups)
    state_indices([(s['sequence_index'], s['timestep']) for s in document['states']],
                  split['sequences'], split['timesteps'])
    path = ROOT/'heldout_state_manifest.json'
    write_json(path, document, frozen=True)
    write_json(ROOT/'state_verification.json', dict(status='verified', all_four_splits_disjoint=True,
        manifest_sha256=sha(path), state_sha256=document['historical_state_sha256'],
        prior_manifest_sha256={str(p): sha(p) for p in prior_paths}, corpus_token_count=len(tokens),
        interval_groups=groups, candidates_sha256=sha(ROOT/'candidates.json')), frozen=True)
    print(json.dumps({'status': 'verified', 'states': 40, 'all_four_splits_disjoint': True}), flush=True)


if __name__ == '__main__':
    main()
