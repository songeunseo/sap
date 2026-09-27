#!/usr/bin/env python3
from __future__ import annotations

import bisect
import copy
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from experiments.dlm_loss_aggregation.run import build_calibration_manifest, historical_state_digest, load_config
from experiments.dlm_role_exchange_prediction.core import ROOT, SEED, TIMESTEPS, atomic_json, feasible_exchanges, json_digest, select_bundles, sha256
from experiments.projection_capacity_followup_65.freeze_new_states import intervals_for
from model import LLaDAConfig

A_PATH=Path("experiments/dlm_dual_role_mini100/role65_mask_manifest.json")
CANDIDATE=Path("experiments/projection_capacity_allocation_65/candidate_mask_manifest.json")
PRIOR=[Path("experiments/dlm_loss_aggregation/exp004/calibration_state_manifest.json"),
       Path("experiments/wanda_failure_characterization/heldout_state_manifest.json"),
       Path("experiments/projection_capacity_followup_65/new_heldout_state_manifest.json")]


def article_rows(rows):
    articles=[]; current=[]
    for index,text in enumerate(rows):
        stripped=text.strip()
        if stripped.startswith("=") and stripped.endswith("=") and current:
            articles.append(current); current=[]
        current.append(index)
    if current: articles.append(current)
    return articles


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    a=json.loads(A_PATH.read_text()); candidate=json.loads(CANDIDATE.read_text())
    levels=list(map(int,a["allocation_levels"])); exchanges=feasible_exchanges(levels,candidate["entries"])
    bundles=select_bundles(exchanges)
    if len(bundles) != 64:
        raise RuntimeError(f"could not select 8 disjoint bundles/four-layer group: {len(bundles)}")

    cfg=copy.deepcopy(load_config("experiments/dlm_loss_aggregation/config.yaml"))
    cfg["calibration"].update(seed=SEED,sequence_count=32,sequence_indices=list(range(32)),timesteps=list(TIMESTEPS))
    tok=AutoTokenizer.from_pretrained(cfg["model"]["id"],revision=cfg["model"]["revision"],trust_remote_code=True)
    model_cfg=LLaDAConfig.from_pretrained(cfg["model"]["id"],revision=cfg["model"]["revision"])
    corpus=load_dataset("Salesforce/wikitext","wikitext-2-raw-v1",split="train")
    texts=corpus["text"]; joined=" ".join(texts)
    encoded=tok(joined,return_tensors="pt",return_offsets_mapping=True)
    ids=encoded.input_ids[0]; offsets=encoded.offset_mapping[0].tolist()
    row_starts=[]; cursor=0
    for text in texts: row_starts.append(cursor); cursor += len(text)+1
    token_by_row=[[] for _ in texts]
    for ti,(start,_end) in enumerate(offsets):
        row=bisect.bisect_right(row_starts,start)-1
        if row>=0: token_by_row[row].append(ti)
    prior_docs=[json.loads(p.read_text()) for p in PRIOR]
    prior_intervals=[x for d in prior_docs for x in intervals_for(d,ids)]
    eligible=[]
    for aid,rows in enumerate(article_rows(texts)):
        tokens=[t for r in rows for t in token_by_row[r]]
        if len(tokens)<258: continue
        start=min(tokens)+1; end=max(tokens)
        if end-start<256: continue
        eligible.append((aid,start,end))
    rng=random.Random(SEED); rng.shuffle(eligible); selected=[]
    for aid,lo,hi in eligible:
        start=rng.randint(lo,hi-256)
        interval=(start,start+256)
        if any(max(start,p["start"])<min(start+256,p["end_exclusive"]) for p in prior_intervals): continue
        selected.append((aid,start))
        if len(selected)==32: break
    if len(selected)!=32: raise RuntimeError("fewer than 32 disjoint WikiText articles available")
    clean=[ids[start:start+256].unsqueeze(0) for _aid,start in selected]
    states=build_calibration_manifest(clean,model_cfg.mask_token_id,cfg)
    states.update(mask_id=model_cfg.mask_token_id,historical_state_sha256=historical_state_digest(states),
                  frozen_before_outcomes=True,purpose="role exchange prediction",
                  documents=[{"sequence_index":i,"article_id":aid,"start":start,"end_exclusive":start+256,
                              "split":"development" if i<16 else "final"} for i,(aid,start) in enumerate(selected)])
    atomic_json(ROOT/"state_manifest.json",states)
    design={"status":"frozen_before_outcomes","seed":SEED,"timesteps":list(TIMESTEPS),
            "development_documents":list(range(16)),"final_documents":list(range(16,32)),
            "layer_folds":[list(range(i*8,(i+1)*8)) for i in range(4)],
            "ridge_alpha":1.0,"primary_models":["P1_pooled_dense","P2_role_dense","P3_role_dense_sparse"],
            "primary_comparisons":[["P2_role_dense","P1_pooled_dense"],["P3_role_dense_sparse","P2_role_dense"],["P2_role_dense","random_role_mean"]],
            "bootstrap_resamples":20000,"simultaneous_interval":"Bonferroni 3-comparison 95%",
            "minimum_practical_relative_mse_reduction":0.10,"exchanges":exchanges,"bundles":bundles,
            "state_digest":states["historical_state_sha256"],"bundle_digest":json_digest(bundles),
            "source_hashes":{str(p):sha256(p) for p in [A_PATH,CANDIDATE,*PRIOR]}}
    atomic_json(ROOT/"design.json",design)
    atomic_json(ROOT/"state_verification.json",{"status":"verified","new_documents":32,"new_states":160,
                "unique_articles":len(set(x[0] for x in selected)),"prior_intervals":len(prior_intervals),
                "overlaps":[],"state_manifest_sha256":sha256(ROOT/"state_manifest.json")})
    print(json.dumps({"event":"prepared","states":160,"exchanges":len(exchanges),"bundles":len(bundles)}))

if __name__=="__main__": main()
