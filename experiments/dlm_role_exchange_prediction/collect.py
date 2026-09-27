#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_loss_aggregation.run import _suffix_logits
from experiments.dlm_role_exchange_prediction.core import ROOT, RUNTIME, atomic_json, sha256
from experiments.dlm_role_validation.core import deterministic_random_role
from experiments.projection_capacity_allocation_65.run import DENSE_SHA, load_dense, read_mask
from experiments.projection_capacity_followup_65.run_heldout import apply_manifest
from experiments.wanda_failure_characterization.run_failure_map import model_sha, state_tensors

A_PATH=Path("experiments/dlm_dual_role_mini100/role65_mask_manifest.json")
CANDIDATE=Path("experiments/projection_capacity_allocation_65/candidate_mask_manifest.json")
SEEDS=(101,202,303)


def event(kind,**values): print(json.dumps({"event":kind,"time":time.time(),**values},sort_keys=True),flush=True)


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(value,tmp); tmp.replace(path)


def validate():
    design=json.loads((ROOT/"design.json").read_text()); states=json.loads((ROOT/"state_manifest.json").read_text())
    for path,digest in design["source_hashes"].items():
        if sha256(path)!=digest: raise RuntimeError(f"frozen source changed: {path}")
    if states["historical_state_sha256"]!=design["state_digest"]: raise RuntimeError("state digest mismatch")
    return design,states,json.loads(A_PATH.read_text()),json.loads(CANDIDATE.read_text())


def role_feature(x, original, base, candidate, bias, groups):
    dense=F.linear(x,original,bias).float(); a=F.linear(x,base,bias).float(); b=F.linear(x,candidate,bias).float()
    change=(b-dense).square()-(a-dense).square(); energy=dense.square()
    result={"pooled_token":float(change.sum().double().cpu()/change.shape[1])}
    for name,group in groups.items():
        group=group.to(change.device)
        result[f"{name}_token"]=float(change[0,group].sum().double().cpu()/int(group.sum()))
        den=energy[0,group].sum().double()
        result[f"{name}_energy"]=float((change[0,group].sum().double()/den).cpu())
    return result


def kl(logits,dense_ref,mask):
    selected=logits[0,mask[0]].float(); ref=dense_ref.float().to(selected.device)
    lp=F.log_softmax(selected,-1); rp=F.log_softmax(ref,-1)
    return float((rp.exp()*(rp-lp)).sum(-1).mean())


@torch.inference_mode()
def load_models(a_manifest):
    dense,dense_map=load_dense(); sparse,sparse_map=load_dense()
    if list(dense_map)!=list(sparse_map): raise RuntimeError("model mappings differ")
    apply_manifest(sparse,sparse_map,a_manifest)
    return dense,dense_map,sparse,sparse_map


@torch.inference_mode()
def candidate_weights(dense_map,candidate,design):
    needed={(x["module_index"],x["new_level"]) for x in design["exchanges"]}
    levels={int(x["module_index"]):int(x["base_level"]) for x in design["exchanges"]}
    weights={}; base={}
    for i,row in enumerate(candidate["entries"]):
        module=dense_map[row["name"]]; original=module.weight.detach()
        mask=read_mask(row,levels[i],original.device); base[i]=original.masked_fill(mask,0); del mask
        for index,level in sorted(x for x in needed if x[0]==i):
            mask=read_mask(row,level,original.device); weights[(index,level)]=original.masked_fill(mask,0); del mask
    return base,weights


@torch.inference_mode()
def smoke():
    design,states,a,candidate=validate(); dense,dm,sparse,sm=load_models(a)
    base,alternatives=candidate_weights(dm,candidate,design)
    state=states["states"][0]; noisy,_,mask=state_tensors(state,next(sparse.parameters()).device)
    exchange=design["exchanges"][0]; i=exchange["module_index"]; name=exchange["name"]; module=sm[name]
    candidate_weight=alternatives[(i,exchange["new_level"])]
    box={}
    def hook(_m,inp,_out):
        ordinary=F.linear(inp[0],base[i],module.bias); changed=F.linear(inp[0],candidate_weight,module.bias)
        mixed=torch.where(mask[0].view(1,-1,1),changed,ordinary)
        mixed=torch.where((~mask[0]).view(1,-1,1),changed,mixed)
        box["mixed_direct"]=float((mixed-changed).abs().max())
        return changed
    h=module.register_forward_hook(hook); hooked=sparse(noisy).logits; h.remove()
    saved=module.weight.detach().clone(); module.weight.copy_(candidate_weight); physical=sparse(noisy).logits; module.weight.copy_(saved)
    repeated=sparse(noisy).logits
    result={"status":"passed","physical_hook_max_abs":float((physical-hooked).abs().max()),
            "mixed_direct_max_abs":box["mixed_direct"],"restored_sham_max_abs":float((repeated-sparse(noisy).logits).abs().max())}
    if any(result[k]!=0 for k in result if k.endswith("max_abs")): raise RuntimeError(result)
    atomic_json(ROOT/"smoke.json",result); print(json.dumps(result)); del dense,sparse; torch.cuda.empty_cache()


@torch.inference_mode()
def collect(split):
    design,states_doc,a,candidate=validate(); wanted=set(design["development_documents"] if split=="development" else design["final_documents"])
    state_rows=[x for x in states_doc["states"] if int(x["sequence_index"]) in wanted]
    dense,dm,sparse,sm=load_models(a); device=next(sparse.parameters()).device
    base,alternatives=candidate_weights(dm,candidate,design)
    by_module={i:[] for i in range(224)}
    for exchange in design["exchanges"]: by_module[exchange["module_index"]].append(exchange)
    path=RUNTIME/f"{split}.pt"; payload=torch.load(path,map_location="cpu",weights_only=False) if path.exists() else {
        "split":split,"design_sha256":sha256(ROOT/"design.json"),"completed_states":0,"states":[]}
    start=int(payload["completed_states"]); started=time.monotonic()
    for local_index,state in enumerate(state_rows[start:],start=start):
        noisy,_,token_mask=state_tensors(state,device); actual=token_mask[0]
        random_groups={}
        for seed in SEEDS:
            random_mask=deterministic_random_role(actual.cpu(),seed,int(state["sequence_index"])*10+int(state["timestep_index"])).to(device)
            random_groups[f"random_{seed}_masked"]=random_mask
            random_groups[f"random_{seed}_unmasked"]=~random_mask
        groups={"masked":actual,"unmasked":~actual,**random_groups}
        dense_inputs={}; sparse_inputs={}; prefixes={}
        handles=[]
        for name,module in dm.items():
            def capture_dense(_m,inp,n=name): dense_inputs[n]=inp[0].detach()
            handles.append(module.register_forward_pre_hook(capture_dense))
        dense_logits=dense(noisy).logits
        for h in handles:h.remove()
        handles=[]
        for layer,block in enumerate(sparse.model.transformer.blocks):
            def capture_prefix(_m,inp,l=layer): prefixes[l]=inp[0].detach()
            handles.append(block.register_forward_pre_hook(capture_prefix))
        for name,module in sm.items():
            def capture_sparse(_m,inp,n=name): sparse_inputs[n]=inp[0].detach()
            handles.append(module.register_forward_pre_hook(capture_sparse))
        sparse_external=sparse(noisy).logits
        for h in handles:h.remove()
        dense_ref=dense_logits[0,actual].cpu(); rows=[]
        for i,(name,module) in enumerate(sm.items()):
            exs=by_module[i]; variants=[base[i]]+[alternatives[(i,x["new_level"])] for x in exs]
            def intervention(mod,inp,_out,ws=variants):
                reference=inp[0][:1]
                if not torch.equal(inp[0],reference.expand_as(inp[0])): raise RuntimeError("variant inputs differ")
                return torch.cat([F.linear(reference,w,mod.bias) for w in ws],0)
            h=module.register_forward_hook(intervention)
            try: logits=_suffix_logits(sparse,prefixes[i//7].expand(len(variants),-1,-1).contiguous(),i//7)
            finally:h.remove()
            baseline_kl=kl(logits[:1],dense_ref,token_mask)
            for j,exchange in enumerate(exs,1):
                original=dm[name].weight.detach(); candidate_weight=variants[j]
                features={"dense":role_feature(dense_inputs[name],original,base[i],candidate_weight,module.bias,groups),
                          "sparse":role_feature(sparse_inputs[name],original,base[i],candidate_weight,module.bias,groups)}
                candidate_kl=kl(logits[j:j+1],dense_ref,token_mask)
                rows.append({"exchange_index":exchange["exchange_index"],"module_index":i,"layer":i//7,
                             "projection_type":exchange["projection_type"],"base_level":exchange["base_level"],
                             "new_level":exchange["new_level"],"parameter_delta":exchange["parameter_delta"],
                             "baseline_kl":baseline_kl,"candidate_kl":candidate_kl,
                             "delta_kl":candidate_kl-baseline_kl,"features":features})
            del logits
        external_sparse_kl=kl(sparse_external,dense_ref,token_mask); bundle_rows=[]
        exchange_lookup={x["exchange_index"]:x for x in design["exchanges"]}
        for bundle in design["bundles"]:
            handles=[]
            for exchange_index in bundle["exchange_indices"]:
                exchange=exchange_lookup[exchange_index]; i=exchange["module_index"]; module=sm[exchange["name"]]
                weight=alternatives[(i,exchange["new_level"])]
                def replace(mod,inp,_out,w=weight): return F.linear(inp[0],w,mod.bias)
                handles.append(module.register_forward_hook(replace))
            try: candidate_logits=sparse(noisy).logits
            finally:
                for h in handles:h.remove()
            candidate_kl=kl(candidate_logits,dense_ref,token_mask)
            bundle_rows.append({"bundle_index":bundle["bundle_index"],"layer_quartile":bundle["layer_quartile"],
                                "exchange_indices":bundle["exchange_indices"],"baseline_kl":external_sparse_kl,
                                "candidate_kl":candidate_kl,"delta_kl":candidate_kl-external_sparse_kl})
            del candidate_logits
        payload["states"].append({"sequence_index":int(state["sequence_index"]),"timestep":float(state["timestep"]),
                                  "rows":rows,"bundle_rows":bundle_rows,"external_sparse_kl":external_sparse_kl})
        payload["completed_states"]=local_index+1; save(path,payload)
        event("state_complete",split=split,state=local_index+1,total=len(state_rows),elapsed=time.monotonic()-started)
        del dense_inputs,sparse_inputs,prefixes,dense_logits,sparse_external; gc.collect(); torch.cuda.empty_cache()
    if model_sha(dense)!=DENSE_SHA: raise RuntimeError("dense model mutated")
    event("collection_complete",split=split,states=len(payload["states"]),path=str(path))


def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=("smoke","collect")); p.add_argument("--split",choices=("development","final"))
    args=p.parse_args()
    if args.command=="smoke": smoke()
    else:
        if not args.split:p.error("--split required")
        collect(args.split)

if __name__=="__main__":main()
