import hashlib, json, platform
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.dlm_loss_aggregation.run import _load_model, _suffix_logits, load_config, validate_config
from experiments.wanda_failure_characterization.propagation_core import tensor_pair_metrics
from experiments.wanda_failure_characterization.run_failure_map import (
    collect, make_masks, metrics, model_sha, modules, state_tensors,
)

ROOT=Path(__file__).parent
STAGES=("branch","c1","same_block","final_pre_norm","final_post_norm","masked_logits")
METRICS=("abs_energy","relative_energy","relative_l2","sham_energy","delta_energy",
         "sham_delta_cosine","aligned_energy_fraction","orthogonal_energy_fraction",
         "masked_abs_energy","unmasked_abs_energy","masked_energy_fraction",
         "masked_position_fraction","masked_enrichment")
BLOCK_METRICS=("abs_energy","relative_energy","relative_l2","masked_abs_energy",
               "unmasked_abs_energy","masked_energy_fraction","masked_enrichment")

def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_metrics(storage,si,mi,value):
    for key in METRICS:
        if key in value:storage[key][si,mi]=value[key]

@torch.inference_mode()
def trace_one(model,prefix,block_index,target_name,target_module,m50,m75,token_mask,storage,si,mi):
    block=model.model.transformer.blocks[block_index]
    side="attention" if target_name.split('.')[-1] in {"q_proj","k_proj","v_proj","attn_out"} else "mlp"
    branch_module=block.attn_out if side=="attention" else block.ff_out
    handles=[]

    def masked_hook(mod,inp,out):
        from experiments.wanda_failure_characterization.core import masked_linear_variants
        return masked_linear_variants(inp[0],mod.weight,mod.bias,[None,m50,m75])
    handles.append(target_module.register_forward_hook(masked_hook))

    def observe(stage,tensor):
        save_metrics(storage[stage],si,mi,tensor_pair_metrics(tensor[0],tensor[1],token_mask[0]))

    def branch_hook(_,inp,out): observe("branch",out)
    handles.append(branch_module.register_forward_hook(branch_hook))

    if side=="attention":
        def post_attention(_,inp): observe("c1",inp[0])
        handles.append(block.ff_norm.register_forward_pre_hook(post_attention))

    for bi in range(block_index,32):
        def block_hook(_,inp,out,j=bi):
            value=out[0] if isinstance(out,tuple) else out
            vals=tensor_pair_metrics(value[0],value[1],token_mask[0])
            for key in BLOCK_METRICS:storage["block_end"][key][si,mi,j]=vals[key]
            if j==block_index:
                observe("same_block",value)
                if side=="mlp":observe("c1",value)
        handles.append(model.model.transformer.blocks[bi].register_forward_hook(block_hook))

    def pre_norm(_,inp): observe("final_pre_norm",inp[0])
    def post_norm(_,inp,out): observe("final_post_norm",out)
    handles.append(model.model.transformer.ln_f.register_forward_pre_hook(pre_norm))
    handles.append(model.model.transformer.ln_f.register_forward_hook(post_norm))
    try:
        logits=_suffix_logits(model,prefix.expand(3,-1,-1).contiguous(),block_index)
    finally:
        for h in handles:h.remove()
    selected=logits[:,token_mask[0]].float()
    vals=tensor_pair_metrics(selected[0],selected[1])
    save_metrics(storage["masked_logits"],si,mi,vals)
    ref_lp=selected[0].log_softmax(-1); var_lp=selected[1].log_softmax(-1)
    token_kl=(ref_lp.exp()*(ref_lp-var_lp)).sum(-1)
    token_norm=(selected[1]-selected[0]).square().mean(-1).sqrt()
    storage["extra"]["logit_delta_rms"][si,mi]=token_norm.square().mean().sqrt().cpu()
    storage["extra"]["token_logitnorm_kl_spearman"][si,mi]=float(spearmanr(token_norm.cpu().numpy(),token_kl.cpu().numpy()).statistic)
    loss,kl,_,_=metrics(logits,logits[:1],storage["clean"],token_mask,storage["p_mask"])
    storage["extra"]["loss_delta"][si,mi]=(loss[1]-loss[0]).cpu();storage["extra"]["kl"][si,mi]=kl[1].cpu()

def allocate(S,N):
    result={stage:{key:torch.full((S,N),float('nan')) for key in METRICS} for stage in STAGES}
    result['block_end']={key:torch.full((S,N,32),float('nan')) for key in BLOCK_METRICS}
    result['extra']={key:torch.empty(S,N) for key in ('logit_delta_rms','token_logitnorm_kl_spearman','loss_delta','kl')}
    return result

@torch.inference_mode()
def main():
    held=json.loads((ROOT/'heldout_state_manifest.json').read_text());old=torch.load(ROOT/'failure_map.pt',map_location='cpu',weights_only=True)['damage'];A=torch.load(ROOT/'wanda_sufficient_statistics.pt',map_location='cpu',weights_only=True)['clean_A']
    config=load_config('experiments/dlm_loss_aggregation/config.yaml');validate_config(config);model,_=_load_model(config);mapping=modules(model);before=model_sha(model)
    masks,meta=make_masks(mapping,A); expected={(e['module'],int(e['sparsity']*100)):e['mask_sha256'] for e in json.loads((ROOT/'wanda_mask_manifest.json').read_text())['entries']}
    for e in meta:
        key=(e['module'],int(e['sparsity']*100))
        if e['mask_sha256']!=expected[key]:raise RuntimeError(f'mask hash mismatch: {key}')
    # Full 40x224 reproduction gate, with the identical existing collection path.
    reproduced=collect(model,held,mapping,masks)
    gate={}
    for key in ('kl','delta_loss'):
        error=(reproduced['fields'][key][:,:,0]-old['fields'][key][:,:,0]).abs()
        gate[key+'_max_abs']=error.max().item();gate[key+'_mean_abs']=error.mean().item()
    if gate['kl_max_abs']>1e-7 or gate['delta_loss_max_abs']>1e-7:
        atomic_json(ROOT/'propagation_reproduction_failure.json',gate);raise RuntimeError(f'reproduction gate failed: {gate}')
    atomic_json(ROOT/'propagation_reproduction_gate.json',gate)
    del reproduced
    names=list(mapping);S=len(held['states']);storage=allocate(S,len(names));dev=model.model.transformer.wte.weight.device
    for si,state in enumerate(held['states']):
        noisy,clean,mask=state_tensors(state,dev);prefixes={};hs=[]
        for bi,b in enumerate(model.model.transformer.blocks):
            def save_prefix(_,inp,j=bi):prefixes[j]=inp[0].detach()
            hs.append(b.register_forward_pre_hook(save_prefix))
        model(noisy)
        for h in hs:h.remove()
        storage['clean']=clean;storage['p_mask']=state['p_mask']
        for mi,name in enumerate(names):
            bi=int(name[6:8]);trace_one(model,prefixes[bi],bi,name,mapping[name],masks[name]['50'],masks[name]['75'],mask,storage,si,mi)
        del storage['clean'],storage['p_mask']
        print(f'trace state {si+1}/{S}',flush=True)
    storage['module_names']=names;storage['sequence_index']=old['sequence_index'];storage['timestep_index']=old['timestep_index'];storage['heldout_sha']=held['historical_state_sha256'];storage['reproduction_gate']=gate
    out=ROOT/'propagation_trace.pt';torch.save(storage,out);after=model_sha(model)
    if before!=after:raise RuntimeError('model weights changed')
    atomic_json(ROOT/'propagation_trace_manifest.json',{'status':'complete','states':S,'modules':len(names),'batch_order':['sham','50','75'],'primary_sparsity':.5,'heldout_sha':held['historical_state_sha256'],'reproduction_gate':gate,'weight_sha_before':before,'weight_sha_after':after,'artifact_sha256':file_sha(out),'environment':{'python':platform.python_version(),'torch':torch.__version__}})

if __name__=='__main__':main()
