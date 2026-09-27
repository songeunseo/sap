import hashlib,json,platform
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_loss_aggregation.run import _load_model,_suffix_logits,load_config,validate_config
from experiments.wanda_failure_characterization.core import masked_linear_variants
from experiments.wanda_failure_characterization.propagation_core import matched_perturbation,tensor_pair_metrics
from experiments.wanda_failure_characterization.run_failure_map import metrics,model_sha,state_tensors

ROOT=Path(__file__).parent

def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@torch.inference_mode()
def l31_sham_hidden(model,prefix):
    block=model.model.transformer.blocks[31];captured={}
    def rowwise_dense(mod,inp,out):return masked_linear_variants(inp[0],mod.weight,mod.bias,[None,None,None])
    def capture(_,inp,out):captured['hidden']=(out[0] if isinstance(out,tuple) else out).detach()
    h1=block.ff_out.register_forward_hook(rowwise_dense);h2=block.register_forward_hook(capture)
    try:logits=_suffix_logits(model,prefix.expand(3,-1,-1).contiguous(),31)
    finally:h1.remove();h2.remove()
    return captured['hidden'][0],logits

@torch.inference_mode()
def final_suffix(model,rows):
    captured={}
    def hook(_,inp,out):captured['hidden']=out.detach()
    h=model.model.transformer.ln_f.register_forward_hook(hook)
    try:logits=_suffix_logits(model,rows,32)
    finally:h.remove()
    return logits,captured['hidden']

def evaluate_rows(model,h31,directions,tau,clean,mask,p_mask):
    perturbations=[];achieved=[]
    for direction in directions:
        perturbation,norm=matched_perturbation(direction,h31,tau);perturbations.append(perturbation);achieved.append(norm)
    rows=torch.stack((h31.float(),h31.float()+perturbations[0],h31.float()+perturbations[1])).to(h31.dtype)
    logits,hidden=final_suffix(model,rows);loss,kl,agree,mae=metrics(logits,logits[:1],clean,mask,p_mask)
    result={k:torch.empty(2) for k in ('kl','loss_delta','top1_agreement','confidence_mae','achieved_relative_l2','norm_deviation','final_hidden_relative_energy','masked_logit_relative_energy','logit_rms')}
    for ci,row in enumerate((1,2)):
        fh=tensor_pair_metrics(hidden[0],hidden[row]);lg=tensor_pair_metrics(logits[0,mask[0]],logits[row,mask[0]])
        result['kl'][ci]=kl[row].cpu();result['loss_delta'][ci]=(loss[row]-loss[0]).cpu();result['top1_agreement'][ci]=agree[row].cpu();result['confidence_mae'][ci]=mae[row].cpu();result['achieved_relative_l2'][ci]=achieved[ci].cpu();result['norm_deviation'][ci]=(achieved[ci]-tau).abs().cpu();result['final_hidden_relative_energy'][ci]=fh['relative_energy'];result['masked_logit_relative_energy'][ci]=lg['relative_energy'];result['logit_rms'][ci]=(logits[row,mask[0]].float()-logits[0,mask[0]].float()).square().mean().sqrt().cpu()
    return result,logits,hidden

@torch.inference_mode()
def main():
    mapping=json.loads((ROOT/'state_swap_mapping.json').read_text());held=json.loads((ROOT/'heldout_state_manifest.json').read_text());delta_art=torch.load(ROOT/'transplant_deltas.pt',map_location='cpu',weights_only=True);prior=torch.load(ROOT/'perturbation_transplant.pt',map_location='cpu',weights_only=True);old=torch.load(ROOT/'failure_map.pt',map_location='cpu',weights_only=True)['damage'];manifest=json.loads((ROOT/'perturbation_transplant_manifest.json').read_text())
    if delta_art['hashes']!=manifest['delta_hashes'] or file_sha(ROOT/'transplant_deltas.pt')!=manifest['delta_artifact_sha256']:raise RuntimeError('stored delta artifact/hash mismatch')
    delta=delta_art['delta31'];tau=prior['tau'];config=load_config('experiments/dlm_loss_aggregation/config.yaml');validate_config(config);model,_=_load_model(config);before=model_sha(model);dev=model.model.transformer.wte.weight.device;S=40
    hiddens=[];native_gate={k:torch.empty(S) for k in ('kl_abs','loss_abs','final_energy_abs','logit_energy_abs','logit_rms_abs')};sham={k:torch.empty(S) for k in ('external_kl','external_loss_abs')};repeat={}
    # Gate phase: no foreign direction is evaluated.
    for si,state in enumerate(held['states']):
        noisy,clean,mask=state_tensors(state,dev);prefix={}
        def capture(_,inp):prefix['x']=inp[0].detach()
        ph=model.model.transformer.blocks[31].register_forward_pre_hook(capture);external=model(noisy).logits;ph.remove();h31,sham_logits=l31_sham_hidden(model,prefix['x']);hiddens.append(h31.cpu())
        direction=delta[si].to(dev);res,logits,hidden=evaluate_rows(model,h31,[direction,direction],tau[si].to(dev),clean,mask,state['p_mask'])
        native_gate['kl_abs'][si]=(res['kl'][0]-prior['results']['kl'][si,1,1]).abs();native_gate['loss_abs'][si]=(res['loss_delta'][0]-prior['results']['loss_delta'][si,1,1]).abs();native_gate['final_energy_abs'][si]=(res['final_hidden_relative_energy'][0]-prior['results']['final_hidden_relative_energy'][si,1,1]).abs();native_gate['logit_energy_abs'][si]=(res['masked_logit_relative_energy'][0]-prior['results']['masked_logit_relative_energy'][si,1,1]).abs();native_gate['logit_rms_abs'][si]=(res['logit_rms'][0]-prior['results']['logit_delta_rms'][si,1,1]).abs()
        rawloss,rawkl,_,_=metrics(sham_logits,external,clean,mask,state['p_mask']);sham['external_kl'][si]=rawkl[0].cpu();sham['external_loss_abs'][si]=(rawloss[0].cpu()-old['dense_loss'][si]).abs()
        if max(v[si].item() for v in native_gate.values())>1e-6:raise RuntimeError(f'native gate failed state {si}: '+str({k:v[si].item() for k,v in native_gate.items()}))
    atomic_json(ROOT/'state_swap_native_gate.json',{k:{'max':v.max().item(),'mean':v.mean().item()} for k,v in native_gate.items()})
    hiddens=torch.stack(hiddens);keys=('kl','loss_delta','top1_agreement','confidence_mae','achieved_relative_l2','norm_deviation','final_hidden_relative_energy','masked_logit_relative_energy','logit_rms');results={k:torch.empty(S,3) for k in keys};cosines=torch.empty(S,2)
    # Primary cyclic phase, frozen and persisted before the secondary phase.
    for si,state in enumerate(held['states']):
        _,clean,mask=state_tensors(state,dev);native=delta[si].to(dev);donor=mapping['cyclic_donors'][si];foreign=delta[donor].to(dev);res,logits,hidden=evaluate_rows(model,hiddens[si].to(dev),[native,foreign],tau[si].to(dev),clean,mask,state['p_mask'])
        for k in keys:results[k][si,0:2]=res[k]
        cosines[si,0]=F.cosine_similarity(native.flatten().float(),foreign.flatten().float(),dim=0).cpu()
        if si==0:
            again,again_logits,again_hidden=evaluate_rows(model,hiddens[si].to(dev),[native,foreign],tau[si].to(dev),clean,mask,state['p_mask']);repeat['cyclic_logits_max_abs']=(again_logits.float()-logits.float()).abs().max().item();repeat['cyclic_hidden_max_abs']=(again_hidden.float()-hidden.float()).abs().max().item()
    torch.save({'mapping_sha':mapping['mapping_sha256'],'native_gate':native_gate,'results_primary':{k:v[:,:2] for k,v in results.items()}},ROOT/'state_swap_primary_frozen.pt')
    # Secondary same-timestep phase.
    for si,state in enumerate(held['states']):
        _,clean,mask=state_tensors(state,dev);native=delta[si].to(dev);donor=mapping['same_timestep_donors'][si];foreign=delta[donor].to(dev);res,logits,hidden=evaluate_rows(model,hiddens[si].to(dev),[native,foreign],tau[si].to(dev),clean,mask,state['p_mask'])
        for k in keys:results[k][si,2]=res[k][1]
        cosines[si,1]=F.cosine_similarity(native.flatten().float(),foreign.flatten().float(),dim=0).cpu()
        if si==0:
            again,again_logits,again_hidden=evaluate_rows(model,hiddens[si].to(dev),[native,foreign],tau[si].to(dev),clean,mask,state['p_mask']);repeat['same_timestep_logits_max_abs']=(again_logits.float()-logits.float()).abs().max().item();repeat['same_timestep_hidden_max_abs']=(again_hidden.float()-hidden.float()).abs().max().item()
    out=ROOT/'state_swap_results.pt';torch.save({'results':results,'tau':tau,'cosines':cosines,'native_gate':native_gate,'sham':sham,'repeatability':repeat,'mapping_sha':mapping['mapping_sha256'],'sequence_index':prior['sequence_index'],'timestep_index':prior['timestep_index'],'delta31_hash':delta_art['hashes'][1]},out);after=model_sha(model)
    if before!=after:raise RuntimeError('weights changed')
    atomic_json(ROOT/'state_swap_manifest.json',{'status':'complete','states':S,'conditions':['native','cyclic_foreign','same_timestep_foreign'],'mapping_sha':mapping['mapping_sha256'],'delta31_hash':delta_art['hashes'][1],'result_sha256':file_sha(out),'primary_frozen_sha256':file_sha(ROOT/'state_swap_primary_frozen.pt'),'weight_sha_before':before,'weight_sha_after':after,'repeatability':repeat,'environment':{'python':platform.python_version(),'torch':torch.__version__}})

if __name__=='__main__':main()
