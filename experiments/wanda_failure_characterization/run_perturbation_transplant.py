import hashlib,json,platform
from pathlib import Path

import torch

from experiments.dlm_loss_aggregation.run import _load_model,_suffix_logits,load_config,validate_config
from experiments.wanda_failure_characterization.core import masked_linear_variants,rowwise_wanda_mask
from experiments.wanda_failure_characterization.propagation_core import matched_perturbation,tensor_pair_metrics
from experiments.wanda_failure_characterization.run_failure_map import metrics,model_sha,modules,state_tensors

ROOT=Path(__file__).parent
NAMES=('block_30.ff_out','block_31.ff_out')

def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def tensor_sha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

@torch.inference_mode()
def native_variant(model,prefix,location,module,m50,m75):
    captured={}
    def mask_hook(mod,inp,out):return masked_linear_variants(inp[0],mod.weight,mod.bias,[None,m50,m75])
    def block_hook(_,inp,out):captured['hidden']=(out[0] if isinstance(out,tuple) else out).detach()
    h1=module.register_forward_hook(mask_hook);h2=model.model.transformer.blocks[location].register_forward_hook(block_hook)
    try:logits=_suffix_logits(model,prefix.expand(3,-1,-1).contiguous(),location)
    finally:h1.remove();h2.remove()
    return captured['hidden'],logits

@torch.inference_mode()
def suffix_from_residual(model,rows,location,capture_hidden=False):
    captured={};handles=[]
    if capture_hidden:
        def norm_hook(_,inp,out):captured['final_hidden']=out.detach()
        handles.append(model.model.transformer.ln_f.register_forward_hook(norm_hook))
    try:logits=_suffix_logits(model,rows,location+1)
    finally:
        for h in handles:h.remove()
    return logits,captured.get('final_hidden')

def allocate(S):
    shape=(S,2,2)
    return {k:torch.empty(shape) for k in ('kl','loss_delta','top1_agreement','confidence_mae','injected_relative_l2','tau_deviation','final_hidden_abs_l2','final_hidden_relative_l2','final_hidden_relative_energy','masked_logit_relative_energy','logit_delta_rms')}

@torch.inference_mode()
def main():
    held=json.loads((ROOT/'heldout_state_manifest.json').read_text());old=torch.load(ROOT/'failure_map.pt',map_location='cpu',weights_only=True)['damage'];A=torch.load(ROOT/'wanda_sufficient_statistics.pt',map_location='cpu',weights_only=True)['clean_A']
    expected={(e['module'],int(e['sparsity']*100)):e['mask_sha256'] for e in json.loads((ROOT/'wanda_mask_manifest.json').read_text())['entries']}
    config=load_config('experiments/dlm_loss_aggregation/config.yaml');validate_config(config);model,_=_load_model(config);mapping=modules(model);before=model_sha(model);dev=model.model.transformer.wte.weight.device
    masks={}
    for name in NAMES:
        mod=mapping[name];score=mod.weight.detach().float().abs()*A[name].to(dev).sqrt()[None,:];masks[name]={}
        for sp in (50,75):
            mask=rowwise_wanda_mask(score,sp/100);digest=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
            if digest!=expected[(name,sp)]:raise RuntimeError(f'mask mismatch {name} {sp}')
            masks[name][sp]=mask
    S=len(held['states']);results=allocate(S);taus=torch.empty(S);natural_r=torch.empty(S,2);deltas=[[],[]];delta_hashes=[]
    native_gate={k:torch.empty(S,2) for k in ('logit_max_abs','kl_abs_error','loss_abs_error')};sham={k:torch.empty(S,2) for k in ('external_kl','external_loss_abs')};repeatability={}
    name_to_old={n:i for i,n in enumerate(old['module_names'])}
    for si,state in enumerate(held['states']):
        noisy,clean,token_mask=state_tensors(state,dev);prefixes={};hs=[]
        for bi in (30,31):
            def save_prefix(_,inp,j=bi):prefixes[j]=inp[0].detach()
            hs.append(model.model.transformer.blocks[bi].register_forward_pre_hook(save_prefix))
        external=model(noisy).logits
        for h in hs:h.remove()
        native=[]
        for di,(name,loc) in enumerate(zip(NAMES,(30,31))):
            hidden,logits=native_variant(model,prefixes[loc],loc,mapping[name],masks[name][50],masks[name][75]);h0=hidden[0].detach();delta=hidden[1].float()-h0.float();deltas[di].append(delta.cpu())
            injected=torch.stack((h0.float(),h0.float()+delta,h0.float())).to(h0.dtype);repro,_=suffix_from_residual(model,injected,loc)
            native_gate['logit_max_abs'][si,di]=(repro[1].float()-logits[1].float()).abs().max().cpu()
            loss,kl,_,_=metrics(repro,repro[:1],clean,token_mask,state['p_mask']);oi=name_to_old[name]
            native_gate['kl_abs_error'][si,di]=(kl[1].cpu()-old['fields']['kl'][si,oi,0]).abs();native_gate['loss_abs_error'][si,di]=((loss[1]-loss[0]).cpu()-old['fields']['delta_loss'][si,oi,0]).abs()
            native.append((h0,delta,logits))
        if max(v[si].max().item() for v in native_gate.values())>1e-6:
            atomic_json(ROOT/'transplant_native_gate_failure.json',{k:v[:si+1].tolist() for k,v in native_gate.items()});raise RuntimeError(f'native injection gate failed at state {si}')
        r30=native[0][1].norm()/native[0][0].float().norm();r31=native[1][1].norm()/native[1][0].float().norm();natural_r[si]=torch.tensor([r30,r31]);tau=torch.sqrt(r30*r31);taus[si]=tau.cpu()
        for li,loc in enumerate((30,31)):
            hloc=native[li][0];pert=[];ach=[]
            for di in range(2):
                p,a=matched_perturbation(native[di][1],hloc,tau);pert.append(p);ach.append(a)
            rows=torch.stack((hloc.float(),hloc.float()+pert[0],hloc.float()+pert[1])).to(hloc.dtype)
            logits,final_hidden=suffix_from_residual(model,rows,loc,capture_hidden=True)
            if si==0:
                again,again_hidden=suffix_from_residual(model,rows,loc,capture_hidden=True)
                repeatability[f'L{loc}_logit_max_abs']=(again.float()-logits.float()).abs().max().item();repeatability[f'L{loc}_hidden_max_abs']=(again_hidden.float()-final_hidden.float()).abs().max().item()
            rawloss,rawkl,_,_=metrics(logits,external,clean,token_mask,state['p_mask']);sham['external_kl'][si,li]=rawkl[0].cpu();sham['external_loss_abs'][si,li]=(rawloss[0].cpu()-old['dense_loss'][si]).abs()
            loss,kl,agree,mae=metrics(logits,logits[:1],clean,token_mask,state['p_mask'])
            for di,row in enumerate((1,2)):
                results['kl'][si,di,li]=kl[row].cpu();results['loss_delta'][si,di,li]=(loss[row]-loss[0]).cpu();results['top1_agreement'][si,di,li]=agree[row].cpu();results['confidence_mae'][si,di,li]=mae[row].cpu();results['injected_relative_l2'][si,di,li]=ach[di].cpu();results['tau_deviation'][si,di,li]=(ach[di]-tau).abs().cpu()
                fh=tensor_pair_metrics(final_hidden[0],final_hidden[row]);ml=tensor_pair_metrics(logits[0,token_mask[0]],logits[row,token_mask[0]])
                results['final_hidden_abs_l2'][si,di,li]=(final_hidden[row].float()-final_hidden[0].float()).norm().cpu();results['final_hidden_relative_l2'][si,di,li]=fh['relative_l2'];results['final_hidden_relative_energy'][si,di,li]=fh['relative_energy'];results['masked_logit_relative_energy'][si,di,li]=ml['relative_energy'];results['logit_delta_rms'][si,di,li]=(logits[row,token_mask[0]].float()-logits[0,token_mask[0]].float()).square().mean().sqrt().cpu()
        print(f'transplant state {si+1}/{S}',flush=True)
    delta30=torch.stack(deltas[0]);delta31=torch.stack(deltas[1]);delta_hashes=[tensor_sha(delta30),tensor_sha(delta31)];delta_path=ROOT/'transplant_deltas.pt';torch.save({'delta30':delta30,'delta31':delta31,'hashes':delta_hashes,'source_locations':[30,31],'heldout_sha':held['historical_state_sha256']},delta_path)
    payload={'results':results,'tau':taus,'natural_relative_l2':natural_r,'native_gate':native_gate,'sham':sham,'repeatability':repeatability,'sequence_index':old['sequence_index'],'timestep_index':old['timestep_index'],'natural_anchors':{name:{'kl':old['fields']['kl'][:,name_to_old[name],0],'loss_delta':old['fields']['delta_loss'][:,name_to_old[name],0]} for name in NAMES},'delta_hashes':delta_hashes}
    out=ROOT/'perturbation_transplant.pt';torch.save(payload,out);after=model_sha(model)
    if before!=after:raise RuntimeError('weights changed')
    atomic_json(ROOT/'perturbation_transplant_manifest.json',{'status':'complete','states':S,'directions':['D30','D31'],'locations':['L30','L31'],'batch_rows':['sham','D30','D31'],'bootstrap_seed':20260905,'heldout_sha':held['historical_state_sha256'],'delta_hashes':delta_hashes,'delta_artifact_sha256':hashlib.sha256(delta_path.read_bytes()).hexdigest(),'result_artifact_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'weight_sha_before':before,'weight_sha_after':after,'repeatability':repeatability,'environment':{'python':platform.python_version(),'torch':torch.__version__}})

if __name__=='__main__':main()
