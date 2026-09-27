import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from experiments.dlm_loss_aggregation.run import _load_clean_calibration, _load_model, _suffix_logits, load_config, validate_config
from experiments.wanda_failure_characterization.core import masked_linear_variants, reconstruction_metrics, rowwise_wanda_mask, threshold_geometry
from lib.prune_llada import find_layers

ROOT = Path(__file__).parent
CONFIG_PATH = "experiments/dlm_loss_aggregation/config.yaml"
HELDOUT = ROOT / "heldout_state_manifest.json"
STATS = ROOT / "wanda_sufficient_statistics.pt"
FAILURE = ROOT / "failure_map.pt"

def atomic_json(path, obj):
    tmp=Path(str(path)+".tmp"); tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n"); tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while b:=f.read(1<<20): h.update(b)
    return h.hexdigest()

def model_sha(model):
    h=hashlib.sha256()
    for n,v in model.state_dict().items():
        h.update(n.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def modules(model):
    result={}
    for layer,block in enumerate(model.model.transformer.blocks):
        for name,module in find_layers(block).items(): result[f"block_{layer:02d}.{name}"]=module
    return result

@torch.inference_mode()
def clean_wanda(model, config, mapping):
    clean,_=_load_clean_calibration(config)
    sums={n:torch.zeros(m.weight.shape[1]) for n,m in mapping.items()}
    handles=[]
    for n,m in mapping.items():
        def hook(_,inp,__ ,name=n):
            x=inp[0].reshape(-1,inp[0].shape[-1]).float(); sums[name].add_(x.square().sum(0).cpu())
        handles.append(m.register_forward_hook(hook))
    dev=model.model.transformer.wte.weight.device
    try:
        for i,ids in enumerate(clean): model(ids.to(dev)); print(f"wanda calibration {i+1}/8",flush=True)
    finally:
        for h in handles:h.remove()
    # WrappedGPT semantics: mean over batch samples (8), token energy is summed.
    return {n:v/8 for n,v in sums.items()}

def make_masks(mapping,A):
    masks={}; meta=[]
    for name,module in mapping.items():
        score=module.weight.detach().float().abs()*A[name].to(module.weight.device).sqrt().unsqueeze(0)
        masks[name]={str(int(sp*100)):rowwise_wanda_mask(score,sp) for sp in (.5,.75)}
        for sp in (.5,.75):
            m=masks[name][str(int(sp*100))]
            meta.append({"module":name,"sparsity":sp,"exact_sparsity":m.float().mean().item(),
                         "mask_sha256":hashlib.sha256(m.cpu().numpy().tobytes()).hexdigest(),
                         "score_mean":score.mean().item(),"score_std":score.std(unbiased=False).item(),
                         "threshold_geometry":threshold_geometry(score,sp)})
    return masks,meta

def state_tensors(state,dev):
    return [torch.tensor(state[k],device=dev,dtype=torch.long if k!="mask" else torch.bool) for k in ("noisy_ids","clean_ids","mask")]

def metrics(logits,ref,clean,mask,p):
    pos=mask[0]; target=clean[0,pos]; selected=logits[:,pos].float(); refsel=ref[0,pos].float()
    loss=F.cross_entropy(selected.reshape(-1,selected.shape[-1]),target.expand(selected.shape[0],-1).reshape(-1),reduction="none").reshape(selected.shape[:2]).sum(1)/p/256
    lp=F.log_softmax(selected,-1); rlp=F.log_softmax(refsel,-1); rp=rlp.exp()
    kl=(rp.unsqueeze(0)*(rlp.unsqueeze(0)-lp)).sum(-1).mean(-1)
    pred=selected.argmax(-1); agree=(pred==refsel.argmax(-1)).float().mean(-1)
    conf=lp.exp().max(-1).values; rconf=rp.max(-1).values
    mae=(conf-rconf.unsqueeze(0)).abs().mean(-1)
    return loss,kl,agree,mae

@torch.inference_mode()
def variant_logits(model,prefix,block_index,module,m50,m75,recbox):
    x=prefix.expand(3,-1,-1).contiguous()
    def hook(mod,inp,out):
        result=masked_linear_variants(inp[0],mod.weight,mod.bias,[None,m50,m75])
        recbox[:] = [reconstruction_metrics(result[0],result[1]),reconstruction_metrics(result[0],result[2])]
        return result
    h=module.register_forward_hook(hook)
    try:return _suffix_logits(model,x,block_index)
    finally:h.remove()

@torch.inference_mode()
def sanity(model,state,mapping,masks):
    dev=model.model.transformer.wte.weight.device; noisy,clean,mask=state_tensors(state,dev)
    prefix={}
    b=model.model.transformer.blocks[0]
    def save_prefix(_, inp):
        prefix["x"] = inp[0].detach()
    ph=b.register_forward_pre_hook(save_prefix)
    dense=model(noisy).logits; ph.remove()
    name=next(iter(mapping)); module=mapping[name]; box=[]
    suffix=variant_logits(model,prefix["x"],0,module,masks[name]["50"],masks[name]["75"],box)
    # Ordinary full-forward equivalence: same functional masked Linear hook, identical batch/path.
    x=noisy.expand(3,-1).contiguous()
    def full_hook(mod,inp,out): return masked_linear_variants(inp[0],mod.weight,mod.bias,[None,masks[name]["50"],masks[name]["75"]])
    hh=module.register_forward_hook(full_hook)
    try:full=model(x).logits
    finally:hh.remove()
    diff=(suffix.float()-full.float()).abs().max().item()
    # cloned ordinary Linear equality at target output
    probe=torch.randn(2,module.weight.shape[1],device=dev,dtype=module.weight.dtype)
    functional=F.linear(probe,module.weight.masked_fill(masks[name]["50"].to(dev),0),module.bias)
    clone=torch.nn.Linear(module.weight.shape[1],module.weight.shape[0],bias=module.bias is not None,device=dev,dtype=module.weight.dtype)
    clone.weight.copy_(module.weight.masked_fill(masks[name]["50"].to(dev),0));
    if module.bias is not None:clone.bias.copy_(module.bias)
    linear_diff=(functional-clone(probe)).float().abs().max().item()
    if diff>1e-6 or linear_diff>1e-6:raise RuntimeError(f"functional/prefix sanity failed {diff=} {linear_diff=}")
    return {"prefix_full_logits_max_abs":diff,"functional_vs_cloned_linear_max_abs":linear_diff}

@torch.inference_mode()
def collect(model,held,mapping,masks):
    names=list(mapping); n=len(names); S=40
    fields={k:torch.empty(S,n,2) for k in ("delta_loss","raw_delta_loss","kl","raw_kl","top1_agreement","confidence_mae","reconstruction","output_cosine","relative_output_l2")}
    sham={k:torch.empty(S,n) for k in ("delta_loss","kl")}; dl=torch.empty(S)
    heldout_A={name:torch.empty(S,module.weight.shape[1]) for name,module in mapping.items()}
    dev=model.model.transformer.wte.weight.device
    byblock={i:[name for name in names if name.startswith(f"block_{i:02d}.")] for i in range(32)}
    for si,state in enumerate(held["states"]):
        noisy,clean,mask=state_tensors(state,dev); prefixes={}; hs=[]
        for i,b in enumerate(model.model.transformer.blocks):
            def save_block_prefix(_, inp, j=i):
                prefixes[j] = inp[0].detach()
            hs.append(b.register_forward_pre_hook(save_block_prefix))
        for name,module in mapping.items():
            def save_activation(_, inp, __, module_name=name):
                x=inp[0].reshape(-1,inp[0].shape[-1]).float()
                heldout_A[module_name][si].copy_(x.square().sum(0).cpu())
            hs.append(module.register_forward_hook(save_activation))
        dense=model(noisy).logits
        for h in hs:h.remove()
        dense_loss=metrics(dense,dense,clean,mask,state["p_mask"])[0][0]; dl[si]=dense_loss.cpu()
        for bi in range(32):
            for name in byblock[bi]:
                mi=names.index(name); box=[]; logits=variant_logits(model,prefixes[bi],bi,mapping[name],masks[name]["50"],masks[name]["75"],box)
                loss,kl,agree,mae=metrics(logits,logits[:1],clean,mask,state["p_mask"])
                rawloss,rawkl,_,_=metrics(logits,dense,clean,mask,state["p_mask"])
                sham["delta_loss"][si,mi]=rawloss[0].cpu()-dl[si]; sham["kl"][si,mi]=rawkl[0].cpu()
                fields["delta_loss"][si,mi]=loss[1:].cpu()-loss[0].cpu(); fields["raw_delta_loss"][si,mi]=rawloss[1:].cpu()-dl[si]
                fields["kl"][si,mi]=kl[1:].cpu(); fields["raw_kl"][si,mi]=rawkl[1:].cpu(); fields["top1_agreement"][si,mi]=agree[1:].cpu(); fields["confidence_mae"][si,mi]=mae[1:].cpu()
                for r in range(2):
                    fields["reconstruction"][si,mi,r]=box[r]["relative_squared_error"]; fields["output_cosine"][si,mi,r]=box[r]["cosine"]; fields["relative_output_l2"][si,mi,r]=box[r]["relative_l2_error"]
        print(f"failure state {si+1}/40",flush=True)
    return {"module_names":names,"fields":fields,"sham":sham,"dense_loss":dl,
            "heldout_A_state":heldout_A,
            "timestep_index":torch.tensor([s["timestep_index"] for s in held["states"]]),"sequence_index":torch.tensor([s["sequence_index"] for s in held["states"]])}

def main(reuse=False):
    config=load_config(CONFIG_PATH);validate_config(config);held=json.loads(HELDOUT.read_text()); model,devices=_load_model(config); mapping=modules(model); before=model_sha(model)
    if len(mapping)!=224:raise RuntimeError("expected 224 modules")
    if reuse: payload=torch.load(STATS,map_location="cpu",weights_only=True);A=payload["clean_A"]
    else:
        A=clean_wanda(model,config,mapping); payload={"clean_A":A,"heldout_sha":held["historical_state_sha256"]};torch.save(payload,STATS)
    masks,maskmeta=make_masks(mapping,A); atomic_json(ROOT/"wanda_mask_manifest.json",{"wanda_formula":"abs(W)*sqrt(scaler_row)","scaler_row":"mean over 8 batch samples of sum_token X^2","entries":maskmeta})
    check=sanity(model,held["states"][0],mapping,masks);atomic_json(ROOT/"execution_sanity.json",check)
    damage=collect(model,held,mapping,masks);torch.save({"heldout_sha":held["historical_state_sha256"],"damage":damage},FAILURE)
    after=model_sha(model)
    if before!=after:raise RuntimeError("weights changed")
    atomic_json(ROOT/"failure_map_manifest.json",{"status":"complete","primary_targets":["Y50_KL","Y50_pos"],"heldout_sha":held["historical_state_sha256"],"module_count":224,"state_count":40,"sparsities":[.5,.75],"sanity":check,"weight_sha_before":before,"weight_sha_after":after,"failure_map_sha256":sha(FAILURE),"environment":{"python":platform.python_version(),"torch":torch.__version__,"transformers":transformers.__version__}})
    print(json.dumps({"status":"failure-map-complete","path":str(FAILURE)},indent=2))

if __name__=="__main__": main()
