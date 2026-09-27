import argparse
import contextlib
import fcntl
import json
import time
import torch
import torch.nn.functional as F
from experiments.dlm_role_exchange_prediction_v2.common import ROOT,OLD,STORE,OLD_STORE,atomic_json,sha256,validate_sources
from experiments.dlm_role_exchange_prediction.collect import load_models,candidate_weights,kl,save,role_feature
from experiments.dlm_role_validation.core import deterministic_random_role
from experiments.wanda_failure_characterization.run_failure_map import state_tensors,model_sha

def event(kind,**kw):
    print(json.dumps({'event':kind,'time':time.time(),**kw}),flush=True)

@contextlib.contextmanager
def physical(mapping,exchanges,base,alternatives):
    try:
        for e in exchanges:
            mapping[e['name']].weight.copy_(alternatives[e['module_index'],e['new_level']])
        yield
    finally:
        for e in exchanges: mapping[e['name']].weight.copy_(base[e['module_index']])

@torch.inference_mode()
def smoke(model,mapping,dense_mapping,base,alt,design,state,old_rows):
    dev=next(model.parameters()).device; noisy,_,mask=state_tensors(state,dev)
    baseline=model(noisy).logits
    if not torch.equal(baseline,model(noisy).logits): raise RuntimeError('baseline sham mismatch')
    # Seven types spread across depth, both directions where available; same helper as production.
    chosen=[]
    for j,typ in enumerate(sorted({e['projection_type'] for e in design['exchanges']})):
        options=[e for e in design['exchanges'] if e['projection_type']==typ]
        distance=min(abs(e['layer']-j*5) for e in options)
        chosen += [e for e in options if abs(e['layer']-j*5)==distance]
    audits=[]
    for e in chosen:
        m=mapping[e['name']]; w=alt[e['module_index'],e['new_level']]; captured={}
        def hook(mod,inp,out):
            captured['x']=inp[0].detach()
            return F.linear(inp[0],w,mod.bias)
        h=m.register_forward_hook(hook)
        try: hooked=model(noisy).logits
        finally: h.remove()
        with physical(mapping,[e],base,alt): direct=model(noisy).logits
        if not torch.equal(hooked,direct): raise RuntimeError('physical/hook mismatch')
        def mixed(mod,inp,out):
            changed=F.linear(inp[0],w,mod.bias)
            composed=out.clone()
            composed[:,mask[0]]=changed[:,mask[0]]
            composed[:,~mask[0]]=changed[:,~mask[0]]
            return composed
        h=m.register_forward_hook(mixed)
        try: composed=model(noisy).logits
        finally: h.remove()
        if not torch.equal(composed,direct): raise RuntimeError('role composition mismatch')
        if not torch.equal(baseline,model(noisy).logits): raise RuntimeError('restore mismatch')
        actual=mask[0]; groups={'masked':actual,'unmasked':~actual}
        for seed in (101,202,303):
            r=deterministic_random_role(actual.cpu(),seed,int(state['sequence_index'])*10+int(state['timestep_index'])).to(dev)
            groups[f'random_{seed}_masked']=r; groups[f'random_{seed}_unmasked']=~r
        f=role_feature(captured['x'],dense_mapping[e['name']].weight,base[e['module_index']],w,m.bias,groups)
        old=old_rows[e['exchange_index']]['features']['sparse']
        if any(abs(f[k]-old[k])>1e-6*max(1.,abs(old[k])) for k in f):
            raise RuntimeError('reused local features failed independent check')
        audits.append({'exchange_index':e['exchange_index'],'physical_hook_max_abs':0.,'mixed_direct_max_abs':0.,'restore_max_abs':0.,'feature_check':'passed'})
    # A multi-projection physical application is compared to independently composed hooks.
    bundle=design['bundles'][0]; es=[design['exchanges'][i] for i in bundle['exchange_indices']]
    handles=[]
    for e in es:
        w=alt[e['module_index'],e['new_level']]
        def hfun(m,inp,out,w=w):return F.linear(inp[0],w,m.bias)
        handles.append(mapping[e['name']].register_forward_hook(hfun))
    try: hooked=model(noisy).logits
    finally:
        for h in handles:h.remove()
    with physical(mapping,es,base,alt):direct=model(noisy).logits
    if not torch.equal(hooked,direct) or not torch.equal(baseline,model(noisy).logits):raise RuntimeError('bundle smoke failed')
    return {'status':'passed','singles':audits,'bundle_physical_hook_max_abs':0.}

@torch.inference_mode()
def run(split,smoke_only=False,limit=None):
    torch.set_num_threads(4)
    cfg=validate_sources()
    design=json.loads((OLD/'design.json').read_text()); states=json.loads((OLD/'state_manifest.json').read_text())
    a=json.loads(open('experiments/dlm_dual_role_mini100/role65_mask_manifest.json').read())
    candidate=json.loads(open('experiments/projection_capacity_allocation_65/candidate_mask_manifest.json').read())
    source=torch.load(OLD_STORE/f'{split}.pt',map_location='cpu',weights_only=False)
    if source['completed_states']!=80:raise RuntimeError('v1 input incomplete')
    previous={(s['sequence_index'],s['timestep']):s for s in source['states']}
    selected=[s for s in states['states'] if s['sequence_index'] in cfg[f'{split}_documents']]
    total=len(selected)
    dense,dm,sparse,sm=load_models(a); device=next(sparse.parameters()).device
    base,alt=candidate_weights(dm,candidate,design)
    before=model_sha(sparse)
    first=selected[0]; old=previous[first['sequence_index'],first['timestep']]
    audit=smoke(sparse,sm,dm,base,alt,design,first,{r['exchange_index']:r for r in old['rows']})
    audit['config_sha256']=sha256(ROOT/'config.json'); atomic_json(ROOT/f'smoke_{split}.json',audit)
    event('smoke_passed',split=split,checks=len(audit['singles']))
    if smoke_only:return
    progress=ROOT/f'progress_{split}.json'; started=time.time(); processed=0
    for si,state in enumerate(selected):
        if limit is not None and si>=limit:break
        key=(state['sequence_index'],state['timestep']); prev=previous[key]
        out=STORE/split/f'state_{si:03d}.pt'
        if out.exists():
            p=torch.load(out,map_location='cpu',weights_only=False)
            if p['config_sha256']!=sha256(ROOT/'config.json'):raise RuntimeError('resume identity mismatch')
            continue
        noisy,_,mask=state_tensors(state,device)
        ref=dense(noisy).logits[0,mask[0]].cpu()
        baseline=sparse(noisy).logits
        if not torch.equal(baseline,sparse(noisy).logits):raise RuntimeError('state sham mismatch')
        bk=kl(baseline,ref,mask); rows=[]; bundles=[]
        lookup={r['exchange_index']:r for r in prev['rows']}
        for e in design['exchanges']:
            with physical(sm,[e],base,alt):logits=sparse(noisy).logits
            ck=kl(logits,ref,mask)
            old=lookup[e['exchange_index']]
            rows.append({**e,'features':old['features'],'baseline_kl':bk,'candidate_kl':ck,'delta_kl':ck-bk,
                         'v1_delta_kl':old['delta_kl'],'v1_baseline_kl':old['baseline_kl']})
        for b in design['bundles']:
            es=[design['exchanges'][i] for i in b['exchange_indices']]
            if sum(e['parameter_delta'] for e in es)!=0:raise RuntimeError('bundle budget mismatch')
            with physical(sm,es,base,alt):logits=sparse(noisy).logits
            ck=kl(logits,ref,mask)
            bundles.append({**b,'baseline_kl':bk,'candidate_kl':ck,'delta_kl':ck-bk})
        if not torch.equal(baseline,sparse(noisy).logits):raise RuntimeError('state restoration mismatch')
        save(out,{'status':'complete','config_sha256':sha256(ROOT/'config.json'),'sequence_index':key[0],
                  'timestep':key[1],'rows':rows,'bundle_rows':bundles,'external_sparse_kl':bk,
                  'sham_max_abs':0.,'restore_max_abs':0.})
        processed+=1; elapsed=time.time()-started
        atomic_json(progress,{'status':'collecting','completed':si+1,'total':total,'elapsed_seconds':elapsed,
                             'eta_seconds':(total-si-1)*elapsed/processed,'updated':time.time()})
        event('state_complete',split=split,completed=si+1,total=total,elapsed_seconds=elapsed)
    if limit is None:
        if before!=model_sha(sparse):raise RuntimeError('sparse model checksum changed')
        atomic_json(ROOT/f'complete_{split}.json',{'status':'complete','config_sha256':sha256(ROOT/'config.json'),'states':total,'model_sha':before})
        atomic_json(progress,{'status':'complete','completed':total,'total':total,'eta_seconds':0,'updated':time.time()})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['development','final'],required=True)
    parser.add_argument('--smoke-only',action='store_true');parser.add_argument('--limit',type=int)
    args=parser.parse_args(); STORE.mkdir(parents=True,exist_ok=True)
    with (STORE/f'{args.split}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.split,args.smoke_only,args.limit)
