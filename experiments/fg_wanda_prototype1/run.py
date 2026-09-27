import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, pearsonr

from experiments.dlm_loss_aggregation.run import _load_model, _suffix_logits, load_config, validate_config, historical_state_digest
from experiments.fg_wanda_prototype1.freeze import ROOT, STORE, sha, write_json
from experiments.fg_wanda_prototype1.core import kappa_diagonal
from experiments.wanda_failure_characterization.geometry_core import score_directions
from experiments.wanda_failure_characterization.run_failure_map import model_sha, state_tensors, metrics
from experiments.wanda_failure_characterization.core import rowwise_wanda_mask, masked_linear_variants, reconstruction_metrics

SOURCE=Path('experiments/wanda_failure_characterization')


def capture_calibration(model,state,device):
    captured={}
    def ff_input(_,args):captured['x']=args[0][0].detach()
    def residual(_,args,output):captured['h']=output[0][0].detach()
    block=model.model.transformer.blocks[31]
    a=block.ff_out.register_forward_pre_hook(ff_input);b=block.register_forward_hook(residual)
    noisy,_,mask=state_tensors(state,device)
    try:model(noisy)
    finally:a.remove();b.remove()
    return captured['x'][mask[0]],captured['h'][mask[0]]


def validate_kappa(model,states,config):
    norm=model.model.transformer.ln_f; head=model.model.transformer.ff_out
    rows=[];device=head.weight.device
    for si in config['kappa_validation_states']:
        _,h=capture_calibration(model,states[si],device)
        h=h[:2]
        formula,audit=kappa_diagonal(h,norm.weight,norm.eps,head.weight)
        for ti in range(len(h)):
            indices=config['kappa_validation_output_indices']
            d=torch.zeros(len(indices),1,4096,device=device)
            d[torch.arange(len(indices),device=device),0,torch.tensor(indices,device=device)]=1
            direct=2*score_directions(h[ti:ti+1],d,norm.weight,norm.eps,head.weight)[:,2]
            calculated=formula[ti,indices]
            for index,f,q in zip(indices,calculated.tolist(),direct.tolist()):
                error=abs(f-q);tol=config['kappa_atol']+config['kappa_rtol']*abs(q)
                rows.append({'state':si,'masked_token_index':ti,'feature':index,'formula':f,'direct':q,
                             'absolute_error':error,'relative_error':error/max(abs(q),1e-20),'allowed_error':tol,'pass':error<=tol})
    doc={'status':'pass' if all(r['pass'] for r in rows) else 'failed','checks':rows,
         'max_absolute_error':max(r['absolute_error'] for r in rows),'max_relative_error':max(r['relative_error'] for r in rows),
         'pearson':float(pearsonr([r['formula'] for r in rows],[r['direct'] for r in rows]).statistic)}
    f=np.array([r['formula'] for r in rows]);q=np.array([r['direct'] for r in rows])
    doc['cosine']=float(f@q/(np.linalg.norm(f)*np.linalg.norm(q)))
    write_json(ROOT/'kappa_validation.json',doc)
    if doc['status']!='pass':raise RuntimeError('kappa numerical gate failed; no score construction')


def collect_scores(model,calibration,config):
    before=model_sha(model)
    expected=json.loads((SOURCE/'failure_map_manifest.json').read_text())['weight_sha_after']
    if before!=expected:raise RuntimeError('pinned model hash mismatch')
    norm=model.model.transformer.ln_f;head=model.model.transformer.ff_out
    if type(norm).__name__!='RMSLayerNorm' or norm.bias is not None or head.bias is not None or model.model.config.scale_logits:
        raise RuntimeError('unexpected readout layout')
    validate_kappa(model,calibration['states'],config)
    print('kappa numerical validation PASS',flush=True)
    device=head.weight.device
    g_sum=torch.zeros(4096,12288,dtype=torch.float64,device=device)
    activation_sum=torch.zeros(12288,dtype=torch.float64,device=device)
    total=0;audits=[];sufficient_hashes={}
    folder=STORE/'calibration';folder.mkdir(exist_ok=True)
    for si,state in enumerate(calibration['states']):
        x,h=capture_calibration(model,state,device)
        k,audit=kappa_diagonal(h,norm.weight,norm.eps,head.weight)
        x2=x.float().square()
        g_sum.add_(k.T@x2.double());activation_sum.add_(x2.double().sum(0));total+=len(x)
        path=folder/f'state_{si:02d}.pt'
        torch.save({'state_index':si,'x_masked':x.cpu(),'kappa':k.cpu(),
                    'masked_positions':torch.tensor(state['mask'][0]).nonzero().flatten()},path)
        sufficient_hashes[str(path)]=sha(path);audits.append(audit)
        print(f'FG calibration {si+1}/80; masked observations {total}',flush=True)
    g=(g_sum/total).float();a=(activation_sum/total).float()
    w=model.model.transformer.blocks[31].ff_out.weight.float().abs()
    clean=torch.load(SOURCE/'wanda_sufficient_statistics.pt',map_location='cpu',weights_only=True)['clean_A']['block_31.ff_out'].to(device)
    scores={'Wanda':w*clean.sqrt()[None,:],'DLMW':w*a.sqrt()[None,:],'FG':w*g.sqrt()}
    torch.save({'G':g.cpu(),'A_DLM':a.cpu(),'A_clean_Wanda':clean.cpu(),'masked_observations':total},STORE/'sufficient_aggregate.pt')
    for name,score in scores.items():torch.save(score.cpu(),STORE/f'score_{name}.pt')
    after=model_sha(model)
    if before!=after:raise RuntimeError('collection changed model weights')
    write_json(ROOT/'collection_manifest.json',{'status':'complete','target':'model.transformer.blocks.31.ff_out',
          'model_sha_before':before,'model_sha_after':after,'masked_observations':total,'state_count':80,
          'sufficient_statistics':sufficient_hashes,'files':{str(p):sha(p) for p in [STORE/'sufficient_aggregate.pt']+[STORE/f'score_{n}.pt' for n in scores]},
          'nonnegativity':{'minimum':min(x['minimum'] for x in audits),'negative_count':sum(x['negative_count'] for x in audits),
                           'material_negative_count':sum(x['material_negative_count'] for x in audits),'per_state':audits}})
    return scores


def gate1(scores,config):
    masks={name:rowwise_wanda_mask(score,.5) for name,score in scores.items()}
    hashes={}
    for name,m in masks.items():
        if not m.sum(1).eq(6144).all():raise RuntimeError('row sparsity violation')
        torch.save(m.cpu(),STORE/f'mask_{name}.pt')
        hashes[name]=__import__('hashlib').sha256(m.cpu().numpy().tobytes()).hexdigest()
    expected=[x for x in json.loads((SOURCE/'wanda_mask_manifest.json').read_text())['entries']
              if x['module']=='block_31.ff_out' and x['sparsity']==.5][0]['mask_sha256']
    if hashes['Wanda']!=expected:raise RuntimeError('standard Wanda mask not bit-exact')
    print('Standard Wanda mask hash matches frozen baseline; computing exact full score ranks.',flush=True)
    cpu={name:score.cpu().numpy() for name,score in scores.items()}
    full={name:rankdata(score.ravel()) for name,score in cpu.items()}
    rowranks={name:rankdata(score,axis=1) for name,score in cpu.items()}
    pairs=[];row_stats={}
    for left,right in [('FG','Wanda'),('FG','DLMW'),('DLMW','Wanda')]:
        xor=(masks[left]!=masks[right]).float().mean(1).cpu().numpy()
        l=rowranks[left];r=rowranks[right]
        l=l-l.mean(1,keepdims=True);r=r-r.mean(1,keepdims=True)
        rho=(l*r).sum(1)/np.sqrt((l*l).sum(1)*(r*r).sum(1))
        x=float(xor.mean());intersection=(masks[left]&masks[right]).sum().item();union=(masks[left]|masks[right]).sum().item()
        keep_intersection=(~masks[left]&~masks[right]).sum().item();keep_union=(~masks[left]|~masks[right]).sum().item()
        key=left+'_vs_'+right;row_stats[key]={'spearman':rho,'xor':xor}
        pairs.append({'pair':key,'full_score_spearman':float(pearsonr(full[left],full[right]).statistic),
                      'row_spearman_mean':float(rho.mean()),'row_spearman_median':float(np.median(rho)),
                      'global_mask_xor':x,'median_row_xor':float(np.median(xor)),'p10_row_xor':float(np.quantile(xor,.1)),
                      'p90_row_xor':float(np.quantile(xor,.9)),'pruned_jaccard':intersection/union,'kept_jaccard':keep_intersection/keep_union})
    primary=pairs[0]
    passed=primary['global_mask_xor']>=.02 and primary['median_row_xor']>=.01
    torch.save(row_stats,STORE/'gate1_row_statistics.pt')
    write_json(ROOT/'gate1.json',{'status':'pass' if passed else 'failed','comparisons':pairs,'mask_raw_hashes':hashes,
                                'exact_pruned_per_row':6144,'rows':4096,'only_target_module_modified':True})
    print('Gate 1:',passed,primary,flush=True)
    return passed,masks


def evaluate_target(model,prefix,masks):
    module=model.model.transformer.blocks[31].ff_out;rec=[]
    def masked(mod,inp,out):
        y=masked_linear_variants(inp[0],mod.weight,mod.bias,masks)
        rec.extend(reconstruction_metrics(y[0],y[i]) for i in range(len(masks)))
        return y
    handle=module.register_forward_hook(masked)
    try:logits=_suffix_logits(model,prefix.expand(len(masks),-1,-1).contiguous(),31)
    finally:handle.remove()
    return logits,rec


def evaluate_target_full(model,noisy,masks):
    """Evaluate every mask in one ordinary full-forward batch.

    This is the Gate-2 reference path.  In particular, it does not combine a
    batch-one prefix with a larger suffix batch, which is numerically distinct
    in BF16 for this model.
    """
    module=model.model.transformer.blocks[31].ff_out;rec=[]
    def masked(mod,inp,out):
        y=masked_linear_variants(inp[0],mod.weight,mod.bias,masks)
        rec.extend(reconstruction_metrics(y[0],y[i]) for i in range(len(masks)))
        return y
    handle=module.register_forward_hook(masked)
    batch_shape=(len(masks),)+(-1,)*(noisy.ndim-1)
    try:logits=model(noisy.expand(*batch_shape).contiguous()).logits
    finally:handle.remove()
    return logits,rec


def gate2(model,evaluation,masks):
    before=model_sha(model);dev=model.model.transformer.wte.weight.device
    results=[];controls=[]
    selected=[None,masks['Wanda'],masks['DLMW'],masks['FG']]
    for si,state in enumerate(evaluation['states']):
        noisy,clean,mask=state_tensors(state,dev)
        dense=model(noisy).logits
        logits,rec=evaluate_target_full(model,noisy,selected)
        if si==0:
            again,_=evaluate_target_full(model,noisy,selected)
            noop,_=evaluate_target_full(model,noisy,[None]*4)
            numerical={'execution_path':'ordinary full forward, fixed [sham,Wanda,DLMW,FG] batch',
                       'repeat_max_logits':(again-logits).float().abs().max().item(),
                       'same_shape_noop_sham_max_logits':(noop[0]-logits[0]).float().abs().max().item(),
                       'same_shape_all_noop_row_spread':(noop-noop[:1]).float().abs().max().item()}
            write_json(ROOT/'gate2_numerical_validation.json',numerical)
            if max(v for v in numerical.values() if isinstance(v,float))>1e-6:
                raise RuntimeError('Gate2 numerical-path failure; stop')
        loss,kl,agree,mae=metrics(logits,logits[:1],clean,mask,state['p_mask'])
        _,external_kl,_,_=metrics(logits[:1],dense,clean,mask,state['p_mask'])
        controls.append({'state':si,'sham_vs_batch1_KL':external_kl.item()})
        for ci,name in enumerate(['sham','Wanda','DLMW','FG']):
            results.append({'state':si,'sequence':state['sequence_index'],'timestep':state['timestep'],'condition':name,
                            'kl':kl[ci].item(),'loss_delta':(loss[ci]-loss[0]).item(),'top1_agreement':agree[ci].item(),
                            'confidence_mae':mae[ci].item(),'local_reconstruction':rec[ci]['relative_squared_error']})
        print(f'Gate2 independent state {si+1}/40',flush=True)
    after=model_sha(model)
    if before!=after:raise RuntimeError('Gate2 weights changed')
    write_json(ROOT/'gate2_per_state.json',results);write_json(ROOT/'gate2_sham_controls.json',controls)
    import pandas as pd
    df=pd.DataFrame(results);df.to_csv(ROOT/'gate2_per_state.csv',index=False)
    pivot=df.pivot(index=['state','sequence','timestep'],columns='condition',values='kl').reset_index()
    pivot['improvement_W']=pivot.Wanda-pivot.FG;pivot['improvement_DLMW']=pivot.DLMW-pivot.FG
    seq=pivot.groupby('sequence').mean(numeric_only=True);ts=pivot.groupby('timestep').mean(numeric_only=True)
    rng=np.random.default_rng(20260905);indices=rng.integers(0,8,(10000,8))
    ci={key:np.quantile(seq[key].to_numpy()[indices].mean(1),[.025,.975]).tolist() for key in ['improvement_W','improvement_DLMW']}
    checks={'lower_mean_KL_than_Wanda':bool(pivot.FG.mean()<pivot.Wanda.mean()),'Wanda_improvement_CI_lower_positive':ci['improvement_W'][0]>0,
            'lower_mean_KL_than_DLMW':bool(pivot.FG.mean()<pivot.DLMW.mean()),
            'positive_timestep_means_at_least4':int((ts.improvement_W>0).sum())>=4,
            'positive_sequence_means_at_least6':int((seq.improvement_W>0).sum())>=6}
    pivot.to_csv(ROOT/'gate2_paired.csv',index=False);seq.to_csv(ROOT/'gate2_sequences.csv');ts.to_csv(ROOT/'gate2_timesteps.csv')
    document={'status':'pass' if all(checks.values()) else 'failed','criteria':checks,
              'mean_kl':{n:pivot[n].mean() for n in ['Wanda','DLMW','FG']},'bootstrap_CI95':ci,
              'positive_timestep_means':int((ts.improvement_W>0).sum()),'positive_sequence_means':int((seq.improvement_W>0).sum()),
              'model_sha_before':before,'model_sha_after':after,'failure_map_scope':'only block31.ff_out on independent states; no aggregate-map claim'}
    write_json(ROOT/'gate2.json',document);print('Gate 2:',document,flush=True)
    return all(checks.values())


@torch.no_grad()
def main(gate2_only=False):
    started=time.time();torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    fixed=json.loads((ROOT/'preregistered.json').read_text())
    for name,key in [('geometry_calibration_manifest.json','geometry_manifest_sha256'),('independent_evaluation_manifest.json','evaluation_manifest_sha256')]:
        if sha(ROOT/name)!=fixed[key]:raise RuntimeError('frozen manifest changed')
    calibration=json.loads((ROOT/'geometry_calibration_manifest.json').read_text())
    evaluation=json.loads((ROOT/'independent_evaluation_manifest.json').read_text())
    cfg=load_config('experiments/dlm_loss_aggregation/config.yaml');validate_config(cfg)
    if historical_state_digest(calibration)!=cfg['calibration']['expected_state_sha256']:raise RuntimeError('EXP001 states differ')
    model,_=_load_model(cfg)
    if gate2_only:
        gate1_document=json.loads((ROOT/'gate1.json').read_text())
        if gate1_document['status']!='pass':raise RuntimeError('cannot resume Gate 2 without a passed Gate 1')
        masks={name:torch.load(STORE/f'mask_{name}.pt',map_location=model.model.transformer.blocks[31].ff_out.weight.device,
                               weights_only=True) for name in ['Wanda','DLMW','FG']}
        passed=True
    else:
        scores=collect_scores(model,calibration,fixed)
        passed,masks=gate1(scores,fixed)
    if not passed:
        write_json(ROOT/'verdict.json',{'verdict':'FAILED AT GATE 1','branch_terminated':True,'elapsed_seconds':time.time()-started})
        return
    if not gate2(model,evaluation,masks):
        write_json(ROOT/'verdict.json',{'verdict':'FAILED AT GATE 2','branch_terminated':True,'elapsed_seconds':time.time()-started})
        return
    write_json(ROOT/'verdict.json',{'verdict':'GATE 3 REQUIRED','gate1_pass':True,'gate2_pass':True,'elapsed_seconds':time.time()-started})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gate2-only',action='store_true')
    main(parser.parse_args().gate2_only)
