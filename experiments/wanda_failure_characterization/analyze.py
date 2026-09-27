import hashlib, json, math, platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, kendalltau, spearmanr

from experiments.dlm_loss_aggregation.run import _load_model, load_config, validate_config
from experiments.wanda_failure_characterization.core import rowwise_wanda_mask
from experiments.wanda_failure_characterization.run_failure_map import modules

ROOT=Path(__file__).parent

def qstats(x,prefix):
    x=np.asarray(x,dtype=np.float64); qs=np.quantile(x,[.5,.9,.99])
    mean=x.mean(); std=x.std()
    return {f'{prefix}_mean':mean,f'{prefix}_median':qs[0],f'{prefix}_std':std,
            f'{prefix}_cv':std/(abs(mean)+1e-30),f'{prefix}_p90':qs[1],f'{prefix}_p99':qs[2],
            f'{prefix}_max':x.max(),f'{prefix}_max_median':x.max()/(qs[0]+1e-30),
            f'{prefix}_p99_median':qs[2]/(qs[0]+1e-30),f'{prefix}_kurtosis':float(kurtosis(x,fisher=False,bias=False))}

def energy_top(x, fractions):
    v=np.sort(np.asarray(x,dtype=np.float64)**2)[::-1]; cs=np.cumsum(v); total=cs[-1]+1e-30
    return [float(cs[max(0,math.ceil(len(v)*f)-1)]/total) for f in fractions]

def sampled_spearman(a,b,cap=200000):
    a=a.reshape(-1);b=b.reshape(-1)
    if a.numel()>cap:
        step=math.ceil(a.numel()/cap);a=a[::step][:cap];b=b[::step][:cap]
    return float(spearmanr(a.float().cpu().numpy(),b.float().cpu().numpy()).statistic)

def safe_spear(a,b):
    r=spearmanr(a,b).statistic
    return None if not np.isfinite(r) else float(r)

def ridge_lolo(df,features,target):
    X=df[features].replace([np.inf,-np.inf],np.nan).fillna(df[features].median()).to_numpy(float)
    y=df[target].to_numpy(float); layers=df.layer.to_numpy(); pred=np.empty_like(y); coefs=[]
    for layer in sorted(set(layers)):
        tr=layers!=layer; te=~tr; mu=X[tr].mean(0); sd=X[tr].std(0); sd[sd<1e-12]=1
        xx=(X[tr]-mu)/sd; yy=y[tr]; ym=yy.mean()
        beta=np.linalg.solve(xx.T@xx+np.eye(xx.shape[1]),xx.T@(yy-ym));pred[te]=(X[te]-mu)/sd@beta+ym;coefs.append(beta)
    denom=((y-y.mean())**2).sum();r2=1-((y-pred)**2).sum()/(denom+1e-30)
    C=np.stack(coefs)
    return {'leave_one_layer_out_r2':float(r2),'rank_correlation':safe_spear(y,pred),
            'coefficient_mean':dict(zip(features,C.mean(0).tolist())),
            'coefficient_sign_stability':dict(zip(features,(np.sign(C)==np.sign(C.mean(0))).mean(0).tolist()))}

def main():
    payload=torch.load(ROOT/'failure_map.pt',map_location='cpu',weights_only=True)['damage']; stats=torch.load(ROOT/'wanda_sufficient_statistics.pt',map_location='cpu',weights_only=True)
    names=payload['module_names']; fields=payload['fields']; A=stats['clean_A']; heldA=payload['heldout_A_state']; ts=payload['timestep_index'].numpy()
    maskmeta={(e['module'],int(e['sparsity']*100)):e for e in json.loads((ROOT/'wanda_mask_manifest.json').read_text())['entries']}
    config=load_config('experiments/dlm_loss_aggregation/config.yaml');validate_config(config);model,_=_load_model(config);mapping=modules(model)
    rows=[]
    for mi,name in enumerate(names):
        layer=int(name[6:8]);typ=name.split('.',1)[1];mod=mapping[name];w=mod.weight.detach().float(); aw=w.abs(); a=A[name].float().to(w.device); score=aw*a.sqrt()[None,:]
        wn=aw.cpu().numpy().reshape(-1); an=a.cpu().numpy(); row={'module':name,'layer':layer,'module_type':typ,'nweight':w.numel()}
        row.update(qstats(wn,'weight')); row.update(qstats(an,'clean_A'))
        for f,v in zip((1,5,10),energy_top(wn,[.01,.05,.1])):row[f'weight_energy_top{f}']=v
        for f,v in zip((1,5,10),energy_top(np.sqrt(an),[.01,.05,.1])):row[f'activation_energy_top{f}']=v
        ha=heldA[name].numpy(); hm=ha.mean(0); row.update(qstats(hm,'heldout_A'))
        row['clean_heldout_A_spearman']=safe_spear(an,hm);row['clean_heldout_A_relative_l2']=float(np.linalg.norm(hm-an)/(np.linalg.norm(an)+1e-30))
        tm=np.stack([ha[ts==t].mean(0) for t in sorted(set(ts))]); energy=tm.mean(1);row['activation_timestep_cv']=float(energy.std()/(abs(energy.mean())+1e-30));row['activation_early_late_ratio']=float(energy[-1]/(energy[0]+1e-30))
        row['wanda_weight_spearman']=sampled_spearman(score,aw);row['wanda_activation_spearman']=sampled_spearman(score,a.sqrt()[None,:].expand_as(score))
        lw=torch.log(aw.clamp_min(1e-30));la=.5*torch.log(a.clamp_min(1e-30))[None,:].expand_as(lw);ls=lw+la
        row['var_log_weight']=lw.var(unbiased=False).item();row['var_half_log_A']=la.var(unbiased=False).item();row['cov_log_terms']=((lw-lw.mean())*(la-la.mean())).mean().item();row['var_log_score']=ls.var(unbiased=False).item()
        for ri,sp in enumerate((50,75)):
            wm=rowwise_wanda_mask(score,sp/100);mm=rowwise_wanda_mask(aw,sp/100); xor=(wm!=mm).float().mean().item();inter=(wm&mm).sum().item();union=(wm|mm).sum().item()
            row[f'wanda_magnitude_xor_{sp}']=xor;row[f'wanda_magnitude_pruned_jaccard_{sp}']=inter/union
            col=wm.float().mean(0);cm=col.mean();row[f'mask_column_prune_cv_{sp}']=(col.std(unbiased=False)/(cm+1e-30)).item()
            p=col.clamp(1e-12,1-1e-12);row[f'mask_entropy_{sp}']=(-(p*p.log()+(1-p)*(1-p).log())).mean().item()
            cs=col.sort(descending=True).values.cumsum(0)/(col.sum()+1e-30)
            for f in (1,5,10):row[f'mask_pruned_concentration_top{f}_{sp}']=cs[math.ceil(len(col)*f/100)-1].item()
            geo=maskmeta[(name,sp)]['threshold_geometry'];row[f'threshold_gap_{sp}']=geo['normalized_gap_mean'];row[f'threshold_within1_{sp}']=geo['within_1pct_mean'];row[f'threshold_within5_{sp}']=geo['within_5pct_mean']
            dl=fields['delta_loss'][:,mi,ri].numpy();kl=fields['kl'][:,mi,ri].numpy();rec=fields['reconstruction'][:,mi,ri].numpy()
            row[f'Y{sp}_signed']=dl.mean();row[f'Y{sp}_pos']=np.maximum(dl,0).mean();row[f'Y{sp}_abs']=np.abs(dl).mean();row[f'Y{sp}_KL']=kl.mean();row[f'Erec{sp}']=rec.mean();row[f'top1_{sp}']=fields['top1_agreement'][:,mi,ri].mean().item();row[f'confidence_mae_{sp}']=fields['confidence_mae'][:,mi,ri].mean().item()
            for metric,vals in [('delta',dl),('KL',kl),('Erec',rec)]:
                tv=np.array([vals[ts==t].mean() for t in sorted(set(ts))]);row[f'{metric}_timestep_cv_{sp}']=tv.std()/(abs(tv.mean())+1e-30);row[f'{metric}_early_late_{sp}']=tv[-1]-tv[0]
        rows.append(row);print(f'statistics {mi+1}/224',flush=True)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'per_module_statistics.csv',index=False)
    del model;torch.cuda.empty_cache()
    # Associations are evaluated only after the failure map and pre-pruning table are frozen.
    targets=['Y50_KL','Y50_pos','Y75_KL','Y75_pos','Y50_signed','Y50_abs','Erec50','Erec75']
    excluded=set(targets+['Y75_signed','Y75_abs']+[c for c in df if c.startswith(('top1_','confidence_mae_','delta_','KL_','Erec'))])
    features=[c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded and c not in ('layer','nweight')]
    associations=[]
    for x in features:
      for y in targets:
        associations.append({'feature':x,'target':y,'scope':'pooled','spearman':safe_spear(df[x],df[y])})
        z=df[x].copy()
        for typ,g in df.groupby('module_type'):
            sd=g[x].std();z.loc[g.index]=(g[x]-g[x].mean())/(sd if sd>0 else 1)
            associations.append({'feature':x,'target':y,'scope':f'type:{typ}','spearman':safe_spear(g[x],g[y])})
        associations.append({'feature':x,'target':y,'scope':'within_type_z','spearman':safe_spear(z,df[y])})
    adf=pd.DataFrame(associations);adf.to_csv(ROOT/'univariate_associations.csv',index=False)
    rec_assoc={f'{x}_{y}':safe_spear(df[x],df[y]) for x in ('Erec50','Erec75') for y in ('Y50_KL','Y50_pos','Y75_KL','Y75_pos')}
    chosen=['weight_cv','weight_kurtosis','weight_energy_top1','clean_A_cv','clean_A_kurtosis','activation_energy_top1','clean_heldout_A_relative_l2','activation_timestep_cv','wanda_magnitude_xor_50','threshold_within1_50','mask_column_prune_cv_50']
    multi={y:ridge_lolo(df,chosen,y) for y in ('Y50_KL','Y50_pos')}
    sham=payload['sham'];noise={'delta_abs_median':sham['delta_loss'].abs().median().item(),'delta_abs_p99':torch.quantile(sham['delta_loss'].abs(),.99).item(),'delta_abs_max':sham['delta_loss'].abs().max().item(),'kl_median':sham['kl'].median().item(),'kl_p99':torch.quantile(sham['kl'],.99).item(),'kl_max':sham['kl'].max().item()}
    summary={'primary_targets':['Y50_KL','Y50_pos'],'numerical_sham_raw_dense_reference':noise,'local_reconstruction_associations':rec_assoc,'multivariate_fixed_ridge':multi,
             'top_modules':{y:df.nlargest(15,y)[['module',y]].to_dict('records') for y in targets},
             'module_type_means':df.groupby('module_type')[targets].mean().to_dict(),'gradient_artifacts':'skipped: persisted EXP-001 artifacts do not contain exact per-module underlying score vectors required for exact mapping'}
    (ROOT/'analysis_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    manifest={'status':'complete','files':{},'environment':{'python':platform.python_version(),'torch':torch.__version__,'pandas':pd.__version__}}
    for path in (ROOT/'per_module_statistics.csv',ROOT/'univariate_associations.csv',ROOT/'analysis_summary.json'):
        manifest['files'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
