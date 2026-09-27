import hashlib,json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).parent

def rho(a,b):
    value=spearmanr(a,b).statistic
    return None if not np.isfinite(value) else float(value)

def ridge_lolo(df,features,target='Y50_KL'):
    X=df[features].to_numpy(float);y=df[target].to_numpy(float);layers=df.layer.to_numpy();pred=np.empty_like(y);coefs=[]
    for layer in sorted(set(layers)):
        tr=layers!=layer;te=~tr;mu=X[tr].mean(0);sd=X[tr].std(0);sd[sd<1e-12]=1;xx=(X[tr]-mu)/sd;ym=y[tr].mean()
        beta=np.linalg.solve(xx.T@xx+np.eye(len(features)),xx.T@(y[tr]-ym));pred[te]=(X[te]-mu)/sd@beta+ym;coefs.append(beta)
    return {'r2':float(1-((y-pred)**2).sum()/(((y-y.mean())**2).sum()+1e-30)),'spearman':rho(y,pred),
            'coefficient_sign_stability':dict(zip(features,(np.sign(coefs)==np.sign(np.mean(coefs,0))).mean(0).tolist()))}

def main():
    trace=torch.load(ROOT/'propagation_trace.pt',map_location='cpu',weights_only=True);old=torch.load(ROOT/'failure_map.pt',map_location='cpu',weights_only=True)['damage'];base=pd.read_csv(ROOT/'per_module_statistics.csv').set_index('module')
    names=trace['module_names'];rows=[]
    for mi,name in enumerate(names):
        layer=int(name[6:8]);typ=name.split('.',1)[1];row={'module':name,'layer':layer,'module_type':typ,'Erec50':old['fields']['reconstruction'][:,mi,0].mean().item(),'Y50_KL':old['fields']['kl'][:,mi,0].mean().item()}
        for stage in ('branch','c1','same_block','final_pre_norm','final_post_norm','masked_logits'):
            for key in ('abs_energy','relative_energy','relative_l2','masked_energy_fraction','masked_enrichment','sham_delta_cosine','aligned_energy_fraction','orthogonal_energy_fraction'):
                values=trace[stage][key][:,mi]
                if torch.isfinite(values).any():
                    finite=values[torch.isfinite(values)];row[f'{stage}_{key}_mean']=finite.mean().item();row[f'{stage}_{key}_median']=finite.median().item();row[f'{stage}_{key}_p10']=torch.quantile(finite,.1).item();row[f'{stage}_{key}_p90']=torch.quantile(finite,.9).item()
        c1=trace['c1']['relative_l2'][:,mi];final=trace['final_post_norm']['relative_l2'][:,mi];kl=old['fields']['kl'][:,mi,0]
        gain=final/c1.clamp_min(1e-30);gkl=kl/trace['c1']['relative_energy'][:,mi].clamp_min(1e-30);gklf=kl/trace['final_post_norm']['relative_energy'][:,mi].clamp_min(1e-30)
        for key,val in [('propagation_gain',gain),('KL_per_C1_energy',gkl),('KL_per_final_energy',gklf)]:
            row[key+'_median']=val.median().item();row[key+'_p10']=torch.quantile(val,.1).item();row[key+'_p90']=torch.quantile(val,.9).item()
        row['c1_denominator_min']=trace['c1']['relative_energy'][:,mi].min().item();row['final_denominator_min']=trace['final_post_norm']['relative_energy'][:,mi].min().item()
        row['logit_delta_rms_mean']=trace['extra']['logit_delta_rms'][:,mi].mean().item();row['token_logitnorm_KL_spearman_mean']=trace['extra']['token_logitnorm_kl_spearman'][:,mi].mean().item()
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'propagation_per_module.csv',index=False)
    chain=[('Erec50','branch_relative_energy_mean'),('branch_relative_energy_mean','c1_relative_energy_mean'),('c1_relative_energy_mean','final_post_norm_relative_energy_mean'),('final_post_norm_relative_energy_mean','Y50_KL'),('masked_logits_relative_energy_mean','Y50_KL'),('Erec50','Y50_KL')]
    chain_result=[]
    for x,y in chain:
        chain_result.append({'x':x,'y':y,'scope':'pooled','spearman':rho(df[x],df[y])})
        for typ,g in df.groupby('module_type'):chain_result.append({'x':x,'y':y,'scope':'type:'+typ,'spearman':rho(g[x],g[y])})
    explanatory=['branch_relative_energy_mean','branch_masked_enrichment_mean','c1_relative_energy_mean','final_post_norm_relative_energy_mean','propagation_gain_median','KL_per_final_energy_median','masked_logits_relative_energy_mean']
    assoc=[]
    for x in explanatory:
        assoc.append({'feature':x,'scope':'pooled','spearman':rho(df[x],df.Y50_KL)})
        z=df[x].copy()
        for typ,g in df.groupby('module_type'):
            sd=g[x].std();z.loc[g.index]=(g[x]-g[x].mean())/(sd if sd>0 else 1);assoc.append({'feature':x,'scope':'type:'+typ,'spearman':rho(g[x],g.Y50_KL)})
        assoc.append({'feature':x,'scope':'within_type_z','spearman':rho(z,df.Y50_KL)})
    pd.DataFrame(chain_result).to_csv(ROOT/'propagation_chain_associations.csv',index=False);pd.DataFrame(assoc).to_csv(ROOT/'propagation_associations.csv',index=False)
    cases={}
    for name in ('block_31.ff_out','block_31.ff_proj','block_31.up_proj','block_30.ff_out','block_00.v_proj'):
        mi=names.index(name);row=df[df.module==name].iloc[0].to_dict();layer=int(name[6:8]);curve=[];baseline=trace['c1']['relative_l2'][:,mi]
        for bi in range(layer,32):
            v=trace['block_end']['relative_l2'][:,mi,bi]
            curve.append({'block':bi,'relative_l2_mean':v.mean().item(),'gain_from_C1_mean':(v/baseline.clamp_min(1e-30)).mean().item(),'masked_enrichment_mean':trace['block_end']['masked_enrichment'][:,mi,bi].mean().item()})
        cases[name]={'summary':row,'block_end_curve':curve}
    fixed=['branch_relative_energy_mean','branch_masked_enrichment_mean','c1_relative_energy_mean','final_post_norm_relative_energy_mean','propagation_gain_median','masked_logits_relative_energy_mean']
    result={'reproduction_gate':trace['reproduction_gate'],'stage_chain':chain_result,'fixed_leave_one_layer_out_ridge':ridge_lolo(df,fixed),'cases':cases,
            'denominator_floors':{'c1_below_1e-12':int((df.c1_denominator_min<1e-12).sum()),'final_below_1e-12':int((df.final_denominator_min<1e-12).sum())}}
    (ROOT/'propagation_analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    files={}
    for p in ('propagation_per_module.csv','propagation_chain_associations.csv','propagation_associations.csv','propagation_analysis.json'):
        files[p]=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    (ROOT/'propagation_analysis_manifest.json').write_text(json.dumps({'status':'complete','files':files},indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
