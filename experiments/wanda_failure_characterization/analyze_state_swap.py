import hashlib,json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).parent;SEED=20260905;B=10000

def summarize(x):
    x=np.asarray(x,float);return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9)),'fraction_positive':float((x>0).mean())}

def bootstrap_ci(values,sequence):
    rng=np.random.default_rng(SEED);ids=np.unique(sequence);draw=[]
    for _ in range(B):
        chosen=rng.choice(ids,len(ids),replace=True);draw.append(np.concatenate([values[sequence==s] for s in chosen]).mean())
    return [float(x) for x in np.quantile(draw,[.025,.975])]

def main():
    p=torch.load(ROOT/'state_swap_results.pt',map_location='cpu',weights_only=True);d=torch.load(ROOT/'transplant_deltas.pt',map_location='cpu',weights_only=True);mapping=json.loads((ROOT/'state_swap_mapping.json').read_text());r=p['results'];seq=p['sequence_index'].numpy();ts=p['timestep_index'].numpy();kl=r['kl'].numpy();labels=('native','cyclic_foreign','same_timestep_foreign')
    advantage={'cyclic':kl[:,0]-kl[:,1],'same_timestep':kl[:,0]-kl[:,2]};rows=[]
    for i in range(40):
        base={'state':i,'sequence':int(seq[i]),'timestep_index':int(ts[i]),'tau':p['tau'][i].item(),'cyclic_donor':mapping['cyclic_donors'][i],'same_timestep_donor':mapping['same_timestep_donors'][i],'cosine_cyclic':p['cosines'][i,0].item(),'cosine_same_timestep':p['cosines'][i,1].item()}
        for ci,label in enumerate(labels):
            for key,val in r.items():base[f'{key}_{label}']=val[i,ci].item()
        base['pairing_advantage_cyclic']=advantage['cyclic'][i];base['pairing_advantage_same_timestep']=advantage['same_timestep'][i];base['ratio_cyclic_native']=kl[i,1]/max(kl[i,0],1e-30);base['ratio_same_timestep_native']=kl[i,2]/max(kl[i,0],1e-30);rows.append(base)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'state_swap_per_state.csv',index=False)
    summaries={};boot={}
    for kind,x in advantage.items():summaries[kind]=summarize(x);boot[kind]={'estimate':float(x.mean()),'ci95':bootstrap_ci(x,seq)}
    by_t=[];by_s=[]
    for ident,mask in [(t,ts==t) for t in np.unique(ts)]:
        by_t.append({'timestep_index':int(ident),'timestep':[.1,.3,.5,.7,.9][int(ident)],**{f'KL_{label}':float(kl[mask,ci].mean()) for ci,label in enumerate(labels)},**{f'pairing_advantage_{k}':float(v[mask].mean()) for k,v in advantage.items()},**{f'fraction_native_gt_{k}':float((v[mask]>0).mean()) for k,v in advantage.items()}})
    for ident,mask in [(s,seq==s) for s in np.unique(seq)]:
        by_s.append({'sequence':int(ident),**{f'KL_{label}':float(kl[mask,ci].mean()) for ci,label in enumerate(labels)},**{f'pairing_advantage_{k}':float(v[mask].mean()) for k,v in advantage.items()}})
    pd.DataFrame(by_t).to_csv(ROOT/'state_swap_by_timestep.csv',index=False);pd.DataFrame(by_s).to_csv(ROOT/'state_swap_by_sequence.csv',index=False)
    magnitude={}
    for foreign,ci in [('cyclic',1),('same_timestep',2)]:
        fhdiff=r['final_hidden_relative_energy'][:,0].numpy()-r['final_hidden_relative_energy'][:,ci].numpy();lgdiff=r['masked_logit_relative_energy'][:,0].numpy()-r['masked_logit_relative_energy'][:,ci].numpy()
        magnitude[foreign]={'final_energy_native_mean':r['final_hidden_relative_energy'][:,0].mean().item(),'final_energy_foreign_mean':r['final_hidden_relative_energy'][:,ci].mean().item(),'logit_energy_native_mean':r['masked_logit_relative_energy'][:,0].mean().item(),'logit_energy_foreign_mean':r['masked_logit_relative_energy'][:,ci].mean().item(),'logit_rms_native_mean':r['logit_rms'][:,0].mean().item(),'logit_rms_foreign_mean':r['logit_rms'][:,ci].mean().item(),'advantage_vs_final_energy_difference_spearman':float(spearmanr(advantage[foreign],fhdiff).statistic),'advantage_vs_logit_energy_difference_spearman':float(spearmanr(advantage[foreign],lgdiff).statistic)}
    # Optional full existing-direction cosine description; no donor selection uses it.
    flat=d['delta31'].flatten(1).float();flat=flat/flat.norm(dim=1,keepdim=True).clamp_min(1e-30);cos=(flat@flat.T).numpy();same=[];different=[]
    for i in range(40):
        for j in range(i+1,40):(same if ts[i]==ts[j] else different).append(cos[i,j])
    cosine={'cyclic':summarize(p['cosines'][:,0].numpy()),'same_timestep_donor':summarize(p['cosines'][:,1].numpy()),'all_same_timestep_cross_sequence':summarize(same),'all_different_timestep':summarize(different),'cyclic_cosine_vs_foreign_KL':float(spearmanr(p['cosines'][:,0],kl[:,1]).statistic),'cyclic_cosine_vs_pairing_advantage':float(spearmanr(p['cosines'][:,0],advantage['cyclic']).statistic),'same_timestep_cosine_vs_foreign_KL':float(spearmanr(p['cosines'][:,1],kl[:,2]).statistic),'same_timestep_cosine_vs_pairing_advantage':float(spearmanr(p['cosines'][:,1],advantage['same_timestep']).statistic)}
    result={'bootstrap_seed':SEED,'bootstrap_replicates':B,'KL_means':dict(zip(labels,kl.mean(0).tolist())),'pairing_advantage':summaries,'sequence_cluster_bootstrap':boot,'retained_damage':{'cyclic':float(kl[:,1].mean()/kl[:,0].mean()),'same_timestep':float(kl[:,2].mean()/kl[:,0].mean())},'foreign_native_ratio_distribution':{'cyclic':summarize(kl[:,1]/np.maximum(kl[:,0],1e-30)),'same_timestep':summarize(kl[:,2]/np.maximum(kl[:,0],1e-30))},'secondary_condition_means':{key:dict(zip(labels,value.mean(0).tolist())) for key,value in r.items() if key in ('loss_delta','top1_agreement','confidence_mae','final_hidden_relative_energy','masked_logit_relative_energy','logit_rms')},'mapping_overlap':{'same_donor_count':sum(a==b for a,b in zip(mapping['cyclic_donors'],mapping['same_timestep_donors'])),'total':40},'norm_error':{'mean':r['norm_deviation'].mean().item(),'p99':torch.quantile(r['norm_deviation'],.99).item(),'max':r['norm_deviation'].max().item()},'magnitude':magnitude,'cosine':cosine,'native_gate':{k:{'mean':v.mean().item(),'max':v.max().item()} for k,v in p['native_gate'].items()},'sham':{k:{'median':v.median().item(),'p99':torch.quantile(v,.99).item(),'max':v.max().item()} for k,v in p['sham'].items()},'repeatability':p['repeatability']}
    (ROOT/'state_swap_analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');files={}
    for name in ('state_swap_per_state.csv','state_swap_by_timestep.csv','state_swap_by_sequence.csv','state_swap_analysis.json'):files[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (ROOT/'state_swap_analysis_manifest.json').write_text(json.dumps({'status':'complete','files':files,'bootstrap_seed':SEED,'bootstrap_replicates':B},indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
