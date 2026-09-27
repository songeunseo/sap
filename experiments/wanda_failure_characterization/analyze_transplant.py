import hashlib,json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from experiments.wanda_failure_characterization.propagation_core import factorial_contrasts

ROOT=Path(__file__).parent;SEED=20260905;BOOTSTRAPS=10000

def summary(x):
    x=np.asarray(x,float)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'p10':float(np.quantile(x,.1)),'p90':float(np.quantile(x,.9)),'fraction_positive':float((x>0).mean())}

def main():
    p=torch.load(ROOT/'perturbation_transplant.pt',map_location='cpu',weights_only=True);r=p['results'];kl=r['kl'];contr=factorial_contrasts(kl);seq=p['sequence_index'].numpy();ts=p['timestep_index'].numpy();seqs=np.unique(seq);times=np.unique(ts)
    rows=[]
    for si in range(40):
        row={'state':si,'sequence':int(seq[si]),'timestep_index':int(ts[si]),'tau':p['tau'][si].item(),'natural_r30':p['natural_relative_l2'][si,0].item(),'natural_r31':p['natural_relative_l2'][si,1].item()}
        for di,d in enumerate(('D30','D31')):
            for li,l in enumerate(('L30','L31')):
                for key,val in r.items():row[f'{key}_{d}_{l}']=val[si,di,li].item()
        for key,val in contr.items():row[key]=val[si].item()
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'transplant_per_state.csv',index=False)
    contrast_summary={k:summary(v.numpy()) for k,v in contr.items()}
    sequence=[];timestep=[]
    for ident,mask in [(s,seq==s) for s in seqs]:
        row={'sequence':int(ident)}
        for di,d in enumerate(('D30','D31')):
            for li,l in enumerate(('L30','L31')):row[f'KL_{d}_{l}']=kl[mask,di,li].mean().item()
        for key,val in contr.items():row[key]=val[mask].mean().item()
        sequence.append(row)
    for ident,mask in [(t,ts==t) for t in times]:
        row={'timestep_index':int(ident),'timestep':[.1,.3,.5,.7,.9][int(ident)]}
        for di,d in enumerate(('D30','D31')):
            for li,l in enumerate(('L30','L31')):row[f'KL_{d}_{l}']=kl[mask,di,li].mean().item()
        for key,val in contr.items():row[key]=val[mask].mean().item()
        timestep.append(row)
    pd.DataFrame(sequence).to_csv(ROOT/'transplant_by_sequence.csv',index=False);pd.DataFrame(timestep).to_csv(ROOT/'transplant_by_timestep.csv',index=False)
    rng=np.random.default_rng(SEED);bootstrap={}
    for key in ('location_main','direction_main','interaction'):
        x=contr[key].numpy();draws=[]
        for _ in range(BOOTSTRAPS):
            selected=rng.choice(seqs,size=len(seqs),replace=True);draws.append(np.concatenate([x[seq==s] for s in selected]).mean())
        bootstrap[key]={'estimate':float(x.mean()),'ci95':[float(v) for v in np.quantile(draws,[.025,.975])]}
    flat_kl=kl.numpy().reshape(-1);fh=r['final_hidden_relative_energy'].numpy().reshape(-1);lg=r['masked_logit_relative_energy'].numpy().reshape(-1)
    magnitude={'KL_vs_final_energy_spearman':float(spearmanr(flat_kl,fh).statistic),'KL_vs_logit_energy_spearman':float(spearmanr(flat_kl,lg).statistic),
               'cell_means':{},'norm_error':{'mean':r['tau_deviation'].mean().item(),'p99':torch.quantile(r['tau_deviation'],.99).item(),'max':r['tau_deviation'].max().item()}}
    for di,d in enumerate(('D30','D31')):
        for li,l in enumerate(('L30','L31')):
            key=f'{d}@{l}';k=kl[:,di,li];f=r['final_hidden_relative_energy'][:,di,li];g=r['masked_logit_relative_energy'][:,di,li]
            magnitude['cell_means'][key]={'KL':k.mean().item(),'loss_delta':r['loss_delta'][:,di,li].mean().item(),'top1_agreement':r['top1_agreement'][:,di,li].mean().item(),'confidence_mae':r['confidence_mae'][:,di,li].mean().item(),'final_hidden_relative_energy':f.mean().item(),'final_hidden_relative_l2':r['final_hidden_relative_l2'][:,di,li].mean().item(),'masked_logit_relative_energy':g.mean().item(),'logit_delta_rms':r['logit_delta_rms'][:,di,li].mean().item(),'median_KL_per_final_energy':(k/f.clamp_min(1e-30)).median().item(),'median_KL_per_logit_energy':(k/g.clamp_min(1e-30)).median().item()}
    natural={name:{'mean_KL':x['kl'].mean().item(),'mean_loss_delta':x['loss_delta'].mean().item()} for name,x in p['natural_anchors'].items()}
    result={'bootstrap_seed':SEED,'bootstrap_replicates':BOOTSTRAPS,'contrast_summaries':contrast_summary,'sequence_cluster_bootstrap':bootstrap,'magnitude_analysis':magnitude,'natural_anchors':natural,'native_gate':{k:{'max':v.max().item(),'mean':v.mean().item()} for k,v in p['native_gate'].items()},'same_path_sham':{k:{'median':v.median().item(),'p99':torch.quantile(v,.99).item(),'max':v.max().item()} for k,v in p['sham'].items()},'repeatability':p['repeatability']}
    (ROOT/'transplant_analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    files={}
    for name in ('transplant_per_state.csv','transplant_by_sequence.csv','transplant_by_timestep.csv','transplant_analysis.json'):files[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (ROOT/'transplant_analysis_manifest.json').write_text(json.dumps({'status':'complete','files':files,'bootstrap_seed':SEED,'bootstrap_replicates':BOOTSTRAPS},indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
