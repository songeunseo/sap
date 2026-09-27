"""Frozen exact KL targets vs three dense predictors. No fitting or rescaling."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

OUT = Path(__file__).parent / 'geometry_kill_gate'
PRED = ['B_hidden', 'B_logit', 'Q_geom']
SEED = 20260905
REPS = 10000
EPS = 1e-12


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x): return {'n':0}
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
            'p01':float(np.quantile(x,.01)),'p10':float(np.quantile(x,.1)),
            'p90':float(np.quantile(x,.9)),'p99':float(np.quantile(x,.99)),
            'min':float(x.min()),'max':float(x.max())}


def correlations(x,y):
    x,y=np.asarray(x),np.asarray(y)
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return {'spearman':None,'pearson':None}
    return {'spearman':float(spearmanr(x,y).statistic),'pearson':float(pearsonr(x,y).statistic)}


def bootstrap_mean(values,sequence):
    values=np.asarray(values); sequence=np.asarray(sequence)
    means=np.array([values[sequence==s].mean() for s in np.unique(sequence)])
    # Five states per sequence; equivalent to sampling entire receiver clusters.
    rng=np.random.default_rng(SEED)
    indices=rng.integers(0,len(means),(REPS,len(means)))
    return np.quantile(means[indices].mean(1),[.025,.975]).tolist()


def contrasts(dataset,c):
    if dataset=='A': return {'state_pairing':c['native']-c['foreign']}
    if dataset=='B':
        nn,nf,fn,ff=(c[key] for key in ['NN','NF','FN','FF'])
        return {'feature':.5*(nn-nf+fn-ff),'token':.5*(nn-fn+nf-ff),
                'interaction':nn-nf-fn+ff,'native_excess':nn-np.maximum(nf,fn)}
    nn,ns,sn,ss=(c[key] for key in ['NN','NS','SN','SS'])
    return {'position':nn-ss,'feature_placement':sn-ns,'pairing':.5*(nn+ss-ns-sn)}


def main():
    run=json.loads((OUT/'run_manifest.json').read_text())
    if run['status']!='complete' or run['records']!=3760:
        raise RuntimeError('collection incomplete')
    if hashlib.sha256((OUT/'per_condition.json').read_bytes()).hexdigest()!=run['predictions_sha256']:
        raise RuntimeError('predictions hash mismatch')
    df=pd.DataFrame(json.loads((OUT/'per_condition.json').read_text()))
    if not np.isfinite(df[['exact_KL']+PRED].values).all(): raise RuntimeError('nonfinite predictions')
    df['Q_ratio_floor']=df.Q_geom<=EPS
    df['exact_over_Q']=df.exact_KL/(df.Q_geom+EPS)
    df['exact_minus_Q']=df.exact_KL-df.Q_geom
    df.to_csv(OUT/'per_condition.csv',index=False)
    cell_means=df.groupby(['dataset','receiver','sequence','timestep','condition'])[['exact_KL']+PRED].mean().reset_index()
    cell_means.to_csv(OUT/'receiver_cell_means.csv',index=False)
    cell_means.groupby(['dataset','condition'])[['exact_KL']+PRED].mean().to_csv(OUT/'family_cell_means.csv')
    rows=[]
    for (dataset,receiver), sub in cell_means.groupby(['dataset','receiver']):
        base={'dataset':dataset,'receiver':int(receiver),'sequence':int(sub.sequence.iloc[0]),'timestep':float(sub.timestep.iloc[0])}
        by_field={field:contrasts(dataset,sub.set_index('condition')[field].to_dict()) for field in ['exact_KL']+PRED}
        for effect in by_field['exact_KL']:
            rows.append({**base,'effect':effect,**{field:by_field[field][effect] for field in by_field}})
    effect_df=pd.DataFrame(rows)
    effect_df.to_csv(OUT/'receiver_contrasts.csv',index=False)
    effect_df.groupby(['dataset','effect','timestep'])[['exact_KL']+PRED].mean().to_csv(OUT/'timestep_contrasts.csv')
    effect_df.groupby(['dataset','effect','sequence'])[['exact_KL']+PRED].mean().to_csv(OUT/'sequence_contrasts.csv')
    associations=[]
    for (dataset,effect),sub in effect_df.groupby(['dataset','effect']):
        exact=sub.exact_KL.to_numpy()
        for pred in PRED:
            vals=sub[pred].to_numpy()
            associations.append({'dataset':dataset,'effect':effect,'predictor':pred,
                                 **correlations(vals,exact),'sign_agreement':float((np.sign(vals)==np.sign(exact)).mean()),
                                 'exact_positive_fraction':float((exact>0).mean()),'predicted_positive_fraction':float((vals>0).mean()),
                                 'exact_mean':float(exact.mean()),'predicted_mean':float(vals.mean()),
                                 'predicted_CI95':bootstrap_mean(vals,sub.sequence),'exact_CI95':bootstrap_mean(exact,sub.sequence)})
    pd.DataFrame(associations).to_csv(OUT/'contrast_prediction.csv',index=False)
    raw=[]; within=[]
    for dataset,sub in [('ALL',df)]+list(df.groupby('dataset')):
        for pred in PRED:
            raw.append({'dataset':dataset,'predictor':pred,'n':len(sub),**correlations(sub[pred],sub.exact_KL)})
        for receiver,part in sub.groupby('receiver'):
            # Native is repeated for every donor/shift. Count once per receiver.
            native=part.condition.isin(['native','NN'])
            unique=pd.concat([part[native].iloc[:1],part[~native]])
            for pred in PRED:
                within.append({'dataset':dataset,'receiver':int(receiver),'predictor':pred,'n':len(unique),
                               **correlations(unique[pred],unique.exact_KL)})
    within_df=pd.DataFrame(within)
    within_df.to_csv(OUT/'within_receiver.csv',index=False)
    pd.DataFrame(raw).to_csv(OUT/'raw_prediction.csv',index=False)
    within_summary=[]
    for (dataset,pred),part in within_df.groupby(['dataset','predictor']):
        within_summary.append({'dataset':dataset,'predictor':pred,**stats(part.spearman.to_numpy())})
    approximation=[]
    for field in ['dataset','receiver','timestep']:
        for group,part in df.groupby(field):
            approximation.append({'group_type':field,'group':str(group),'ratio':stats(part.exact_over_Q),
                                  'residual':stats(part.exact_minus_Q),'floor_count':int(part.Q_ratio_floor.sum())})
    for (dataset,condition),part in df.groupby(['dataset','condition']):
        approximation.append({'group_type':'family_condition','group':dataset+':'+condition,
                              'ratio':stats(part.exact_over_Q),'residual':stats(part.exact_minus_Q),
                              'floor_count':int(part.Q_ratio_floor.sum())})
    summary={'raw_prediction':raw,'within_receiver_spearman':within_summary,
             'causal_contrasts':associations,'approximation':approximation,
             'bootstrap':{'seed':SEED,'replicates':REPS,'cluster':'receiver sequence, keeping five timesteps; donors/shifts averaged first'},
             'ratio_epsilon':EPS,'no_fitted_rescaling':True}
    (OUT/'analysis.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in OUT.iterdir()
           if path.suffix in ['.csv','.png'] or path.name=='analysis.json'}
    (OUT/'analysis_manifest.json').write_text(json.dumps({'status':'complete','files':files},indent=2,sort_keys=True)+'\n')
    print(pd.DataFrame(associations)[['dataset','effect','predictor','spearman','sign_agreement','exact_mean','predicted_mean']].to_string(index=False))
    print('\nWithin receiver Spearman summaries:')
    print(pd.DataFrame(within_summary).to_string(index=False))


if __name__=='__main__':main()
