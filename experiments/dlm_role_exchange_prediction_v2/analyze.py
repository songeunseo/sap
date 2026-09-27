import json
import numpy as np
import torch
from scipy.stats import spearmanr
from experiments.dlm_role_exchange_prediction.analyze import features as old_features, TYPES
from experiments.dlm_role_exchange_prediction_v2.common import ROOT,OLD,STORE,ridge,equal_layer_weights,bootstrap_cells,atomic_json,sha256,validate_sources

BASE=['P0_structure','P1_pooled_dense','P2_role_dense','P3_role_dense_sparse','P4_role_energy']
EXTRA=['E0_pooled_context','E1_role_contrast','E2_role_interactions']
RANDOM=[f'random_{s}' for s in (101,202,303)]

def feature(row,model):
    if model in BASE:return old_features(row,model)
    if model.startswith('random_'):return old_features(row,'P2_role_dense',int(model.split('_')[1]))
    # Nested comparisons retain the pooled signal, cardinality and common context in both models.
    z=old_features(row,'P1_pooled_dense')
    z += [row['timestep'],row['masked_fraction']]
    if model=='E0_pooled_context':return z
    f=row['features']['dense']; contrast=f['masked_token']-f['unmasked_token']
    z += [contrast]
    if model=='E1_role_contrast':return z
    z += [contrast*row['timestep']]
    z += [contrast*float(row['projection_type']==t) for t in TYPES]
    return z

def load(split):
    marker=json.loads((ROOT/f'complete_{split}.json').read_text())
    if marker['config_sha256']!=sha256(ROOT/'config.json'):raise RuntimeError('completion identity mismatch')
    manifest=json.loads((OLD/'state_manifest.json').read_text())
    state_lookup={(s['sequence_index'],s['timestep']):s for s in manifest['states']}
    rows=[]; bundles=[]
    for i in range(marker['states']):
        p=torch.load(STORE/split/f'state_{i:03d}.pt',map_location='cpu',weights_only=False)
        if p['config_sha256']!=marker['config_sha256'] or len(p['rows'])!=385 or len(p['bundle_rows'])!=64:
            raise RuntimeError('incomplete/changed state')
        key=p['sequence_index'],p['timestep']; state=state_lookup[key]
        if p['sham_max_abs'] or p['restore_max_abs']:raise RuntimeError('sham gate failed')
        common={'document':key[0],'timestep':key[1],'masked_fraction':float(np.mean(state['mask']))}
        rows += [{**r,**common} for r in p['rows']]
        bundles += [{**r,**common,'layer':4*r['layer_group']} for r in p['bundle_rows']]
    if len({(r['document'],r['timestep'],r['exchange_index']) for r in rows})!=len(rows):raise RuntimeError('duplicate observations')
    return rows,bundles

def metric(y,p,rows):
    w=equal_layer_weights(rows);w/=w.sum()
    return {'mse':float(np.sum(w*(y-p)**2)),'mae':float(np.sum(w*abs(y-p))),
            'spearman':float(spearmanr(y,p).statistic),'sign_accuracy':float(np.sum(w*(np.sign(y)==np.sign(p)))),
            'calibration_slope':float(np.linalg.lstsq(np.column_stack([np.ones(len(p)),p])*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0][1])}

def interval_decision(comparison,improvement):
    lo,hi=comparison['simultaneous_ci']
    if hi<0 and improvement>=.1:return 'supported_for_this_predictor_and_target'
    if lo>0:return 'worse_under_this_specification'
    if hi<0:return 'improvement_below_practical_gate'
    return 'inconclusive'

def main():
    validate_sources();train,_=load('development');test,bundles=load('final')
    models=BASE+RANDOM+EXTRA; y=np.array([r['delta_kl'] for r in test]);pred={m:np.full(len(y),np.nan) for m in models}
    fits={}
    for fold in range(4):
        tr=[r for r in train if r['layer']//8!=fold];ids=[i for i,r in enumerate(test) if r['layer']//8==fold]
        te=[test[i] for i in ids];w=equal_layer_weights(tr)
        for m in models:
            values,fit=ridge([feature(r,m) for r in tr],[r['delta_kl'] for r in tr],[feature(r,m) for r in te],w)
            pred[m][ids]=values;fits[f'{fold}:{m}']=fit
    if any(not np.isfinite(p).all() for p in pred.values()):raise RuntimeError('OOF predictions nonfinite')
    metrics={m:metric(y,p,test) for m,p in pred.items()}
    errors={m:(p-y)**2 for m,p in pred.items()}
    errors['random_average']=np.mean([errors[m] for m in RANDOM],axis=0)
    comparisons={}
    for name,l,r in [('H1','P2_role_dense','P1_pooled_dense'),('H2','P3_role_dense_sparse','P2_role_dense'),('random_control','P2_role_dense','random_average')]:
        ci=bootstrap_cells(errors[l]-errors[r],test)
        weights=equal_layer_weights(test);weights/=weights.sum()
        gain=float(1-np.sum(weights*errors[l])/np.sum(weights*errors[r]))
        comparisons[name]={**ci,'relative_mse_reduction':gain,'decision':interval_decision(ci,gain)}
    exploratory={}
    for l,r in [(EXTRA[1],EXTRA[0]),(EXTRA[2],EXTRA[1])]:
        exploratory[f'{l}_minus_{r}']=bootstrap_cells(errors[l]-errors[r],test,family=2)
    # All model outputs and strata retained; no selecting a favorable subgroup for the headline.
    strata={}
    for field,values in [('timestep',sorted({r['timestep'] for r in test})),('projection_type',TYPES),('layer_fold',range(4)),('direction',[-1,1])]:
        for value in values:
            indices=[i for i,r in enumerate(test) if (r['layer']//8 if field=='layer_fold' else np.sign(r['parameter_delta']) if field=='direction' else r[field])==value]
            sub=[test[i] for i in indices]
            strata[f'{field}:{value}']={m:metric(y[indices],pred[m][indices],sub) for m in BASE+EXTRA}
    lookup={(r['document'],r['timestep'],r['exchange_index']):i for i,r in enumerate(test)}
    indices=[[lookup[b['document'],b['timestep'],e] for e in b['exchange_indices']] for b in bundles]
    by=np.array([b['delta_kl'] for b in bundles]); summed=np.array([y[ix].sum() for ix in indices])
    bundle_metrics={};bundle_ci={}
    for m in BASE+EXTRA:
        bp=np.array([pred[m][ix].sum() for ix in indices]);nonadd=by-summed;est=summed-bp
        bundle_metrics[m]={**metric(by,bp,bundles),'nonadditivity_rmse':float(np.sqrt(np.mean(nonadd**2))),
                           'individual_prediction_rmse':float(np.sqrt(np.mean(est**2))),
                           'mean_nonadditivity':float(nonadd.mean()),'cross_term':float(2*np.mean(nonadd*est))}
        if m!='P1_pooled_dense':
            ref=np.array([pred['P1_pooled_dense'][ix].sum() for ix in indices])
            bundle_ci[m]=bootstrap_cells((bp-by)**2-(ref-by)**2,bundles,family=len(BASE+EXTRA)-1)
    safety={}
    for norm in ['token','energy']:
        dm=np.array([sum(test[i]['features']['sparse'][f'masked_{norm}'] for i in ix) for ix in indices])
        du=np.array([sum(test[i]['features']['sparse'][f'unmasked_{norm}'] for i in ix) for ix in indices])
        good=(dm<=0)&(du<=0);bad=good&(by>0)
        safety[norm]={'unit':'exact-budget bundle/state','both_improve':int(good.sum()),'kl_worsens':int(bad.sum()),
                      'fraction':float(bad.sum()/good.sum()) if good.any() else None,
                      'mean_kl_change_when_both_improve':float(by[good].mean()) if good.any() else None,
                      'caution':'sum of local reconstruction changes on fixed A inputs; no universal safety conclusion'}
    old=np.array([r['v1_delta_kl'] for r in test])
    drift={'median_abs_label_change':float(np.median(abs(y-old))),
           'p95_abs_label_change':float(np.quantile(abs(y-old),.95)),
           'sign_disagreement_fraction':float(np.mean(np.sign(y)!=np.sign(old))),
           'label_spearman':float(spearmanr(y,old).statistic)}
    result={'status':'complete','study_type':'corrective exploratory rerun, previously seen final data',
            'models':metrics,'comparisons':comparisons,'exploratory_nested_comparisons':exploratory,
            'strata':strata,'bundle_models':bundle_metrics,'bundle_comparisons':bundle_ci,
            'bundle_role_reconstruction_check':safety,'measurement_correction':drift,'fits':fits,
            'interpretation':'Only conclusions about this target, features, and fixed predictor; no blanket rejection of roles.'}
    atomic_json(ROOT/'analysis.json',result)
    from experiments.dlm_role_exchange_prediction.collect import save
    save(STORE/'oof_predictions.pt',{'keys':list(lookup),'y':y,'predictions':pred,'config_sha256':sha256(ROOT/'config.json')})
    lines=['# Role exchange 수정 재분석','', '이전에 확인한 데이터를 동일 설정으로 재측정했다. 새로운 독립 확증은 아니다.','',
           '| 모델 | 가중 MSE | Spearman | 부호 정확도 |','|---|---:|---:|---:|']
    for m,v in metrics.items():lines.append(f"| {m} | {v['mse']:.7g} | {v['spearman']:.4f} | {v['sign_accuracy']:.4f} |")
    lines += ['','## 수정된 측정 경로','',json.dumps(drift,ensure_ascii=False),'','## 사전 비교','']
    for k,v in comparisons.items():lines.append(f"- {k}: {v['decision']}, MSE 감소 {v['relative_mse_reduction']:.2%}, CI {v['simultaneous_ci']}")
    lines += ['','## 동일 예산 bundle','',json.dumps(safety,ensure_ascii=False,indent=2),'',
              '유의한 개선을 확인하지 못한 경우 해당 예측기의 근거 부족으로 해석한다. 역할 분리 전체를 기각하지 않는다.',
              'Bootstrap은 고정된 OOF 모델에 조건부이며, 학습 표본 변동 전체를 포함하지 않는다.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(ROOT/'decision.json',{'status':'complete','comparisons':comparisons,'scope':'this predictor/target only','role_separation':'retained','further_method_search':'not automated'})
    print(json.dumps({'event':'analysis_complete','comparisons':comparisons}),flush=True)

if __name__=='__main__':main()
