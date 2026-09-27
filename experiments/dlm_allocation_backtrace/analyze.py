"""CPU-only trace of existing allocation statistics. Never builds pruning masks."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import rankdata, spearmanr
from experiments.dlm_dual_role_allocation.io import atomic_write_json as write, file_sha256 as sha

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]; EXP=REPO/'experiments'
SNAP=Path('/home/tmluser1/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2')
SOURCES={}
READ_TYPES={'q_proj','k_proj','v_proj','up_proj','ff_proj'}

def read(path):
    path=Path(path); SOURCES[str(path)]=sha(path)
    return json.loads(path.read_text())

def rho(x,y):
    x,y=np.asarray(x),np.asarray(y)
    if np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(spearmanr(x,y).statistic)

def profile(v):
    v=np.asarray(v,dtype=float)
    return dict(depth_rho=rho(np.arange(32),v),quartile_means=v.reshape(4,8).mean(1).tolist(),
                min=float(v.min()),max=float(v.max()),values=v.tolist())

def factorization(weight,a):
    w=weight.float().abs(); r=a.float().sqrt()
    mw=float(w.double().mean()); ma=float(r.double().mean())
    q=float((w.double().mean(0)*r.double()).mean())
    return dict(mean_abs_weight=mw,mean_channel_rms=ma,alignment=q/(mw*ma),mean_score=q,
                input_energy_per_channel=float(a.double().mean()))

def normalized_mapping(q,width,sign):
    q=np.asarray(q,float)
    return .65+sign*2*width*(q-q.mean())/np.ptp(q)

def event(stage,**kw):
    p=dict(stage=stage,time=time.time(),**kw);write(ROOT/'progress.json',p);print(json.dumps(p),flush=True)

def association(values,byseq,draws):
    """Exploratory CI resamples sequences only, conditionally on frozen scores."""
    x=rankdata(values); x=x-x.mean()
    targets=byseq[:,draws].mean(2)
    y=rankdata(targets,axis=0);y-=y.mean(0)
    boot=(x@y)/(np.linalg.norm(x)*np.linalg.norm(y,axis=0))
    return dict(rho=rho(values,byseq.mean(1)),conditional_ci95=np.quantile(boot,[.025,.975]).tolist(),
                per_sequence=[rho(values,byseq[:,i]) for i in range(8)])

def main():
    torch.set_num_threads(8); started=time.time();event('verify_sources')
    from experiments.dlm_allocation_sequential65 import run as seqrun
    from experiments.dlm_lsa_projection65 import run as lsacrun
    seqrun.validate();lsacrun.validate()
    v=read(EXP/'dlm_lsa_mask_mismatch/verification.json')
    for p,h in v['sources'].items():assert sha(p)==h,p
    ref=read(EXP/'projection_capacity_allocation_65/candidate_mask_manifest.json')['entries']
    names=[r['name'] for r in ref];N=np.array([r['weights'] for r in ref]);N2=N.reshape(32,7)
    types=np.array([n.split('.')[1] for n in names]); layer=np.arange(224)//7
    def block(v):return np.asarray(v).reshape(32,7).mean(1)
    def weighted(v):return (np.asarray(v).reshape(32,7)*N2).sum(1)/N2.sum(1)
    stats_path=EXP/'cgq_wanda_structured_diagnostic/sufficient_statistics.pt'
    SOURCES[str(stats_path)]=sha(stats_path)
    stats=torch.load(stats_path,map_location='cpu',weights_only=False)['statistics']
    owl=read(EXP/'dlm_owl65/allocation.json')['blocks']
    dlp=np.array([r['mean'] for r in read(EXP/'dlm_allocation_baselines65/dlp/statistics.json')['rows']])
    lsa_rows=read(EXP/'dlm_allocation_baselines65/lsa/statistics.json')['rows']
    alpha_rows=read(EXP/'dlm_allocation_baselines65/alpha/statistics.json')['rows']
    assert names==[r['name'] for r in lsa_rows]==[r['name'] for r in alpha_rows]
    super_path=EXP/'dlm_super_outlier_statistics/projection_statistics.csv'
    SOURCES[str(super_path)]=sha(super_path)
    with super_path.open() as f:super_rows=list(csv.DictReader(f))
    assert names==[r['name'] for r in super_rows]
    super_cfg=read(EXP/'dlm_super_outlier_statistics/config.json')
    for p,h in super_cfg['sources'].items():assert sha(REPO/p)==h,p
    index=read(SNAP/'model.safetensors.index.json')['weight_map']
    for file in sorted(set(index.values())):
        p=SNAP/file;digest=sha(p);assert digest==p.resolve().name
        SOURCES[str(p)]=digest
    cal=read(EXP/'dlm_loss_aggregation/exp004/calibration_state_manifest.json')['states']
    sequence=np.array([r['sequence_index'] for r in cal])
    damage=[]
    for i,n in enumerate(names):
        r=read(EXP/'dlm_lsa_mask_mismatch/damage'/f'{i:03d}.json');assert r['name']==n
        damage.append(r['delta_ce'])
    damage=np.array(damage);assert damage.shape==(224,80)
    byseq=np.stack([damage[:,sequence==q].mean(1) for q in range(8)],axis=1)
    features=[]; owl_blocks=[]
    for b in range(32):
        event('score_decomposition',completed=b,total=32)
        scores=[]; fs=[]
        for i in range(b*7,b*7+7):
            name=names[i];typ=types[i];key=f'model.transformer.blocks.{b}.{typ}.weight'
            with safe_open(SNAP/index[key],framework='pt',device='cpu') as sf:w=sf.get_tensor(key).float()
            assert list(w.shape)==ref[i]['shape']
            a=stats[name]['overall_uniform'].float();f=factorization(w,a)
            f.update(name=name,layer=b,type=str(typ),weights=int(N[i]))
            fs.append(f);scores.append(w.abs()*a.sqrt()[None,:]);del w
        q=np.array([f['mean_score'] for f in fs]);sizes=N2[b];total=int(sizes.sum())
        pooled=float(np.average(q,weights=sizes));assert abs(pooled/dlp[b]-1)<2e-6
        threshold=owl[b]['threshold'];assert abs(threshold/(5*pooled)-1)<2e-6
        counts=[int((s>threshold).sum()) for s in scores]
        assert abs(sum(counts)-owl[b]['outlier_count'])<=max(5,owl[b]['outlier_count']*2e-6)
        super_count=0;super_sum=0.;removed=0
        for s,f in zip(scores,fs):
            if f['type'] in READ_TYPES:
                super_count+=int((s[:,3848]>threshold).sum());super_sum+=float(s[:,3848].double().sum());removed+=s.shape[0]
        threshold_excl=5*(pooled*total-super_sum)/(total-removed)
        counts_excl=[]
        for s,f in zip(scores,fs):
            c=int((s>threshold_excl).sum())
            if f['type'] in READ_TYPES:c-=int((s[:,3848]>threshold_excl).sum())
            counts_excl.append(c)
        norms=np.array([f['mean_channel_rms'] for f in fs])
        normalized_threshold=5*np.average(q/norms,weights=sizes)
        normalized_counts=[int((s>normalized_threshold*z).sum()) for s,z in zip(scores,norms)]
        own_counts=[int((s>5*f['mean_score']).sum()) for s,f in zip(scores,fs)]
        for f,c,ce,cn,co in zip(fs,counts,counts_excl,normalized_counts,own_counts):
            f.update(owl_count=c,owl_excluded_count=ce,owl_activation_normalized_count=cn,owl_own_threshold_count=co)
        features.extend(fs)
        owl_blocks.append(dict(block=b,original_percent=100*sum(counts)/total,
            original_threshold=threshold,super_outlier_fraction=super_count/sum(counts),
            super_score_mass_fraction=super_sum/(pooled*total),
            excluded_fixed_threshold_percent=100*(sum(counts)-super_count)/(total-removed),
            excluded_rethreshold_percent=100*sum(counts_excl)/(total-removed),
            activation_normalized_percent=100*sum(normalized_counts)/total,
            own_projection_threshold_percent=100*sum(own_counts)/total))
        write(ROOT/'collection.json',dict(features=features,owl_blocks=owl_blocks))
        del scores
    event('analyze')
    fields={k:np.array([f[k] for f in features]) for k in ['mean_score','mean_abs_weight','mean_channel_rms','alignment','input_energy_per_channel']}
    q=fields['mean_score'];wm=fields['mean_abs_weight'];am=fields['mean_channel_rms']
    assert np.allclose(q,wm*am*fields['alignment'],rtol=1e-12)
    dlp_cf=dict(original=weighted(q),remove_activation_scale=weighted(q/am),
                remove_weight_scale=weighted(q/wm),remove_both_scales=weighted(q/(wm*am)))
    block_factors=dict(weight=weighted(wm),activation=weighted(am))
    block_factors['alignment']=weighted(q)/(block_factors['weight']*block_factors['activation'])
    slope=lambda x:float(np.polyfit(np.arange(32),np.log(x),1)[0])
    dlp_factors={k:dict(profile=profile(x),log_depth_slope=slope(x)) for k,x in block_factors.items()}
    assert abs(sum(x['log_depth_slope'] for x in dlp_factors.values())-slope(weighted(q)))<1e-10
    gap_contrib={str(t):float((q[types==t][24:].mean()-q[types==t][:8].mean())*N[types==t][0]/N2[0].sum()) for t in sorted(set(types))}
    assert abs(sum(gap_contrib.values())-(weighted(q)[24:].mean()-weighted(q)[:8].mean()))<1e-12
    E=np.array([r['metric'] for r in lsa_rows]);alpha=np.array([r['alpha'] for r in alpha_rows])
    outenergy=np.array([float(r['output_energy']) for r in super_rows])
    lsa_cf=dict(raw=abs(block(E)),per_parameter=abs(block(E/N)),
                input_scale_removed=abs(block(E/fields['input_energy_per_channel'])),
                output_relative=abs(block(E/outenergy)))
    def type_summary(values):
        mat=np.asarray(values).reshape(32,7);full=mat.mean(1)
        return dict(by_type={str(types[j]):profile(mat[:,j]) for j in range(7)},
                    leave_one_type_out={str(types[j]):dict(depth_rho=rho(np.arange(32),np.delete(mat,j,axis=1).mean(1)),
                        original_rank_rho=rho(full,np.delete(mat,j,axis=1).mean(1))) for j in range(7)})
    fitD=np.array([r['D'] for r in alpha_rows]);tail=np.array([min(r['shape'])-a['fit_index'] for r,a in zip(ref,alpha_rows)])
    alpha_info=dict(profile=profile(block(alpha)),type_analysis=type_summary(alpha),
        alpha_vs_fit_D=rho(alpha,fitD),alpha_vs_tail_count=rho(alpha,tail),
        fit_D_quantiles=np.quantile(fitD,[0,.25,.5,.75,1]).tolist(),tail_count_quantiles=np.quantile(tail,[0,.25,.5,.75,1]).tolist(),
        limitation='No saved full eigenspectra. Tail-range refit and top-eigenvalue attribution not performed; D is a fit discrepancy, not confidence interval.')
    raw_proxies=dict(owl=np.array([r['original_percent'] for r in owl_blocks]),dlp=weighted(q),lsa=abs(block(E)),alpha=block(alpha))
    actual={};mapping_checks={}
    for method in ('uniform','owl','dlp','alpha','lsa','dsa'):
        folder=EXP/'dlm_allocation_sequential65'/method;m=read(folder/'mask_manifest.json');res=read(folder/'results.json')
        assert res['manifest_sha256']==sha(folder/'mask_manifest.json')
        assert [r['name'] for r in m['entries']]==names
        counts=np.array([r['selected_mask']['pruned'] for r in m['entries']]);assert counts.sum()==4536008704
        sp=counts.reshape(32,7).sum(1)/N2.sum(1);actual[method]=dict(profile=profile(sp),correct=res['correct'])
        if method in ('owl','dlp','lsa'):
            width={'owl':.08,'dlp':.15,'lsa':.1}[method];sign=-1 if method=='owl' else 1
            ideal=normalized_mapping(raw_proxies[method],width,sign)
            mapping_checks[method]=dict(raw_to_sparsity_rho=rho(raw_proxies[method],sp),
                formula_vs_rounded_max_pp=float(abs(ideal-sp).max()*100),full_range_pp=2*width*100,
                raw_min_layer=int(raw_proxies[method].argmin()),raw_max_layer=int(raw_proxies[method].argmax()),
                middle80_fraction_of_range=float(np.ptp(np.quantile(raw_proxies[method],[.1,.9]))/np.ptp(raw_proxies[method])))
    rng=np.random.default_rng(2026);draws=rng.integers(0,8,(1000,8));blockseq=byseq.reshape(32,7,8).mean(1)
    tests={k:association(x,blockseq,draws) for k,x in raw_proxies.items()}
    tests.update({f'lsa_{k}':association(x,blockseq,draws) for k,x in lsa_cf.items() if k!='raw'})
    tests.update({f'dlp_{k}':association(x,blockseq,draws) for k,x in dlp_cf.items() if k!='original'})
    result=dict(status='complete',elapsed_seconds=time.time()-started,
        scope='Exploratory CPU statistics audit. No new pruning, model forward or downstream.',
        actual_allocations=actual,mapping=mapping_checks,
        dlp=dict(counterfactuals={k:profile(x) for k,x in dlp_cf.items()},factorization=dlp_factors,
                 total_log_depth_slope=slope(weighted(q)),late_minus_early_contribution_by_type=gap_contrib,type_analysis=type_summary(q)),
        owl={k:profile([r[k] for r in owl_blocks]) for k in owl_blocks[0] if k!='block'},
        lsa=dict(counterfactuals={k:profile(x) for k,x in lsa_cf.items()},type_analysis=type_summary(E),
                 raw_vs_output_energy_rho=rho(E,outenergy),raw_vs_input_energy_rho=rho(E,fields['input_energy_per_channel']),
                 negative_metric_count=int((E<0).sum()),normalization_note='Mean state output-energy sum from frozen super-outlier collector; scale diagnostic only, not a new algorithm.'),
        alpha=alpha_info,block_proxy_vs_signed_ce_damage=tests,
        limitations=['Block target averages seven separate 50% projection interventions, not joint block pruning.',
            'CE signed changes: improvements remain negative; not absolute damage and not GSM8K.',
            'Only 8 in-calibration sequences. Bootstrap conditions on fixed proxy estimates; all intervals exploratory, not multiplicity-adjusted.',
            '50% dense-background probe is not 65% sparse-background marginal utility.',
            'No matched AR model; no proof that observed mechanism is DLM-specific.',
            'Counterfactual statistics are not deployed masks and establish no downstream improvement.',
            'Alpha fitting robustness needs full eigenvalues, not saved in this audit.'],sources=SOURCES)
    SOURCES[str(Path(__file__))]=sha(__file__)
    write(ROOT/'results.json',result)
    with (ROOT/'projection_features.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(features[0]));writer.writeheader();writer.writerows(features)
    event('complete',elapsed_seconds=result['elapsed_seconds'])

if __name__=='__main__':
    try:main()
    except BaseException as e:event('failed',error=repr(e));raise
