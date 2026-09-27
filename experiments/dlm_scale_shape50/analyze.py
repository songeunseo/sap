"""Offline figures and paired article-level analysis for the frozen diagnostic."""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.dlm_ppl50.sequential import read,write,sha
from experiments.dlm_wikitext_ppl.core import digest
ROOT=Path(__file__).resolve().parent


def distribution_report():
    rows=read(ROOT/'statistics.json')['rows']
    flat=[]
    for row in rows:
        flat.append(dict(name=row['name'],layer=row['layer'],type=row['type'],weights=row['weights'],
            weight_mean=row['weight']['mean_abs'],rms_mean=row['channel_rms_dense']['mean_abs'],
            score_mean=row['score_dense']['mean_abs'],score_log_variance=row['score_dense']['positive_log_variance'],
            score_zero_fraction=row['score_dense']['zero_fraction'],outlier_fraction=row['score_dense']['outlier5_fraction'],
            outlier_energy=row['score_dense']['outlier5_energy_fraction'],
            clean_score_mean=row['score_clean']['mean_abs'],clean_log_variance=row['score_clean']['positive_log_variance']))
    with (ROOT/'projection_features.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    types=sorted({r['type'] for r in rows})
    colors=['#2166ac','#67a9cf','#ef8a62','#b2182b']
    for field,title in [('score_dense','Dense Wanda score'),('weight','Absolute weight'),('activation_samples_dense','Sampled absolute input activation')]:
        for normalized in (False,True):
            fig,axes=plt.subplots(2,4,figsize=(14,6),layout='constrained')
            for ax,typ in zip(axes.flat,types):
                for q in range(4):
                    subset=[r for r in rows if r['type']==typ and r['layer']//8==q]
                    edgekey='log10_mean_normalized_edges' if normalized else 'log10_abs_edges'
                    histkey='log10_mean_normalized_hist' if normalized else 'log10_abs_hist'
                    edges=np.array(subset[0][field][edgekey])
                    # Equal projection mean, conditional on nonzero samples within displayed range.
                    hist=np.array([r[field][histkey] for r in subset],float)
                    hist=hist/np.maximum(hist.sum(1,keepdims=True),1)
                    y=hist.mean(0)/np.diff(edges)
                    ax.plot((edges[1:]+edges[:-1])/2,y,label=f'blocks {q*8}-{q*8+7}',color=colors[q],lw=1.4)
                ax.set_title(typ);ax.grid(alpha=.15)
                ax.set_xlim((-3.5,2.5) if normalized else (-6,3))
                ax.set_xlabel('log10(|x| / mean |x|)' if normalized else 'log10 |x|')
                ax.set_ylabel('Sample density')
            axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
            axes.flat[-1].legend(handles,labels,loc='center')
            fig.suptitle(title+(' | mean-normalized' if normalized else ' | raw scale')+' | sampled histograms; zeros reported separately')
            fig.savefig(ROOT/f'{field}_{"normalized" if normalized else "raw"}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for typ in types:
        subset=[r for r in flat if r['type']==typ]
        for ax,key,label in zip(axes,['score_mean','score_log_variance','outlier_energy'],['Mean Wanda score','Var(log positive Wanda score)','Energy share above 5 x mean']):
            ax.plot([r['layer'] for r in subset],[r[key] for r in subset],label=typ);ax.set_title(label)
            ax.set_xlabel('Block');ax.grid(alpha=.2)
    axes[0].set_yscale('log');axes[-1].legend(fontsize=8)
    fig.savefig(ROOT/'scale_shape_profiles.png',dpi=170);plt.close(fig)
    rho=lambda a,b:float(spearmanr(a,b).statistic)
    d=dict(clean_corrupted_mean_rho=rho([r['score_mean'] for r in flat],[r['clean_score_mean'] for r in flat]),
        clean_corrupted_logvariance_rho=rho([r['score_log_variance'] for r in flat],[r['clean_log_variance'] for r in flat]),
        score_mean_depth_rho=rho([r['score_mean'] for r in flat],[r['layer'] for r in flat]),
        score_logvariance_depth_rho=rho([r['score_log_variance'] for r in flat],[r['layer'] for r in flat]))
    write(ROOT/'distribution_summary.json',d)
    (ROOT/'distribution_report.md').write_text('# Distribution survey\n\nAll224 projections, original dense weights, frozen80 corrupted inputs and same8 clean inputs. Exact moments of weight/channel-RMS/score; random scalar activation samples1024 per state. Quantile/histogram summaries use at most131072 values and are not full distributions. The positive-log variance excludes zero scores; zero mass is separately retained. Plots show equal-projection conditional nonzero densities; use weights column in CSV for parameter-weighted aggregation.\n\n'+
        '\n'.join(f'- {k}: {v:.6f}' for k,v in d.items())+'\n\nThese are descriptive associations; clean inputs are a distribution control, not a clean-input likelihood target or an AR model control.\n')


def main():
    cfg=read(ROOT/'config.json');pairs=read(ROOT/'selection.json')['pairs'];sources=cfg['evaluation_blocks']
    def load(label):
        return np.array([read(ROOT/'evaluation'/label/f'{digest(s["block_id"])[:20]}.json')['sample_token_nelbo'] for s in sources])
    base=load('baseline');raw=np.stack([load(p['id']+'_raw') for p in pairs]);shape=np.stack([load(p['id']+'_shape') for p in pairs])
    if raw.shape!=(12,16,128) or not np.isfinite(raw).all() or not np.isfinite(shape).all():raise RuntimeError('Incomplete data')
    delta=raw-shape
    rng=np.random.default_rng(cfg['seed']);draws=rng.integers(0,16,size=(5000,16))
    def stats(ids,alpha=.05):
        values=delta[ids].mean(0);article=values.mean(1)
        return dict(raw_minus_shape=float(article.mean()),
            paired_article_ci=np.quantile(article[draws].mean(1),[alpha/2,1-alpha/2]).tolist(),
            ci_level=1-alpha,paired_mc_se=float(np.sqrt(values.var(1,ddof=1).sum()/128/16**2)),
            group_A=float(article[:8].mean()),group_B=float(article[8:].mean()),
            raw_minus_baseline=float((raw[ids]-base).mean()),shape_minus_baseline=float((shape[ids]-base).mean()),
            article_differences=article.tolist())
    primary=stats(list(range(12)))
    strata={key:stats([i for i,p in enumerate(pairs) if p['stratum']==key],alpha=.05/3) for key in sorted({p['stratum'] for p in pairs})}
    individual=[]
    for i,p in enumerate(pairs):
        individual.append(dict(**p,raw_minus_shape=float(delta[i].mean()),raw_minus_baseline=float((raw[i]-base).mean()),
                               shape_minus_baseline=float((shape[i]-base).mean())))
    result=dict(status='complete',primary=primary,strata=strata,pairs=individual,
        interpretation='Positive raw-minus-shape favors pruning the higher log-dispersion projection; negative favors higher mean. Signed changes retained. Feature-selected pairs, not a general pruning-policy score.',
        uncertainty='Paired article bootstrap conditions on this selected pair set and proxy estimates; MC SE covers masking noise only. Same16 development articles; no final-test or downstream claim.',
        config_sha256=sha(ROOT/'config.json'),statistics_sha256=sha(ROOT/'statistics.json'),selection_sha256=sha(ROOT/'selection.json'))
    write(ROOT/'results.json',result)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    vals=[r['raw_minus_shape'] for r in individual]
    ax.barh(range(12),vals,color=['#2166ac' if v>0 else '#b2182b' for v in vals])
    ax.set_yticks(range(12),[p['id']+' / '+p['stratum'].replace('_',' ') for p in pairs],fontsize=8)
    ax.axvline(0,color='black',lw=.8);ax.set_xlabel('NELBO(raw direction) - NELBO(shape direction); positive favors shape')
    ax.set_title('Equal-budget opposite exchanges around Uniform50 | descriptive pair means')
    fig.savefig(ROOT/'exchange_results.png',dpi=170);plt.close(fig)
    lo,hi=primary['paired_article_ci']
    rows='\n'.join(f"| {k} | {v['raw_minus_shape']:+.7f} | [{v['paired_article_ci'][0]:+.7f}, {v['paired_article_ci'][1]:+.7f}] | {v['raw_minus_baseline']:+.7f} | {v['shape_minus_baseline']:+.7f} |" for k,v in strata.items())
    text=f'''# Scale versus shape at50%: paired intervention diagnostic

## Objective / hypothesis
Test whether mean Wanda score and positive-log dispersion recommend different useful pruning exchanges. Shape may improve loss under matched budgets; no direction is assumed. This is a diagnostic of selected decisions, not a new full allocation method.

## Actual setup
Frozen LLaDA revision and80-state calibration. Same8 clean inputs added for distributions only. All224 sequential Uniform50 masks exactly reproduced. Twelve feature-selected, disjoint pairs: four early/late same type, four nearby-depth same type, four same-layer same shape. Each direction moves exactly786432 pruned weights between two projections; global budget remains3489660928. Same original row-wise Wanda ranking in both directions; no prefix recalibration per intervention. Sixteen distinct512-token validation article chunks, exact-k MC128, original shared seed/draws, baseline +24 variants. These are previously used development-validation articles, not new final-test documents. Two fixed8-article halves are descriptive replication checks.

## Results
Primary average over all12 pairs: raw-minus-shape NELBO **{primary['raw_minus_shape']:+.7f}**, paired article95% CI [{lo:+.7f}, {hi:+.7f}], paired MC SE {primary['paired_mc_se']:.7f}. Positive favors shape. GroupA {primary['group_A']:+.7f}, groupB {primary['group_B']:+.7f}. Raw-minus-baseline {primary['raw_minus_baseline']:+.7f}; shape-minus-baseline {primary['shape_minus_baseline']:+.7f}.

| Stratum | Raw minus shape | Article98.33% CI | Raw minus baseline | Shape minus baseline |
|---|---:|---|---:|---:|
{rows}

## Interpretation
Interpret the sign and interval jointly. Cross-depth advantage alone does not show information beyond depth; same-layer comparisons hold depth exactly fixed but vary projection type. Nearby-depth pairs hold projection type fixed and restrict depth distance. The controls do not isolate a single causal activation feature or establish DLM specificity. Log dispersion is related to the existing DSA family, not a novel proxy. Histograms are sampled while recorded weight/score moments are exact for their tensors. Equal transfer counts produce different percentage changes for different matrix sizes.

## Verification
Historical dense activation and mean score controls; all224 mask hashes; baseline per-draw likelihood agreement within2e-6; exact physical weight restoration and baseline sham after every variant; config/source revalidation. See receipts.

## Decision
Finish this fixed diagnostic and retain all signed results. No automatic score inversion, new coefficient, full allocation, test or GSM8K evaluation. A useful next criterion needs one explicit mechanism and a matched ablation; neither a favorable aggregate mean nor a histogram alone establishes it.

## Artifacts
statistics.json, projection_features.csv, distribution_summary.json, selection.json, evaluation/, restoration/, baseline_control.json, results.json, and static PNGs. Large input samples and boundary changes stored under /DATA/tmluser1/dlm-scale-shape50.
'''
    (ROOT/'report.md').write_text(text)
    print(text,flush=True)


if __name__=='__main__':main()
