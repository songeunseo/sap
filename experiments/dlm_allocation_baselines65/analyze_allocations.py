"""Post-hoc allocation audit; no fitting, GPU evaluation or curve extrapolation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,binomtest

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
EXP=REPO/'experiments'
OUT=ROOT/'allocation_analysis'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rho(a,b):return None if np.ptp(a)==0 or np.ptp(b)==0 else float(spearmanr(a,b).statistic)
def holm(p):
    keys=sorted(p,key=p.get);out={};last=0.
    for i,k in enumerate(keys):last=max(last,min(1.,p[k]*(len(keys)-i)));out[k]=last
    return out

def main():
    OUT.mkdir(exist_ok=True)
    manifests={
      'Uniform':EXP/'projection_capacity_allocation_65/uniform65_mask_manifest.json',
      'Role':EXP/'dlm_dual_role_mini100/role65_mask_manifest.json',
      'Aggregate':EXP/'dlm_dual_role_mini100/aggregate65_mask_manifest.json',
      'OWL':EXP/'dlm_owl65/mask_manifest.json',
      **{m:ROOT/m/'mask_manifest.json' for m in ('dlp','dsa','alpha','lsa')}}
    predictions={
      'Uniform':EXP/'projection_capacity_allocation_65/gsm8k/uniform_100_predictions.jsonl',
      'Role':EXP/'dlm_dual_role_mini100/gsm8k/role_100_predictions.jsonl',
      'Aggregate':EXP/'dlm_dual_role_mini100/gsm8k/aggregate_100_predictions.jsonl',
      'OWL':EXP/'dlm_owl65/predictions.jsonl',
      **{m:ROOT/m/'predictions.jsonl' for m in ('dlp','dsa','alpha','lsa')}}
    curve_path=EXP/'projection_capacity_allocation_65/capacity_curves_raw.json'
    curves=read(curve_path)['projections'];reference=read(manifests['Uniform'])['entries']
    names=[e['name'] for e in reference];weights=np.array([e['weights'] for e in reference],dtype=np.int64)
    layers=np.arange(224)//7;types=np.array([n.split('.')[1] for n in names]);width=np.array([e['shape'][1] for e in reference])
    assert [c['name'] for c in curves]==names
    damage=np.array([c['curves'][3]['summary']['mean_kl'] for c in curves])
    increment=np.array([c['curves'][4]['summary']['mean_kl']-c['curves'][3]['summary']['mean_kl'] for c in curves])
    gain=np.array([(int(w*.7)-int(w*.65))*r['shape'][0] for w,r in zip(width,reference)])
    marginal=increment/gain
    top=np.argsort(-damage,kind='stable')[:56];top_cost=np.argsort(-marginal,kind='stable')[:56]
    from experiments.dlm_dual_role_mini100.run import _validate_rows
    baseline=[json.loads(x) for x in predictions['Uniform'].read_text().splitlines()]
    protocol=read(ROOT/'config.json')['protocol_hash']
    methods={};rates={};counts={};sources={str(curve_path):sha(curve_path),str(Path(__file__)):sha(__file__)}
    for method,path in manifests.items():
        m=read(path);entries=m['entries'];assert len(entries)==224 and [e['name'] for e in entries]==names
        assert all(e['shape']==r['shape'] for e,r in zip(entries,reference))
        pruned=np.array([e['selected_mask']['pruned'] for e in entries],dtype=np.int64)
        assert all(e['selected_mask']['prune_per_row']*e['shape'][0]==e['selected_mask']['pruned'] for e in entries)
        assert pruned.sum()==m['pruned']==4536008704 and weights.sum()==6979321856
        s=pruned/weights;rates[method]=s;counts[method]=pruned
        rows=[json.loads(x) for x in predictions[method].read_text().splitlines()];_validate_rows(rows,baseline,protocol)
        if method in ('dlp','dsa','alpha','lsa'):
            result=read(ROOT/method/'results.json')
            assert result['manifest_sha256']==sha(path) and result['predictions_sha256']==sha(predictions[method])
            assert result['config_sha256']==sha(ROOT/method/'config.json')
            assert sum(x['correct'] for x in rows)==result['correct']
        new=sum(not b['correct'] and r['correct'] for b,r in zip(baseline,rows))
        lost=sum(b['correct'] and not r['correct'] for b,r in zip(baseline,rows))
        block_s=pruned.reshape(32,7).sum(1)/weights.reshape(32,7).sum(1)
        delta=pruned-counts['Uniform']
        data=dict(correct=sum(x['correct'] for x in rows),global_sparsity=float(pruned.sum()/weights.sum()),
          min=float(s.min()),max=float(s.max()),layer_spearman=rho(np.arange(32),block_s),
          layer_parameter_weighted=block_s.tolist(),quartile_parameter_weighted=[float(pruned[layers//8==q].sum()/weights[layers//8==q].sum()) for q in range(4)],
          quartile_projection_mean=[float(s[layers//8==q].mean()) for q in range(4)],
          type_sparsity={t:float(pruned[types==t].sum()/weights[types==t].sum()) for t in sorted(set(types))},
          within_layer_max_spread=float(np.ptp(s.reshape(32,7),axis=1).max()),
          density_vs_D65_spearman=None if method=='Uniform' else rho(1-s,damage),
          density_vs_marginal_per_parameter_spearman=None if method=='Uniform' else rho(1-s,marginal),
          above75_projections=int((s>.75).sum()),below50_projections=int((s<.5-1e-12).sum()),
          top_D65_quartile_extra_pruned=int(delta[top].sum()),top_marginal_quartile_extra_pruned=int(delta[top_cost].sum()),
          changed_projection_count=int((delta!=0).sum()),restored_vs_uniform=int(-delta[delta<0].sum()),
          added_vs_uniform=int(delta[delta>0].sum()),
          paired_vs_uniform=dict(gained=new,lost=lost,p=float(binomtest(new,new+lost,.5).pvalue) if new+lost else 1.))
        assert data['restored_vs_uniform']==data['added_vs_uniform']
        methods[method]=data;sources[str(path)]=sha(path);sources[str(predictions[method])]=sha(predictions[method])
    adjusted=holm({m:methods[m]['paired_vs_uniform']['p'] for m in ('dlp','dsa','alpha','lsa')})
    for m,p in adjusted.items():methods[m]['paired_vs_uniform']['holm_four']=p
    dsa=read(ROOT/'dsa/statistics.json')['rows']
    vulnerable=[dict(name=names[i],D65=float(damage[i]),sparsity={m:float(s[i]) for m,s in rates.items()}) for i in np.argsort(-damage,kind='stable')[:10]]
    result=dict(status='complete_offline',methods=methods,top10_D65=vulnerable,
        dsa_nan_blocks=[r['block'] for r in dsa if r['nan_mapped_to_zero']],sources=sources,
        limitations=['Post-hoc descriptive; no causal intervention.','Capacity curves single-module dense-background; not joint-model damage.',
        'No extrapolation outside50-75; D65 and marginal sensitivity only characterize budget direction.',
        'Small selected set of methods, no cross-method accuracy correlation inference.',
        'Same mini repeatedly used for development; no fresh confirmatory holdout.'])
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (OUT/'projection_allocations.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['projection','weights','D65','marginal_KL_per_parameter',*rates])
        for i,n in enumerate(names):writer.writerow([n,int(weights[i]),damage[i],marginal[i],*[s[i] for s in rates.values()]])
    lines=['# Allocation 실패 패턴: 사후 offline 분석','',
      '## 설정과 범위','',
      '동일224 projection·실제 제거수4,536,008,704·동일mini100 예측 검증. 새GPU/성능실험 없음. 모든 비율은 실제 정수 mask count 기준.',
      'Layer 구간은 parameter 가중 평균. Role의 기존 projection 단순평균과 혼동하지 않는다. Obsidian 연결 불가로 로컬 기록만 저장.','',
      '## 전체 allocation','',
      '|방법|mini100|B00–07|B08–15|B16–23|B24–31|최대 sparsity|layer ρ|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m,d in methods.items():
        q='|'.join(f'{100*x:.2f}%' for x in d['quartile_parameter_weighted']);r='NA' if d['layer_spearman'] is None else f'{d["layer_spearman"]:.3f}'
        lines.append(f'|{m}|{d["correct"]}|{q}|{100*d["max"]:.2f}%|{r}|')
    lines+=['','## Projection type','', '|방법|'+'|'.join(sorted(set(types)))+'|','|---|'+'---:|'*7]
    for m,d in methods.items():lines.append('|'+m+'|'+'|'.join(f'{v*100:.2f}%' for v in d['type_sparsity'].values())+'|')
    lines+=['','## 기존 functional vulnerability와 예산 방향','',
      'Top56은 D65 또는65→70 추가 제거 parameter당 KL cost 상위25%. 양수는 Uniform보다 그 집합에서 더 제거했다는 뜻. 실제 joint KL 증가량이 아니다.','',
      '|방법|density vs D65 ρ|density vs marginal cost ρ|Top56 D65 추가제거(M)|Top56 marginal 추가제거(M)|75%초과 projection 수|',
      '|---|---:|---:|---:|---:|---:|']
    def fmt(x):return 'NA' if x is None else f'{x:.3f}'
    for m,d in methods.items():lines.append(f'|{m}|{fmt(d["density_vs_D65_spearman"])}|{fmt(d["density_vs_marginal_per_parameter_spearman"])}|{d["top_D65_quartile_extra_pruned"]/1e6:.2f}|{d["top_marginal_quartile_extra_pruned"]/1e6:.2f}|{d["above75_projections"]}|')
    lines+=['','## D65 상위 projection','', '|projection|D65|'+'|'.join(rates)+'|','|---|---:|'+'---:|'*len(rates)]
    for r in vulnerable:lines.append('|'+r['name']+f'|{r["D65"]:.5f}|'+'|'.join(f'{v*100:.2f}%' for v in r['sparsity'].values())+'|')
    lines+=['','## Paired mini100 결과','', '|방법|새 정답|소실 정답|exact p|Holm(4)|','|---|---:|---:|---:|---:|']
    for m in ('dlp','dsa','alpha','lsa'):
        d=methods[m]['paired_vs_uniform'];lines.append(f'|{m}|{d["gained"]}|{d["lost"]}|{d["p"]:.6f}|{d["holm_four"]:.6f}|')
    lines+=['','## 해석상의 제한','',*['- '+x for x in result['limitations']],
      '- Uniform density 상관은 정수 row-floor 차이만 반영하므로 NA로 처리한다.',
      '- DSA NaN→0 blocks: '+str(result['dsa_nan_blocks']),
      '- Layer pattern, projection별 granularity, sparsity 범위, proxy가 함께 다르므로 Role의 성능 차이를 masked/unmasked 분리 하나로 귀속할 수 없다.',
      '- 불량 candidate를 보고 curve나allocation을 수정하지 않았다.','']
    lines += ['## 핵심 해석과 다음 판단','',
      '1. 네 새 baseline 모두 early 보호/late 추가 제거 패턴. Role/Aggregate는 앞8block에서 더 많이 제거한다. 배분 방향 불일치가 공통 단서다.',
      '2. DLP/Alpha/LSA는75%를 넘는 projection이 존재한다. 동일 global budget이 동일 local aggressiveness를 의미하지 않는다.',
      '3. Role은 모든 late projection을 보호하지 않는다. D65 상위 B31 MLP 세 개는 모두75%다. 따라서 취약 late MLP 보호로 Role 성공을 단정할 수 없다.',
      '4. Role은 type별로 attn_out/v를 상대적으로 보호한다. 네 baseline은 같은block 안7projection에 거의 동일 sparsity라 이런 차등 배분이 없다.',
      '5. D65 절대 damage와 parameter당 marginal cost는 다른 기준이다. Role도 D65 상위56개에서 Uniform보다 더 제거하지만 marginal 상위56개에서는 덜 제거한다.',
      '6. Aggregate도 비슷한 depth/type 구조를 가지며19점이다. 이 비교는 masked/unmasked 또는 max의 고유 효과를 식별하지 않는다.',
      '7. 다음 개입으로 layer 총예산을 유지한 type 균일화가 within-layer allocation 효과를 분리하는 통제가 될 수 있다. 실행/후보 선택은 하지 않았다.',
      '8. 이번 결과로 전체 AR allocation 실패나 특정부위 pruning의 인과효과를 확정하지 않는다.','']
    (OUT/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))

if __name__=='__main__':main()
