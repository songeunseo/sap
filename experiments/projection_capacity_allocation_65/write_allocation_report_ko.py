"""Render a Korean allocation-process snapshot from existing artifacts only."""
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    config = read('config.json')
    a = read('allocation.json')
    s = read('allocation_summary.json')
    marginal = read('marginal_costs.json')
    m = marginal['summary']
    raw = read('capacity_curves_raw.json')['projections']
    held = read('heldout_dlm_results.json')
    gate = held['gate']
    downstream = read('downstream.json') if (ROOT / 'downstream.json').exists() else None
    verification = read('state_verification.json')
    names = {r['module_index']: r['name'] for r in a['assignments']}
    byname = {r['name']: r for r in a['assignments']}
    timestamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    n = a['weights']
    initial = n // 2
    theoretical = .65 * n
    changed = sum(r['assigned_sparsity'] != .65 for r in a['assignments'])
    affected = sum(v > 0 for v in m['per_projection_violation_counts'].values())
    u = held['methods']['uniform']['summary']
    c = held['methods']['capacity']['summary']
    improvement = (u['mean_kl'] - c['mean_kl']) / u['mean_kl']
    lines = []

    def add(text=''):
        lines.append(text)

    def table(headers, rows):
        add('| ' + ' | '.join(headers) + ' |')
        add('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        for row in rows:
            add('| ' + ' | '.join(str(cell).replace('|', '\\|') for cell in row) + ' |')
        add()

    add('# Projection별 기능적 용량 기반 sparsity 배분 @ 65% — 중간 과정 보고서')
    add()
    add(f'작성 시점: {timestamp}. 기존 실험 산출물을 읽어 작성한 시점별 보고서이며, 실행 중인 평가나 allocation을 변경하지 않았다.')
    add()
    add('## 1. 현재까지의 결론')
    add()
    add(f'224개 projection의 단독 가지치기 손상 곡선을 측정한 뒤, 같은 총 가지치기 예산 안에서 sparsity를 재배분했다. '
        f'모든 projection을 동시에 가지치기한 held-out 평가에서 평균 KL은 {u["mean_kl"]:.6f}에서 {c["mean_kl"]:.6f}로 '
        f'{improvement:.2%} 감소했다. 사전 등록한 DLM kill gate는 통과했다.')
    if downstream:
        mini = downstream['evaluations'][0]
        add(f'고정 GSM8K mini-100에서도 Uniform {mini["uniform"]["correct"]}/100, Capacity {mini["capacity"]["correct"]}/100으로 '
            '방향성 개선을 확인했다. 전체 GSM8K의 최종 결과와는 구분해야 한다.')
    add()
    add('배분 과정에서 앞쪽 레이어를 더 많이 가지치기하는 경향과 projection 유형별 차이가 나타났다. '
        '따라서 이번 결과는 allocation의 효용을 지지하지만, 새로운 DLM 전용 신호나 독창적인 레이어 패턴을 발견했다는 의미는 아니다.')
    add()
    add('## 2. 무엇을 고정하고 무엇을 바꿨는가')
    add()
    table(['항목', '설정'], [
        ['모델', 'GSAI-ML/LLaDA-8B-Base'],
        ['revision', '`0f2787f2d87eac5eed8a087d5ecd24277e6255b2`'],
        ['대상', '32 blocks × 7 Linear = 224 projections; 기존 모듈 이름·순서 유지'],
        ['명목상 전역 sparsity', '65%; 결과를 보고 변경하지 않음'],
        ['projection별 후보', '50 / 55 / 60 / 65 / 70 / 75%'],
        ['projection 내부 선택', '`abs(W.float()) * sqrt(A.float())`, 행별 stable 오름차순 정렬'],
        ['A의 의미', '기존 uniform sweep의 80개 DLM 상태에서 token별 X² 합을 상태 평균한 unweighted `overall_uniform`'],
        ['실험에서 바뀐 것', '각 projection에서 얼마나 가지치기할지'],
        ['배분에 사용한 값', '단독 projection 가지치기의 평균 masked-token Dense||Sparse KL만 사용'],
    ])
    add('**보정 데이터 구분:** 이 실험의 Standard Wanda는 인용된 GSM8K uniform sweep과 동일한 DLM 상태 기반 activation 보정을 사용했다. '
        '기존 causal failure map의 clean 8-sequence Wanda 보정과는 다르다. 실제로 65% 마스크 224개의 해시가 sweep과 모두 일치했다. '
        '캐시 파일은 과거 다른 진단 실험 디렉터리에 있지만, 읽은 activation 값은 `overall_uniform`뿐이며 CGQ·confidence·gradient 등으로 배분하지 않았다.')
    add()
    add('## 3. 데이터 분리 및 실행 경로 검증')
    add()
    table(['용도', '구성', '상태 수'], [
        ['용량 곡선 측정', 'WikiText-2 8 spans × t={0.05, 0.15, …, 0.95}', 80],
        ['held-out 평가', '다른 8 spans × t={0.1, 0.3, 0.5, 0.7, 0.9}', 40],
    ])
    add('두 manifest의 저장 digest를 다시 계산해 일치 여부를 확인했고, 120개 corruption mask를 기존 seed로 재생성하여 모두 정확히 일치함을 확인했다. '
        'WikiText token stream에서 원래 span 시작·끝 위치도 복원했다. 두 split 간 interval overlap은 0이며, 각 split 내부의 span끼리도 겹치지 않는다. '
        '단순히 sequence ID가 다르다는 이유로 분리됐다고 판단하지 않았다.')
    add()
    for split in verification['splits']:
        add(f'- {split["states"]}개 상태 digest: `{split["state_sha256"]}`')
    add()
    add('단독 projection 측정에서는 해당 block 직전의 dense hidden state를 캐시하고, dense sham 1개와 후보 sparsity 6개를 같은 7개 batch 경로로 실행했다. '
        '해당 Linear 하나만 후보 마스크로 치환하고 나머지 223개는 dense로 유지했다. 비교 기준은 같은 경로의 dense sham이다. '
        '첫 대상에서 같은 batch의 full forward와 suffix forward를 비교한 최대 logit 오차는 0이었다. '
        '이 검사는 모든 projection에 대한 별도 full-forward 동일성 검사는 아니다.')
    add()
    add('전체 sparse 모델 비교는 dense·Uniform·Capacity 모두 동일한 batch-one full forward와 실제 weight masking을 사용했다. '
        f'held-out dense forward를 반복한 sham 검사 최대 오차도 {held["dense_sham_max_abs"]}이었다.')
    add()
    add('## 4. 224 × 6 용량 곡선의 측정 방식')
    add()
    add('각 상태에서 masked token에 대한 `KL(p_dense || p_single_projection_sparse)`를 평균하고, 그 상태 평균들을 80개 상태에 동일 가중치로 평균해 `D_g(r)`를 정의했다. '
        '즉 masked token이 많은 상태에 더 큰 가중치를 주는 pooled-token 평균과는 구분된다. pooled-token 평균과 median은 별도 진단값으로 저장했다.')
    add()
    add('총 1,344개 projection–sparsity 조합을 각각 80개 상태에서 평가했다. 아래 분포는 각 sparsity에서 projection 224개의 `D_g(r)` 분포이다. '
        'CV는 표준편차/평균이며, 값이 클수록 projection 간 손상 차이가 크다.')
    add()
    table(['sparsity', '평균 KL', '중앙값', 'p10', 'p90', '최댓값', 'CV'], [
        [f'{float(r):.0%}'] + [f'{d[k]:.6g}' for k in ['mean','median','p10','p90','max','coefficient_of_variation']]
        for r, d in m['damage_by_sparsity'].items()])
    add('sparsity가 커질수록 평균 손상뿐 아니라 projection 간 편차도 커졌다. 이는 동일 sparsity가 모든 projection에 같은 기능적 부담을 주지 않는다는 증거다. '
        '다만 이 결과만으로는 모든 projection을 동시에 가지치기했을 때의 성능 개선을 보장하지 않는다.')
    add()
    add('### 4.1 5%p 추가 가지치기의 한계비용')
    add()
    add('원시 한계비용은 `D_g(r+0.05) − D_g(r)`다. 아래 값은 parameter 수로 나누기 전의 KL 증가량이다.')
    add()
    table(['구간', '평균 ΔKL', '중앙값', 'p10', 'p90', '최솟값', '최댓값', 'CV'], [
        [label] + [f'{d[k]:.6g}' for k in ['mean','median','p10','p90','min','max','coefficient_of_variation']]
        for label, d in m['marginal_by_increment'].items()])
    add(f'전체 1,120개 increment 중 {len(m["monotonicity_violations"])}개({len(m["monotonicity_violations"])/1120:.2%})가 음수였고, '
        f'{affected}개 projection에서 발생했다. 가지치기를 더 했는데 측정 KL이 조금 낮아진 구간이다. '
        '원인을 이 실험만으로 확정할 수 없으며, 이를 유용한 보정 효과라고 해석하지 않았다. '
        '**원시 값을 그대로 사용했고 monotone envelope·clipping·수동 수정은 적용하지 않았다.**')
    add()
    table(['projection', '구간', '음수 ΔKL'], [
        [r['name'], f'{r["from"]:.0%}→{r["to"]:.0%}', f'{r["marginal_kl"]:.8g}'] for r in m['monotonicity_violations']])
    add('### 4.2 가장 비싼/저렴한 increment')
    add()
    add('아래 순위는 실제 배분 기준인 `ΔKL / (0.05 × N_g)` 순위다. 읽기 쉽게 추가 제거 parameter 백만 개당 KL 비용으로 표시했다. '
        '이는 모든 increment의 정적 순위이며, 실제 선택 순서는 선행 increment 제약 때문에 이 표와 다를 수 있다. '
        '정규화 전 ΔKL 기준 상·하위 20개도 `marginal_costs.json`에 별도 저장돼 있다.')
    add()
    for title, key in [('비싼 상위 20개', 'top20_expensive_per_parameter'), ('저렴한 상위 20개', 'top20_cheapest_per_parameter')]:
        add(f'**{title}**')
        add()
        table(['순위', 'projection', '구간', '원시 ΔKL', '백만 parameter당 비용'], [
            [i+1, r['name'], f'{r["from"]:.0%}→{r["to"]:.0%}', f'{r["marginal_kl"]:.7g}', f'{r["cost_per_nominal_parameter"]*1e6:.7g}']
            for i, r in enumerate(m[key])])
    add('## 5. 실제 allocation은 어떻게 진행됐는가')
    add()
    add('1. 모든 projection을 50%에서 시작한다.\n'
        '2. 각 projection에서 현재 sparsity의 **바로 다음 5%p increment만** 후보로 둔다. 예를 들어 50% 상태에서 60→65% increment를 먼저 고를 수 없다.\n'
        '3. 후보를 원시 `ΔKL / (0.05 × N_g)`가 작은 순으로 비교한다. 정확히 같은 비용이면 기존 module 순서로 결정한다.\n'
        '4. 선택 후 해당 projection의 다음 increment를 후보로 갱신한다.\n'
        '5. Uniform-65의 실제 제거 parameter 수와 같아질 때까지 반복한다.')
    add()
    add('**정수 예산을 위한 구현상의 추가 조건:** 순수 greedy 순서만 따르면 행별 floor 때문에 마지막 잔여 예산을 정확히 채우지 못할 수 있다. '
        '따라서 후보를 받아들이기 전에, 남은 projection별 grid 선택으로 잔여 예산에 정확히 도달할 수 있는지 정수 subset-sum 검사를 수행했다. '
        '불가능한 후보는 건너뛰고 다음으로 저렴한 후보를 검토했다. 이는 KL 계수를 조절하거나 특정 모듈을 보호하는 규칙이 아니라 '
        '사전에 기록한 정확한 예산 충족 조건이다. 다만 제약 없는 순수 greedy와 완전히 같은 알고리즘이라고 표현해서는 안 된다.')
    add()
    add('비용의 분모는 요청대로 명목상 `0.05 × N_g`를 유지했다. 실제 예산 차감에는 행별 floor 적용 뒤의 정수 제거 개수 차이를 사용했다. '
        '곡선을 convex하게 만들지 않았으므로 선택된 한계비용이 시간순으로 항상 증가하지는 않는다. 또한 이 greedy 규칙이 전역 최적 allocation을 구했다는 주장은 하지 않는다.')
    add()
    table(['단계', '실제 제거 parameter 수'], [
        ['모두 50%인 시작점', f'{initial:,}'],
        ['추가 제거해야 하는 수', f'{a["pruned"]-initial:,}'],
        ['최종 제거 수', f'{a["pruned"]:,}'],
        ['선택한 5%p increment 수', len(a['trace'])],
        ['정수 도달 가능성 검사 단위', f'{a["integer_budget_unit"]:,} parameters'],
    ])
    add('### 5.1 실제 선택 순서의 앞·뒤 구간')
    add()
    chosen = list(enumerate(a['trace'][:10], 1)) + list(enumerate(a['trace'][-10:], len(a['trace'])-9))
    table(['선택 번호', 'projection', '변경', '원시 ΔKL', '실제 추가 제거 수', '남은 제거 수'], [
        [i, names[r['module_index']], f'{r["from"]:.0%}→{r["to"]:.0%}', f'{r["marginal_kl"]:.7g}',
         f'{r["actual_parameter_gain"]:,}', f'{r["remaining"]:,}'] for i, r in chosen])
    add('### 5.2 정수 예산 때문에 건너뛴 후보')
    add()
    skips = a['feasibility_skips']
    add(f'총 {len(skips)}번의 후보 검토가 도달 불가능 판정을 받았다. 최초 발생 시점은 '
        f'{skips[0]["step"]}개 increment를 이미 선택한 뒤였다. 같은 후보를 여러 단계에서 다시 검토할 수 있으므로 28개 모듈을 제외했다는 뜻은 아니다.')
    add()
    grouped = {}
    for row in skips:
        key = (row['module_index'], row['from'])
        grouped.setdefault(key, []).append(row['step'])
    table(['projection', '보류한 다음 increment', '검토 횟수', '발생 step 범위(0 기준)'], [
        [names[i], f'{r:.0%}→{r+.05:.0%}', len(steps), f'{min(steps)}–{max(steps)}'] for (i, r), steps in grouped.items()])
    add('전체 선택 trace와 모든 skip 기록은 `allocation.json`에 보존돼 있다. 최종 예산 오차는 0이다.')
    add()
    add('## 6. 동일 예산 검증과 최종 배분')
    add()
    table(['항목', 'Uniform-65', 'Capacity-65'], [
        ['대상 parameter 수', f'{n:,}', f'{n:,}'],
        ['제거 parameter 수', f'{a["uniform_pruned"]:,}', f'{a["pruned"]:,}'],
        ['실제 전역 sparsity', f'{s["uniform_parameter_weighted_sparsity"]:.8%}', f'{s["global_parameter_weighted_sparsity"]:.8%}'],
        ['projection sparsity 단순 평균', '65.0000%', f'{s["nominal_projection_sparsity"]["mean"]:.4%}'],
        ['projection sparsity 중앙값', '65%', '65%'],
        ['projection sparsity 최솟값 / 최댓값', '65% / 65%', '50% / 75%'],
    ])
    add(f'수학적인 `0.65 × N`은 {theoretical:,.1f}이지만, 기존 Wanda의 행별 `floor(in_features × sparsity)`가 '
        f'실제로 제거하는 개수는 {a["pruned"]:,}이다. 본 실험은 후자와 정확히 맞췄다. '
        '전역 sparsity의 분모는 지정된 224개 Linear weight만이며 embedding 등 비대상 parameter를 포함하지 않는다.')
    add()
    add('projection 크기가 다르므로 224개 sparsity의 단순 평균 63.8839%와 parameter 가중 전역 sparsity 64.9921%가 다르다. '
        '예산 일치 여부는 후자로 판단했다.')
    add()
    table(['배정 sparsity', 'Uniform projection 수', 'Capacity projection 수'], [
        [f'{float(r):.0%}', 224 if float(r)==.65 else 0, count] for r, count in s['counts'].items()])
    add(f'Uniform 대비 sparsity가 달라진 projection은 {changed}/224개다. 동일 순위에 대한 중첩 마스크를 사용했으므로 '
        'sparsity가 달라진 모듈에서는 제거 범위도 달라진다. mask XOR의 별도 수치는 본 보고서에서 새로 추정하지 않았다.')
    add()
    add('### 6.1 224개 projection 전체 배분')
    add()
    types = ['q_proj','k_proj','v_proj','attn_out','up_proj','ff_proj','ff_out']
    add('아래 표는 읽기 쉽게 projection 유형별 열로 정리했다. allocation 계산과 동률 처리에는 저장된 기존 module 순서를 사용했다.')
    add()
    table(['Block']+types+['단순 평균'], [
        [f'B{layer:02d}']+[f'{byname[f"block_{layer:02d}.{t}"]["assigned_sparsity"]:.0%}' for t in types]
        +[f'{s["by_layer"][str(layer)]["mean_nominal_sparsity"]:.2%}'] for layer in range(32)])
    add('## 7. 레이어·유형별 패턴과 해석의 한계')
    add()
    table(['레이어 구간', '배정 sparsity 단순 평균'], [
        [f'B{q*8:02d}–B{q*8+7:02d}', f'{value:.2%}'] for q,value in enumerate(s['quartile_mean_sparsity'])])
    add(f'레이어 인덱스와 sparsity의 Spearman 상관은 **{s["correlation_layer_sparsity"]["spearman"]:.4f}**다. '
        '깊어질수록 덜 가지치기하는, 즉 Earlier-Is-Sparser 경향이 뚜렷하다. 완전히 단조로운 레이어별 스케줄은 아니다.')
    add()
    table(['projection 유형', '배정 sparsity 평균', 'D_g(65%)의 유형 평균'], [
        [t, f'{s["by_type"][t]["mean_nominal_sparsity"]:.2%}', f'{m["by_type"][t]["damage"][3]["mean"]:.7g}'] for t in types])
    add('유형별로는 `ff_out`에 더 큰 sparsity, `attn_out`과 `v_proj`에 더 작은 sparsity가 배정됐다. '
        '`block_31.ff_proj` 같은 비싼 increment도 측정 결과에서 나타난 것이며 이름을 보고 별도 보호한 것이 아니다. '
        '동일 유형이라도 레이어별 최종 배정값은 위의 전체 표처럼 달라진다.')
    add()
    table(['비교', 'Spearman', 'Pearson'], [
        [label, f'{s[key]["spearman"]:.4f}', f'{s[key]["pearson"]:.4f}'] for label,key in [
            ['D_g(65%) vs 배정 density(1−sparsity)', 'correlation_D65_assigned_density'],
            ['다섯 원시 한계비용의 평균 vs 배정 sparsity', 'correlation_mean_marginal_assigned_sparsity'],
            ['parameter당 다섯 한계비용의 평균 vs 배정 sparsity', 'correlation_mean_marginal_per_parameter_assigned_sparsity']]])
    add('마지막의 강한 음의 상관은 이 비용을 기준으로 allocation을 만든 결과이므로, 독립적인 predictor 검증으로 볼 수 없다. '
        '또한 단순 Earlier-Is-Sparser 스케줄과 같은 예산에서 직접 비교한 실험은 수행하지 않았다. '
        '따라서 이번 allocation이 그 단순 패턴을 넘어서는 고유한 이점을 갖는지, 기존 Layer Collapse 패턴과 얼마나 중복되는지는 아직 확정할 수 없다. '
        '새로운 레이어 배분 패턴이라는 신규성 주장은 약하다.')
    add()
    add('## 8. 단독 손상이 전체 sparse 모델에서도 유효했는가')
    add()
    add('allocation을 고정한 다음 모든 224개 projection을 동시에 가지치기했다. 단독 손상 곡선을 단순 합산해 전체 성능을 대신하지 않고, '
        '별도의 40개 held-out 상태에서 두 전체 모델을 직접 비교했다.')
    add()
    table(['지표', 'Uniform', 'Capacity'], [
        [label, f'{u[key]:.7g}', f'{c[key]:.7g}'] for label,key in [
            ['평균 masked-token KL(상태 동일 가중)', 'mean_kl'],
            ['masked-token KL 중앙값(pooled)', 'median_kl'],
            ['Top-1 agreement', 'top1_agreement'],
            ['Confidence MAE', 'confidence_mae'],
            ['mean max(상태 ΔLoss, 0)', 'positive_delta_loss']]])
    add(f'Capacity−Uniform의 상태별 평균 차이는 **{gate["mean_difference"]:.7f}**, 상태별 차이의 중앙값은 '
        f'**{gate["median_difference"]:.7f}**였다. paired state bootstrap 20,000회(seed 0)의 percentile 95% CI는 '
        f'**[{gate["bootstrap_95_ci"][0]:.7f}, {gate["bootstrap_95_ci"][1]:.7f}]**이다.')
    add()
    add(f'40개 상태 중 {gate["states_improved"]}개가 개선되고 {gate["states_worsened"]}개가 악화됐다. '
        f'시퀀스 평균은 {gate["sequence_means_improved"]}/8, timestep 평균은 {gate["timestep_means_improved"]}/5가 개선돼 세 가지 사전 기준을 모두 충족했다.')
    add()
    for key,label,field in [('per_sequence','시퀀스','sequence_index'),('per_timestep','timestep','timestep')]:
        table([label, 'Uniform KL', 'Capacity KL', '차이(C−U)'], [
            [r[field], f'{r["uniform_mean_kl"]:.6f}', f'{r["capacity_mean_kl"]:.6f}', f'{r["difference"]:.6f}'] for r in gate[key]])
    add('판정: **SUPPORTED**. 이 frozen rule과 예산에서 projection별 기능적 용량에 따른 배분이 held-out 전체 sparse 모델의 DLM fidelity를 개선했다. '
        '이는 현재 설정에서의 allocation 효용에 대한 판정이다. cheap DLM predictor를 개발하거나 다른 sparsity·grid로 결과를 일반화한 것은 아니다. '
        'CI의 재표집 단위는 요청대로 상태이며, 같은 시퀀스의 여러 timestep을 묶어 재표집한 sequence-cluster CI는 아니다.')
    add()
    add('## 9. Downstream 진행 상황')
    add()
    if downstream:
        for outcomes in downstream['evaluations']:
            table(['평가', 'Uniform', 'Capacity'], [[
                f'GSM8K {outcomes["uniform"]["limit"]}개',
                f'{outcomes["uniform"]["correct"]}/{outcomes["uniform"]["limit"]}',
                f'{outcomes["capacity"]["correct"]}/{outcomes["capacity"]["limit"]}']])
        add('mini-100은 동일한 고정 예제, 5-shot prompt, generation 설정, seed, strict EM 평가를 사용했다. '
            '12→20개 정답은 +8%p의 방향성 개선이다. 이 100개 결과에 대해 별도의 유의성을 주장하지 않는다.')
    status = read('status.json')
    if status['status'] != 'complete_and_audited':
        log = (ROOT / 'logs/run.log').read_text()
        progress = re.findall(r'Generating[^\r\n]*', log)
        add('작성 시점에는 전체 평가와 최종 audit이 아직 완료되지 않았다. 아래는 실행 로그의 최신 진행 표시이며, 전체 평가 정확도를 뜻하지 않는다.')
        if progress:
            add()
            add('```text\n' + progress[-1] + '\n```')
    else:
        add('전체 파이프라인과 최종 audit이 완료된 상태다.')
    add()
    add('## 10. 실행 중 발생한 문제와 보존된 근거')
    add()
    add('첫 projection 결과를 저장한 뒤 진행 로그 함수의 `name` 인자가 충돌해 초기 실행이 중단됐다. '
        '이 충돌을 재현하는 테스트를 추가하고 logger 인자 이름만 수정했다. 6개 테스트가 통과한 뒤 저장된 마스크·곡선에서 재개했다. '
        '실험 config나 손상 값, allocation 기준은 이 수정으로 바꾸지 않았다. 초기 오류 로그도 보존했다. '
        '현재 실행을 이 한국어 보고서 작성 때문에 재시작하거나 수정하지 않았다.')
    add()
    table(['근거', '파일'], [
        ['사전 고정 설정', '[config.json](config.json)'],
        ['상태·span 분리 검증', '[state_verification.json](state_verification.json)'],
        ['224×6 원시 곡선 및 상태별 진단', '[capacity_curves_raw.json](capacity_curves_raw.json)'],
        ['한계비용 분포·순위·비단조 구간·layer/type 통계', '[marginal_costs.json](marginal_costs.json)'],
        ['전체 선택 trace·skip·224개 배분', '[allocation.json](allocation.json)'],
        ['예산·분포·상관 통계', '[allocation_summary.json](allocation_summary.json)'],
        ['Uniform 마스크', '[uniform65_mask_manifest.json](uniform65_mask_manifest.json)'],
        ['Capacity 마스크', '[capacity65_mask_manifest.json](capacity65_mask_manifest.json)'],
        ['held-out paired 평가', '[heldout_dlm_results.json](heldout_dlm_results.json)'],
        ['완료된 downstream 결과', '[downstream.json](downstream.json)'],
        ['현재 실행 상태', '[status.json](status.json)'],
        ['초기 logging 오류', '[run_initial_logging_error.log](logs/run_initial_logging_error.log)'],
    ])
    target = ROOT / 'allocation_process_ko.md'
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(target)
    print(f'projection rows: {len(byname)}, selected increments: {len(a["trace"])}, negative increments: {len(m["monotonicity_violations"])}')


if __name__ == '__main__':
    main()
