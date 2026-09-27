# Multi / A-only / Uniform：별도 100문항 결과

분석일: 2026-09-27. 실행은 2026-09-26 22:52:37 KST에 완료됐다.

## 결과

| 방법 | 기존 개발용 100문항 | 별도 100문항 |
|---|---:|---:|
| Multi |63|59|
| 같은 multiscale bank의 A-only |60|58|
| 같은 계열의 native Uniform |54|52|

모두 동일한 기존 마스크와 원래 살아남은 가중치, 정확한 전체50% sparsity,5-shot/256생성토큰/256denoising steps/temperature0/strict-match를 사용했다. Uniform은 과거62점의 dense-calibration 모델과 다르다. A-only도 과거 legacy A가 아닌 같은 multiscale bank의 A 모델이다.

별도 표본은 9월22일에 해시 규칙으로 고정한100문항이다. 현재 Multi 개발에 사용한 ID0–99와 겹치지 않는다. 프로젝트 전체에서 과거에 전혀 본 적 없는 문항이라는 주장은 하지 않는다.

| 비교 | 새로 맞힘 | 잃음 | 순증 | exact McNemar p | 고정3비교 Holm p |
|---|---:|---:|---:|---:|---:|
| Multi−A |6|5|+1|1.0000|1.0000|
| Multi−Uniform |13|6|+7|0.1671|0.5012|
| A−Uniform |13|7|+6|0.2632|0.5264|

세 비교 모두 유의하지 않았다. Multi와 A의 정오가 달라진 문항은11개이고, Multi만 맞힌6개와 A만 맞힌5개가 거의 균형을 이룬다. 전체 strict-invalid는 Multi6개, A8개, Uniform5개다. 이 수만으로 모델 간 능력 차이의 원인을 단정할 수 없다.

## 해석과 결정

1. Multi와 A 모두 Uniform보다 높은 점수를 얻는 방향은 두 표본에서 유지됐다. Multi의 차이는+9→+7점, A의 차이는+6→+6점이다. 비균일 배분의 유용성에 관한 탐색 근거가 이어지지만, 별도100의 통계로 일반적 우위를 확정하지 않는다.
2. 핵심인 C의 추가 효과는 아직 입증되지 않았다. Multi−A 차이는 개발 표본+3점에서 별도 표본+1점으로 줄었다. 최종 점수 순서가 같다는 것만으로 multiscale response 보존의 필요성을 주장할 수 없다.
3. 같은 점수라는 동등성 결론도 성립하지 않는다. 현재100문항과11개 불일치 문항은 작은 효과를 판단하기에 제한적이다.
4. 기존 Multi는 후보로 유지하되, 결과를 성공적인 C 기여 입증으로 표현하지 않는다. 모델을 더 복잡하게 만드는 근거도 이번 결과에서 나오지 않는다. 새 실험은 자동 실행하지 않았다.

## 검증과 비용

원본 source validation과 세 완료 receipt를 새로 검증했다. 각 receipt는 마스크 파일, sparse-model identity,100개 문항 checkpoint와 결과 집계를 묶는다. 문항/프롬프트/정답/평가 hash의 모델 간 일치를 확인했고, exact McNemar와 Holm 값도 독립적으로 재계산해 저장 결과와 일치했다.

추가 probe나 calibration forward 없이300개 답변을 생성했다. 총76,800 forwards, 병렬 소요37.36분, 재시도나 미집계 attempt 없음.

원본: experiments/dlm_multi_validation10050/output/report.json.
파생 자료: summary.json, questions.csv, sources.json. 요약 재생성: summarize.py.
