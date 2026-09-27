# 향후 DLM pruning 기본 성능지표 결정

## Decision
2026-09-15 사용자 지시: 향후 신규 실험의 기본 성능지표는 공식 likelihood 기반 WikiText perplexity upper bound로 한다. GSM8K는 추가적인 task capability 확인에 사용한다. 기존 실행 중인 실험의 동결된 목적함수와 평가 기준은 소급 변경하지 않는다.

## Definition
LLaDA 공식 get_log_likelihood.py 및 논문의 likelihood bound를 따른다. 정확한 joint-likelihood PPL이 아닌 Monte Carlo로 추정한 PPL upper bound임을 명시한다. 고정 timestep 평균 masked CE, pseudo-perplexity, 외부 모델 generative PPL과 혼동하지 않는다. 최적화/통계 비교에는 token-normalized negative ELBO를 사용하고 표에는 그 지수인 PPL bound를 함께 보고할 수 있다(순위 동일).

## Setup still to freeze
WikiText version/split, unconditional vs prefix-conditioned block evaluation, context/target length, corpus processing, special tokens, Monte Carlo sample count, masking seeds, batch shape. 새 실험 전에 이 조건을 동결하고 calibration/search/evaluation 문장을 분리한다. 방법 간 같은 평가 문장과 동일한 Monte Carlo masking draws를 사용한다. 아직 구체적인 새 평가 프로토콜을 구현하거나 실행한 것은 아니다.

## Runtime / interpretation
전체 생성 trajectory가 필요하지 않아 GSM8K 생성 평가보다 빠를 것으로 예상하나, corpus 규모/context/MC 수에 좌우된다. 실제 실행 시간/ETA 또는 downstream 예측력은 아직 측정되지 않았다. 첫 diagnostic에서 시간을 확인한다. PPL 개선을 GSM8K 개선으로 자동 해석하지 않는다.

## Memory
이 턴에는 Obsidian 도구가 노출되지 않아 Research-State 조회/업데이트 불가. 대화에서 사용자가 명시한 결정만 로컬에 기록했으며, 연결 복구 시 연구 노트에 동기화한다.

## Sources
- https://github.com/ML-GSAI/LLaDA/blob/main/get_log_likelihood.py
- https://arxiv.org/html/2502.09992v3
- https://arxiv.org/html/2406.04329v2
