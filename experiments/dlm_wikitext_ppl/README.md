# 기본 DLM WikiText likelihood 지표와 시간 측정 pilot

사용자 결정: 앞으로 기본 지표는 WikiText DLM likelihood의 PPL upper bound 추정치.
GSM8K는 자동 실행하지 않는다. 기존 실험과 DSA fitness는 소급 변경하지 않는다.

## 동결 설정

- LLaDA-8B-Base, revision 0f2787f2d87eac5eed8a087d5ecd24277e6255b2.
- Salesforce/wikitext, wikitext-2-raw-v1: validation 개발, test 최종 확인.
- top-level article 내 비중첩 512-token 비조건부 chunk, 짧은 tail 포함.
- special token 추가 없음, CFG 0, batch 1, BF16 forward / FP32 CE.
- chunk마다 k~Uniform{1,...,L}, k개 위치 비복원 균등 masking, MC128.
- Eq.6의 (L/k)*sum CE를 L로 나눈 token NELBO를 token 수로 가중평균.
  exp(NELBO)를 보고. finite MC 추정값 자체가 반드시 진짜 PPL보다 크다는 보장은 없음.
- seed2025, CPU RNG와 per-chunk seed+mask SHA256 동결. 모든 후보 동일 draws.
- article/chunk 중복 및 기존 DLM states·다른 split과 32-token 문자열을 공유하는 chunk 제외.
  필터는 점수에 무관하며 제외 내역 저장. 필터된 corpus 지표로 명시.

## 이번 실행 범위

완료된 sequential Uniform65 mask를 재사용해 validation의 8개 서로 다른 article에서
첫 full-length chunk를 MC128씩 평가. 총1024forward의 timing pilot일 뿐, 이 점수로
계수를 선택하거나 full validation/test 점수라고 하지 않는다. 전체 sweep 자동 실행 없음.
전체 validation/test protocol은 함께 준비하고 corpus/hash/mask draws를 저장한다.
128MC가 pruning candidate 순위에 충분히 안정적인지는 아직 별도 검증되지 않았다.

출처: LLaDA Eq.6/Algorithm3 https://arxiv.org/html/2502.09992v3
공식 get_log_likelihood.py https://github.com/ML-GSAI/LLaDA/blob/main/get_log_likelihood.py

테스트: 저장소 환경에서 python -m unittest experiments.dlm_wikitext_ppl.test_core.
실행: CUDA_VISIBLE_DEVICES=3 bash experiments/dlm_wikitext_ppl/run.sh (tmux 필수).
진행: python3 experiments/dlm_wikitext_ppl/status.py.
