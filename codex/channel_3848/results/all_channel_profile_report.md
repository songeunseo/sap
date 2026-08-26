# LLaDA-8B Dense 전 채널 Activation 측정

## 측정 설정

- 모델: `GSAI-ML/LLaDA-8B-Base`
- 데이터: WikiText2, seed 0의 고정 8개 sequence
- 범위: 32개 transformer block × 4096개 residual channel 전수
- 주 지표: 각 block의 `q_proj` 입력에서 sample·token 전체 `mean(abs(activation))`, 이후 32개 block 평균
- 보조 지표: 각 block 출력 residual의 동일 통계
- 채널 3848은 측정 또는 정렬 함수의 입력으로 사용하지 않음

## 전체 순위

| 측정 위치 | 1위 | 평균 절대 activation | 2위 | 평균 절대 activation | 1위/2위 |
|---|---:|---:|---:|---:|---:|
| `q_proj` 입력 | **3848** | **16.3971** | 753 | 6.3279 | **2.59×** |
| Post-block residual | **3848** | **91.4623** | 753 | 73.7790 | **1.24×** |

`q_proj` 입력 기준 상위 5개 채널은 `3848, 753, 653, 2374, 2471`이고,
post-block 기준 상위 5개는 `3848, 753, 1863, 2374, 653`이다.

## Block별 순위

- `q_proj` 입력: 채널 3848은 block 1–25에서 모두 1위, block 26–30에서 2위,
  block 31에서 5위였다. Block 0 입력에서는 3814위였으며, 첫 block을 지난 뒤
  outlier가 생성되는 흐름과 일치한다.
- Post-block residual: 채널 3848은 block 0–25에서 모두 1위, block 26–30에서
  2위, block 31에서 21위였다.
- 후기 block의 1위는 채널 753으로 전환됐다.

## 결론

논문이 제시한 채널 번호를 선택 기준으로 사용하지 않고 전 채널을 정렬해도,
채널 3848이 두 측정 위치 모두에서 전체 1위로 발견됐다. 따라서 **LLaDA-8B의
초·중반부에 지속되는 dominant activation channel이 3848이라는 관찰은 이번
환경에서도 독립적으로 재현됐다.** 이 결과는 activation magnitude의 재현이며,
채널의 과제별 인과 효과 자체를 증명하지는 않는다.

## 원본 산출물

- `dense-all-channel-profile.json`: 전체 집계값, 전체 순위, block별 전체 값과 순위
- `dense-all-channel-profile.csv`: 262,144개 channel-block-location 행
- `dense-all-channel-profile.log`: 실행 명령과 종료 코드
