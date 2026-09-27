# 기존 결과 감사 보고서

상태: `complete`

## 마스크와 held-out KL

| 방법 | 224 순서/shape | 선택/전체 파라미터 | mean KL |
|---|---:|---:|---:|
| uniform | 검증 | 4,536,008,704/6,979,321,856 | 0.562117 |
| capacity | 검증 | 4,536,008,704/6,979,321,856 | 0.452808 |
| reconstruction | 검증 | 4,536,008,704/6,979,321,856 | 0.631180 |
| eis_type | 검증 | 4,536,008,704/6,979,321,856 | 0.498892 |

## GSM8K strict exact match

lm-eval 0.4.8 GSM8K strict-match 정규화(쉼표/$/`#### ` 앞부분/마지막 마침표 제거, 대소문자 무시)를 `extracted_answer`와 `reference_answer`에 다시 적용했다. 재계산 EM이 저장된 `correct`와 전 행 일치하며, example identity와 evaluation config hash도 교차 검증했다.

| 방법 | mini100 | full1319 |
|---|---:|---:|
| uniform | 12/100 | 139/1319 |
| capacity | 20/100 | 250/1319 |
| reconstruction | 19/100 | 255/1319 |
| eis_type | 22/100 | 263/1319 |

## Paired exact McNemar

### n=100

- uniform_vs_capacity: discordant=18, p=0.096252441
- uniform_vs_reconstruction: discordant=15, p=0.11846924
- uniform_vs_eis_type: discordant=12, p=0.0063476562
- capacity_vs_reconstruction: discordant=17, p=1
- capacity_vs_eis_type: discordant=20, p=0.82380295
- reconstruction_vs_eis_type: discordant=15, p=0.60723877

### n=1319

- uniform_vs_capacity: discordant=223, p=5.1722769e-14
- uniform_vs_reconstruction: discordant=216, p=9.8741136e-16
- uniform_vs_eis_type: discordant=238, p=3.1580745e-16
- capacity_vs_reconstruction: discordant=227, p=0.79070279
- capacity_vs_eis_type: discordant=205, p=0.40202361
- reconstruction_vs_eis_type: discordant=236, p=0.64872604

## 해석 제한

- EIS+type은 oracle에서 유도한 multiset-sorted 설명용 대조군이며 공식 cheap EIS baseline이 아니다.
- Reconstruction의 KL이 더 나쁘다는 사실만으로 downstream 성능도 더 나쁘다고 결론낼 수 없다.
- 존재하지 않거나 미완료인 full 기록은 검증되었다고 표시하지 않았다.
