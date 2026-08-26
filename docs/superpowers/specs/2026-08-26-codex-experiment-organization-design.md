# Codex 실험 산출물 정리 설계

## 목표

Codex가 수행한 모든 실험의 설정, 실행 스크립트, 설계 문서, 결과를 저장소 루트의 `codex/` 아래에 실험별로 모은다. 원본 파일을 복제하지 않고 이동하며, 공용 모델 구현과 테스트는 기존 위치에 유지한다.

## 대상

다음 여섯 실험 묶음을 정리한다.

1. `temporal_rho`: LLaDA temporal perturbation propagation phase1/phase2 실험
2. `channel_3848`: 채널 3848 및 비교 채널의 인과적 보호 실험
3. `time_risk_sensitivity`: Time-Risk DLM gradient-sensitivity 타당성 실험
4. `mean_dlm_sweep`: Mean-DLM의 GSM8K 및 WinoGrande sparsity sweep
5. `controlled_winogrande`: Dense, Wanda, SparseGPT, Sink-Aware의 동일 조건 WinoGrande 비교
6. `calibration_16x512`: Wanda, SparseGPT, Mean-DLM의 8×256 대비 16×512 calibration 비교

## 디렉터리 구조

```text
codex/
├── README.md
├── temporal_rho/
│   └── results/
├── channel_3848/
│   ├── configs/
│   ├── design.md
│   ├── plan.md
│   ├── pruned_weights/
│   └── results/
├── time_risk_sensitivity/
│   ├── config.json
│   ├── design.md
│   ├── plan.md
│   └── results/
├── mean_dlm_sweep/
│   ├── run_gsm8k.sh
│   ├── run_winogrande.sh
│   └── results/
├── controlled_winogrande/
│   ├── run.sh
│   └── results/
└── calibration_16x512/
    ├── config.json
    ├── run.sh
    └── results/
```

## 이동 매핑

| 기존 위치 | 새 위치 |
|---|---|
| 기본 worktree의 `results/phase1/`, `results/phase2/` | `codex/temporal_rho/results/phase1/`, `phase2/` |
| `experiments/channel_3848/` | `codex/channel_3848/configs/` |
| `results/channel_3848/` | `codex/channel_3848/results/` |
| 기본 worktree의 `pruned_weights/channel_3848/` | `codex/channel_3848/pruned_weights/` |
| channel 3848 설계·계획 문서 | `codex/channel_3848/design.md`, `plan.md` |
| `experiments/time_risk/pilot.json` | `codex/time_risk_sensitivity/config.json` |
| `results/time_risk/` | `codex/time_risk_sensitivity/results/` |
| Time-Risk 설계·계획 문서 | `codex/time_risk_sensitivity/design.md`, `plan.md` |
| Mean-DLM 실행 스크립트 2개 | `codex/mean_dlm_sweep/` |
| `results/mean_dlm/` | `codex/mean_dlm_sweep/results/` |
| controlled WinoGrande 실행 스크립트 | `codex/controlled_winogrande/run.sh` |
| `results/controlled_winogrande/` | `codex/controlled_winogrande/results/` |
| `experiments/time_risk/calibration_16x512.json` | `codex/calibration_16x512/config.json` |
| 16×512 실행 스크립트 | `codex/calibration_16x512/run.sh` |
| `results/calibration_16x512/` | `codex/calibration_16x512/results/` |

`dlm_gradient_sensitivity.py`, `lib/dlm_gradient_sensitivity.py`, `main_llada.py`, `eval_llada.py`와 테스트는 여러 실험이 공유하는 실행 구현이므로 루트의 기존 위치에 둔다.

## 경로와 재현성

실행 스크립트는 어느 디렉터리에서 호출해도 저장소 루트로 이동한 뒤 실행되도록 유지한다. 설정, 결과, artifact 경로는 모두 새 `codex/` 경로를 사용한다. Mean-DLM sweep은 Time-Risk sensitivity의 단일 공용 설정과 mask artifact를 참조하며 설정 파일을 복제하지 않는다.

루트 `.gitignore`의 `results/`와 `pruned_weights/` 규칙은 중첩된 같은 이름의 디렉터리에도 적용된다. 기존에 추적된 compact 결과와 평가 sample은 이동 후에도 계속 추적한다. 기존에 무시된 18GB raw mask/state, 로그, 기본 worktree의 temporal-rho 결과와 45GB channel checkpoint는 디스크상 이동하되 새로 Git에 추가하지 않는다.

## README 내용

`codex/README.md`는 한국어로 다음을 제공한다.

- 공통 모델, calibration dataset, 평가 설정과 용어
- 실험별 목적, 방법, 실행 상태 및 파일 위치
- JSON 원본에서 추출한 정확한 결과표
- Dense 및 논문 표와 비교할 때의 조건 차이
- standard error와 paired confidence interval을 포함한 통계적 해석
- OOM, 중단, 미실행 항목과 결과 해석의 한계
- 현재 경로를 사용하는 재현 명령

README는 결과를 복제 생성하지 않고 기존 JSON·Markdown·로그를 근거로 작성한다.

## 검증

1. 이동 전후 각 대상 디렉터리의 파일 수와 총 byte 수가 일치하는지 확인한다. 기본 worktree에만 존재하는 ignored 산출물은 fast-forward 직전에 별도로 측정하고 반영 후 다시 확인한다.
2. 기존 대상 경로가 실행 스크립트, 테스트, 이동된 문서에 남지 않았는지 `rg`로 확인한다.
3. 모든 JSON을 `jq empty`, shell runner를 `bash -n`으로 검사한다.
4. 관련 pytest를 실행한다. 기존에 확인된 외부 `Dataset` monkeypatch 실패는 이번 경로 이동과 분리해 보고한다.
5. README 표의 수치를 결과 JSON에서 다시 추출해 대조한다.
6. Git status와 diff를 검토해 사용자 소유의 무관한 변경이 포함되지 않았는지 확인한다.

## 커밋과 반영

구조 이동과 실행 경로 수정, 한국어 README를 각각 독립적인 커밋으로 남긴다. 검증 후 `exp/time-risk-dlm-gradient-sensitivity`를 기본 `outlier` worktree에 fast-forward하여 `/home/tmluser1/sap/codex`에서 결과를 볼 수 있게 한다. 기본 worktree의 기존 untracked `.agents/`와 `skills-lock.json`은 건드리지 않는다.
