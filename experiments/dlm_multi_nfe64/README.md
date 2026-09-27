# Multi at 64 denoising steps

User-approved exploratory follow-up, 2026-09-27. Reuse the original physical Multi exact50% mask. Generate only Multi64 on the same previously exposed200 GSM8K questions. No new mask, calibration, diagnostic, or coefficient tuning.

## Contract

- Goal: compare primary Multi64−A64; secondary (Multi64−A64)−(Multi256−A256).
- Operation: execute; authority: create new experiment artifacts and update project Obsidian memory.
- Fixed: original revision/BF16/mask/surviving weights/native decoder/5shot/seeds/length256/block256/strict-match. Only Multi denoising steps change from256 to64.
- Sources: completed `dlm_pruning_nfe50` A64/A256 and frozen requests; `dlm_crosschain_control50` Multi mask/model and Multi256.
- Quality gates:600 imported answers regraded,200 prompt token hashes,224 packed masks,physical dense and Multi hashes,checkpoint coverage and final regrade,paired statistics tests,source/code freeze.
- Statistics: question-paired10,000 bootstrap draws,seed20260927; exact McNemar for primary. Secondary descriptive. Counts and absolute percentage-point differences reported.
- Boundaries:64 was selected after observing Dense/A; all200 questions already exposed. This is exploratory and does not establish a mechanism or general superiority. No automatic wider experiment follows.
- Open now: controller prepares/executes; GPT-6 Sol/high agent implements and checks analysis with separate file ownership. Parent reviews its statistics. Writing remains deferred.
- New generation cost:12,800 forward calls; reused controls115,200 historical calls. Failed/retried calls are recorded separately. No new validation generation planned because physical model hashes match archived weights and the native generation path is unchanged.
- Stop on source/mask/protocol mismatch or occupied assigned GPU. Preserve checkpoints and earlier experiments.

## Run

Prepare and seal on CPU using `bash experiments/dlm_multi_nfe64/run.sh prepare prepare`, then `prepare seal`. The launch audit must reference both hashes. Run `runner` under a dedicated tmux session with one idle `CUDA_VISIBLE_DEVICES`. Status is `output/progress/Multi.json`; final `output/report.json` and `report.md` are written only after full validation.
