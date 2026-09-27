# Multi at 32 denoising steps

User-approved exploratory follow-up, 2026-09-27. Same original physical Multi exact50% mask and exposed200 GSM8K questions as the completed Multi64 comparison.

## Contract

- Goal: primary Multi32−A32; secondary (Multi32−A32)−(Multi256−A256).
- Operation: execute; create new experiment artifacts and a new Obsidian experiment note.
- Fixed: LLaDA revision/BF16/mask/surviving weights/native generator/5shot/seeds/length256/block256/strict-match. Multi steps change to32.
- New generation:200answers,6,400forward calls. Reuse A32=15/200,A256=118/200,Multi256=122/200 after600 official regrades and source checks. Historical reused cost108,800calls.
- Statistics:10,000 question-paired bootstrap draws,seed20260927; exact McNemar primary. Secondary descriptive. Same200 questions retained regardless of results.
- Exploration: selected after Dense/A and Multi64 outcomes; previously exposed development data. Report all32/64/256 observations when interpreting results. Low32-step absolute scores can limit the comparison.
- Gates:600 source grades,200 tokenized prompts,224packed masks,actual dense/Multi physical hashes,exact budget,200sealed answer checkpoints,32forwards per successful answer,all attempts including failures,final official regrade and code/source receipts.
- CPU tests cover paired signs/counts,questioncoverage,failed-forwardcosts,noncanonicalduplicatefiles,andtamperedcheckpoints.
- Separate role: GPT-6 Sol/high independently examines the source diff and the user's100-vs200 question. Its artifacts live under research/multi_sample_audit_2026-09-27; no edits to frozen source experiments.
- Stop on identity/protocol mismatch or occupied assignedGPU; run in tmux dlm_multi_nfe32. No additional model conditions automatically launched.

## Outputs

Progress: output/progress/Multi.json. Final report.json/report.md are emitted only after all gates pass. Config,code_receipt,launch_audit preserve the actual setup. Prior experiments remain authoritative for their original conditions.
