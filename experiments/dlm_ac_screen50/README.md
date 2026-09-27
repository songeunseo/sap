# Ten-arm AC mini100 screen

Specification: `openspec/changes/parallel-ac-mini100-screen/design.md`.
Exact 50% budget (3,489,660,928 / 6,979,321,856 weights), original row-wise
native Wanda rankings, original surviving weights, batch-one native LLaDA.
No full GSM8K, confirmation, coefficient sweeps or candidate combinations.

The imported MS-A/Short/Path/All/Multi run retains its original files and hashes.
Legacy Uniform54/A55/AC61 are cached references. Historical Uniform62 uses a
different construction; layer-global Uniform is not the native row-wise reference.
The old `A` arm maps to `MS-A` only in this screen's display.

Square uses four branches, equal endpoint and edge means, and no additional
interaction penalty. Vector uses original 80 pairs, ALL emitted vocabulary logits,
FP32 raw teacher files and FP64 global-vocabulary centering/reduction. Pair means
are equally weighted. DKD is a different, deferred probability objective.
Exchange starts from the legacy AC mask, moves d=41/3d row quotas between two
blocks, and accepts only full-bank measured scalar AC improvements. Eight offers
per round, three rounds; epsilon is frozen from the first of two initial checks.
Fixed old-cost shortlists provide no global convergence guarantee.

Square: 160 teacher calibration +160 teacher diagnostic +160 Uniform reference
+10,240 probe +640 final-state evaluations =11,360 before generation.
Vector has the same nominal state count, different query/cache cost. Exchange
has <=4,160 calibration/search and <=320 diagnostics, plus160 teacher states
when not shared. Smoke adds exactly five forwards, capped at32.
All new GPU jobs have a receipt and actual forward counters. Source hashes cover
science and implementation; tasks.md progress checkboxes are not scientific inputs.

## Commands

All CPU commands hide CUDA and cap numerical threads to one. No 8B CPU load.

```bash
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/run.sh prepare
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/run.sh validate
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/run.sh launch --gpus 2,3 --dry-run
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/run.sh launch --gpus 2,3
watch -n 10 'bash /home/tmluser1/sap/experiments/dlm_ac_screen50/run.sh status'
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/status_with_history.sh
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/run.sh status --json
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/run.sh report
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/output/audits/control_receipt_resume_2026-09-26.sh --check
bash /home/tmluser1/sap/experiments/dlm_ac_screen50/output/audits/control_receipt_resume_2026-09-26.sh stop
```

Resume uses the same launch with unchanged scientific manifest. Status never polls
GPUs; ETA is stage-based and unknown before measurements. Allocation/import work
is CPU-only. Actual launch checks only assigned GPUs and never kills unrelated jobs.
Each GPU has at most one worker. tmux keeps workers alive across app/SSH closure.

`status_with_history.sh` prints the frozen live mini-100 status followed by the
completed historical 50% WikiText-2 NELBO/PPL receipts and the separately
constructed cached Uniform-Wanda mini-100 reference. These metrics have
different protocols. The live `run.sh`/`run.py` files are part of the frozen
source hashes, so the historical display remains a separate read-only command
while the screen runs.
During the receipt-amended run, use the audited `control_receipt_resume`
command to stop it; the frozen `run.sh stop` recognizes only its original
controller command line. If an interruption requires a restart, use
`output/audits/launch_receipt_resume_2026-09-26.sh`.

## First launch transition (2026-09-26)

While the new implementation was being prepared, the user-authorized legacy five
candidates were resumed in `ac_screen50_legacy_gpu3`. Short and Multi completed
61/100 and63/100. A controlled handoff stops only that owned bootstrap process
and its worker, retains every committed document, and releases both leases before
the audited unified scheduler launches. Normal scheduling rotates ready families;
there is no requirement that all five legacy evaluations finish before a new family.
`--after-legacy` is retained only as a deprecated CLI compatibility flag; the
controller uses the normal DAG and requires the old owner to have released its leases.
The controller and its owned worker process groups execute in one dedicated tmux
session; individual workers do not require separate tmux sessions.

Obsidian experiment note was created and verified before stage one:
`Research/DLM-Pruning/Experiments/2026-09-26 AC Mini100 Ten-Arm Screen GPU3.md`.
The wrapper's `legacy_resume.json` and screen `execution.json` track actual stages.
Completion/failure records use `obsidian_sync=pending` until MCP reconciliation;
a successful remote write is distinct from its verification result.

## Interpretation

Seven fixed within-family paired comparisons; Holm only after all are complete.
Repeated mini100 remains development even after correction. Partial results use
common IDs and never extrapolate. Cross-family results are end-to-end comparisons;
raw scalar/vector losses have different units. Path has unequal endpoint degrees.
The old all-pair conditional flip statistic is not the theorem's unconditional
Multi-edge event. Square groups do not identify semantic circuits. An Exchange
win does not establish the necessity of C under search. Negative results are valid.

## Prelaunch audit amendment

`output/audits/prelaunch_revision_1/` preserves the original implementation and
preparation manifest. No new-family model work was performed with that revision.
The amended manifest additionally anchors its own hash, exact checkpoint shard
hashes and all224 projection keys, the legacy dense teacher, explicit objective
contracts and controls. Completed old results and old source/config files are
unchanged. New readouts/metrics/probes and exchange decisions have content hashes;
restart verifies output receipts as well as configuration identity.

Both full-vocabulary teacher banks require about18.824GB decimal on disk; the
largest streamed pair reduction estimates747.225MB RAM, plus1GiB headroom checked
before production. Each producer reserves its remaining bytes with filesystem
`posix_fallocate` before its first forward. Allocated blocks count against free
space for concurrent producers. No top-k or hidden-state fallback is permitted.
Actual peak RSS/CUDA and file bytes are recorded separately from these estimates.
